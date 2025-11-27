# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
import time
from collections.abc import Callable, Mapping
from functools import partial
from typing import TYPE_CHECKING, Any, Optional

import torch

from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.v1.engine import (
    EngineCoreEvent,
    EngineCoreEventType,
    EngineCoreRequest,
    FinishReason,
)
from vllm.v1.structured_output.request import StructuredOutputRequest
from vllm.v1.utils import ConstantList

if TYPE_CHECKING:
    from vllm.lora.request import LoRARequest
    from vllm.v1.core.kv_cache_utils import BlockHash


class Request:
    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int] | None,
        sampling_params: SamplingParams | None,
        pooling_params: PoolingParams | None,
        eos_token_id: int | None,
        client_index: int = 0,
        arrival_time: float | None = None,
        prompt_embeds: torch.Tensor | None = None,
        mm_features: list[MultiModalFeatureSpec] | None = None,
        lora_request: Optional["LoRARequest"] = None,
        cache_salt: str | None = None,
        priority: int = 0,
        trace_headers: Mapping[str, str] | None = None,
        block_hasher: Callable[["Request"], list["BlockHash"]] | None = None,
    ) -> None:
        self.request_id = request_id
        self.client_index = client_index
        self.priority = priority
        self.sampling_params = sampling_params
        self.pooling_params = pooling_params
        # Because of LoRA, the eos token id can be different for each request.
        self.eos_token_id = eos_token_id
        self.lora_request = lora_request
        self.structured_output_request = StructuredOutputRequest.from_sampling_params(
            sampling_params
        )
        self.arrival_time = arrival_time if arrival_time is not None else time.time()

        self.status = RequestStatus.WAITING
        self.events: list[EngineCoreEvent] = []
        self.stop_reason: int | str | None = None

        # P/D: Connector-specific KV transfer parameters.
        self.kv_transfer_params: dict[str, Any] | None = None

        if pooling_params is not None:
            # Pooling models.
            self.max_tokens = 1
        elif sampling_params is not None:
            # Generative models.
            assert sampling_params.max_tokens is not None
            self.max_tokens = sampling_params.max_tokens
            if self.structured_output_request is not None:
                self.status = RequestStatus.WAITING_FOR_FSM

            if sampling_params.extra_args is not None:
                self.kv_transfer_params = sampling_params.extra_args.get(
                    "kv_transfer_params"
                )
        else:
            raise ValueError("sampling_params and pooling_params can't both be unset")

        self.prompt_token_ids = prompt_token_ids
        self.prompt_embeds = prompt_embeds
        self.num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
            prompt_token_ids, prompt_embeds
        )
        self._output_token_ids: list[int] = []
        self._all_token_ids: list[int] = (
            self.prompt_token_ids.copy()
            if self.prompt_token_ids is not None
            else [0] * self.num_prompt_tokens
        )
        self.num_output_placeholders = 0  # Used in async scheduling.
        self.spec_token_ids: list[int] = []
        self.num_computed_tokens = 0
        self.cache_salt: str | None = cache_salt

        # Multi-modal related
        self.mm_features = mm_features or []
        self.num_encoder_inputs = len(self.mm_features)
        self.has_encoder_inputs = self.num_encoder_inputs > 0

        # Read-only views
        # Prevent directly appending to these lists since
        # they should also be updated simultaneously.
        self.output_token_ids = ConstantList(self._output_token_ids)
        self.all_token_ids = ConstantList(self._all_token_ids)
        # trace_headers
        self.trace_headers = trace_headers
        # State
        # The number of tokens with prefix cache hits.
        self.num_cached_tokens = -1

        # The number of NaNs in logits. A value greater than 0
        # indicates that the output is corrupted
        self.num_nans_in_logits = 0

        # The number of requests being preempted by the scheduler
        self.num_preemptions = 0

        # The number of tokens that have been computed remotely.
        self.num_external_computed_tokens = 0

        self.block_hashes: list[BlockHash] = []
        self.get_hash_new_full_blocks: Callable[[], list[BlockHash]] | None = None
        if block_hasher is not None:
            self.get_hash_new_full_blocks = partial(block_hasher, self)
            self.block_hashes = self.get_hash_new_full_blocks()

        self.skip_reading_prefix_cache = self.get_skip_reading_prefix_cache()

        # DLLM-specific fields
        self.is_dllm = (
            sampling_params is not None and sampling_params.dllm_enabled
        )
        if self.is_dllm:
            from vllm.v1.sample.dllm_constants import DLLMMaskState

            self.dllm_block_size = sampling_params.dllm_block_size
            self.dllm_denoising_steps = (
                sampling_params.dllm_denoising_steps
                if sampling_params.dllm_denoising_steps is not None
                else self.dllm_block_size
            )
            self.dllm_unmasking_strategy = sampling_params.dllm_unmasking_strategy
            self.dllm_confidence_threshold = sampling_params.dllm_confidence_threshold
            self.dllm_mask_token_id = sampling_params.dllm_mask_token_id

            # Track mask state for each token (MASKED/UNMASKED/CACHED)
            self._dllm_mask: list[int] = []
            # Number of tokens that are unmasked (confirmed)
            self._num_valid_tokens = 0
            # Current denoising iteration within the block
            self._dllm_iteration = 0
            # Decoding order: iteration when each token was unmasked
            # -1 = not yet unmasked, 0+ = iteration number
            self._dllm_decoding_order: list[int] = []
        else:
            self.dllm_block_size = 0
            self.dllm_denoising_steps = 0
            self.dllm_unmasking_strategy = ""
            self.dllm_confidence_threshold = 0.0
            self.dllm_mask_token_id = None
            self._dllm_mask = []
            self._num_valid_tokens = 0
            self._dllm_iteration = 0
            self._dllm_decoding_order = []

    @classmethod
    def from_engine_core_request(
        cls,
        request: EngineCoreRequest,
        block_hasher: Callable[["Request"], list["BlockHash"]] | None,
    ) -> "Request":
        return cls(
            request_id=request.request_id,
            client_index=request.client_index,
            prompt_token_ids=request.prompt_token_ids,
            prompt_embeds=request.prompt_embeds,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            eos_token_id=request.eos_token_id,
            arrival_time=request.arrival_time,
            lora_request=request.lora_request,
            cache_salt=request.cache_salt,
            priority=request.priority,
            trace_headers=request.trace_headers,
            block_hasher=block_hasher,
        )

    def append_output_token_ids(
        self,
        token_ids: int | list[int],
    ) -> None:
        if isinstance(token_ids, int):
            self._output_token_ids.append(token_ids)
            self._all_token_ids.append(token_ids)
        else:
            self._output_token_ids.extend(token_ids)
            self._all_token_ids.extend(token_ids)

        if self.get_hash_new_full_blocks is not None:
            self.block_hashes.extend(self.get_hash_new_full_blocks())

    @property
    def use_structured_output(self) -> bool:
        return self.structured_output_request is not None

    @property
    def num_tokens(self) -> int:
        return len(self._all_token_ids)

    @property
    def num_tokens_with_spec(self) -> int:
        return len(self._all_token_ids) + len(self.spec_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self._output_token_ids)

    def get_skip_reading_prefix_cache(self) -> bool:
        if (
            self.sampling_params is not None
            and self.sampling_params.skip_reading_prefix_cache is not None
        ):
            return self.sampling_params.skip_reading_prefix_cache
        elif (
            self.pooling_params is not None
            and self.pooling_params.skip_reading_prefix_cache is not None
        ):
            return self.pooling_params.skip_reading_prefix_cache
        return False

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def get_finished_reason(self) -> FinishReason | None:
        return RequestStatus.get_finished_reason(self.status)

    def get_num_encoder_tokens(self, input_id: int) -> int:
        assert input_id < len(self.mm_features)
        num_tokens = self.mm_features[input_id].mm_position.length
        return num_tokens

    def record_event(
        self,
        event_type: EngineCoreEventType,
        timestamp: float | None = None,
    ) -> None:
        self.events.append(EngineCoreEvent.new_event(event_type, timestamp))

    def take_events(self) -> list[EngineCoreEvent] | None:
        if not self.events:
            return None
        events, self.events = self.events, []
        return events

    # DLLM-specific methods
    def dllm_get_mask(self) -> list[int]:
        """Get the current DLLM mask for output tokens."""
        if not self.is_dllm:
            return []
        return self._dllm_mask.copy()

    def dllm_update_mask(self, new_mask: list[int]) -> None:
        """Update the DLLM mask for the current block."""
        if not self.is_dllm:
            return
        from vllm.v1.sample.dllm_constants import DLLMMaskState

        # Track which tokens became unmasked in this iteration
        old_mask = self._dllm_mask.copy() if self._dllm_mask else []

        # Update mask
        self._dllm_mask = new_mask.copy()

        # Ensure decoding_order has same length as mask
        while len(self._dllm_decoding_order) < len(new_mask):
            self._dllm_decoding_order.append(-1)

        # Update decoding order for newly unmasked tokens
        block_start = max(0, len(self._dllm_mask) - self.dllm_block_size)
        for i in range(block_start, len(new_mask)):
            # Check if token became unmasked in this iteration
            old_state = old_mask[i] if i < len(old_mask) else DLLMMaskState.MASKED
            new_state = new_mask[i]

            if (old_state == DLLMMaskState.MASKED and
                new_state == DLLMMaskState.UNMASKED and
                self._dllm_decoding_order[i] == -1):
                # Token was just unmasked, record the iteration
                self._dllm_decoding_order[i] = self._dllm_iteration

        # Update num_valid_tokens
        # Count unmasked tokens in current block
        block_mask = self._dllm_mask[block_start:]
        num_unmasked = sum(
            1 for m in block_mask if m == DLLMMaskState.UNMASKED
        )

        # Update iteration counter
        self._dllm_iteration += 1

    def dllm_is_current_block_complete(self) -> bool:
        """Check if the current block is fully unmasked."""
        if not self.is_dllm or not self._dllm_mask:
            return False
        from vllm.v1.sample.dllm_constants import DLLMMaskState

        # Check last block
        block_start = max(0, len(self._dllm_mask) - self.dllm_block_size)
        block_mask = self._dllm_mask[block_start:]

        # Block is complete if all tokens are unmasked
        return all(m == DLLMMaskState.UNMASKED for m in block_mask)

    def dllm_advance_to_next_block(self) -> None:
        """Mark current block as cached and prepare for next block."""
        if not self.is_dllm:
            return
        from vllm.v1.sample.dllm_constants import DLLMMaskState

        # Mark current block as cached
        block_start = max(0, len(self._dllm_mask) - self.dllm_block_size)
        for i in range(block_start, len(self._dllm_mask)):
            self._dllm_mask[i] = DLLMMaskState.CACHED

        # Update valid tokens
        self._num_valid_tokens += self.dllm_block_size

        # Reset iteration counter
        self._dllm_iteration = 0

    def dllm_init_new_block(self, token_ids: list[int]) -> None:
        """Initialize a new masked block."""
        if not self.is_dllm:
            return
        from vllm.v1.sample.dllm_constants import DLLMMaskState

        # Pad to block size if needed
        num_tokens = len(token_ids)
        if num_tokens < self.dllm_block_size:
            # Pad with masked tokens
            padding = [self.dllm_mask_token_id] * (
                self.dllm_block_size - num_tokens
            )
            token_ids.extend(padding)

        # Initialize all tokens as masked except possibly the first
        new_mask = [DLLMMaskState.MASKED] * self.dllm_block_size
        # First token can be unmasked for sequential generation
        if num_tokens > 0:
            new_mask[0] = DLLMMaskState.UNMASKED

        self._dllm_mask.extend(new_mask)

        # Initialize decoding order for new block
        # If first token is unmasked, mark it as iteration 0
        new_order = [-1] * self.dllm_block_size
        if num_tokens > 0:
            new_order[0] = 0
        self._dllm_decoding_order.extend(new_order)

        self._dllm_iteration = 0

    @property
    def dllm_num_valid_tokens(self) -> int:
        """Get number of valid (unmasked) tokens."""
        return self._num_valid_tokens if self.is_dllm else self.num_output_tokens

    @property
    def dllm_current_iteration(self) -> int:
        """Get current denoising iteration."""
        return self._dllm_iteration if self.is_dllm else 0

    def dllm_get_decoding_order(self) -> list[int]:
        """Get the decoding order for all tokens."""
        if not self.is_dllm:
            return []
        return self._dllm_decoding_order.copy()


class RequestStatus(enum.IntEnum):
    """Status of a request."""

    WAITING = enum.auto()
    WAITING_FOR_FSM = enum.auto()
    WAITING_FOR_REMOTE_KVS = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    # Note: anything after PREEMPTED will be considered
    # as a finished status.
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()

    def __str__(self):
        return self.name

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status > RequestStatus.PREEMPTED

    @staticmethod
    def get_finished_reason(status: "RequestStatus") -> FinishReason | None:
        return _FINISHED_REASON_MAP.get(status)


# Mapping of finished statuses to their finish reasons.
# NOTE: The ignored requests are the requests whose prompt lengths
# are longer than the model's length cap. Therefore, the stop
# reason should also be "length" as in OpenAI API.
_FINISHED_REASON_MAP = {
    RequestStatus.FINISHED_STOPPED: FinishReason.STOP,
    RequestStatus.FINISHED_LENGTH_CAPPED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ABORTED: FinishReason.ABORT,
    RequestStatus.FINISHED_IGNORED: FinishReason.LENGTH,
}
