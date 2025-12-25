# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DLLM generation logic for autoregressive models."""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.v1.request import Request


class DLLMGenerator:
    """Handles DLLM block-based generation for AR models."""

    def __init__(self):
        """Initialize DLLM generator."""
        pass

    def prepare_dllm_input(
        self,
        request: "Request",
        current_tokens: list[int],
    ) -> list[int]:
        """Prepare input tokens for DLLM generation.

        For DLLM, we need to include the current block being refined,
        with masked positions filled with the mask token.

        Args:
            request: The DLLM request
            current_tokens: Current token sequence (prompt + generated)

        Returns:
            Token IDs including masked positions for current block
        """
        if not request.is_dllm:
            return current_tokens

        # Get current mask state
        mask = request.dllm_get_mask()
        if not mask:
            # First block - initialize
            block_tokens = [request.dllm_mask_token_id] * request.dllm_block_size
            request.dllm_init_new_block(block_tokens)
            mask = request.dllm_get_mask()

        from vllm.v1.sample.dllm_constants import DLLMMaskState

        # Build input: prompt + generated + current_block
        # Current block includes masked positions
        block_start = max(0, len(mask) - request.dllm_block_size)
        block_mask = mask[block_start:]

        # For each position in current block:
        # - If CACHED: use the actual token (already in current_tokens)
        # - If UNMASKED: use the actual token (already in current_tokens)
        # - If MASKED: use mask_token_id

        input_tokens = current_tokens.copy()

        # Add current block with masks
        block_tokens = []
        for i, mask_state in enumerate(block_mask):
            if mask_state == DLLMMaskState.MASKED:
                block_tokens.append(request.dllm_mask_token_id)
            else:
                # Token should be in output already
                token_idx = block_start + i
                if token_idx < len(request._all_token_ids):
                    block_tokens.append(request._all_token_ids[token_idx])
                else:
                    block_tokens.append(request.dllm_mask_token_id)

        # Only add the current block being refined
        # (previous blocks are already in current_tokens as CACHED)
        return input_tokens + block_tokens

    def process_dllm_output(
        self,
        request: "Request",
        sampled_token_ids: list[int],
        logits: torch.Tensor | None,
    ) -> tuple[list[int], list[int], bool]:
        """Process DLLM output and apply unmasking.

        Args:
            request: The DLLM request
            sampled_token_ids: Tokens sampled from the model
            logits: Model logits for confidence-based unmasking

        Returns:
            Tuple of (tokens_to_keep, new_mask, block_complete)
            - tokens_to_keep: Tokens to append to output (only unmasked)
            - new_mask: Updated mask state for current block
            - block_complete: Whether current block is finished
        """
        if not request.is_dllm:
            return sampled_token_ids, [], False

        from vllm.v1.sample.dllm_constants import DLLMMaskState
        from vllm.v1.sample.dllm_unmasking import DLLMUnmaskingProcessor

        # Get current mask state
        current_mask = request.dllm_get_mask()
        block_size = request.dllm_block_size
        block_start = max(0, len(current_mask) - block_size)
        block_mask = current_mask[block_start:]

        # Convert sampled_token_ids to match block
        # We only care about the last block_size tokens
        if len(sampled_token_ids) > block_size:
            block_tokens = sampled_token_ids[-block_size:]
        else:
            block_tokens = sampled_token_ids

        # Apply unmasking strategy
        processor = DLLMUnmaskingProcessor()

        # Get input tokens for the block (for context)
        block_input_ids = []
        for i in range(block_size):
            token_idx = block_start + i
            if token_idx < len(request._all_token_ids):
                block_input_ids.append(request._all_token_ids[token_idx])
            else:
                block_input_ids.append(request.dllm_mask_token_id)

        # Apply unmasking
        new_tokens, new_mask = processor(
            logits=logits,
            input_ids=block_input_ids,
            token_ids=block_tokens,
            dllm_mask=block_mask,
            dllm_strategy=request.dllm_unmasking_strategy,
            dllm_confidence_threshold=request.dllm_confidence_threshold,
        )

        # Update request mask state (THIS POPULATES DECODING ORDER)
        full_new_mask = current_mask[:block_start] + new_mask
        request.dllm_update_mask(full_new_mask)

        # Extract only newly unmasked tokens
        tokens_to_append = []
        for i, (new_token, mask_state) in enumerate(zip(new_tokens, new_mask)):
            if mask_state == DLLMMaskState.UNMASKED:
                tokens_to_append.append(new_token)

        # Check if block is complete
        block_complete = request.dllm_is_current_block_complete()

        if block_complete:
            # Prepare for next block
            request.dllm_advance_to_next_block()
            # Initialize next block if not done
            if request.dllm_num_valid_tokens < request.max_tokens:
                next_block = [request.dllm_mask_token_id] * block_size
                request.dllm_init_new_block(next_block)

        return tokens_to_append, new_mask, block_complete

    def should_continue_block_refinement(self, request: "Request") -> bool:
        """Check if we should continue refining the current block.

        Args:
            request: The DLLM request

        Returns:
            True if should continue refining, False if should advance
        """
        if not request.is_dllm:
            return False

        # Check iteration count
        if request.dllm_current_iteration >= request.dllm_denoising_steps:
            return False

        # Check if block is complete
        if request.dllm_is_current_block_complete():
            return False

        return True
