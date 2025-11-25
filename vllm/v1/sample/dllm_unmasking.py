# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DLLM unmasking processor for iterative token refinement."""

import torch

from vllm.v1.sample.dllm_constants import DLLMMaskState, DLLMUnmaskingStrategy


class DLLMUnmaskingProcessor:
    """Processor for unmasking tokens during DLLM generation.

    Implements three strategies:
    1. SEQUENTIAL: Unmask tokens left-to-right sequentially
    2. LOW_CONFIDENCE_DYNAMIC: Unmask tokens with confidence below threshold
    3. LOW_CONFIDENCE_STATIC: Unmask top-K lowest confidence tokens per block
    """

    def __init__(
        self,
        block_size: int,
        denoising_steps: int,
        confidence_threshold: float = 0.85,
    ):
        """Initialize the unmasking processor.

        Args:
            block_size: Size of each generation block
            denoising_steps: Number of denoising iterations
            confidence_threshold: Threshold for dynamic unmasking
        """
        self.block_size = block_size
        self.denoising_steps = denoising_steps
        self.confidence_threshold = confidence_threshold

    def _get_scores(
        self, logits: torch.Tensor, token_ids: torch.Tensor
    ) -> torch.Tensor:
        """Get probability scores for sampled tokens.

        Args:
            logits: Model logits [batch_size * block_size, vocab_size]
            token_ids: Sampled token IDs [batch_size * block_size]

        Returns:
            Probability scores for each token [batch_size * block_size]
        """
        scores = logits.softmax(dim=-1)
        scores = scores.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
        return scores

    def _get_denoise_num(self) -> int:
        """Get number of tokens to unmask per iteration."""
        num = self.block_size // self.denoising_steps
        num = max(1, min(num, self.block_size))
        return num

    def low_confidence_static(
        self,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
        dllm_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Unmask top-K lowest confidence tokens per block.

        Selects the K tokens with lowest confidence scores among masked
        positions and unmasks them.

        Args:
            logits: Model logits
            token_ids: Sampled token IDs
            dllm_mask: Current mask state

        Returns:
            Updated mask tensor
        """
        topk = self._get_denoise_num()
        scores = self._get_scores(logits, token_ids)

        # Only consider masked tokens
        is_masked = dllm_mask == DLLMMaskState.MASKED
        scores = torch.where(is_masked, scores, torch.zeros_like(scores))

        # Reshape to [batch_size, block_size]
        scores = scores.view(-1, self.block_size)
        dllm_mask = dllm_mask.view(-1, self.block_size)

        # Get top-K lowest confidence (highest scores mask)
        _, indices = scores.topk(topk, dim=-1)

        # Unmask selected tokens
        dllm_unmasked = dllm_mask.scatter(
            -1, indices, DLLMMaskState.UNMASKED
        )

        # Only update masked positions
        is_masked = is_masked.view_as(dllm_mask)
        dllm_mask = torch.where(is_masked, dllm_unmasked, dllm_mask)

        return dllm_mask.flatten()

    def low_confidence_dynamic(
        self,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
        dllm_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Unmask tokens with confidence score above threshold.

        Unmasks all masked tokens whose confidence exceeds the threshold.

        Args:
            logits: Model logits
            token_ids: Sampled token IDs
            dllm_mask: Current mask state

        Returns:
            Updated mask tensor
        """
        threshold = self.confidence_threshold
        scores = self._get_scores(logits, token_ids)

        # Only consider masked tokens
        is_masked = dllm_mask == DLLMMaskState.MASKED
        scores = torch.where(is_masked, scores, torch.zeros_like(scores))

        # Reshape to [batch_size, block_size]
        scores = scores.view(-1, self.block_size)
        dllm_mask = dllm_mask.view(-1, self.block_size)

        # Find highest confidence token per block
        _, indices = scores.topk(1, dim=-1)
        scores = scores.scatter(-1, indices, threshold)

        # Unmask tokens above threshold
        is_masked = is_masked.view_as(dllm_mask)
        is_masked &= scores >= threshold
        dllm_mask = torch.where(
            is_masked, DLLMMaskState.UNMASKED, dllm_mask
        )

        return dllm_mask.flatten()

    def sequential(self, dllm_mask: torch.Tensor) -> torch.Tensor:
        """Unmask tokens sequentially from left to right.

        Args:
            dllm_mask: Current mask state

        Returns:
            Updated mask tensor
        """
        denoise_num = self._get_denoise_num()
        dllm_mask = dllm_mask.view(-1, self.block_size)
        is_masked = dllm_mask == DLLMMaskState.MASKED

        # Get first masked position in each block
        indices = is_masked.int().argmax(dim=1)

        # Create range for sequential unmasking
        ranges = torch.arange(
            0, denoise_num, device=indices.device, dtype=indices.dtype
        )
        indices = (indices[:, None] + ranges[None, :]) % self.block_size

        # Unmask sequential tokens
        dllm_unmasked = dllm_mask.clone()
        dllm_unmasked = dllm_unmasked.scatter(
            -1, indices, DLLMMaskState.UNMASKED
        )
        dllm_mask = torch.where(is_masked, dllm_unmasked, dllm_mask)

        return dllm_mask.flatten()

    def __call__(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        token_ids: torch.Tensor,
        dllm_mask: torch.Tensor,
        strategy: DLLMUnmaskingStrategy,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply unmasking strategy.

        Args:
            logits: Model logits [batch_size * block_size, vocab_size]
            input_ids: Input token IDs [batch_size * block_size]
            token_ids: Newly sampled token IDs [batch_size * block_size]
            dllm_mask: Current mask state [batch_size * block_size]
            strategy: Unmasking strategy to use

        Returns:
            Tuple of (updated_mask, updated_token_ids)
        """
        # Reshape to [num_blocks, block_size]
        dllm_mask = dllm_mask.unflatten(0, (-1, self.block_size))

        # Check for fully unmasked blocks
        is_same = (dllm_mask == dllm_mask[:, :1]).all(dim=1)
        first_mask = dllm_mask[:, 0]

        # Mark fully unmasked blocks as cached
        is_block_unmasked = is_same & (
            first_mask == DLLMMaskState.UNMASKED
        )
        dllm_mask[is_block_unmasked] = DLLMMaskState.CACHED

        # Flatten back
        dllm_mask = dllm_mask.flatten()

        # Update token_ids: use input_ids for non-masked positions
        token_ids = torch.where(
            dllm_mask != DLLMMaskState.MASKED, input_ids, token_ids
        )

        # Apply unmasking strategy
        if strategy == DLLMUnmaskingStrategy.LOW_CONFIDENCE_STATIC:
            dllm_mask = self.low_confidence_static(
                logits, token_ids, dllm_mask
            )
        elif strategy == DLLMUnmaskingStrategy.LOW_CONFIDENCE_DYNAMIC:
            dllm_mask = self.low_confidence_dynamic(
                logits, token_ids, dllm_mask
            )
        elif strategy == DLLMUnmaskingStrategy.SEQUENTIAL:
            dllm_mask = self.sequential(dllm_mask)
        else:
            raise RuntimeError(f"Unknown strategy: {strategy}")

        return dllm_mask, token_ids
