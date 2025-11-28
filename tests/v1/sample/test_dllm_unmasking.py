# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for DLLM unmasking strategies."""

import pytest
import torch

from vllm.v1.sample.dllm_constants import DLLMMaskState, DLLMUnmaskingStrategy
from vllm.v1.sample.dllm_unmasking import DLLMUnmaskingProcessor


class TestDLLMConstants:
    """Test DLLM constants and enums."""

    def test_mask_states(self):
        """Test DLLMMaskState enum values."""
        assert DLLMMaskState.MASKED == 0
        assert DLLMMaskState.UNMASKED == 1
        assert DLLMMaskState.CACHED == 2

    def test_unmasking_strategy_from_str(self):
        """Test DLLMUnmaskingStrategy.from_str() conversion."""
        assert DLLMUnmaskingStrategy.from_str("SEQUENTIAL") == DLLMUnmaskingStrategy.SEQUENTIAL
        assert DLLMUnmaskingStrategy.from_str("LOW_CONFIDENCE_DYNAMIC") == DLLMUnmaskingStrategy.LOW_CONFIDENCE_DYNAMIC
        assert DLLMUnmaskingStrategy.from_str("LOW_CONFIDENCE_STATIC") == DLLMUnmaskingStrategy.LOW_CONFIDENCE_STATIC

        # Test case insensitivity
        assert DLLMUnmaskingStrategy.from_str("sequential") == DLLMUnmaskingStrategy.SEQUENTIAL

        # Test invalid strategy
        with pytest.raises(ValueError):
            DLLMUnmaskingStrategy.from_str("INVALID_STRATEGY")


class TestDLLMUnmaskingProcessor:
    """Test DLLMUnmaskingProcessor unmasking strategies."""

    def setup_method(self):
        """Set up test fixtures."""
        self.block_size = 4
        self.denoising_steps = 4
        self.processor = DLLMUnmaskingProcessor(
            block_size=self.block_size,
            denoising_steps=self.denoising_steps,
            confidence_threshold=0.85,
        )

    def test_sequential_unmasking(self):
        """Test sequential unmasking strategy."""
        # Start with all masked
        mask = torch.tensor([
            DLLMMaskState.MASKED,
            DLLMMaskState.MASKED,
            DLLMMaskState.MASKED,
            DLLMMaskState.MASKED,
        ], dtype=torch.long)

        # Apply sequential unmasking
        updated_mask = self.processor.sequential(mask)

        # Should unmask first token
        assert updated_mask[0] == DLLMMaskState.UNMASKED
        assert updated_mask[1] == DLLMMaskState.MASKED
        assert updated_mask[2] == DLLMMaskState.MASKED
        assert updated_mask[3] == DLLMMaskState.MASKED

    def test_sequential_unmasking_progressive(self):
        """Test sequential unmasking progresses left-to-right."""
        # Start with first token unmasked
        mask = torch.tensor([
            DLLMMaskState.UNMASKED,
            DLLMMaskState.MASKED,
            DLLMMaskState.MASKED,
            DLLMMaskState.MASKED,
        ], dtype=torch.long)

        # Apply sequential unmasking
        updated_mask = self.processor.sequential(mask)

        # Should unmask second token
        assert updated_mask[0] == DLLMMaskState.UNMASKED
        assert updated_mask[1] == DLLMMaskState.UNMASKED
        assert updated_mask[2] == DLLMMaskState.MASKED
        assert updated_mask[3] == DLLMMaskState.MASKED

    def test_low_confidence_dynamic(self):
        """Test low confidence dynamic unmasking strategy."""
        # Create logits with varying confidence
        vocab_size = 100
        logits = torch.randn(self.block_size, vocab_size)

        # Make first and third tokens high confidence
        logits[0, 50] = 10.0  # High logit for position 0, token 50
        logits[2, 30] = 10.0  # High logit for position 2, token 30

        tokens = torch.tensor([50, 20, 30, 40], dtype=torch.long)
        mask = torch.zeros(self.block_size, dtype=torch.long)  # All masked

        # Apply low confidence dynamic
        updated_mask, updated_tokens = self.processor.low_confidence_dynamic(
            logits, tokens, mask
        )

        # Tokens with high confidence should be unmasked
        assert updated_mask[0] == DLLMMaskState.UNMASKED
        assert updated_mask[2] == DLLMMaskState.UNMASKED

    def test_low_confidence_static(self):
        """Test low confidence static unmasking strategy."""
        vocab_size = 100
        logits = torch.randn(self.block_size, vocab_size)
        tokens = torch.randint(0, vocab_size, (self.block_size,))
        mask = torch.zeros(self.block_size, dtype=torch.long)  # All masked

        # Apply low confidence static
        updated_mask, updated_tokens = self.processor.low_confidence_static(
            logits, tokens, mask
        )

        # Should unmask k tokens (k = block_size / denoising_steps = 1)
        num_unmasked = (updated_mask == DLLMMaskState.UNMASKED).sum().item()
        expected_k = max(1, self.block_size // self.denoising_steps)
        assert num_unmasked == expected_k

    def test_processor_call_with_sequential(self):
        """Test processor __call__ with sequential strategy."""
        vocab_size = 100
        logits = torch.randn(self.block_size, vocab_size)
        input_ids = torch.randint(0, vocab_size, (10,))
        token_ids = torch.randint(0, vocab_size, (self.block_size,))
        mask = torch.zeros(self.block_size, dtype=torch.long)

        updated_mask, updated_tokens = self.processor(
            logits=logits,
            input_ids=input_ids,
            token_ids=token_ids,
            dllm_mask=mask,
            strategy=DLLMUnmaskingStrategy.SEQUENTIAL,
        )

        # Sequential should unmask first token
        assert updated_mask[0] == DLLMMaskState.UNMASKED
        assert (updated_mask[1:] == DLLMMaskState.MASKED).all()

    def test_processor_call_with_low_confidence_dynamic(self):
        """Test processor __call__ with low confidence dynamic strategy."""
        vocab_size = 100
        logits = torch.randn(self.block_size, vocab_size)

        # Make all tokens high confidence
        for i in range(self.block_size):
            logits[i, i] = 10.0

        input_ids = torch.randint(0, vocab_size, (10,))
        token_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        mask = torch.zeros(self.block_size, dtype=torch.long)

        updated_mask, updated_tokens = self.processor(
            logits=logits,
            input_ids=input_ids,
            token_ids=token_ids,
            dllm_mask=mask,
            strategy=DLLMUnmaskingStrategy.LOW_CONFIDENCE_DYNAMIC,
        )

        # All tokens should be unmasked (high confidence)
        assert (updated_mask == DLLMMaskState.UNMASKED).all()

    def test_block_completion_detection(self):
        """Test that processor correctly detects block completion."""
        # All tokens unmasked
        mask = torch.ones(self.block_size, dtype=torch.long)  # All unmasked

        # Sequential on completed block should not change anything
        updated_mask = self.processor.sequential(mask)
        assert (updated_mask == DLLMMaskState.UNMASKED).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
