"""Simple test script to verify DLLM implementation."""

import torch
from vllm.v1.sample.dllm_constants import DLLMMaskState, DLLMUnmaskingStrategy
from vllm.v1.sample.dllm_unmasking import DLLMUnmaskingProcessor

def test_constants():
    """Test DLLM constants."""
    print("Testing DLLM constants...")
    assert DLLMMaskState.MASKED == 0
    assert DLLMMaskState.UNMASKED == 1
    assert DLLMMaskState.CACHED == 2
    print("✓ DLLMMaskState values correct")

    assert DLLMUnmaskingStrategy.from_str("SEQUENTIAL") == DLLMUnmaskingStrategy.SEQUENTIAL
    assert DLLMUnmaskingStrategy.from_str("LOW_CONFIDENCE_DYNAMIC") == DLLMUnmaskingStrategy.LOW_CONFIDENCE_DYNAMIC
    print("✓ DLLMUnmaskingStrategy.from_str() works")

def test_sequential_unmasking():
    """Test sequential unmasking."""
    print("\nTesting sequential unmasking...")
    processor = DLLMUnmaskingProcessor(
        block_size=4,
        denoising_steps=4,
        confidence_threshold=0.85,
    )

    # All masked initially
    mask = torch.tensor([0, 0, 0, 0], dtype=torch.long)
    updated_mask = processor.sequential(mask)

    # Should unmask first token
    assert updated_mask[0] == DLLMMaskState.UNMASKED
    assert updated_mask[1] == DLLMMaskState.MASKED
    print("✓ Sequential unmasking works (first token unmasked)")

    # Now unmask second
    updated_mask = processor.sequential(updated_mask)
    assert updated_mask[0] == DLLMMaskState.UNMASKED
    assert updated_mask[1] == DLLMMaskState.UNMASKED
    assert updated_mask[2] == DLLMMaskState.MASKED
    print("✓ Sequential unmasking progresses left-to-right")

def test_low_confidence_dynamic():
    """Test low confidence dynamic unmasking."""
    print("\nTesting low confidence dynamic unmasking...")
    processor = DLLMUnmaskingProcessor(
        block_size=4,
        denoising_steps=4,
        confidence_threshold=0.85,
    )

    vocab_size = 100
    logits = torch.randn(4, vocab_size)

    # Make first and third tokens high confidence
    logits[0, 50] = 10.0  # Very high logit
    logits[2, 30] = 10.0  # Very high logit

    tokens = torch.tensor([50, 20, 30, 40], dtype=torch.long)
    mask = torch.zeros(4, dtype=torch.long)

    updated_mask, updated_tokens = processor.low_confidence_dynamic(logits, tokens, mask)

    # High confidence tokens should be unmasked
    assert updated_mask[0] == DLLMMaskState.UNMASKED
    assert updated_mask[2] == DLLMMaskState.UNMASKED
    print("✓ Low confidence dynamic unmasks high-confidence tokens")

def test_processor_call():
    """Test processor __call__ method."""
    print("\nTesting processor __call__...")
    processor = DLLMUnmaskingProcessor(
        block_size=4,
        denoising_steps=4,
        confidence_threshold=0.85,
    )

    vocab_size = 100
    logits = torch.randn(4, vocab_size)
    input_ids = torch.randint(0, vocab_size, (10,))
    token_ids = torch.randint(0, vocab_size, (4,))
    mask = torch.zeros(4, dtype=torch.long)

    updated_mask, updated_tokens = processor(
        logits=logits,
        input_ids=input_ids,
        token_ids=token_ids,
        dllm_mask=mask,
        strategy=DLLMUnmaskingStrategy.SEQUENTIAL,
    )

    assert updated_mask[0] == DLLMMaskState.UNMASKED
    assert (updated_mask[1:] == DLLMMaskState.MASKED).all()
    print("✓ Processor __call__ works with SEQUENTIAL strategy")

def main():
    """Run all tests."""
    print("=" * 60)
    print("DLLM Implementation Verification")
    print("=" * 60)

    try:
        test_constants()
        test_sequential_unmasking()
        test_low_confidence_dynamic()
        test_processor_call()

        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        print("\nDLLM implementation is working correctly.")
        print("\nNext steps:")
        print("1. Test with actual model (Qwen or SDAR)")
        print("2. Verify end-to-end generation")
        print("3. Check decoding order tracking")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
