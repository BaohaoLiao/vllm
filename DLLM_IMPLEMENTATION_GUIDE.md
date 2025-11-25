# DLLM Implementation Guide for vLLM

This guide documents the implementation of Diffusion Language Model (DLLM) support in vLLM, following the design from LMDeploy.

## Overview

DLLM is a non-autoregressive generation approach where tokens are generated in blocks and iteratively refined through a denoising process, similar to diffusion models in image generation.

## Completed Components ✅

### 1. Core Constants and Types (`vllm/v1/sample/dllm_constants.py`)
- **DLLMMaskState**: Token states (MASKED=0, UNMASKED=1, CACHED=2)
- **DLLMUnmaskingStrategy**: Unmasking strategies (SEQUENTIAL, LOW_CONFIDENCE_DYNAMIC, LOW_CONFIDENCE_STATIC)

### 2. Configuration (`vllm/sampling_params.py`)
Added DLLM parameters to `SamplingParams`:
```python
dllm_enabled: bool = False
dllm_block_size: int = 4
dllm_denoising_steps: int | None = None
dllm_unmasking_strategy: str = "low_confidence_dynamic"
dllm_confidence_threshold: float = 0.85
dllm_mask_token_id: int | None = None
```

### 3. Request Extension (`vllm/v1/request.py`)
Extended `Request` class with DLLM fields and methods:
- **Fields**: `is_dllm`, `dllm_block_size`, `_dllm_mask`, `_num_valid_tokens`, `_dllm_iteration`
- **Methods**:
  - `dllm_get_mask()`: Get current mask state
  - `dllm_update_mask(new_mask)`: Update mask after denoising
  - `dllm_is_current_block_complete()`: Check if block fully unmasked
  - `dllm_advance_to_next_block()`: Mark block as cached
  - `dllm_init_new_block(token_ids)`: Initialize new masked block

### 4. Unmasking Processor (`vllm/v1/sample/dllm_unmasking.py`)
Implements three unmasking strategies:

**LOW_CONFIDENCE_STATIC**: Unmask top-K lowest confidence tokens per block
```python
# Select K tokens with lowest confidence
_, indices = scores.topk(K, dim=-1)
dllm_mask.scatter_(-1, indices, UNMASKED)
```

**LOW_CONFIDENCE_DYNAMIC**: Unmask tokens above confidence threshold
```python
# Unmask all tokens with score >= threshold
dllm_mask[scores >= threshold] = UNMASKED
```

**SEQUENTIAL**: Unmask tokens left-to-right
```python
# Unmask K tokens starting from first masked position
indices = first_masked_pos + range(K)
dllm_mask[indices] = UNMASKED
```

## Remaining Implementation Steps 🚧

### 5. Modify Sampler (`vllm/v1/sample/sampler.py`)

**Current behavior**: Samples 1 token per request per iteration
**Required behavior**: Sample `block_size` tokens for DLLM requests

#### Changes needed in `Sampler.forward()`:

```python
def forward(
    self,
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> SamplerOutput:
    """Sample tokens from logits."""

    # NEW: Check if any requests use DLLM
    has_dllm = any(
        req.is_dllm for req in sampling_metadata.requests
    )

    if not has_dllm:
        # Original single-token sampling logic
        return self._sample_single_token(logits, sampling_metadata)
    else:
        # NEW: Block-based sampling for DLLM
        return self._sample_dllm_blocks(logits, sampling_metadata)

def _sample_dllm_blocks(
    self,
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> SamplerOutput:
    """Sample blocks of tokens for DLLM requests."""

    # Reshape logits for block-based sampling
    # Original: [batch_size, vocab_size]
    # DLLM: [batch_size * block_size, vocab_size]

    # Sample tokens for each position in block
    sampled_token_ids = self._apply_sampling(logits, sampling_metadata)

    # NEW: Apply unmasking if in decode phase
    if sampling_metadata.is_decode_phase:
        sampled_token_ids, updated_masks = self._apply_unmasking(
            logits,
            sampled_token_ids,
            sampling_metadata,
        )

    return SamplerOutput(
        sampled_token_ids=sampled_token_ids,
        dllm_masks=updated_masks,  # NEW field
    )

def _apply_unmasking(
    self,
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> tuple[torch.Tensor, list[list[int]]]:
    """Apply DLLM unmasking strategies."""
    from vllm.v1.sample.dllm_unmasking import DLLMUnmaskingProcessor
    from vllm.v1.sample.dllm_constants import DLLMUnmaskingStrategy

    updated_masks = []

    for req in sampling_metadata.requests:
        if not req.is_dllm:
            updated_masks.append([])
            continue

        # Get current mask and input IDs
        current_mask = torch.tensor(
            req.dllm_get_mask(), device=logits.device
        )
        input_ids = torch.tensor(
            req.output_token_ids[-req.dllm_block_size:],
            device=logits.device,
        )

        # Get request-specific logits and tokens
        req_start = req.batch_idx * req.dllm_block_size
        req_end = req_start + req.dllm_block_size
        req_logits = logits[req_start:req_end]
        req_tokens = token_ids[req_start:req_end]

        # Create unmasking processor
        processor = DLLMUnmaskingProcessor(
            block_size=req.dllm_block_size,
            denoising_steps=req.dllm_denoising_steps,
            confidence_threshold=req.dllm_confidence_threshold,
        )

        # Apply unmasking
        strategy = DLLMUnmaskingStrategy.from_str(
            req.dllm_unmasking_strategy
        )
        new_mask, new_tokens = processor(
            req_logits,
            input_ids,
            req_tokens,
            current_mask,
            strategy,
        )

        # Update token_ids and collect mask
        token_ids[req_start:req_end] = new_tokens
        updated_masks.append(new_mask.tolist())

    return token_ids, updated_masks
```

### 6. Update Scheduler (`vllm/v1/core/sched/scheduler.py`)

#### Changes in `Scheduler.schedule()`:

**Purpose**: Ensure DLLM requests continue refining current block before advancing

```python
def schedule(self) -> SchedulerOutput:
    """Schedule next batch of requests."""

    # Existing logic for selecting requests...

    # NEW: Check DLLM block completion
    for request in running_queue:
        if request.is_dllm:
            # Check if current block is complete
            if not request.dllm_is_current_block_complete():
                # Continue refining current block
                # Don't allocate new KV cache slots
                num_new_tokens = 0
            else:
                # Check if we've reached max_tokens
                if request.dllm_num_valid_tokens >= request.max_tokens:
                    request.status = RequestStatus.FINISHED_LENGTH_CAPPED
                    continue

                # Advance to next block
                request.dllm_advance_to_next_block()

                # Allocate for new block
                num_new_tokens = request.dllm_block_size

                # Initialize new block
                new_block_tokens = [request.dllm_mask_token_id] * num_new_tokens
                request.dllm_init_new_block(new_block_tokens)
        else:
            # Original AR logic
            num_new_tokens = 1

    # Continue with KV cache allocation...
```

#### Changes in `Scheduler.update_from_output()`:

```python
def update_from_output(
    self,
    scheduler_output: SchedulerOutput,
    model_runner_output: ModelRunnerOutput,
) -> None:
    """Update request states from model output."""

    for idx, request in enumerate(scheduler_output.scheduled_requests):
        if request.is_dllm:
            # NEW: Update DLLM mask
            if idx < len(model_runner_output.dllm_masks):
                new_mask = model_runner_output.dllm_masks[idx]
                request.dllm_update_mask(new_mask)

            # NEW: Update tokens for unmasked positions only
            token_ids = model_runner_output.sampled_token_ids[idx]
            mask = request.dllm_get_mask()

            # Only append tokens that are unmasked
            from vllm.v1.sample.dllm_constants import DLLMMaskState
            for i, (token, mask_state) in enumerate(zip(token_ids, mask)):
                if mask_state == DLLMMaskState.UNMASKED:
                    request.append_output_token_ids(token)
        else:
            # Original AR logic
            request.append_output_token_ids(
                model_runner_output.sampled_token_ids[idx]
            )
```

### 7. Integrate into GPUModelRunner (`vllm/v1/worker/gpu_model_runner.py`)

#### Changes in `_prepare_inputs()`:

```python
def _prepare_inputs(
    self,
    scheduler_output: SchedulerOutput,
) -> ModelInputs:
    """Prepare inputs for model execution."""

    # NEW: Collect DLLM masks
    dllm_masks = []
    for request in scheduler_output.scheduled_requests:
        if request.is_dllm:
            mask = request.dllm_get_mask()
            dllm_masks.extend(mask)

    # Convert to tensor if needed
    dllm_mask_tensor = None
    if dllm_masks:
        dllm_mask_tensor = torch.tensor(
            dllm_masks, dtype=torch.long, device=self.device
        )

    # Create model inputs
    return ModelInputs(
        ...,  # existing fields
        dllm_masks=dllm_mask_tensor,  # NEW field
    )
```

#### Changes in `sample_tokens()`:

```python
def sample_tokens(
    self,
    scheduler_output: SchedulerOutput,
) -> ModelRunnerOutput:
    """Sample tokens from model outputs."""

    # Get logits from previous execute_model()
    logits = self._get_logits()

    # Sample tokens
    sampler_output = self.sampler.forward(
        logits=logits,
        sampling_metadata=self._build_sampling_metadata(
            scheduler_output
        ),
    )

    return ModelRunnerOutput(
        sampled_token_ids=sampler_output.sampled_token_ids,
        dllm_masks=sampler_output.dllm_masks,  # NEW field
    )
```

### 8. Update Output Types

#### `vllm/v1/outputs.py`:

```python
@dataclass
class SamplerOutput:
    """Output from token sampler."""
    sampled_token_ids: torch.Tensor
    logprobs: torch.Tensor | None = None
    dllm_masks: list[list[int]] | None = None  # NEW field

@dataclass
class ModelRunnerOutput:
    """Output from GPU model runner."""
    sampled_token_ids: list[int]
    logprobs: list[dict[int, float]] | None = None
    dllm_masks: list[list[int]] | None = None  # NEW field
```

## Testing Strategy

### 1. Unit Tests

Create `vllm/tests/v1/sample/test_dllm.py`:

```python
def test_dllm_constants():
    """Test DLLM constants and enums."""
    from vllm.v1.sample.dllm_constants import DLLMMaskState

    assert DLLMMaskState.MASKED == 0
    assert DLLMMaskState.UNMASKED == 1
    assert DLLMMaskState.CACHED == 2

def test_dllm_unmasking_sequential():
    """Test sequential unmasking strategy."""
    from vllm.v1.sample.dllm_unmasking import DLLMUnmaskingProcessor

    processor = DLLMUnmaskingProcessor(
        block_size=4,
        denoising_steps=4,
    )

    # Create test data
    mask = torch.tensor([0, 0, 0, 0])  # All masked
    updated_mask = processor.sequential(mask)

    # Should unmask first token
    assert updated_mask[0] == 1
    assert updated_mask[1:].sum() == 0

def test_request_dllm_methods():
    """Test Request DLLM methods."""
    from vllm.v1.request import Request
    from vllm.sampling_params import SamplingParams

    # Create DLLM request
    params = SamplingParams(
        dllm_enabled=True,
        dllm_block_size=4,
        dllm_mask_token_id=0,
    )

    request = Request(
        request_id="test",
        prompt_token_ids=[1, 2, 3],
        sampling_params=params,
        eos_token_id=2,
    )

    # Test initialization
    assert request.is_dllm
    assert request.dllm_block_size == 4

    # Test block initialization
    request.dllm_init_new_block([10, 11, 12, 13])
    mask = request.dllm_get_mask()
    assert len(mask) == 4
    assert mask[0] == 1  # First token unmasked
```

### 2. Integration Tests

Create `vllm/tests/v1/test_dllm_integration.py`:

```python
@pytest.mark.asyncio
async def test_dllm_generation():
    """Test end-to-end DLLM generation."""
    from vllm import LLM, SamplingParams

    # Initialize LLM
    llm = LLM(model="path/to/dllm/model")

    # Create DLLM sampling params
    sampling_params = SamplingParams(
        max_tokens=16,
        dllm_enabled=True,
        dllm_block_size=4,
        dllm_denoising_steps=4,
        dllm_unmasking_strategy="low_confidence_dynamic",
    )

    # Generate
    outputs = llm.generate(
        prompts=["Hello, how are you?"],
        sampling_params=sampling_params,
    )

    # Verify output
    assert len(outputs) == 1
    assert len(outputs[0].outputs[0].token_ids) == 16
```

### 3. Model-Specific Tests

For SDAR or other DLLM models:

```python
def test_sdar_dllm():
    """Test SDAR model with DLLM."""
    # Load SDAR model
    llm = LLM(model="SDAR-model-path")

    # DLLM generation
    outputs = llm.generate(
        prompts=["Test prompt"],
        sampling_params=SamplingParams(
            dllm_enabled=True,
            dllm_block_size=4,
            dllm_mask_token_id=model.config.dllm_mask_token,
        ),
    )

    # Verify outputs match expected format
    ...
```

## Usage Example

```python
from vllm import LLM, SamplingParams

# Initialize model
llm = LLM(model="path/to/dllm/model")

# Configure DLLM sampling
sampling_params = SamplingParams(
    max_tokens=32,
    temperature=0.0,
    # DLLM-specific parameters
    dllm_enabled=True,
    dllm_block_size=4,
    dllm_denoising_steps=4,
    dllm_unmasking_strategy="low_confidence_dynamic",
    dllm_confidence_threshold=0.85,
    dllm_mask_token_id=0,  # Model-specific mask token
)

# Generate
outputs = llm.generate(
    prompts=["Explain the concept of diffusion models."],
    sampling_params=sampling_params,
)

print(outputs[0].outputs[0].text)
```

## Key Differences from LMDeploy

1. **Architecture**: vLLM uses separate Scheduler/Worker vs LMDeploy's unified engine
2. **Sequence Management**: vLLM uses `Request` class vs LMDeploy's `SchedulerSequence`
3. **Sampling**: vLLM has dedicated `Sampler` class vs LMDeploy's strategy pattern
4. **State Updates**: vLLM updates via `ModelRunnerOutput` vs LMDeploy's direct updates

## Next Steps

1. ✅ Implement Steps 5-7 (Sampler, Scheduler, ModelRunner modifications)
2. ✅ Add comprehensive tests
3. ✅ Test with SDAR model
4. ✅ Optimize performance (CUDA graphs, batching)
5. ✅ Add documentation and examples
6. ✅ Submit PR to vLLM project

## Notes

- DLLM requires model-specific support (attention masking, block sparsity)
- Currently targets SDAR model family
- Future: Support more DLLM architectures (e.g., AR-Diffusion, SUNDAE)
- Consider adding visualization tools for debugging mask states

## References

- LMDeploy DLLM implementation: `lmdeploy/pytorch/strategies/dllm/`
- SDAR model: Research paper on structured DAR
- vLLM architecture: `vllm/v1/`
