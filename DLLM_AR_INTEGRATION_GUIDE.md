# DLLM Integration Guide for AR Models

## Quick Start

I've created a `DLLMGenerator` class that handles block-based generation for any autoregressive model. Here's how to integrate it.

## Files Created

1. **`vllm/v1/worker/dllm_generator.py`** - DLLM generation logic
   - `prepare_dllm_input()` - Prepares block input with masks
   - `process_dllm_output()` - Applies unmasking and updates state
   - `should_continue_block_refinement()` - Checks if should continue

## Integration Steps

### Step 1: Import in Model Runner

In `vllm/v1/worker/gpu_model_runner.py`:

```python
from vllm.v1.worker.dllm_generator import DLLMGenerator

class GPUModelRunner:
    def __init__(self, ...):
        # ... existing init ...
        self.dllm_generator = DLLMGenerator()
```

### Step 2: Prepare DLLM Input

Before the forward pass, prepare input for DLLM requests:

```python
def _prepare_model_input(self, ...):
    # For each request in batch:
    for req_id, request in requests.items():
        if request.is_dllm:
            # Use DLLM input preparation
            input_tokens = self.dllm_generator.prepare_dllm_input(
                request,
                current_tokens
            )
        else:
            # Standard AR input
            input_tokens = current_tokens
```

### Step 3: Process DLLM Output

After sampling, process DLLM outputs:

```python
def _process_model_outputs(self, sampled_tokens, logits, requests):
    for req_id, request in requests.items():
        if request.is_dllm:
            # Apply DLLM unmasking
            tokens_to_keep, new_mask, block_complete = \
                self.dllm_generator.process_dllm_output(
                    request,
                    sampled_tokens[req_id],
                    logits[req_id] if logits else None
                )

            # Only append unmasked tokens
            output_tokens = tokens_to_keep
        else:
            # Standard AR output
            output_tokens = sampled_tokens[req_id]
```

### Step 4: Handle Iteration Loop

Check if DLLM requests need more iterations:

```python
def _should_continue_generation(self, request):
    if request.is_dllm:
        # Check if should refine current block
        return self.dllm_generator.should_continue_block_refinement(request)
    else:
        # Standard stopping condition
        return not request.is_finished()
```

## Minimal Example Integration

Here's a simplified example showing the key touchpoints:

```python
# In gpu_model_runner.py

from vllm.v1.worker.dllm_generator import DLLMGenerator

class GPUModelRunner:
    def __init__(self, ...):
        self.dllm_generator = DLLMGenerator()

    def execute_model(self, scheduler_output):
        # ... existing code ...

        # BEFORE forward pass:
        for request in scheduler_output.scheduled_requests:
            if request.is_dllm:
                # Prepare block input with masks
                input_ids = self.dllm_generator.prepare_dllm_input(
                    request,
                    request.all_token_ids
                )
            else:
                input_ids = request.all_token_ids

        # Forward pass (unchanged)
        hidden_states = self.model(input_ids, ...)

        # Sample (unchanged)
        sampler_output = self.sampler(logits, sampling_metadata)

        # AFTER sampling:
        for request in scheduler_output.scheduled_requests:
            if request.is_dllm:
                # Process DLLM output
                tokens, mask, complete = \
                    self.dllm_generator.process_dllm_output(
                        request,
                        sampler_output.sampled_token_ids[req_idx],
                        logits[req_idx]
                    )
                # Append only unmasked tokens
                request.append_output_token_ids(tokens)
            else:
                # Standard append
                request.append_output_token_ids(
                    sampler_output.sampled_token_ids[req_idx]
                )
```

## How It Works

### Block Initialization

When a DLLM request starts:
```python
# First time: Initialize block with masks
input = [prompt_tokens, mask, mask, mask, mask]
#                      └─ block_size = 4 ─┘
```

### Iteration 1

```python
# Forward pass with masks
input = [prompt, mask, mask, mask, mask]
output = model(input)
sampled = [t0, t1, t2, t3]

# Apply unmasking (e.g., keep high-confidence tokens)
unmasked_tokens = [t0]  # Only t0 is confident
new_mask = [UNMASKED, MASKED, MASKED, MASKED]

# Update request
request.dllm_update_mask(new_mask)  # ← Records decoding order!

# Prepare next iteration
input = [prompt, t0, mask, mask, mask]
```

### Iteration 2

```python
# Forward pass with partial unmask
input = [prompt, t0, mask, mask, mask]
output = model(input)
sampled = [t0, t1, t2, t3]  # t0 unchanged, others resampled

# Apply unmasking
unmasked_tokens = [t0, t1, t2]  # t1, t2 now confident
new_mask = [UNMASKED, UNMASKED, UNMASKED, MASKED]

# Update request
request.dllm_update_mask(new_mask)  # ← Records t1, t2 at iteration 1

# Prepare next iteration
input = [prompt, t0, t1, t2, mask]
```

### Iteration 3

```python
# Forward pass
input = [prompt, t0, t1, t2, mask]
output = model(input)
sampled = [t0, t1, t2, t3]

# Apply unmasking
unmasked_tokens = [t0, t1, t2, t3]  # All confident
new_mask = [UNMASKED, UNMASKED, UNMASKED, UNMASKED]

# Block complete!
request.dllm_update_mask(new_mask)  # ← Records t3 at iteration 2
request.dllm_advance_to_next_block()

# Final decoding order: [0, 1, 1, 2]
```

## Benefits

- ✅ Works with ANY AR model (Qwen, LLaMA, etc.)
- ✅ Automatic decoding order tracking
- ✅ Configurable unmasking strategies
- ✅ Block-based generation
- ✅ Iterative refinement

## Testing

Test with your Qwen model:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="/mnt/nushare2/data/baliao/PLLMs/qwen/Qwen3-1.7B-Base")

# DLLM parameters
sampling_params = SamplingParams(
    max_tokens=16,  # Total tokens
    dllm_enabled=True,
    dllm_block_size=4,  # Generate 4 tokens per block
    dllm_denoising_steps=3,  # 3 iterations per block
    dllm_unmasking_strategy="low_confidence_dynamic",
    dllm_confidence_threshold=0.85,
    dllm_mask_token_id=0,  # Use token 0 as mask
    dllm_return_decoding_order=True,
)

outputs = llm.generate(
    prompts=["Explain the concept of diffusion models."],
    sampling_params=sampling_params,
)

# Should now have decoding order!
print(outputs[0].outputs[0].dllm_decoding_order)
# Expected: [0, 1, 1, 2, 0, 1, 2, 2, 0, 0, 1, 2, 0, 1, 1, 2]
#           └─ Block 1 ─┘ └─ Block 2 ─┘ └─ Block 3 ─┘ └─ Block 4 ─┘
```

## Next Steps

To fully enable DLLM for AR models, integrate the `DLLMGenerator` into the model runner:

1. Add `dllm_generator = DLLMGenerator()` to model runner init
2. Call `prepare_dllm_input()` before forward pass
3. Call `process_dllm_output()` after sampling
4. Handle iteration loop for block refinement

The infrastructure is ready - just needs these integration points!
