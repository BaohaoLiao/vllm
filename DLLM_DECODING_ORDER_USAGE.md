# DLLM Decoding Order Usage Guide

## Overview

The decoding order feature allows you to see **which denoising iteration** each token was finalized at during DLLM generation. This provides insight into the diffusion process and helps analyze model behavior.

## Quick Start

### 1. Enable Decoding Order Output

Add `dllm_return_decoding_order=True` to your `SamplingParams`:

```python
from vllm import LLM, SamplingParams

# Initialize model
llm = LLM(model="path/to/sdar-model")

# Configure DLLM with decoding order output
sampling_params = SamplingParams(
    max_tokens=16,
    dllm_enabled=True,
    dllm_block_size=4,
    dllm_denoising_steps=4,
    dllm_unmasking_strategy="low_confidence_dynamic",
    dllm_return_decoding_order=True,  # ← Enable decoding order output
)

# Generate
outputs = llm.generate(["Tell me about AI"], sampling_params=sampling_params)

# Access decoding order
for output in outputs:
    for completion in output.outputs:
        print(f"Text: {completion.text}")
        print(f"Tokens: {completion.token_ids}")
        print(f"Decoding order: {completion.dllm_decoding_order}")
```

### 2. Understanding the Output

```python
# Example output:
text = "Artificial intelligence is the"
token_ids = [9,  234,  5896,  374,  279]
dllm_decoding_order = [0, 1, 1, 2, 3]

# Interpretation:
# - Token 9 ("Artificial"): unmasked at iteration 0
# - Token 234 ("intelligence"): unmasked at iteration 1
# - Token 5896 ("is"): unmasked at iteration 1 (same as previous)
# - Token 374 ("the"): unmasked at iteration 2
# - Token 279 (next word): unmasked at iteration 3
```

## Data Flow

### 1. SamplingParams (User Input)

```python
SamplingParams(
    dllm_enabled=True,
    dllm_return_decoding_order=True,  # ← User sets flag
    # ... other params
)
```

### 2. Request (Internal Tracking)

```python
# Inside Request class
self._dllm_decoding_order: list[int] = []

# Updated during unmasking in dllm_update_mask()
for token in newly_unmasked_tokens:
    self._dllm_decoding_order[token_pos] = self._dllm_iteration
```

### 3. ModelRunnerOutput (Batch Processing)

```python
ModelRunnerOutput(
    req_ids=["req-1", "req-2"],
    sampled_token_ids=[[token_ids_req1], [token_ids_req2]],
    dllm_decoding_order=[[0, 1, 2], [0, 0, 1]],  # ← Per request
)
```

### 4. CompletionOutput (User Output)

```python
CompletionOutput(
    text="Generated text",
    token_ids=[9, 234, 5896],
    dllm_decoding_order=[0, 1, 1],  # ← User receives this
)
```

## Practical Examples

### Example 1: Sequential Unmasking

```python
sampling_params = SamplingParams(
    max_tokens=8,
    dllm_enabled=True,
    dllm_block_size=4,
    dllm_unmasking_strategy="sequential",
    dllm_return_decoding_order=True,
)

output = llm.generate(["Hello"], sampling_params=sampling_params)[0]
print(output.outputs[0].dllm_decoding_order)
# Output: [0, 1, 2, 3, 0, 1, 2, 3]
#         ├─ Block 1 ─┤ ├─ Block 2 ─┤
# Each token unmasked in separate iteration
```

### Example 2: Low-Confidence Dynamic

```python
sampling_params = SamplingParams(
    max_tokens=8,
    dllm_enabled=True,
    dllm_block_size=4,
    dllm_unmasking_strategy="low_confidence_dynamic",
    dllm_confidence_threshold=0.85,
    dllm_return_decoding_order=True,
)

output = llm.generate(["Hello"], sampling_params=sampling_params)[0]
print(output.outputs[0].dllm_decoding_order)
# Output: [0, 1, 1, 2, 0, 0, 1, 2]
#         ├─ Block 1 ─┤ ├─ Block 2 ─┤
# Multiple tokens can unmask in same iteration
```

### Example 3: Analyzing Confidence Patterns

```python
def analyze_decoding_pattern(output):
    """Analyze which tokens required more iterations."""
    decoding_order = output.outputs[0].dllm_decoding_order
    token_ids = output.outputs[0].token_ids

    # Group by iteration
    iterations = {}
    for token_id, iter_num in zip(token_ids, decoding_order):
        if iter_num not in iterations:
            iterations[iter_num] = []
        iterations[iter_num].append(token_id)

    # Print analysis
    for iter_num in sorted(iterations.keys()):
        tokens = iterations[iter_num]
        print(f"Iteration {iter_num}: {len(tokens)} tokens unmasked")
        print(f"  Token IDs: {tokens}")

    # Calculate average
    avg_iter = sum(decoding_order) / len(decoding_order)
    print(f"Average iteration: {avg_iter:.2f}")

# Use it
output = llm.generate(["Example"], sampling_params=sampling_params)[0]
analyze_decoding_pattern(output)

# Output:
# Iteration 0: 2 tokens unmasked
#   Token IDs: [9, 234]
# Iteration 1: 3 tokens unmasked
#   Token IDs: [5896, 374, 279]
# Iteration 2: 1 token unmasked
#   Token IDs: [1234]
# Average iteration: 1.17
```

## Use Cases

### 1. Model Behavior Analysis

```python
# Compare different unmasking strategies
strategies = ["sequential", "low_confidence_dynamic", "low_confidence_static"]

for strategy in strategies:
    params = SamplingParams(
        max_tokens=16,
        dllm_enabled=True,
        dllm_unmasking_strategy=strategy,
        dllm_return_decoding_order=True,
    )
    output = llm.generate(["Test"], params)[0]

    order = output.outputs[0].dllm_decoding_order
    avg_iter = sum(order) / len(order)
    print(f"{strategy}: avg iteration = {avg_iter:.2f}")
```

### 2. Quality Metrics

```python
def calculate_convergence_speed(decoding_order, block_size):
    """Calculate how quickly tokens converge per block."""
    blocks = [
        decoding_order[i:i+block_size]
        for i in range(0, len(decoding_order), block_size)
    ]

    speeds = []
    for block in blocks:
        # How many iterations until all tokens unmasked
        max_iter = max(block) if block else 0
        speeds.append(max_iter + 1)

    return speeds

output = llm.generate(["Test"], sampling_params)[0]
speeds = calculate_convergence_speed(
    output.outputs[0].dllm_decoding_order,
    block_size=4
)
print(f"Block convergence speeds: {speeds}")
# Output: [3, 4, 2, 4]  # iterations per block
```

### 3. Token Confidence Visualization

```python
import matplotlib.pyplot as plt

def visualize_decoding_order(output, block_size=4):
    """Visualize when each token was unmasked."""
    decoding_order = output.outputs[0].dllm_decoding_order
    tokens = output.outputs[0].token_ids

    # Create heatmap
    num_blocks = (len(decoding_order) + block_size - 1) // block_size
    data = []

    for block_idx in range(num_blocks):
        start = block_idx * block_size
        end = min(start + block_size, len(decoding_order))
        block_order = decoding_order[start:end]
        data.append(block_order + [-1] * (block_size - len(block_order)))

    plt.imshow(data, cmap='viridis', aspect='auto')
    plt.colorbar(label='Iteration')
    plt.xlabel('Position in Block')
    plt.ylabel('Block Number')
    plt.title('DLLM Decoding Order Heatmap')
    plt.show()

# Use it
output = llm.generate(["Example"], sampling_params)[0]
visualize_decoding_order(output)
```

## Performance Impact

### Without `dllm_return_decoding_order` (Default)

```python
sampling_params = SamplingParams(
    dllm_enabled=True,
    dllm_return_decoding_order=False,  # Default
)

# What happens:
# - Decoding order is tracked internally (for scheduler)
# - NOT returned in output (saves memory/bandwidth)
# - output.outputs[0].dllm_decoding_order is None
```

### With `dllm_return_decoding_order=True`

```python
sampling_params = SamplingParams(
    dllm_enabled=True,
    dllm_return_decoding_order=True,
)

# What happens:
# - Decoding order is tracked internally (same as before)
# - Included in output (small overhead: ~4 bytes per token)
# - output.outputs[0].dllm_decoding_order is list[int]
```

**Memory overhead**: ~4 bytes per token (int32)
- 100 tokens: ~400 bytes
- 1000 tokens: ~4 KB
- **Negligible for most use cases**

## API Reference

### SamplingParams

```python
dllm_return_decoding_order: bool = False
```

**Type**: `bool`
**Default**: `False`
**Description**: Whether to return the decoding order in the output. Similar to `logprobs`, this adds extra information showing which denoising iteration finalized each token.

### CompletionOutput

```python
dllm_decoding_order: list[int] | None = None
```

**Type**: `list[int] | None`
**Default**: `None`
**Description**:
- For DLLM requests with `dllm_return_decoding_order=True`: List of iteration numbers (0-indexed) indicating when each token was unmasked
- Length matches `len(token_ids)`
- `None` for non-DLLM requests or when flag is disabled

**Example Values**:
- `[0, 1, 2, 3]` - Sequential unmasking, one per iteration
- `[0, 0, 1, 2]` - First two tokens unmasked together at iteration 0
- `[-1, -1, -1, -1]` - All tokens still masked (shouldn't appear in output)

## Troubleshooting

### Issue: `dllm_decoding_order` is always `None`

**Cause**: Flag not enabled
**Solution**: Set `dllm_return_decoding_order=True` in SamplingParams

```python
# Wrong
params = SamplingParams(dllm_enabled=True)

# Correct
params = SamplingParams(
    dllm_enabled=True,
    dllm_return_decoding_order=True
)
```

### Issue: Decoding order has unexpected values

**Cause**: Different unmasking strategies produce different patterns
**Solution**: Check your `dllm_unmasking_strategy` setting

```python
# Sequential: [0, 1, 2, 3, ...]
params = SamplingParams(
    dllm_enabled=True,
    dllm_unmasking_strategy="sequential",
    dllm_return_decoding_order=True,
)

# Dynamic: [0, 0, 1, 2, ...] (varies)
params = SamplingParams(
    dllm_enabled=True,
    dllm_unmasking_strategy="low_confidence_dynamic",
    dllm_return_decoding_order=True,
)
```

### Issue: Non-DLLM request has `dllm_decoding_order`

**Cause**: This shouldn't happen (field should be `None`)
**Solution**: Verify `dllm_enabled=True` is set

## Summary

**To enable decoding order output**:

```python
sampling_params = SamplingParams(
    dllm_enabled=True,
    dllm_return_decoding_order=True,  # ← Add this flag
    # ... other DLLM params
)
```

**To access decoding order**:

```python
output = llm.generate(prompts, sampling_params)[0]
decoding_order = output.outputs[0].dllm_decoding_order
```

**What you get**:
- List of integers (0-indexed)
- Same length as `token_ids`
- Shows iteration number when each token was finalized
- `None` if disabled or non-DLLM request

**Minimal overhead**: ~4 bytes per token, negligible for most use cases.
