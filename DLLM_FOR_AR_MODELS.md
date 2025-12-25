# DLLM Generation for Autoregressive Models

## Concept

DLLM (Diffusion Language Model) generation is a **generation strategy** that can be applied to any autoregressive transformer model, not just specialized models like SDAR.

## Key Difference from Standard AR

**Standard Autoregressive:**
```
Token 1 → Token 2 → Token 3 → Token 4 → ...
(one at a time, left to right)
```

**DLLM with AR Model:**
```
Block 1: [t0, M, M, M]  ← Initialize with masks
      ↓  (iteration 1)
         [t0, t1, M, M]  ← Unmask based on confidence
      ↓  (iteration 2)
         [t0, t1, t2, M]
      ↓  (iteration 3)
         [t0, t1, t2, t3]  ← Block complete!
      ↓
Block 2: [M, M, M, M]  ← Start next block
         ...
```

## Implementation Strategy

### 1. Block Initialization
Replace masked positions with a special mask token ID:
```python
# Instead of generating one token:
input_ids = [prompt_tokens, <next_token>]

# DLLM generates a block:
input_ids = [prompt_tokens, mask_token, mask_token, mask_token, mask_token]
```

### 2. Iterative Refinement
Run multiple forward passes over the same block:
```python
for iteration in range(dllm_denoising_steps):
    # Forward pass
    logits = model(input_ids)

    # Sample tokens
    sampled = sample(logits)

    # Decide which tokens to keep (unmasking strategy)
    updated_tokens, updated_mask = apply_unmasking(
        sampled, logits, mask, strategy
    )

    # Update input for next iteration
    input_ids[masked_positions] = updated_tokens[masked_positions]
```

### 3. Unmasking Strategies

**Low Confidence Dynamic:**
- Keep tokens with probability > threshold
- Iteratively unmask more tokens

**Low Confidence Static:**
- Unmask top-K tokens each iteration

**Sequential:**
- Unmask left-to-right, one per iteration

### 4. Integration Points

The key changes needed:

1. **Scheduler**: Recognize DLLM requests and schedule blocks
2. **Model Runner**: Implement block-based generation loop
3. **Sampler**: Apply unmasking after sampling
4. **Request**: Track mask state and iteration count

## Architecture

```
┌─────────────────────────────────────────────┐
│           Scheduler                         │
│  - Detects DLLM request                     │
│  - Schedules block (not single token)       │
│  - Tracks iteration count                   │
└─────────────────┬───────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│           Model Runner                      │
│  - Prepares block input (with masks)        │
│  - Runs forward pass                        │
│  - Calls sampler                            │
└─────────────────┬───────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│           Sampler                           │
│  - Samples tokens from logits               │
│  - Applies unmasking strategy               │
│  - Updates mask state                       │
└─────────────────┬───────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│           Request                           │
│  - Stores current block tokens              │
│  - Tracks mask state per token              │
│  - Records decoding order                   │
└─────────────────────────────────────────────┘
```

## Implementation Plan

### Phase 1: Minimal DLLM Loop (In Model Runner)

Create a DLLM generation loop that:
1. Checks if request is DLLM
2. Generates blocks instead of single tokens
3. Applies unmasking
4. Updates request state

### Phase 2: Unmasking Integration

Connect the unmasking processor to:
1. Evaluate token confidence
2. Update mask state
3. Track decoding order

### Phase 3: Scheduler Integration

Update scheduler to:
1. Recognize DLLM block generation
2. Schedule multiple iterations
3. Handle block completion

## Key Files to Modify

1. **`vllm/v1/worker/gpu_model_runner.py`**
   - Add DLLM generation loop
   - Integrate unmasking processor

2. **`vllm/v1/core/sched/scheduler.py`**
   - Handle DLLM scheduling
   - Use `dllm_helper.py` functions

3. **`vllm/v1/sample/sampler.py`**
   - Implement unmasking in sampler
   - Update mask state

## Benefits of DLLM with AR Models

1. **Exploration**: Can revise tokens based on future context
2. **Quality**: Iterative refinement can improve coherence
3. **Controllability**: Can guide generation via unmasking
4. **Analysis**: Decoding order shows generation process

## Challenges

1. **Slower**: Multiple passes per block
2. **Memory**: Need to store mask state
3. **Compatibility**: Requires careful integration

## Next Steps

I'll implement the DLLM generation loop in the model runner to make this work with Qwen and other AR models.
