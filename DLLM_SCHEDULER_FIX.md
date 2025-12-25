# DLLM Scheduler Fix - Resolving Stuck Generation

## Problem

When using DLLM generation, the system would get stuck and never produce output:

```python
sampling_params = SamplingParams(
    max_tokens=32,
    dllm_enabled=True,
    dllm_block_size=4,
    dllm_denoising_steps=4,
    ...
)
outputs = llm.generate(["Explain the concept of diffusion models."], sampling_params)
# Gets stuck here - never returns
```

## Root Cause

The scheduler had three critical issues with DLLM block refinement:

### Issue 1: Scheduler Skipping DLLM Refinement Requests

**Location**: `scheduler.py` lines 288-302

**Problem**: During DLLM block refinement, we set `num_new_tokens = 0` because we're refining existing tokens, not allocating new ones. However, the scheduler has logic that skips requests when `num_new_tokens == 0`:

```python
if num_new_tokens == 0:
    # The request cannot be scheduled...
    req_index += 1
    continue  # ← SKIPS the request!
```

This caused DLLM refinement requests to be skipped entirely, preventing them from ever being processed.

**Fix**: Added an exception for DLLM requests:

```python
if num_new_tokens == 0:
    # ... comments ...
    # DLLM exception: Allow DLLM requests with 0 new tokens (refinement mode)
    if not request.is_dllm:
        req_index += 1
        continue
    # DLLM refinement - proceed to schedule with existing KV cache
```

### Issue 2: KV Cache Allocation During Refinement

**Location**: `scheduler.py` lines 307-320

**Problem**: Even when `num_new_tokens = 0` for DLLM refinement, the code would try to allocate new KV cache slots:

```python
new_blocks = self.kv_cache_manager.allocate_slots(
    request,
    num_new_tokens,  # ← 0 for DLLM refinement
    ...
)
```

This could cause issues or unnecessary allocation attempts.

**Fix**: Skip KV cache allocation when refining, and set `new_blocks = []` for DLLM:

```python
if num_new_tokens > 0:
    with record_function_or_nullcontext("schedule: allocate_slots"):
        while True:
            new_blocks = self.kv_cache_manager.allocate_slots(...)
            if new_blocks is not None:
                break
            # ... preemption logic ...
else:
    # DLLM refinement: No new blocks needed, use existing KV cache
    new_blocks = []
```

### Issue 3: Missing First Block Initialization

**Location**: `scheduler.py` lines 236-265

**Problem**: When a DLLM request first starts, it doesn't have any blocks initialized. The original logic only initialized blocks when advancing from one block to the next, but not for the first block.

**Fix**: Added first block initialization check:

```python
if request.is_dllm:
    # Check if first block needs initialization
    if not hasattr(request, '_dllm_mask') or len(request._dllm_mask) == 0:
        # Initialize first block
        num_new_tokens = request.dllm_block_size
        new_block_tokens = [request.dllm_mask_token_id] * num_new_tokens
        request.dllm_init_new_block(new_block_tokens)
    elif should_continue_refining_block(request):
        # Refining current block...
        num_new_tokens = 0
    else:
        # Advance to next block...
```

## Complete DLLM Scheduling Flow

### 1. First Time (Block Initialization)
```
Request enters RUNNING queue
    ↓
Check: _dllm_mask is empty
    ↓
Initialize first block:
    - num_new_tokens = block_size (e.g., 4)
    - Create masked tokens: [mask_token_id, mask_token_id, ...]
    - Call request.dllm_init_new_block()
    ↓
Allocate KV cache for block_size tokens
    ↓
Schedule request for execution
```

### 2. Refinement (Iterations 1-N)
```
Request is RUNNING, has current block
    ↓
Check: should_continue_refining_block(request)
    ↓ (iteration < denoising_steps)
Refine current block:
    - num_new_tokens = 0  ← No new slots!
    - Skip KV cache allocation
    - Use existing KV cache (new_blocks = [])
    ↓
Schedule request for execution
    ↓
Model forward pass with masked tokens
    ↓
Sampler applies unmasking strategy
    ↓
Request.dllm_update_mask() updates state
    ↓ (Loop back to refinement)
```

### 3. Block Complete (Advancement)
```
Request completes current block
    ↓
Check: should_continue_refining_block(request)
    ↓ (iteration >= denoising_steps)
Block is complete:
    - Call request.dllm_advance_to_next_block()
    - num_new_tokens = block_size
    - Initialize new block with mask tokens
    ↓
Allocate KV cache for new block
    ↓
Schedule request for execution
    ↓ (Start refinement loop for new block)
```

### 4. Generation Complete
```
Check: dllm_num_valid_tokens >= max_tokens
    ↓
Set status = FINISHED_LENGTH_CAPPED
    ↓
Stop scheduling, return output
```

## Files Modified

1. **`vllm/v1/core/sched/scheduler.py`**
   - Lines 243-248: Added first block initialization
   - Lines 301-305: Added DLLM exception for `num_new_tokens == 0`
   - Lines 309-372: Wrapped KV allocation in `if num_new_tokens > 0:`, added `else:` for DLLM

## Testing

After these fixes, DLLM generation should work correctly:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="/path/to/qwen/Qwen3-1.7B-Base")

sampling_params = SamplingParams(
    max_tokens=16,
    dllm_enabled=True,
    dllm_block_size=4,
    dllm_denoising_steps=4,
    dllm_unmasking_strategy="sequential",
    dllm_mask_token_id=0,
    dllm_return_decoding_order=True,
)

# Should now complete without getting stuck!
outputs = llm.generate(["Hello world"], sampling_params)
print(outputs[0].outputs[0].text)
print(outputs[0].outputs[0].dllm_decoding_order)
```

Expected behavior:
- First block (4 tokens) generated with 4 denoising iterations
- Second block (4 tokens) generated with 4 denoising iterations
- Continues until max_tokens reached
- Returns decoding order showing which iteration each token was finalized

## Summary

The fixes enable DLLM block-based generation to work properly by:
1. ✅ Allowing DLLM requests to be scheduled during refinement (`num_new_tokens = 0`)
2. ✅ Skipping KV cache allocation during refinement
3. ✅ Initializing the first block when DLLM requests start

The scheduler now correctly handles the iterative refinement process that is core to DLLM generation!
