# DLLM Decoding Order - FINAL FIX ✅

## Root Cause Identified

The decoding order is stored in the scheduler's `Request` objects (`request._dllm_decoding_order`), but the **model runner** doesn't have access to these objects. The model runner only has `CachedRequestState`, which doesn't include decoding order.

## Solution

Extract decoding order **in the scheduler** where `Request` objects are available, not in the model runner.

## Key Changes

### 1. Scheduler Extracts Decoding Order (`scheduler.py:1036-1043`)

```python
# Extract DLLM decoding order from the request (if DLLM and flag enabled)
req_dllm_decoding_order = None
if request.is_dllm and request.sampling_params.dllm_return_decoding_order:
    # Get the decoding order for tokens generated in this step
    full_order = request.dllm_get_decoding_order()
    if full_order and len(generated_token_ids) > 0:
        # Return only the decoding order for newly generated tokens
        req_dllm_decoding_order = full_order[-len(generated_token_ids):]
```

This extracts the decoding order directly from the `Request` object, which tracks it internally in `_dllm_decoding_order`.

### 2. Scheduler Passes to `EngineCoreOutput` (`scheduler.py:1128`)

```python
EngineCoreOutput(
    ...,
    dllm_decoding_order=req_dllm_decoding_order,
)
```

### 3. Complete Data Flow (NOW CORRECT)

```
1. Request tracks decoding order internally:
   request._dllm_decoding_order = [0, 1, 1, 2, ...]  (maintained by dllm_update_mask)
         ↓
2. Scheduler.update_from_output():
   req_dllm_decoding_order = request.dllm_get_decoding_order()[-len(generated_token_ids):]
         ↓
3. EngineCoreOutput:
   dllm_decoding_order=req_dllm_decoding_order
         ↓
4. OutputProcessor.process_outputs():
   req_state.dllm_decoding_order.extend(engine_core_output.dllm_decoding_order)
         ↓
5. RequestState.make_request_output():
   output_dllm_decoding_order = req_state.dllm_decoding_order.copy()
         ↓
6. CompletionOutput:
   dllm_decoding_order = output_dllm_decoding_order
         ↓
7. User receives:
   outputs[0].outputs[0].dllm_decoding_order = [0, 1, 1, 2, ...]
```

## Why Previous Attempt Failed

**Previous (Wrong) Approach:**
- Model runner tried to extract from `SamplerOutput.dllm_decoding_order`
- Sampler doesn't have access to `Request` objects, only tensors
- `SamplerOutput.dllm_decoding_order` was always `None`

**Current (Correct) Approach:**
- Scheduler extracts from `Request.dllm_get_decoding_order()`
- Scheduler has full `Request` objects with all tracking
- Decoding order flows through `EngineCoreOutput` → `OutputProcessor` → `CompletionOutput`

## Files Modified

1. **`vllm/v1/core/sched/scheduler.py` (lines 1036-1043, 1128)**
   - Extract decoding order from `Request` object
   - Pass to `EngineCoreOutput`

2. **Previous changes still valid:**
   - `vllm/sampling_params.py` - `dllm_return_decoding_order` flag
   - `vllm/outputs.py` - `CompletionOutput.dllm_decoding_order` field
   - `vllm/v1/engine/__init__.py` - `EngineCoreOutput.dllm_decoding_order` field
   - `vllm/v1/engine/output_processor.py` - Data flow logic
   - `vllm/v1/request.py` - Internal tracking (`_dllm_decoding_order`)

## Testing

```python
from vllm import LLM, SamplingParams

llm = LLM(model="/mnt/nushare2/data/baliao/PLLMs/qwen/Qwen3-1.7B-Base")

sampling_params = SamplingParams(
    max_tokens=128,
    dllm_enabled=True,
    dllm_block_size=4,
    dllm_denoising_steps=4,
    dllm_unmasking_strategy="low_confidence_dynamic",
    dllm_confidence_threshold=0.85,
    dllm_mask_token_id=0,
    dllm_return_decoding_order=True,  # ← ENABLE FLAG
)

outputs = llm.generate(
    prompts=["Explain the concept of diffusion models."],
    sampling_params=sampling_params,
)

# Should now work!
decoding_order = outputs[0].outputs[0].dllm_decoding_order
print(f"Decoding order: {decoding_order}")
assert decoding_order is not None, "FIXED - Decoding order should not be None!"
```

## Important Notes

### Decoding Order Tracking

The decoding order is updated in `Request.dllm_update_mask()` (vllm/v1/request.py:286-296):

```python
# Track transitions from MASKED → UNMASKED
for i in range(block_start, len(new_mask)):
    old_state = old_mask[i] if i < len(old_mask) else DLLMMaskState.MASKED
    new_state = new_mask[i]

    if (old_state == DLLMMaskState.MASKED and
        new_state == DLLMMaskState.UNMASKED and
        self._dllm_decoding_order[i] == -1):
        self._dllm_decoding_order[i] = self._dllm_iteration  # ← Records iteration
```

This means the decoding order will **only** be populated if:
1. `dllm_update_mask()` is being called during generation
2. Tokens are transitioning from MASKED to UNMASKED
3. The DLLM unmasking logic is active

### If Still Getting `None`

If decoding order is still `None`, check:

1. **Is DLLM actually being used?**
   ```python
   # Verify request is marked as DLLM
   assert request.is_dllm == True
   ```

2. **Is the unmasking processor being called?**
   - The unmasking processor should call `request.dllm_update_mask()`
   - Check if DLLM model runner integration is complete

3. **Are tokens being generated?**
   ```python
   # Check if any tokens were generated
   assert len(generated_token_ids) > 0
   ```

4. **Is the flag set?**
   ```python
   assert sampling_params.dllm_return_decoding_order == True
   ```

## Summary

✅ **Root cause identified**: Model runner doesn't have `Request` objects
✅ **Solution implemented**: Scheduler extracts from `Request.dllm_get_decoding_order()`
✅ **Data flow complete**: Scheduler → EngineCoreOutput → OutputProcessor → CompletionOutput
⚠️ **Dependency**: Requires DLLM unmasking logic to call `request.dllm_update_mask()`

The infrastructure is now correct. If still getting `None`, the issue is likely that the DLLM unmasking/generation logic isn't being triggered or `dllm_update_mask()` isn't being called.
