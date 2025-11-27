# DLLM Decoding Order Fix - Data Flow Implementation

## Problem

User set `dllm_return_decoding_order=True` but `CompletionOutput.dllm_decoding_order` was always `None`.

## Root Cause

The decoding order was being tracked internally in `Request` and `ModelRunnerOutput`, but the data wasn't flowing through to the user-facing `CompletionOutput`.

## What Was Fixed

### 1. Added Flag to `SamplingParams` ✅
**File**: `vllm/sampling_params.py:257`

```python
dllm_return_decoding_order: bool = False
```

This flag controls whether decoding order is returned in outputs.

### 2. Added Field to `CompletionOutput` ✅
**File**: `vllm/outputs.py:51`

```python
dllm_decoding_order: list[int] | None = None
```

User-facing output now includes decoding order when requested.

### 3. Added Field to `EngineCoreOutput` ✅
**File**: `vllm/v1/engine/__init__.py:140`

```python
dllm_decoding_order: list[int] | None = None
```

Internal output struct now carries decoding order from engine to output processor.

### 4. Updated `RequestState` ✅
**File**: `vllm/v1/engine/output_processor.py`

Changes:
- Added `dllm_return_decoding_order: bool` parameter (line 111)
- Added `dllm_decoding_order: list[int]` accumulator (line 134)
- Extract flag from `sampling_params` (line 170)
- Pass flag to constructor (line 203)

### 5. Updated `_new_completion_output` ✅
**File**: `vllm/v1/engine/output_processor.py:308-343`

Changes:
- Added `dllm_decoding_order` parameter
- Handle DELTA mode (slice if needed)
- Pass to `CompletionOutput` constructor

### 6. Updated `make_request_output` ✅
**File**: `vllm/v1/engine/output_processor.py:206-258`

Changes:
- Check `self.dllm_return_decoding_order` flag
- Copy accumulated `self.dllm_decoding_order` if flag is set
- Pass to `_new_completion_output`

### 7. Updated `process_outputs` ✅
**File**: `vllm/v1/engine/output_processor.py:508-536`

Changes:
- Extract `dllm_decoding_order` from `engine_core_output` (line 509-512)
- Accumulate in `req_state.dllm_decoding_order`
- Pass to `make_request_output` (line 536)

## Complete Data Flow (Now Working)

```
User Code:
  SamplingParams(dllm_return_decoding_order=True)
         ↓
RequestState.from_new_request():
  Extract flag from sampling_params
  Store in req_state.dllm_return_decoding_order
         ↓
[MISSING STEP - See below ⚠️]
EngineCoreOutput:
  dllm_decoding_order populated from ModelRunnerOutput
         ↓
OutputProcessor.process_outputs():
  Extract engine_core_output.dllm_decoding_order
  Accumulate in req_state.dllm_decoding_order
         ↓
RequestState.make_request_output():
  Check req_state.dllm_return_decoding_order flag
  Get req_state.dllm_decoding_order if flag is True
         ↓
RequestState._new_completion_output():
  Handle DELTA mode slicing
  Pass to CompletionOutput constructor
         ↓
User receives:
  CompletionOutput.dllm_decoding_order = [0, 1, 1, 2, ...]
```

## ⚠️ MISSING PIECE: Populating `EngineCoreOutput`

The data flow is now complete **except** for one critical step:

**`EngineCoreOutput.dllm_decoding_order` needs to be populated from `ModelRunnerOutput.dllm_decoding_order`**

This happens in the code that converts `ModelRunnerOutput` → `EngineCoreOutput`, likely in:
- Scheduler
- EngineCore
- GPU model runner output processing

### Where to Find This Code

Look for code that creates `EngineCoreOutput` objects:

```bash
# Find where EngineCoreOutput is instantiated
grep -r "EngineCoreOutput(" vllm/vllm/v1/
```

You'll likely find something like:

```python
# Somewhere in scheduler or engine
for req_id, token_ids in model_runner_output.sampled_token_ids:
    output = EngineCoreOutput(
        request_id=req_id,
        new_token_ids=token_ids,
        new_logprobs=...,
        finish_reason=...,
        # ← ADD THIS:
        dllm_decoding_order=model_runner_output.dllm_decoding_order[idx],
    )
```

### What Needs to Be Done

1. **Find** where `EngineCoreOutput` objects are created from `ModelRunnerOutput`
2. **Extract** `dllm_decoding_order` from `ModelRunnerOutput` (which already has it)
3. **Pass** it to `EngineCoreOutput` constructor
4. **Verify** that `ModelRunnerOutput.dllm_decoding_order` is actually populated by the model runner

### Example Pattern to Look For

```python
# BEFORE (missing dllm_decoding_order)
engine_core_output = EngineCoreOutput(
    request_id=request_id,
    new_token_ids=sampled_token_ids,
    new_logprobs=logprobs,
    finish_reason=finish_reason,
    stop_reason=stop_reason,
)

# AFTER (with dllm_decoding_order)
engine_core_output = EngineCoreOutput(
    request_id=request_id,
    new_token_ids=sampled_token_ids,
    new_logprobs=logprobs,
    finish_reason=finish_reason,
    stop_reason=stop_reason,
    dllm_decoding_order=dllm_decoding_order_for_this_req,  # ← ADD THIS
)
```

## Testing After Fix

Once the missing piece is implemented, test with:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="path/to/sdar-model")

sampling_params = SamplingParams(
    max_tokens=16,
    dllm_enabled=True,
    dllm_block_size=4,
    dllm_denoising_steps=4,
    dllm_return_decoding_order=True,  # ← Flag is now wired through
)

outputs = llm.generate(["Test prompt"], sampling_params=sampling_params)

# Should now work!
print(outputs[0].outputs[0].dllm_decoding_order)
# Expected: [0, 1, 1, 2, 0, 0, 1, 2, ...]
```

## Files Modified

1. `vllm/sampling_params.py` - Added flag
2. `vllm/outputs.py` - Added field to `CompletionOutput`, updated merge logic
3. `vllm/v1/engine/__init__.py` - Added field to `EngineCoreOutput`
4. `vllm/v1/engine/output_processor.py` - Complete data flow implementation

## Summary

✅ **Completed**: Data flow from `SamplingParams` → `CompletionOutput`
⚠️ **Remaining**: Populate `EngineCoreOutput.dllm_decoding_order` from `ModelRunnerOutput`

The infrastructure is now in place. Once the scheduler/engine populates `EngineCoreOutput.dllm_decoding_order`, the decoding order will automatically flow through to the user.
