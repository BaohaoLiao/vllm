# DLLM Scheduler Integration Guide

This document provides detailed instructions for integrating DLLM support into the vLLM V1 Scheduler.

## Overview

The Scheduler needs to be modified in two key methods:
1. `schedule()` - To handle block-aligned scheduling
2. `update_from_output()` - To process DLLM outputs and update request state

## Helper Module

We've created `vllm/v1/core/sched/dllm_helper.py` with utility functions:
- `should_continue_refining_block()` - Check if block needs more refinement
- `get_dllm_num_new_tokens()` - Calculate tokens to schedule
- `prepare_dllm_next_block()` - Initialize new block
- `update_dllm_from_output()` - Process model outputs
- `check_dllm_ready_for_next_block()` - Check if ready to advance

## Integration Points

### 1. In `schedule()` method

**Location**: After line 227 where `num_new_tokens` is calculated

```python
# Around line 227-234 in scheduler.py
num_new_tokens = (
    request.num_tokens_with_spec
    + request.num_output_placeholders
    - request.num_computed_tokens
)
if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
    num_new_tokens = self.scheduler_config.long_prefill_token_threshold
num_new_tokens = min(num_new_tokens, token_budget)

# ADD THIS BLOCK:
# DLLM: Override num_new_tokens for DLLM requests
if request.is_dllm:
    from vllm.v1.core.sched.dllm_helper import (
        should_continue_refining_block,
        get_dllm_num_new_tokens,
        check_dllm_ready_for_next_block,
        prepare_dllm_next_block,
    )

    if should_continue_refining_block(request):
        # Continue refining current block - no new KV cache slots
        num_new_tokens = 0
        # Don't skip this request, it needs to run for refinement
        # Fall through to normal scheduling logic
    elif check_dllm_ready_for_next_block(request):
        # Block is complete, prepare next block
        prepare_dllm_next_block(request)
        num_new_tokens = request.dllm_block_size
        # Allocate new KV cache slots for next block
    else:
        # Still refining and not yet complete
        num_new_tokens = 0
```

**Rationale**:
- DLLM doesn't generate 1 token at a time like AR models
- During refinement iterations: `num_new_tokens = 0` (no new KV slots)
- When advancing to next block: `num_new_tokens = block_size` (allocate new slots)

### 2. In `schedule()` - Handle num_new_tokens == 0 for DLLM

**Location**: Around line 263-272 where `num_new_tokens == 0` is handled

```python
if num_new_tokens == 0:
    # The request cannot be scheduled because one of the following
    # reasons:
    # 1. No new tokens to schedule. This may happen when
    #    (1) PP>1 and we have already scheduled all prompt tokens
    #    but they are not finished yet.
    #    (2) Async scheduling and the request has reached to either
    #    its max_total_tokens or max_model_len.
    # 2. The encoder budget is exhausted.
    # 3. The encoder cache is exhausted.

    # ADD THIS CHECK:
    # DLLM: Special case - allow scheduling with 0 tokens for refinement
    if request.is_dllm:
        from vllm.v1.core.sched.dllm_helper import should_continue_refining_block

        if should_continue_refining_block(request):
            # Schedule for block refinement even with 0 new tokens
            # This is a DLLM denoising iteration
            scheduled_running_reqs.append(request)
            num_scheduled_tokens[request.request_id] = request.dllm_block_size
            req_index += 1
            continue  # Don't skip, process this request

    # Original logic for non-DLLM or DLLM advancing to next block
    if encoder_inputs_to_schedule is None:
        req_index += 1
        continue
```

**Rationale**:
- DLLM needs to run model forward pass even when `num_new_tokens = 0`
- This is for denoising iterations on the current block
- We still schedule `block_size` tokens for the model to process

### 3. In `update_from_output()` method

**Location**: Around line 1032-1034 where `generated_token_ids` are extracted

```python
# Around line 1032-1034
req_index = model_runner_output.req_id_to_index[req_id]
generated_token_ids = (
    sampled_token_ids[req_index] if sampled_token_ids else []
)

# ADD THIS BLOCK:
# DLLM: Extract mask and process tokens
dllm_mask = None
if request.is_dllm and model_runner_output.dllm_masks:
    dllm_mask = model_runner_output.dllm_masks[req_index]

    # Process DLLM output
    from vllm.v1.core.sched.dllm_helper import update_dllm_from_output

    tokens_to_append, dllm_stopped = update_dllm_from_output(
        request,
        generated_token_ids,
        dllm_mask,
    )

    # Override tokens and stop signal for DLLM
    generated_token_ids = tokens_to_append
    if dllm_stopped:
        stopped = True
```

**Location**: Around line 1076-1082 where tokens are appended to request

```python
# Around line 1076-1082
# Append the generated tokens to the request.
for token_id in generated_token_ids:
    request.append_output_token_ids(token_id)

# MODIFY TO:
# DLLM: Only append unmasked tokens (already filtered in update_dllm_from_output)
for token_id in generated_token_ids:
    request.append_output_token_ids(token_id)

# For DLLM, generated_token_ids already contains only unmasked tokens
# No additional filtering needed here
```

**Rationale**:
- DLLM generates multiple tokens but only some are unmasked
- `update_dllm_from_output()` filters to only unmasked tokens
- Updates request's internal DLLM mask state

## Complete Integration Example

Here's a minimal example showing the key modifications:

```python
# In Scheduler.schedule()
def schedule(self) -> SchedulerOutput:
    # ... existing code ...

    while req_index < len(self.running) and token_budget > 0:
        request = self.running[req_index]

        # Calculate num_new_tokens
        num_new_tokens = (
            request.num_tokens_with_spec
            + request.num_output_placeholders
            - request.num_computed_tokens
        )
        num_new_tokens = min(num_new_tokens, token_budget)

        # DLLM INTEGRATION POINT 1
        if request.is_dllm:
            from vllm.v1.core.sched.dllm_helper import (
                should_continue_refining_block,
                check_dllm_ready_for_next_block,
                prepare_dllm_next_block,
            )

            if should_continue_refining_block(request):
                num_new_tokens = 0
            elif check_dllm_ready_for_next_block(request):
                prepare_dllm_next_block(request)
                num_new_tokens = request.dllm_block_size

        # DLLM INTEGRATION POINT 2
        if num_new_tokens == 0:
            if request.is_dllm and should_continue_refining_block(request):
                scheduled_running_reqs.append(request)
                num_scheduled_tokens[request.request_id] = request.dllm_block_size
                req_index += 1
                continue
            # ... existing skip logic ...

        # ... rest of scheduling logic ...


# In Scheduler.update_from_output()
def update_from_output(
    self,
    scheduler_output: SchedulerOutput,
    model_runner_output: ModelRunnerOutput,
) -> dict[int, EngineCoreOutputs]:
    # ... existing code ...

    for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
        request = self.requests.get(req_id)
        req_index = model_runner_output.req_id_to_index[req_id]
        generated_token_ids = sampled_token_ids[req_index]

        # DLLM INTEGRATION POINT 3
        if request.is_dllm:
            from vllm.v1.core.sched.dllm_helper import update_dllm_from_output

            dllm_mask = (
                model_runner_output.dllm_masks[req_index]
                if model_runner_output.dllm_masks
                else None
            )

            generated_token_ids, dllm_stopped = update_dllm_from_output(
                request,
                generated_token_ids,
                dllm_mask,
            )

            if dllm_stopped:
                stopped = True

        # Append tokens (already filtered for DLLM)
        for token_id in generated_token_ids:
            request.append_output_token_ids(token_id)

        # ... rest of update logic ...
```

## Testing the Integration

After integration, test with:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="sdar-model-path")

sampling_params = SamplingParams(
    max_tokens=16,
    dllm_enabled=True,
    dllm_block_size=4,
    dllm_denoising_steps=4,
    dllm_unmasking_strategy="low_confidence_dynamic",
)

outputs = llm.generate(
    prompts=["Test prompt"],
    sampling_params=sampling_params,
)

# Should generate 16 tokens in 4 blocks of 4 tokens each
# Each block should go through 4 denoising iterations
assert len(outputs[0].outputs[0].token_ids) == 16
```

## Debugging Tips

Add logging to track DLLM scheduling:

```python
from vllm.v1.core.sched.dllm_helper import get_dllm_schedule_info

if request.is_dllm:
    info = get_dllm_schedule_info(request)
    logger.debug(f"DLLM schedule info for {req_id}: {info}")
```

This will log:
- Current iteration
- Block completion status
- Number of valid tokens
- Whether refinement should continue

## Summary

The key insight for DLLM scheduling:
1. **During refinement**: Schedule with `num_new_tokens = 0` but still run model
2. **When advancing**: Schedule with `num_new_tokens = block_size` and allocate KV cache
3. **In updates**: Only append unmasked tokens to output

This allows DLLM to iteratively refine tokens in blocks while maintaining compatibility with vLLM's scheduling architecture.
