# DLLM KV Cache Strategy

## Overview

DLLM uses a **different KV caching strategy** than autoregressive models:
- **Autoregressive**: Append-only KV cache (1 new token = 1 new KV slot)
- **DLLM**: In-place overwrite during refinement (reuse KV slots for current block)

## Key Concept: Two Cache Regions

```
KV Cache Memory Layout:
┌───────────────────────┬─────────────────────┐
│   history_cache       │   current_block     │
│   (immutable)         │   (mutable)         │
│   [CACHED blocks]     │   [REFINING block]  │
└───────────────────────┴─────────────────────┘
        ↑                        ↑
   num_history_ids         num_history_ids +
                           block_size
```

### History Cache (Immutable)
- Completed blocks marked as `CACHED`
- KV values are **frozen** and won't change
- Positions: `[0, num_history_ids)`

### Current Block (Mutable)
- Block being refined through denoising
- KV values **overwritten** each iteration
- Positions: `[num_history_ids, num_history_ids + block_size)`

## Generation Flow

### Example: Generate 8 tokens with block_size=4

```
=== BLOCK 1 ===

Iteration 1:
  Positions:    [0, 1, 2, 3]
  Tokens:       [t0, m, m, m]  (m = mask token)
  Mask:         [U, M, M, M]
  KV Cache:
    [0:4] ← compute_kv([t0, m, m, m])  # WRITE

Iteration 2:
  Positions:    [0, 1, 2, 3]  # Same positions!
  Tokens:       [t0, t1, m, m]
  Mask:         [U, U, M, M]
  KV Cache:
    [0:4] ← compute_kv([t0, t1, m, m])  # OVERWRITE

Iteration 3:
  Positions:    [0, 1, 2, 3]
  Tokens:       [t0, t1, t2, m]
  Mask:         [U, U, U, M]
  KV Cache:
    [0:4] ← compute_kv([t0, t1, t2, m])  # OVERWRITE

Iteration 4:
  Positions:    [0, 1, 2, 3]
  Tokens:       [t0, t1, t2, t3]
  Mask:         [U, U, U, U]  # All unmasked!
  KV Cache:
    [0:4] ← compute_kv([t0, t1, t2, t3])  # FINAL OVERWRITE

Block 1 Complete:
  Mask → [C, C, C, C]  # Mark as CACHED
  num_history_ids: 0 → 4  # Advance cache boundary

=== BLOCK 2 ===

Iteration 1:
  Positions:    [4, 5, 6, 7]  # New positions
  Tokens:       [m, m, m, m]
  Mask:         [M, M, M, M]
  KV Cache:
    [0:4] = Block 1 (frozen)
    [4:8] ← compute_kv([m, m, m, m])  # WRITE NEW BLOCK

Iteration 2-4: (similar refinement of Block 2)
  ...

Block 2 Complete:
  num_history_ids: 4 → 8
```

## vLLM Integration

### 1. Scheduler Changes

```python
# In Scheduler.schedule()

if request.is_dllm:
    if should_continue_refining_block(request):
        # REFINEMENT: Reuse existing KV slots
        num_new_tokens = 0

        # Don't allocate new blocks
        # Model will overwrite existing KV cache
        allocated_blocks = []  # Empty!

        # Track which positions to compute
        compute_positions = range(
            request.num_history_tokens,
            request.num_history_tokens + request.dllm_block_size
        )

    elif check_dllm_ready_for_next_block(request):
        # ADVANCEMENT: Allocate new KV slots
        num_new_tokens = request.dllm_block_size

        # Allocate fresh KV blocks
        allocated_blocks = kv_cache_manager.allocate(
            request,
            num_tokens=num_new_tokens
        )

        # Initialize new block
        request.dllm_advance_to_next_block()
```

### 2. KV Cache Manager

Need to support **rewriting** existing slots:

```python
class KVCacheManager:

    def get_reusable_slots(self, request: Request) -> List[int]:
        """Get KV cache slots for current block (reuse during refinement)."""
        if not request.is_dllm:
            return []

        # Return slots for current block
        start_pos = request.num_history_tokens
        block_size = request.dllm_block_size

        # These slots were allocated when block started
        # Now we're reusing them for refinement
        return request.kv_cache_blocks[start_pos:start_pos + block_size]

    def mark_block_complete(self, request: Request) -> None:
        """Mark current block as complete (immutable)."""
        if not request.is_dllm:
            return

        # Freeze these KV slots
        start_pos = request.num_history_tokens
        block_size = request.dllm_block_size

        for slot in request.kv_cache_blocks[start_pos:start_pos + block_size]:
            slot.mark_immutable()
```

### 3. Model Runner

```python
# In GPUModelRunner.execute_model()

def execute_model(self, scheduler_output):
    model_input = self._prepare_inputs(scheduler_output)

    for request in scheduler_output.scheduled_requests:
        if request.is_dllm:
            # Determine if refining or advancing
            if should_continue_refining_block(request):
                # Reuse existing KV slots
                kv_mode = "OVERWRITE"

                # Get positions to recompute
                start = request.num_history_tokens
                positions = range(start, start + request.dllm_block_size)

            else:
                # New block, use append mode
                kv_mode = "APPEND"

                # Get new positions
                positions = range(
                    request.num_tokens,
                    request.num_tokens + request.dllm_block_size
                )

    # Pass kv_mode to attention
    outputs = self.model(
        input_ids=model_input.input_ids,
        positions=positions,
        kv_cache=model_input.kv_cache,
        kv_mode=kv_mode,  # NEW: "APPEND" or "OVERWRITE"
    )
```

## Memory Efficiency

### Comparison (16 tokens total):

**Autoregressive**:
```
KV Cache Growth:
Step  1: [K₀,V₀]
Step  2: [K₀,V₀, K₁,V₁]
Step  3: [K₀,V₀, K₁,V₁, K₂,V₂]
...
Step 16: [K₀,V₀, ..., K₁₅,V₁₅]

Peak memory: 16 KV pairs
Allocations: 16 (incremental)
```

**DLLM (block_size=4)**:
```
KV Cache Growth:
Iter 1-1: [K₀,V₀, K₁,V₁, K₂,V₂, K₃,V₃]          # Allocate
Iter 1-2: [K₀,V₀, K₁,V₁, K₂,V₂, K₃,V₃]          # Overwrite
Iter 1-3: [K₀,V₀, K₁,V₁, K₂,V₂, K₃,V₃]          # Overwrite
Iter 1-4: [K₀,V₀, K₁,V₁, K₂,V₂, K₃,V₃]          # Overwrite
---
Iter 2-1: [..., K₄,V₄, K₅,V₅, K₆,V₆, K₇,V₇]     # Allocate
Iter 2-2: [..., K₄,V₄, K₅,V₅, K₆,V₆, K₇,V₇]     # Overwrite
...

Peak memory: 16 KV pairs (same)
Allocations: 4 (by blocks)
Model calls: 16 (4 blocks × 4 iterations)
```

### Trade-offs:
- ✅ **Fewer allocations** (block-wise vs token-wise)
- ✅ **Same peak memory** (final size identical)
- ❌ **More computation** (recompute KV each iteration)
- ❌ **Higher latency** (multiple passes per block)

## Attention Mechanism

The attention layer needs to handle **position reuse**:

```python
class Attention:
    def forward(self, q, k, v, kv_cache, positions, kv_mode="APPEND"):
        if kv_mode == "APPEND":
            # Normal AR behavior
            # Append new KV to cache
            kv_cache.append(k, v, positions)

        elif kv_mode == "OVERWRITE":
            # DLLM refinement
            # Overwrite existing KV at these positions
            kv_cache.write(k, v, positions)  # In-place update

        # Compute attention (same for both modes)
        output = scaled_dot_product_attention(q, kv_cache.k, kv_cache.v)
        return output
```

## Important Considerations

### 1. Block Alignment
- All operations must be block-aligned
- Partial blocks should be padded with mask tokens

### 2. Position IDs
- During refinement: **Same position IDs** across iterations
- After completion: **New position IDs** for next block

### 3. Cache Invalidation
- Once block is CACHED, those slots are **immutable**
- Preemption must respect block boundaries

### 4. Prefix Caching
- Can cache completed blocks across requests
- Must include both token_ids AND mask state

## Summary

**Key Insight**: DLLM uses **in-place KV cache updates** during block refinement, then **freezes** the block when complete.

**Implementation Requirements**:
1. Track `num_history_tokens` (cached positions)
2. Support KV cache **overwrite** mode
3. Allocate by **blocks**, not tokens
4. Mark completed blocks as **immutable**

**Memory Efficiency**:
- Same peak memory as AR
- Fewer allocations (block-wise)
- More computation (iterative refinement)

This strategy enables efficient DLLM generation while maintaining compatibility with vLLM's KV cache architecture.
