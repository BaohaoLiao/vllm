# Adding SDAR Model Support to vLLM

## Overview

SDAR (Structured Discrete AutoRegressive) is a diffusion-based language model currently only supported in LMDeploy. This guide explains how to add SDAR support to vLLM.

## Current Status

- ❌ SDAR not in vLLM
- ✅ SDAR in LMDeploy at `lmdeploy/pytorch/models/sdar.py`
- ✅ DLLM infrastructure ready in vLLM (from our implementation)

## Step-by-Step Implementation

### 1. Copy SDAR Model Implementation

**Source**: `lmdeploy/lmdeploy/pytorch/models/sdar.py`
**Destination**: `vllm/vllm/model_executor/models/sdar.py`

**Key components to port**:
- `SDARAttention` - Attention with block sparsity
- `SDARDecoderLayer` - Decoder layer
- `SDARModel` - Base model
- `SDARForCausalLM` - CausalLM wrapper

### 2. Adapt Imports

Change LMDeploy imports to vLLM equivalents:

```python
# LMDeploy imports:
from lmdeploy.pytorch.nn import (
    ApplyRotaryEmb,
    Attention,
    RMSNorm,
    SiluAndMul,
)
from lmdeploy.pytorch.nn.linear import (
    build_qkv_proj,
    build_o_proj,
    build_gateup_linear,
)

# vLLM equivalents:
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.attention import Attention, AttentionMetadata
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
    MergedColumnParallelLinear,
)
```

### 3. Adapt Attention Mechanism

**LMDeploy SDAR Attention**:
```python
self.attn_fwd = Attention(
    num_heads,
    head_dim,
    num_kv_heads=num_key_value_heads,
    v_head_size=head_dim,
    sliding_window=config.sliding_window,
    block_sparse_size=dllm_block_length,  # DLLM-specific
)
```

**vLLM Adaptation**:
```python
from vllm.attention import Attention

self.attn = Attention(
    num_heads=num_heads,
    head_size=head_dim,
    scale=1.0 / math.sqrt(head_dim),
    num_kv_heads=num_key_value_heads,
    # DLLM-specific: Add block sparsity parameter
    block_sparse_size=config.dllm_block_length,
)
```

### 4. Register SDAR in Model Registry

**File**: `vllm/vllm/model_executor/models/registry.py`

Add to `_TEXT_GENERATION_MODELS`:
```python
_TEXT_GENERATION_MODELS = {
    # ... existing models ...
    "SDARForCausalLM": ("sdar", "SDARForCausalLM"),
    "SdarForCausalLM": ("sdar", "SDARForCausalLM"),  # Alternative naming
}
```

### 5. Configuration Handling

Ensure DLLM config fields are recognized:

**File**: `vllm/config.py` or model-specific config

```python
# In model config, ensure these are loaded:
dllm_block_length: int = 4
dllm_mask_token: int = 0  # Special mask token ID
attention_bias: bool = False
sliding_window: Optional[int] = None
```

### 6. Minimal SDAR Implementation Template

```python
# vllm/vllm/model_executor/models/sdar.py

import math
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors


class SDARAttention(nn.Module):
    """SDAR attention with block sparsity for DLLM."""

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads

        # QKV projection
        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.num_heads,
            self.num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
        )

        # Attention
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            scale=1.0 / math.sqrt(self.head_dim),
            num_kv_heads=self.num_kv_heads,
        )

        # O projection
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
        )

        # Q/K normalization (SDAR-specific)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Rotary embeddings
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [
                self.num_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
            ],
            dim=-1,
        )

        # Apply Q/K normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply rotary embeddings
        q, k = self.rotary_emb(positions, q, k)

        # Attention
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)

        # Output projection
        output, _ = self.o_proj(attn_output)
        return output


class SDARMLP(nn.Module):
    """SDAR MLP."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_up_proj = MergedColumnParallelLinear(
            self.hidden_size,
            [self.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
        )

        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
        )

        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class SDARDecoderLayer(nn.Module):
    """SDAR decoder layer."""

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.self_attn = SDARAttention(config, cache_config, quant_config)
        self.mlp = SDARMLP(config, quant_config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states, kv_cache, attn_metadata)
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class SDARModel(nn.Module):
    """SDAR base model."""

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )

        self.layers = nn.ModuleList(
            [
                SDARDecoderLayer(config, cache_config, quant_config)
                for _ in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)

        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                positions,
                hidden_states,
                kv_caches[i],
                attn_metadata,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


class SDARForCausalLM(nn.Module):
    """SDAR model for causal language modeling with DLLM support."""

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = SDARModel(config, cache_config, quant_config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches, attn_metadata)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states, sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
```

### 7. Testing SDAR

```python
from vllm import LLM, SamplingParams

# Load SDAR model
llm = LLM(model="path/to/sdar-model")

# DLLM sampling
sampling_params = SamplingParams(
    max_tokens=16,
    dllm_enabled=True,
    dllm_block_size=4,
    dllm_denoising_steps=4,
    dllm_unmasking_strategy="low_confidence_dynamic",
)

# Generate
outputs = llm.generate(["Test prompt"], sampling_params=sampling_params)
print(outputs[0].outputs[0].text)
```

## Summary

To add SDAR support to vLLM:

1. ✅ DLLM infrastructure (already done)
2. ⏳ Port SDAR model from LMDeploy
3. ⏳ Adapt to vLLM's layer interfaces
4. ⏳ Register in model registry
5. ⏳ Test with actual SDAR weights

**Effort Estimate**: 1-2 days for experienced vLLM developer

**Key Challenges**:
- Attention layer adaptation (block sparsity)
- Q/K normalization integration
- Weight loading compatibility
- DLLM-specific features

**Alternative**: Use LMDeploy for SDAR until vLLM support is added.
