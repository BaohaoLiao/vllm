# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Constants and enums for DLLM (Diffusion Language Model) support."""

from enum import IntEnum, auto


class DLLMMaskState(IntEnum):
    """DLLM mask states for tokens.

    Attributes:
        MASKED: Token is masked and needs refinement
        UNMASKED: Token is predicted and revealed
        CACHED: Token is finalized and cached
    """
    MASKED = 0
    UNMASKED = 1
    CACHED = 2


class DLLMUnmaskingStrategy(IntEnum):
    """Strategies for unmasking tokens during DLLM generation.

    Attributes:
        SEQUENTIAL: Unmask tokens left-to-right sequentially
        LOW_CONFIDENCE_DYNAMIC: Unmask tokens with confidence below threshold
        LOW_CONFIDENCE_STATIC: Unmask top-K lowest confidence tokens per block
    """
    SEQUENTIAL = auto()
    LOW_CONFIDENCE_DYNAMIC = auto()
    LOW_CONFIDENCE_STATIC = auto()

    @classmethod
    def from_str(cls, name: str) -> "DLLMUnmaskingStrategy":
        """Convert string to DLLMUnmaskingStrategy enum."""
        name_upper = name.upper()
        if name_upper == "SEQUENTIAL":
            return cls.SEQUENTIAL
        elif name_upper == "LOW_CONFIDENCE_DYNAMIC":
            return cls.LOW_CONFIDENCE_DYNAMIC
        elif name_upper == "LOW_CONFIDENCE_STATIC":
            return cls.LOW_CONFIDENCE_STATIC
        else:
            raise ValueError(f"Unknown unmasking strategy: {name}")
