"""Attention dispatch and FlashAttention backends."""

from vforge.ops._attention.dispatch import (
    attention_debug_enabled,
    get_attn_implementation,
    get_flash_attention_capability,
    set_attention_debug,
    set_flash_attention_backend,
    set_flash_attention_deterministic,
    set_sdpa_backend,
    unified_attention,
    unified_attention_varlen,
)
from vforge.ops._attention.flash import (
    AttentionBackend,
    AttentionCallMetadata,
    FA3Backend,
    FA4Backend,
    load_fa3_backend,
    load_fa4_backend,
)
from vforge.ops._attention.policy import AttentionCallPolicy
from vforge.ops._attention.segmented import (
    INVALID_SEGMENT,
    SegmentedSparsity,
    segmented_available,
    segmented_dense_mask,
)
from vforge.ops._attention.teacher_forcing import (
    TeacherForcingSparsity,
    teacher_forcing_available,
    teacher_forcing_dense_mask,
    teacher_forcing_deterministic_available,
    teacher_forcing_sparsity_stats,
)

__all__ = [
    "AttentionBackend",
    "AttentionCallPolicy",
    "AttentionCallMetadata",
    "FA3Backend",
    "FA4Backend",
    "INVALID_SEGMENT",
    "SegmentedSparsity",
    "TeacherForcingSparsity",
    "attention_debug_enabled",
    "get_attn_implementation",
    "get_flash_attention_capability",
    "load_fa3_backend",
    "load_fa4_backend",
    "set_attention_debug",
    "set_flash_attention_backend",
    "set_flash_attention_deterministic",
    "set_sdpa_backend",
    "segmented_available",
    "segmented_dense_mask",
    "teacher_forcing_available",
    "teacher_forcing_dense_mask",
    "teacher_forcing_deterministic_available",
    "teacher_forcing_sparsity_stats",
    "unified_attention",
    "unified_attention_varlen",
]
