"""Model-independent contracts and storage for block-causal inference."""

from vforge.models.causal.cache import (
    AttentionFn,
    CausalKVCache,
    CausalKVCacheLayer,
    allocate_kv_cache,
)
from vforge.models.causal.context import (
    IncrementalKVAttentionContext,
    get_incremental_kv_cache,
    use_incremental_kv_cache,
)
from vforge.models.causal.executor import (
    IncrementalKVBlockForward,
    IncrementalKVExecutor,
    IncrementalKVGuidanceCaches,
    allocate_incremental_kv_guidance_caches,
)
from vforge.models.causal.protocol import IncrementalKVTransformer

__all__ = [
    "AttentionFn",
    "CausalKVCache",
    "CausalKVCacheLayer",
    "IncrementalKVBlockForward",
    "IncrementalKVExecutor",
    "IncrementalKVGuidanceCaches",
    "IncrementalKVAttentionContext",
    "IncrementalKVTransformer",
    "allocate_incremental_kv_guidance_caches",
    "allocate_kv_cache",
    "get_incremental_kv_cache",
    "use_incremental_kv_cache",
]
