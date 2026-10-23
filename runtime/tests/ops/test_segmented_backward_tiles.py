"""Short SM100 queries need independently tiled sparse backward metadata."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from vforge.ops._attention import segmented


@pytest.mark.parametrize("length", [32, 96, 128, 129, 256])
def test_short_query_backward_reclassifies_tail(monkeypatch, length):
    # Metadata construction is a CPU contract; the actual FA4 kernel is tested
    # separately on B300. The carrier has no computational behavior.
    carrier = ModuleType("flash_attn.cute.block_sparsity")
    carrier.BlockSparseTensorsTorch = SimpleNamespace
    monkeypatch.setitem(sys.modules, carrier.__name__, carrier)
    monkeypatch.setattr(segmented, "segmented_available", lambda: (True, ""))
    segmented._SEGMENTED_METADATA_CACHE.clear()
    spec = segmented.SegmentedSparsity((0,) * length, (0,) * 128, ((0, 0),))
    try:
        fwd, bwd = segmented.build_segmented_metadata(spec, device="cpu")
        assert fwd.block_size == (128 if length <= 128 else 256, 128)
        assert bwd.block_size == (256, 128)
        # With a padded Q tail the whole backward block must invoke mask_mod.
        assert int(bwd.full_block_cnt.sum()) == (1 if length == 256 else 0)
        assert int(bwd.mask_block_cnt.sum()) == (0 if length == 256 else 1)
        if length == 128:
            assert int(fwd.full_block_cnt.sum()) == 1
            assert int(fwd.mask_block_cnt.sum()) == 0
        again = segmented.build_segmented_metadata(spec, device="cpu")
        assert again[0] is fwd and again[1] is bwd
    finally:
        segmented._SEGMENTED_METADATA_CACHE.clear()
