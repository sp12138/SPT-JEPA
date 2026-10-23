# Frozen codec provenance

Only model inference components are adapted. Upstream training loops,
registries, distributed code, logging, downloads and attention dispatchers are
not runtime dependencies. All attention calls use `vforge.ops`.

- VideoRAE frozen encoder/compressor/decoder: local upstream revision
  `ac6fc3c9b8be8fe7ae96c8ec3507cffc874e4aa0`, adapted in `videorae.py`.
  The MIT license is preserved in `licenses/VIDEORAE-LICENSE`.
  Only the published 3D 1024x64 and 1D 512x32 inference paths are adapted;
  REPA alignment heads, upstream training, registries and downloaders are not.
  Embedded V-JEPA2/V-JEPA2.1 teachers retain their respective rotary/fusion
  behavior, including V-JEPA2's historical frequency repetition.

- V-RAE architecture and released-checkpoint contract:
  <https://github.com/V-RAE/V-RAE>, local upstream revision
  `3ab8d037905086542c4bf46ac8c428b23767a923`.
  Adapted pooling, reconstruction decoder, encoder feature-fusion and weight
  names. The inspected upstream checkout has no root license file; no broader
  redistribution permission is asserted here. Review upstream licensing before
  distributing these adaptations outside the authorized research workspace.
- EUPE backbone: the V-RAE revision's `third_party/eupe`, adapted in `eupe.py`.
  Copyright Meta Platforms, Inc. and affiliates. The original FAIR
  Noncommercial Research License and Acceptable Use Policy are preserved in
  `licenses/EUPE-LICENSE.md`; this component is not relicensed by vforge.
- V-JEPA2.1 backbone: the V-RAE revision's `third_party/vjepa2`.
  Copyright Meta Platforms, Inc. and affiliates. Original license files are
  preserved in `licenses/VJEPA-LICENSE` and `licenses/VJEPA-APACHE-LICENSE`.
  `encoders.py` reuses the Native projection/attention blocks and retains the
  official frozen feature-fusion semantics.

Model weights are external user-supplied assets, not included in this repo.
Their own licenses still apply. V-RAE checkpoints include the frozen encoder
weights, so Native loading does not download or initialize an extra encoder
checkpoint before replacing it with the exact same stored state.
