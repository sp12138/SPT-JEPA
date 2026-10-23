"""Wan 2.2 TI2V 5B bidirectional model definition.

Bidirectional text+image-to-video: the reference frame is in-painted into
latent frame 0 rather than fed through an image side branch, so this plugin
carries a per-token timestep and a widened patch embedding instead of the
2.1 cross-attention image path. Every token attends to every other one, which
is the delta against the block-causal sibling in
:mod:`vforge.models.wan.v22.ti2v_5b.teacher_forcing`.

Variant-neutral roots, frozen encoders, conditioning contracts, and sampling
helpers live in :mod:`vforge.models.wan`; this package keeps only TI2V's
per-token timestep and channel-concatenation behavior.

Heavy modeling imports (diffusers transformer/encoder) stay lazy so importing
this package works in CPU/Hydra smoke environments. Task assembly lives in
``vforge.tasks.wan.v22.ti2v_5b``; nothing here may import the training layer.
"""
