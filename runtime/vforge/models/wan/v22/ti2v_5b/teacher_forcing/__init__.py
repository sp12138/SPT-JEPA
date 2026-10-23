"""Wan 2.2 TI2V 5B teacher-forcing model definition.

The block-causal autoregressive teacher: the transformer sees a doubled
``[clean | noisy]`` sequence and each noisy token may attend to its own noisy
block plus strictly-earlier clean frames, never to its own clean answer and
never to the future. Training a bidirectional checkpoint this way turns it into
a teacher that can supervise autoregressive students.

Weights, channel layout and conditioning are those of
:mod:`vforge.models.wan.v22.ti2v_5b.ct`, which this plugin subclasses; the delta is
the doubled forward and its mask. The mask itself lives in
:mod:`vforge.ops` as :class:`~vforge.ops.TeacherForcingSparsity`, because
skipping the forbidden tiles is a kernel concern rather than a model one.

Heavy modeling imports stay lazy so importing this package works in CPU/Hydra
smoke environments. Task assembly lives in ``vforge.tasks.wan.v22.ti2v_5b.teacher_forcing``;
nothing here may import the training layer.
"""
