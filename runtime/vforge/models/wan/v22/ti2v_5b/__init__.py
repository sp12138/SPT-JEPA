"""Wan 2.2 TI2V 5B model definitions.

Two forward variants share one checkpoint, channel layout and conditioning
contract:

- :mod:`~vforge.models.wan.v22.ti2v_5b.ct` is the bidirectional definition;
  every token attends to the whole sequence.
- :mod:`~vforge.models.wan.v22.ti2v_5b.teacher_forcing` subclasses it with a
  block-causal forward over a doubled ``[clean | noisy]`` sequence.

This package holds no modeling code of its own, so importing one variant never
drags in the other.
"""
