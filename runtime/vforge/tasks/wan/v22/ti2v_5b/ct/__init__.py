"""Wan 2.2 TI2V 5B bidirectional training assembly.

Training only. The two forward variants share one checkpoint and one cache
format, so only the step recipe is version-specific: ``shared.py``,
``encode.py`` and ``eval.py`` stay one level up and serve both this package and
:mod:`vforge.tasks.wan.v22.ti2v_5b.teacher_forcing`.

The model definition this assembles is
:mod:`vforge.models.wan.v22.ti2v_5b.ct`.
"""
