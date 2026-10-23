"""Wan 2.2 TI2V 5B task assembly.

Composition builders live here, pure model definition lives in
``vforge.models.wan.v22.ti2v_5b``.

The two forward variants differ only in the training step, so the layout
mirrors the model package: ``ct`` and ``teacher_forcing`` own a ``train``
module each, while ``shared.py``, ``encode.py`` and ``eval.py`` sit at this
level and serve both. A teacher-forcing run reads exactly the caches a
bidirectional run reads, and evaluates through the same sampler, so there is
nothing for the variant packages to specialise beyond the step.

The Wan-family helpers this plugin needs are imported from
``vforge.tasks.wan.v21.shared`` rather than from a neutral shared module, so
each helper still has one owner.

Nothing is re-exported here so that importing this package stays cheap; the
registry imports the verb module it actually needs.
"""
