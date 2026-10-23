"""Wan 2.2 TI2V 5B teacher-forcing task assembly.

Training only. Data, conditioning and cache format are those of the
bidirectional sibling :mod:`vforge.tasks.wan.v22.ti2v_5b.ct`, so everything
except the step recipe comes from the parent package -- a teacher-forcing run
reads exactly the caches a bidirectional run reads. Encoding and evaluation
stay up there for the same reason; there is nothing for this one to specialise.

Nothing is re-exported here so that importing this package stays cheap; the
registry imports the verb module it actually needs.
"""
