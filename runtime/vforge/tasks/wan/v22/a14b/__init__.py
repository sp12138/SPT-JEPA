"""Wan 2.2 A14B task assembly.

The data and conditioning contract is the Wan2.1 one -- same encoder, same
encoded-bundle fields, same scalar timestep -- so this plugin reuses
``vforge.tasks.wan.v21.shared`` for those and owns only what the expert pair
changes: which timestep range a step may sample from, and the fact that the
training root loads from two published subfolders instead of one.
"""
