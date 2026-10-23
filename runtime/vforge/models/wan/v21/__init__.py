"""Wan 2.1 native model definition.

Heavy modeling imports (diffusers transformer/encoder) stay lazy so importing
this package works in CPU/Hydra smoke environments. Task assembly lives in
``vforge.tasks.wan.v21``; nothing here may import the training layer.
"""
