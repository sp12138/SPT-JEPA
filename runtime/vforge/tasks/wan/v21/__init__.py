"""Wan 2.1 task assembly.

Composition builders for ``vforge.models.wan.v21``: one module per verb, plus
the pieces more than one verb needs.

This is the oldest plugin in the family and the other versions were written
against it, so ``shared.py`` still owns Wan-family helpers that v22 imports
from here rather than from a neutral module -- each helper keeps one owner.
"""
