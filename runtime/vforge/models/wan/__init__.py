"""Shared primitives and versioned Wan model implementations.

Reusable modules live directly in this package, while concrete models live in
the ``v21`` and ``v22`` subpackages. Nothing is eagerly re-exported so importing
the package does not load heavyweight model dependencies.
"""
