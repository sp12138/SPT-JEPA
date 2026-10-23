"""Standalone data utilities: datasets, resumable loaders, and latent caches.

Everything the training data path needs is a flat module here. Most of it is
content-agnostic -- ``cache``, ``dataset``, ``loader`` and ``mixed_dataset`` move
bundles, read manifests and hand out a resumable loader without knowing what a
sample carries. The rest is named for the corpus or contract it was built
against: ``collate`` assembles Wan's condition streams, ``taichu_index_cache``
reads the sharded layout taichu-era caches were written in, and the three
dataset modules decode the corpora they are named after.

Nothing under this package imports from ``vforge.tasks``: reading a request,
picking a data source and deriving the resume fingerprint are assembly
concerns. ``tests/data/test_layering.py`` holds that line.
"""
