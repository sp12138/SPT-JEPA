"""Task assembly for the Wan model family.

Mirrors ``vforge/models/wan`` one level at a time -- ``v21`` here assembles
``models/wan/v21``, ``v22/a14b`` assembles ``models/wan/v22/a14b``, and so on --
so a reader who knows where a model lives knows where its builders live.

The split against ``vforge/models`` is composition versus definition: modules
under this package read a request and wire a model together with data, an
optimizer and step semantics, then hand the trainer or the evaluation runner a
components object. The model packages stay pure definition and may not import
anything from here.

Nothing is re-exported at any level so that importing a package stays cheap;
the registry imports the one verb module it needs.
"""
