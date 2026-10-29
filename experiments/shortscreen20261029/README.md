# Short-budget method screen

The user corrected the long16run matrix: inspect short-run evidence before deciding whether to continue. All16 old methods/seeds were gracefully paused around9.6-10.3k updates with finite full optimizer/scheduler/RNG/data cursor checkpoints. No long continuation is automatic.

Only diagnostic training seeds1(high original performer) and2(weaker original performer) of the four methods resume to exactly12491updates, the first epoch, while keeping the124910-step LR schedule. Compare each to the matching original beta=.25 checkpoint already saved at12491updates. Other seeds stay paused.

P0 uses two paired evaluation seeds(42/43),50episodes each; generation64cases,one sample,50Eulersteps; counterfactual16cases,50steps. These are diagnostic screens, not final benchmark confirmation. See [frozen protocol](protocol.json). Continue only evidence-supported directions; publish all informative outcomes.
