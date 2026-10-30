# Empirical validation of retained ideas, 2026-10-31

The user explicitly requested a new empirical campaign filling dev2/dev13/dev15. This supersedes the previous operational hold for new, separately named experiments; it does not clear old holds or imply that ten original algorithms have already been established.

One bounded batch uses24single-GPU runs: six arms × TwoRoom/Cube × training seeds1/2. Stop at600updates for full-state, gradient, information, causal-transition and unchanged strict50-case planning review. No automatic longer grid.

## Six arms, two substantive hypotheses

1. Fixed initialized spatialRP64 carrier, identity chart.
2. Same frozen carrier, freely learned residual chart.
3. Same frozen carrier, residual chart with an actual distance lower bound.
4. Native temporal-forecast plus per-frame query-symbol supervision.
5. Same native model/head plus scalar query-completion supervision.
6. Same native model/head plus the complete five-class path-transformation target.

The carrier controls test whether learnable coordinates improve prediction without destroying existing information. All three have identical tensors at initialization; the identity chart and backbone are intentionally frozen. Report trainable and total parameters separately. Carrier normalization uses fixed training-only frames, with identical statistics across its three arms. An invertible observation chart does not impose invertible contact dynamics.

The query controls test whether the complete transformation adds something beyond extra visual supervision or one success scalar. A query describes a factual recorded path; it does not label an action optimal. Query prototypes only enter the auxiliary readout and never the causal predictor. Within-clip sampling has known shortcuts, especially in TwoRoom; these are explicitly reasons for a short falsification, not grounds for claimed success. All three have the same head tensors; differences in active output rows and loss supports must be reported.

The six arms are not six unrelated original ideas. Carrier and query families differ in parameter and training-compute costs; direct mechanism comparisons are within family. Historical TF600 and released LeWM remain contextual references. The evaluator, test cases, action/observation access at planning, goal semantics and planner budget stay unchanged. No physical labels train either hypothesis.

## Gates and provenance

Freeze a committed source before GPU execution. First prepare four training-only carrier calibrations (two tasks × two seeds). Use those results to finalize immutable configs. Run real two-update checks for all six arms in both tasks, plus explicit full-state/gradient checks. Only after those pass launch the24fresh600runs. Each saves initial weights and recoverable checkpoints at200/400/600. Parent run directories and failed experiment snapshots remain unchanged.

The native trainer, optimizer/scheduler, sampler, checkpoint and export implementation are retained. A shared explicit model factory extends construction/export/audit dispatch and preserves historical variant precedence. Frozen parameters require explicit no-gradient audits rather than a misleading assertion that every encoder must receive gradients.

Frozen diagnostics distinguish observed encoding, teacher-forced one-step prediction, shifted-observed carry, and full causal rollout. The teacher-forced and shifted-carry conditions receive additional real observations during the diagnostic and are never used as deployed planning policies. Test initial-state and recorded-action changes without moving final padding actions into valid transitions. State readability, transition accuracy, generator condition use and native planning are separate endpoints.
