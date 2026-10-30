# Empirical validation of retained ideas, 2026-10-31

## Completed result

All 24 fresh runs reached 600 updates; all 120 planned endpoints exited successfully. The 24 original-state comparisons checked all 550 native tensors and exact within-family initial states. Full checkpoint, optimizer, gradient and common-case checks passed. The three hosts ran eight simultaneous training jobs each. All checkpoints are preserved; after review the three verified idle dispatchers were closed and all 24 GPUs were empty at approximately 18:27 UTC on 2026-10-30. No run continued beyond 600.

| Arm | TwoRoom seed 1 / 2 | Cube seed 1 / 2 |
|---|---:|---:|
| Fixed carrier, identity chart | 44 / 30% | 4 / 6% |
| Fixed carrier, free chart | 34 / 30% | 4 / 8% |
| Fixed carrier, bounded chart | 46 / 42% | 10 / 6% |
| Path query: symbols | 6 / 12% | 0 / 0% |
| Path query: completion | 10 / 0% | 0 / 0% |
| Path query: full transformation | 2 / 4% | 0 / 0% |

Each model uses the same 50 strict validation cases for its task and the unchanged native planner. Historical temporal-forecast600 gives TwoRoom 6/8% and Cube 14/10%, but different total/trainable parameters, observation calibration and training computation prevent treating that comparison as an isolated mechanism effect. Full matched comparisons, per-case outcomes and plots are in [the complete report](results/gate600/RESULTS.md).

Three findings survive the direct controls. First, fixed observation information can support useful dynamics without end-to-end encoder learning: carrier rollout error is about .58 of initial-copy error on TwoRoom and .52–.54 on Cube. Teacher-forced prediction also beats copying the latest observed state. Correct initial states and recorded actions improve prediction. Nevertheless Cube planning stays weak. Information preservation and short-horizon latent prediction are therefore insufficient for this control task; these measurements do not separately identify goal geometry, missing orientation/contact information, rollout support or search as the remaining cause.

Second, the bounded chart improves TwoRoom planning relative to its free control (+12 pp in each seed), but the free charts themselves remain globally invertibility-certified: residual Lipschitz bounds .112–.150, versus .097–.114 when bounded. Observed probes and dynamics are close across the three arms. This is a limited parameterization/optimization signal, not evidence that preventing irreversible information deletion caused the gain. The large difference from historical TF is already mostly present in the fixed carrier. Two seeds and repeatedly used validation cases do not establish broad stability or novelty.

A further CPU-only check inverts the saved predictions through their certified charts into the identical fixed-carrier coordinates, avoiding a moving target metric. Observation inversion agrees within 4.8e-7. Common-coordinate rollout/copy errors for identity/free/bounded are TwoRoom .5811/.5854, .5856/.5825, .5802/.5821; Cube .5278/.5176, .5252/.5155, .5256/.5152. These are tiny differences (less than .7% relative improvement), with no consistent bounded-versus-free prediction winner. The apparent own-coordinate prediction worsening in Cube disappears after alignment. A coordinate change must not be mistaken for more accurate physical prediction. Details, probe limitations and paired successes are in [the carrier review](carrier/INDEPENDENT-PREFLIGHT-REVIEW.md).

Third, the current path-query objective fails its intended function. All six Cube task scores are zero. Completion accuracy equals the majority class for each full/half context: 73.5677% on TwoRoom and 72.8516% on Cube. Swapping real initial states hardly changes head loss, although replacing all codes by zeros does. Thus a shared nonzero code/time pattern can support auxiliary loss improvement without scene-specific prediction. Free-rollout latent error is far worse than initial carry; both the small true-variation denominator and substantial prediction drift are retained in the diagnostics. Clean probes still contain some information, so the result is not literal information absence or a proof that every compositional objective must fail. See [the path-query review](pathquery/review600/REPORT.md).

Decision: hold every run at 600. Retain the TwoRoom carrier signal for a future controlled confirmation; no present candidate has demonstrated substantial gains on both tasks. This batch validates two hypotheses and their controls, not ten successful original ideas. The earlier ten-idea requirement remains unfulfilled rather than being filled with renamed controls or negative findings.

The executable protocol and original design follow.

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
