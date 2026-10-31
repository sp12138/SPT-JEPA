# Fixed-model NGC × action-correlation control

All 48 new correlated-proposal endpoints exited 0 and are held. The 48 completed white-proposal endpoints are retained as the other two cells. No training, graph encoding, additional cases, or parameter selection occurred.

The two public controller changes do not show a general additive benefit on Cube. Native improves from 24/29% white-L2 to 36/36% correlated-L2, then only to 37/38% with NGC (primary/fresh). TF changes from 20/15% to 33/31%, then 33/28%; local1024 changes from 19/22% to 23/29%, then 23/22%. All within-correlated Cube NGC-effect intervals include zero. Native has the highest observed Cube mean in all four cells, subject to different capacities and exposure.

The clearest interaction contrast is TF on fresh Cube: NGC adds 10pp under white proposals and −3pp under correlated proposals, giving −13pp [−24, −2]. This is evidence against a simple additive account in these fixed runs. It is not proof of a general negative interaction: intervals omit simulator-repeat variance and population training-seed uncertainty, and no multiplicity correction was applied.

On TwoRoom, local1024 retains a large NGC benefit under correlated proposals: +39pp primary and +22pp fresh. Correlation itself does not improve its NGC result (95/91% versus white-NGC 98/94%). Local1024 ranks below native under L2 but above native under NGC in both proposal regimes; TF remains weak. This preserves the task-dependent evidence for an evaluation-cost limitation without establishing a new learning-algorithm gain or a single cross-task winner.

Each cell evaluates the same frozen checkpoint on the same 50 cases. W is the original diagonal CEM; C is the frozen training-only correlation parameterization. L2 is native GoalMSE; NGC is the public graph cost with the uniform FP32 direct-distance numerical repair. The CEM budget remains 300 candidates × 30 iterations, five model steps, history three and receding horizon five. Correlation preserves the initial per-coordinate marginal scale; later elite distributions can differ.

## Two-seed means and paired contrasts

Percent success; contrast intervals are case-paired 95% bootstrap intervals conditional on these two trained seeds. Interaction = (C-NGC − C-L2) − (W-NGC − W-L2).

| Task | Model | Cases | W-L2 | W-NGC | C-L2 | C-NGC | NGC effect within C, pp | Interaction, pp |
|---|---|---|---:|---:|---:|---:|---|---|
| tworoom | local1024 | primary | 60 | 98 | 56 | 95 | +39 [27, 51] | +1 [-6, 8] |
| tworoom | local1024 | fresh | 66 | 94 | 69 | 91 | +22 [11, 33] | -6 [-15, 3] |
| tworoom | native-raw | primary | 76 | 81 | 68 | 80 | +12 [4, 20] | +7 [-4, 17] |
| tworoom | native-raw | fresh | 73 | 85 | 74 | 79 | +5 [-4, 14] | -7 [-19, 4] |
| tworoom | TF64 | primary | 3 | 3 | 5 | 4 | -1 [-3, 0] | -1 [-3, 0] |
| tworoom | TF64 | fresh | 3 | 3 | 4 | 4 | +0 [0, 0] | +0 [0, 0] |
| cube | local1024 | primary | 19 | 19 | 23 | 23 | +0 [-9, 10] | +0 [-11, 11] |
| cube | local1024 | fresh | 22 | 21 | 29 | 22 | -7 [-18, 5] | -6 [-21, 9] |
| cube | native-raw | primary | 24 | 33 | 36 | 37 | +1 [-10, 12] | -8 [-23, 7] |
| cube | native-raw | fresh | 29 | 35 | 36 | 38 | +2 [-10, 14] | -4 [-19, 10] |
| cube | TF64 | primary | 20 | 24 | 33 | 33 | +0 [-9, 9] | -4 [-17, 9] |
| cube | TF64 | fresh | 15 | 25 | 31 | 28 | -3 [-12, 6] | -13 [-24, -2] |

## Every training seed

| Task | Model | Seed | Cases | W-L2 | W-NGC | C-L2 | C-NGC |
|---|---|---:|---|---:|---:|---:|---:|
| tworoom | local1024 | 1 | primary | 60 | 98 | 56 | 96 |
| tworoom | local1024 | 2 | primary | 60 | 98 | 56 | 94 |
| tworoom | local1024 | 1 | fresh | 72 | 96 | 72 | 94 |
| tworoom | local1024 | 2 | fresh | 60 | 92 | 66 | 88 |
| tworoom | native-raw | 1 | primary | 80 | 84 | 74 | 82 |
| tworoom | native-raw | 2 | primary | 72 | 78 | 62 | 78 |
| tworoom | native-raw | 1 | fresh | 76 | 90 | 80 | 84 |
| tworoom | native-raw | 2 | fresh | 70 | 80 | 68 | 74 |
| tworoom | TF64 | 1 | primary | 2 | 2 | 6 | 4 |
| tworoom | TF64 | 2 | primary | 4 | 4 | 4 | 4 |
| tworoom | TF64 | 1 | fresh | 4 | 4 | 6 | 6 |
| tworoom | TF64 | 2 | fresh | 2 | 2 | 2 | 2 |
| cube | local1024 | 1 | primary | 16 | 14 | 24 | 26 |
| cube | local1024 | 2 | primary | 22 | 24 | 22 | 20 |
| cube | local1024 | 1 | fresh | 24 | 16 | 24 | 24 |
| cube | local1024 | 2 | fresh | 20 | 26 | 34 | 20 |
| cube | native-raw | 1 | primary | 22 | 32 | 36 | 42 |
| cube | native-raw | 2 | primary | 26 | 34 | 36 | 32 |
| cube | native-raw | 1 | fresh | 32 | 32 | 32 | 36 |
| cube | native-raw | 2 | fresh | 26 | 38 | 40 | 40 |
| cube | TF64 | 1 | primary | 20 | 24 | 36 | 32 |
| cube | TF64 | 2 | primary | 20 | 24 | 30 | 34 |
| cube | TF64 | 1 | fresh | 14 | 18 | 32 | 30 |
| cube | TF64 | 2 | fresh | 16 | 32 | 30 | 26 |

## Audit and interpretation limits

All 24 within-correlated L2/NGC pairs have identical first-iteration action, goal, and predicted endpoint hashes on all 1200 case banks, with identical saved L2/NGC score arrays. Later CEM populations may diverge. White and correlated physical candidate banks deliberately differ; their Gaussian innovation equivalence is supported by the frozen CPU contracts, not an unrecorded cross-run tensor equality claim.

All 48 complete model state hashes are unchanged. Actual model imports, card scalers, checkpoint/source/graph/case/config provenance and fixed Cholesky matrix hashes pass (2136 unique immutable inputs checked). The first returned action-map error is zero in every run; mapped objective metadata is checked by the runtime wrapper. The twelve graph artifacts are reused unchanged.

The white-proposal cells are the immediately preceding completed evaluations of these same checkpoints/cases, not simultaneous repeated evaluations in the new correlated batch. White-L2 historical replay differences remain separate: 21/24 exact case vectors. No historical or contemporary best-of baseline is selected. The repeated simulator protocol is not outcome-bitwise reproducible; paired intervals do not include this repeat variability.

All models use ordinary recorded-action conditioning. The public controller changes use no physical-state or optimal-action labels. NGC uses 5,000 training-only observation nodes; the correlation matrix uses the fixed 128 training clips. These are controller controls, not new sample-level learning objectives.

Capacity/exposure differ across model families: local1024 is approximately 56M parameters at 6,000 updates on both tasks; native is approximately 18M with width192 at 6,000 TwoRoom / 12,000 Cube; TF is approximately 51M with width64 at the same respective budgets. The local family is local1024 on both tasks, not a task-specific replacement with F64.

Case-paired bootstrap10000/fixedseed20261031; average the two seed differences per case before resampling. Conditional on these two trained seeds; CI excludes environment repeat variability and seed-population uncertainty. Primary/fresh separate. Contemporary white L2, white NGC, correlated L2 and correlated NGC are retained without historical or best-of selection. The 2x2 interaction is a difference-in-differences of success indicators, not a causal proof excluding simulation repeat noise. First-iteration common-bank proof applies within a proposal regime; later CEM populations diverge, and white/correlated banks intentionally differ. Both arms pay both-score recording cost, so runtime is not bare L2 latency. Model dimensions/capacities/exposures differ; controller comparison is not a new learned algorithm.

Runtime includes both-score instrumentation in both cost arms, so it is not bare L2 latency. Per-case outcomes, all five factorial contrasts and conditional intervals are retained in the local results.json; full candidate bank metadata and arrays remain on EFS.
