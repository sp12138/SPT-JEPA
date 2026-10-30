# Independent-bank query sampler: one CPU control

Predeclared once: 64 training-frame prototypes per task, drawn from 64 training episodes excluded from all 128 audit-training and 128 validation episodes; bank seed 20261101. Eight ordered triples are drawn once with seed 20261102 and reused across every window and split. No query or bank frame was selected using labels. The prototype indices/timestamps are not readout inputs.

This adds access to 64 RGB frames per task for the auxiliary query bank; it adds no interaction, action, success label, or validation image to the bank. The eight triples use exactly 20 distinct bank frames per task; all 64 accessed bank frames are charged. No GPU, world-model training, tuning, additional seed search, or new idea is involved.

The target remains the complete five-class A→B transformation of the whole window and its two halves. Fixed readouts and preprocessing are unchanged: linear/Gaussian-kernel ridge 1; normalization and median kernel bandwidth from training inputs only. Episode cohorts and start times match the prior within-clip audit. Kernel geometry changes when initial features are added, so this is not a strict capacity proof.

| Task / split | I / A / B / AB / BA (whole) | I / A / B / AB / BA (halves pooled) | Constant symbol path | Initial-symbol carry agreement (whole / first / second) |
|---|---|---|---:|---|
| tworoom / train | 227/438/172/132/55 | 580/929/358/119/62 | 56.15% | 71.09%/83.89%/69.82% |
| tworoom / validation | 243/438/170/122/51 | 591/930/359/104/64 | 58.98% | 74.12%/85.16%/72.66% |
| cube / train | 49/267/332/298/78 | 205/625/788/279/151 | 16.99% | 42.48%/59.08%/45.02% |
| cube / validation | 44/261/322/304/93 | 212/610/749/280/197 | 14.26% | 38.48%/55.76%/41.02% |

| Task / readout | Majority validation | Queries+actions | +Initial image | Initial increment | Paired episode 95% CI |
|---|---:|---:|---:|---:|---:|
| tworoom / linear | 44.53% | 74.80% | 64.84% | -9.96pp | [-13.90, -6.32]pp |
| tworoom / rbf | 44.53% | 74.87% | 75.91% | +1.04pp | [-1.27, +3.22]pp |
| cube / linear | 34.86% | 52.21% | 57.62% | +5.40pp | [+3.22, +7.68]pp |
| cube / rbf | 34.86% | 61.20% | 67.06% | +5.86pp | [+4.10, +7.65]pp |

The paired intervals resample validation episodes, not individual queries; they do not include training-seed uncertainty. Full per-segment and per-class metrics remain in the JSON, including the unchanged action-only/query-only linear controls.


The cohort/access check confirms bitwise equality of the initial features, recorded actions and episode IDs against the earlier within-clip audit for both splits/tasks. Every bank episode is in the training split and outside the entire validation split. Only episode generalization is tested: the same eight query triples occur in training and validation, so this does not establish generalization to new queries.

| Task | Linear query-only accuracy | RBF macro recall, queries+actions | RBF macro recall, +initial |
|---|---:|---:|---:|
| TwoRoom | 74.84% | 49.15% | 51.80% |
| Cube | 59.51% | 42.45% | 48.67% |

The task distinction is substantive. On TwoRoom, query-only prediction already reaches 74.84%, almost identical to queries+actions; adding the initial image has no reliable accuracy gain. Fixed-query class priors are therefore a strong shortcut in this particular bank. The 58.98% constant-path rate and high initial-carry agreement further limit a dynamics interpretation. This sampler does not establish a useful state-and-action forcing objective there.

On Cube, adding initial information improves both classifier families, and only 14.26% of query symbol paths are constant. This is stronger target-support evidence than the previous within-clip sampler. Nevertheless, the query-only prior already reaches 59.51%; the 67.06% RBF result is an audit of a handcrafted observation target, not a trained world-model result or a proof of causal action dependence. No state-use theorem, planning gain, or new algorithm claim follows from it.

**Interpretation limit:** an independent bank removes the guarantee that every prototype appears, but can make targets mostly constant-region identity. If initial-symbol carry already matches most targets, a larger initial-image increment mainly supports static query grounding; it does not establish action-conditioned path prediction or planning gains. A small increment leaves even this state-use incentive weak. Either outcome is reported without changing bank size, query seed, bandwidth rule or episode selection.

This completes the one requested sampler control. No follow-up search or GPU run is implied.
