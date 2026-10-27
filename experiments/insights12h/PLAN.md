# Twelve-hour AS-JEPA insight campaign

User authorization: 2026-10-27, dev2, replacement dev13 and dev15 (24 H200 total after 12:04 UTC), deadline 2026-10-27 22:24:58 UTC. Single-seed screening and early stopping take priority over completing the original three-seed grid. All authoritative files live under /efs/prescottsun/SPT-JEPA.

## Questions and evidence

1. Does the rate-priced state learn rather than collapse? Compare L2 beta 1/4/16/64/256 with matched L1 (SIGReg plus renderer) and L0. Track held-out reconstruction, open-loop motion, effective rank, rate, future-state probes and decoder intervention sensitivity. Low training loss alone is not success.
2. Does the anchor let the state omit persistent appearance? Standard data first; paired color-varied dataset with/without anchor and unpriced L1. Probe color with episode-disjoint splits and inspect cross-episode anchor swaps. The color-only affine appearance dataset is an explicit controlled variant, not the original Appendix A simulator variation dataset.
3. Does a stochastic renderer omit transient detail? Use deterministic per-frame random-dot overlays with flow versus MSE renderers, plus an unpriced control. Probe dot properties and dynamics; assess that reduced leakage is not simply complete collapse.
4. Which choices affect slow dynamics and collapse? Conditional ablations: first-frame anchor, no anchor, direct instead of identity prior, sigma0=.03 and longer window. Do not run every combination.
5. Does the state support action-conditioned rollouts and planning? Common held-out windows, shuffled-action controls, simulator counterfactuals and standard PushT planning for working L2 versus L0/L1. Generation and planning are reported separately.
6. Does the result transfer to SPT modules? Add the S2 Native build if the LeWM line is functional and time permits. An omitted or unsuccessful transfer is reported explicitly, never inferred from the LeWM result.

## Resource and stop rules

- Start with a measured full optimizer-step benchmark and CPU contract checks. The LeWM encoder/prior match the referenced architecture. The exploration renderer is width 192/depth 6 instead of width 384/depth 12. Record exact parameter counts.
- Data uses the original read-only HDF5 expert frames, not Lance JPEG. Export one frameskip-aligned phase: every fifth raw frame, 224px encoder and uint8-rounded 64px renderer. This cuts redundant window starts by approximately five; report steps and windows seen, not equivalence to the paper's epochs.
- First stage: beta grid and baselines on standard data. Stop demonstrably collapsed arms only after enough training for the renderer to use state (with an evaluator floor and matched controls); high early loss or weak first-step gradients are not collapse.
- Continue the useful beta(s), run mechanism ablations, and evaluate. Use one seed first; spend replication only on conclusions that could change the next decision.
- A hard controller deadline writes STOP requests, saves full Lightning/Native checkpoints and exits. The original 12-hour wall-clock deadline is unchanged when hosts are added. Actual GPU-hours are logged per run; the original 16-card allocation was expanded by the user at 12:04 UTC. Use single-GPU independent runs by default.
- Keep committed immutable source snapshots, exact configurations, RNG state, data offsets, package versions and recovery instructions. Shared projects, datasets and environments are read-only.

## Reporting boundaries

This campaign seeks actionable evidence, not a guarantee that every claim is settled in twelve hours. Distinguish implementation checks, exploratory learning evidence, formal generation gates and statistically supported model comparisons. Never describe missing measurements as a failure or a passing result.
