# Session playbook: how the shared agent sessions are directed

Written 2026-11-21. Peng shares agent sessions on his machines with Lingjing's
side. Lingjing's agent sends each session one phase of work at a time and reads the
reply in the session. This file is the standing brief for those sessions: a chat
message names a phase below and adds only what is specific to that moment.

Background for every phase is in [../results-index.md](../results-index.md), which
lists what has been run, with sources.

## 1. Standing rules for every session

1. `AGENTS.md` stays in force. If a phase conflicts with it, or with Peng's running
   work, stop and say so in the session. Do not choose between them.
2. Never signal, stop, reprioritise or reconfigure a job or process that the
   session did not launch for one of these phases.
3. Use only the machines and GPUs that a message says Peng assigned. An idle GPU
   is not an assigned GPU.
4. Work in a separate git worktree per session, never in the checkout of Peng's
   own hourly-check session. Commit to the branch the phase names; do not push to
   `as-jepa/lingjing-fullspec-20261102` directly, open a pull request.
5. Do not change an objective, a named quantity, an arm or a decision rule. If one
   cannot be run as written, report why and stop.
6. A failed prediction or a negative result is a result. Report it with the same
   care as a positive one.
7. Record every launched job in the worktree (`RUNS.md`: command, machine, GPUs,
   start time, output directory on the EFS) and follow the repository's rule that
   raw results stay on the EFS, with compact conclusions in git.
8. Reply format: numbered items that mirror the phase's "Report" list, the relevant
   lines of output, then one line "Could not determine: ...". No proposals for next
   steps unless the phase asks for them.

## 2. Who does what

| Where | Work | Why there |
| --- | --- | --- |
| Session S1 (Peng's Mac, reaches dev13 and the EFS) | Anything that needs existing checkpoints or the repository's test environment: the anchor-gap diagnostic, generation numbers of existing models, the test dry run for the repository cleanup | The checkpoints and the environment exist only there |
| Session S2 (requested, own worktree) | The evaluation harness on RoboCasa | The simulator, the RoboJEPA and V-JEPA weights and the Reach evaluation code are there; this is long-running code work that should not block S1 |
| Lingjing's cluster | All new small-benchmark training from random weights: the pixel-free PushT test, and the original framework on PushT, TwoRoom and Cube | Keeps dev13 free for the RoboCasa work; needs only public datasets |

If the small-benchmark setup on Lingjing's cluster fails, the same phases (B1, A1 to
A3) move to a third session on Peng's machines, with GPUs assigned by Peng.

## 3. Phases for session S1

### S1-0 Orientation (read-only)

Sent as a chat message on 2026-11-21. Reports the workspace, reach to dev13 and the
EFS, GPU state, the PushT assets and launch record, the test command, and the
archived `reader_mse` values.

### S1-1 Anchor-gap diagnostic (needs one assigned GPU, under one hour, no training)

- **Do:** Part G of [decoder-free/handoff-anchor-reader.md](decoder-free/handoff-anchor-reader.md),
  Section 4, with the commands of Section 11.1, on: PushT temporal-forecast at
  37,473 updates (seeds 1 and 2); Cube temporal-forecast at 23,342 updates (both
  seeds); and the last RoboCasa Reach3D model that still trained the renderer (the
  width-384 future-mean model, latest full hour).
- **Why:** on RoboCasa the renderer could redraw the future frame without the state
  (swapping in the current code raised the decode error by only 1.05 to 1.30
  times). The diagnostic measures, per distance between anchor and frame, how much
  the renderer relies on the state and on the anchor. That tells us whether the
  anchor rule needs to change before the original framework is run again.
- **Do not change:** the tool, its seeds, the bucket rule.
- **Report:** per checkpoint, the table the tool prints (per gap bucket:
  `render_loss`, `state_reliance`, `anchor_reliance`, rate, probe error), the
  Spearman correlations, and which row of the "Reading the result" table applies.

### S1-2 Generation numbers of the existing best PushT models (one assigned GPU)

- **Do:** run the existing quality tools (`as_jepa_tools/quality_complete.py` and
  the pose-estimator rollout evaluation used in the completion campaign) on F8 at
  24,982 updates and temporal-forecast at 37,473 updates, both seeds, with the same
  256-clip protocol as the completion campaign.
- **Why:** generation quality was measured only on early models. The best models
  have planning numbers but no generation numbers.
- **Report:** FVD on 256 clips, LPIPS, SSIM, the 7-step agent and block errors as a
  ratio to copying the last frame, per model and seed, next to the completion
  campaign's values for the small model.

### S1-3 Test dry run for the repository cleanup (CPU only)

- **Do:** in a fresh worktree, run the test suite on `as-jepa/lingjing-fullspec-20261102`
  and record the baseline. Then check out the cleanup branch named in the message,
  run the same suite, and run `tests/test_portability.py` and one import of
  `experiments.robocasa_stable_scaling.train`.
- **Why:** the cleanup moves 25 reference experiment folders to `archive/experiments/`
  and removes 53 folders that nothing depends on. The five `robocasa_*` folders,
  `runtime/`, `as_jepa_lewm/` and `as_jepa_tools/` do not move, so running jobs and
  saved checkpoints are unaffected. This phase verifies that claim before the pull
  request is merged.
- **Report:** pass and fail counts before and after, every test whose status
  changed with its error, and whether any frozen-source or hash check fired.

## 4. Phases for session S2 (evaluation harness)

These wait for one decision between Peng and Lingjing: the evaluation stage needs
its own rule set, separate from the rules for pretraining a method. The proposal is
that `AGENTS.md` keeps governing how a model is pretrained (from scratch, RGB and
executed actions only, per-sample losses), while the evaluation stage may train a
policy head on action sequences toward a later frame of the same episode, may
fine-tune the encoder in the fine-tuning regimes, and may evaluate public pretrained
models as reference rows. Results from the two rule sets are never mixed in one
table.

### S2-0 Confirm the inventory (read-only)

- **Do:** check Section "Line C" of the results index against the code: the Reach
  simulator loop, per-case replay, frozen-feature caching, the RoboJEPA and V-JEPA
  2.1 adapters and the CEM planner. Report what the index has wrong.

### S2-1 Reference rows from public models (needs assigned GPUs)

- **Do:** on the 64 development cases of the Reach3D protocol, with RGB-only input
  and the 2 mm rule, evaluate the public RoboJEPA predictors (50M, 300M, 1B, 2B)
  twice: with RoboJEPA's own CEM planner, and with the existing linear readout on
  frozen features.
- **Why:** this is the first scaling result of the evaluation line and needs no
  pretraining. It also gives the protocol-matched RoboJEPA comparison that the
  current mainline lacks.
- **Report:** strict successes, mean final distance and seconds per decision, per
  model size and readout, with the 4-case reproduction (0.8866) as a check.

### S2-2 The missing pieces

In this order, each with the success checks of
[as-jepa-evaluation-handoff.md](as-jepa-evaluation-handoff.md) Section 6, Step 1:
the state probes on RoboCasa (logging simulator state for the probe's training
logs), the chunked policy head on frozen features, the policy-plus-planning loop,
then the fine-tuning regimes.

## 5. Phases on Lingjing's cluster

All runs start from random weights and use public datasets. Budgets are filled in
from the measured seconds per update of the smoke test.

### B1 The pixel-free PushT test

[decoder-free/handoff-two-tracks.md](decoder-free/handoff-two-tracks.md) Section 4,
with two changes: the contrastive control (Q5) runs only after a denoiser arm
passes, so the first batch is four arms on two seeds (Q1, Q2, Q3 with `rho = 0.8`,
and Q0 with `rho = 0`) to 12,491 updates. Pass rule, continuation and fallback are
Sections 4.3 to 4.6, unchanged.

### A1 Original framework on PushT, matched against LeWM

- **Recipe (fixed):** the round-2 cell F8: renderer trained on one-step forecasts
  (`lam = 1`), time-correlated posterior noise (`rho = 0.8`), `sigma0 = 0.1`,
  `beta = 0.25`, no initial-scale calibration. It ties the collaborator's
  temporal-forecast mix at equal budget (88.50 against 89.00 on the same planning
  subset) and has fewer moving parts.
- **Do:** F8 and native LeWM in the same campaign, training seeds 1, 2 and 3, 37,473
  updates each, the same planning seeds (42, 43, 44 and 1042, 1043, 1044).
- **Report:** planning success per seed, the paired difference with an interval
  clustered by episode, and next to it the things that cannot be matched: frames
  per window (8 against 4), trained parameters, GPU-hours.
- **Decision:** the framework holds on PushT if F8 is within 5 points of LeWM or
  better on both sets of planning seeds.

### A2 The same recipe on TwoRoom and Cube

- **Do:** F8 with the anchor rules already implemented for these tasks, two seeds:
  TwoRoom to 6,000 updates with the state-use check at 2,000; Cube to 12,000.
  Native LeWM once per task as a check that this cluster reproduces 80 / 72 and
  26 / 27.
- **Why:** the working recipe was never run on these tasks. The earlier runs there
  used independent noise, the setting that also fails on PushT.
- **Decision:** if the TwoRoom state is trapped near zero spread again, or Cube
  trails LeWM by more than 5 points, the original framework is a PushT-scale result
  and is reported as such.

### A3 Generation numbers for A1

The quality tools of S1-2 on the A1 checkpoints.
