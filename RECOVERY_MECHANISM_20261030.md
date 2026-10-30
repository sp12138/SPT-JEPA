# Active mechanism research: TwoRoom and Cube

User explicitly authorized parallel research on dev2/dev13/dev15, using the 24 H200s productively, on 2026-10-30. Parallel agents are authorized for this request. All large assets remain under `/efs/prescottsun/SPT-JEPA`; old campaigns, checkpoints, sources and STOP markers remain held and unchanged.

## Completed first evidence

`runs/mechanism_information_20261030`: all eight frozen information audits complete. Source `1bbb7f0`. Published, visually reviewed report: `experiments/mechanism20261030/results/information` (results commit `0116632`, included in pushed `0d2feae`).

- TwoRoom temporal-forecast clean agent xy probe R2 is .141/.255 and .113/.291; posterior noise sigma=.1 removes useful linear predictability. This is weak information, not a proof of zero information.
- Removing noise and scaling the existing signal by 100x changes frozen renderer error by less than .032%. Rescaling alone is not an effective repair.
- Moving pixels occupy 9.37% of TwoRoom but account for 83.71–90.50% of residual high-corruption error. Current background domination is not supported.
- Temporal-forecast priced-rate/flow state-gradient norm ratios are .203/.311, not evidence of rate dominating all current gradients.
- Cube block xyz R2 is .909/.951/.546 and .909/.950/.720; effector xyz exceeds .979. State/action interventions strongly affect denoising. Planning failure is downstream of substantial encoded information.

`runs/mechanism_rate600_20261030`: all16 training and16 health endpoints exit0; held after review. Source `2a117d6`. Same task/seed pairs share all550 fresh initial tensors or exact full update-2000 state copies. New paired rate-on/.25 versus rate-off/0 runs stop at600 fresh or2600 resumed; no continuation.

TwoRoom fresh600 training feature std: on .0007865/.0008758, off .0007943/.0008905. Resumed2600: on .0005693/.0005961, off .000622/.000634. Removing rate does not restore the state channel at this budget. Cube resumed2600 increases from .289/.259 to .373/.370; information amplitude is not planning performance. Full `REVIEW.json`, initialization tensor equality and resume-copy audits remain in the EFS root.

## Active bounded evaluation queues

- `runs/mechanism_rateeval_20261030`:16 frozen information audits plus8 fixed50-case strict-validation planning jobs for the resumed endpoints. dev2 all8 slots for TwoRoom, dev13 all8 slots for Cube. Source is the immutable `0d2feae` diagnostic snapshot. Fresh600 is evaluated for mechanism, not promoted as mature performance. No training here.
- `runs/mechanism_cubeaudit_recovery_20261030`: released LeWM's CEM proposals,8 physical candidate-bank shards,7 model scorers, on dev15 GPUs0–7. Reuses six completed internal-model proposal sets from the parent. Each of50 cases has32 diagnostic base proposals plus7 native CEM proposals, evaluated for25 raw steps. Recorded actions and simulator state are oracle diagnostics only. This is not the50-step closed-loop benchmark.
- Parent `runs/mechanism_cubeaudit_20261030` is held. Its own/released scorers and native CEM preflight passed. Six complete internal CEM proposal sets are retained. Released proposal failed because two HDF5 readers used conflicting SWMR flags; `0d2feae` changes only the read flag. CPU action/qpos/pixel equality passed. The failed endpoint and all old source snapshots are preserved.

The Cube audit's interpretation must separate physical candidate coverage, goal metric on actual encoded endpoints, and predicted endpoint ranking. Strict pose/hold success, position and orientation errors are retained. Simulation duplicate-restore error was zero in the two-case preflight; recorded replay position errors were .366mm/.0124mm. No conclusions may be drawn from just those two cases.

## Next discriminating work under preparation

After the completed rate review, test TwoRoom's two information channels: latent sampling noise on/off crossed with ordinary target corruption/pure-noise t=1. Six new paired600 gates (two seeds and three new settings), with existing matching ordinary controls, are under implementation and must pass exact default-loss/gradient/RNG regression and GPU execution preflight before launch. These are causal diagnostics, not a new algorithm or a claimed valid ELBO after channel modification.

Also prepare frozen spatial probes comparing QueryPool, CLS, global patch mean and spatial patch pooling, with fixed64-dimensional random-projection controls and episode-disjoint fitting. This tests whether position is lost before or after state pooling. No new model training is needed for this check.

Use existing Trainer/checkpoint code and durable workflow queues. Freeze new committed source/configs before execution. Do not clear old holds or automatically extend beyond600. Preserve every seed and report negative controls. `experiments/hardtasks20261030/THEORY-QUERY-SUFFICIENCY-20261030.md` records theoretical identities, counterexamples to prior assumptions and literature; it is not an innovation claim.
