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

## Completed bounded evaluations

- `runs/mechanism_rateeval_20261030`: all24 endpoints exit0. Resumed TwoRoom rate-on/off P0 is4/4 versus16/2; Cube10/12 versus8/6. Removing rate does not consistently repair planning. Published `results/rate-controls`.
- `runs/mechanism_cubeaudit_recovery_20261030`: all16 endpoints exit0, retaining six completed internal CEM proposal sets from the parent. Each of50 cases has32 diagnostic base proposals plus7 native CEM proposals, evaluated for25 raw steps. Recorded actions and simulator state are oracle diagnostics only. This is not the50-step closed-loop benchmark. Published `results/cube-audit`.
- Parent `runs/mechanism_cubeaudit_20261030` is held. Its own/released scorers and native CEM preflight passed. Six complete internal CEM proposal sets are retained. Released proposal failed because two HDF5 readers used conflicting SWMR flags; `0d2feae` changes only the read flag. CPU action/qpos/pixel equality passed. The failed endpoint and all old source snapshots are preserved.

The full Cube audit has zero duplicate-restore error and maximum recorded endpoint position error3.07mm. Nonoracle predicted/actual latent-cost rank is.948/.951 for temporal versus.556 LeWM, but temporal's goal cost has effector partial rank.928/.935 and block-position partial rank.034/.022. This is a goal-geometry association, not proof that the transition is physically better in every direction.

`runs/mechanism_metric_20261030`: all3 frozen interventions exit0. All shared128train/128validation episodes and original costs verified. Deleting the fitted effector rank3 span leaves newly refitted effector R2 at.994/.996 and nonoracle task success unchanged20/12. It does not semantically erase the effector, and is not a validated repair. Fixed block-position probe R2 is.785/.811 versusLeWM.989 on held-out frames; do not claim information is literally missing. Published `results/metric`.

`runs/mechanism_spatial_20261030`: all14 frozen spatial probes exit0. TwoRoom update2000 QueryPool64 agent xy R2=.141/.255 and.113/.291, versus fixed spatialRP64=.958/.938 and.951/.893. This gap already exists at initialization. Cube mature pools are broadly competitive. Full3072dim ridge overfits on TwoRoom; do not treat it as an information gold standard. Published `results/spatial`.

`runs/mechanism_channels600_20261030`: all26 endpoints exit0, all six new runs match550 initial tensors. P0 base6/8, clean-state4/4, pure-target-noise6/14, both8/8. Clean position probes barely change; under clean sigma0/t1, constant-state replacement changes renderer error by less than.001%. All six remain held at600. Added sigma.1 probes are counterfactual for noise-off arms; pure-t1 scalar training loss is not directly comparable. Published `results/channels`.

All completed mechanism dispatchers were verified idle and closed; endpoints, failed preflight logs and all source snapshots remain preserved. The earlier three two-step channel health failures came from missing100-step gradient logs; subsequent explicit full-state/backward audits passed. They were not evidence of zero gradients.

## Current bounded spatial-addressing experiment

`runs/mechanism_pooling600_20261030`: source `b3f27e6`. Eight fresh600 gates, two tasks x two seeds x quadrants/shuffled pooling, on dev13 GPUs0–7. Reuse the verified matching full-attention fresh controls. Both new modes keep all550 parameters/tensors, four slots (64dimensions total) and sharedCLS; each query reads64patches. The shuffled partition controls support count and dilution. This changes readout support, not the globally attending ViT receptive field. It is a standard spatial prior diagnostic, not an innovation claim. Two additional Cube600 baseline planning evaluations use dev15 GPUs4/5.

Five CPU contracts pass: exact original default loss/metrics/gradients/RNG, parameter/init identity, mask coverage and RNG isolation, direct readout locality and sample-level finite functional gradients. Separate `mechanism_pooling_preflight_20261030` has all8 real-model two-step/full-state/backward endpoints exit0; reviewed before main launch. No automatic continuation beyond600.

`runs/mechanism_poolspatial_20261030`:16 frozen probes on initial and600 weights of all eight arms. Initial probes use dev2 GPUs0–7 after training has started; final probes use dev15 GPUs0–7 after stage completion. Same128/128 episodes and fixed projection as prior spatial audit. These distinguish initialization/readout effects from learning and never train a privileged policy.

Use existing Trainer/checkpoint code and durable workflow queues. Freeze new committed source/configs before execution. Do not clear old holds or automatically extend beyond600. Preserve every seed and report negative controls. `experiments/hardtasks20261030/THEORY-QUERY-SUFFICIENCY-20261030.md` records theoretical identities, counterexamples to prior assumptions and literature; it is not an innovation claim.
