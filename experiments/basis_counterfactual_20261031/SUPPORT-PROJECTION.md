# Training support and a single saved-bank planner control

The full9000 training episodes contain828,712 HDF5 action rows:819,712 finite commands and exactly9000 terminal NaN rows. Every finite coordinate is in[-1,1], with per-axis minima−1 and maxima+1; **0/1,639,424 finite coordinates exceed the bound**. The union of all valid8-frame/stride5 training windows covers8992 episodes and819,454 outgoing action rows (1,638,908 finite coordinates), again with0 out-of-range values.1,045,742 coordinates equal±1. This is the full available training support, not the number sampled by a2000-update run. Only training action rows were scanned; no heldout actions or images.

Current pinned collection code logs the **command passed into the environment**: `stable_worldmodel/wrapper/default.py:293,309` calls `env.step(action)` then stores that incoming `action`; `world/world.py:357–358` aligns it with the preceding observation and moves the initial NaN to the terminal row. The TwoRoom collection script selects noisy `ExpertPolicy` (`scripts/data/collect_tworooms.py:18`), whose return clips commands to[-1,1] (`envs/two_room/expert_policy.py:193`). Therefore this code path's logged commands are already bounded before the environment clips again. It does not read back the actuator's post-clip value. The downloaded H5 is pinned to `quentinll/lewm-tworooms@6903a2de048b13819d812da0b4dd661290bc01e4`, archive SHA494b1a02…80ca, but its acquisition record does **not** pin a historical generator commit. The current script writes Lance, so it must not be claimed as a forensic identification of the historical H5 generator. The actual H5 range and terminal-row findings above are independent of that uncertainty.

Prepare one uniform input treatment, with the original64 CUDA candidates and their already executed endpoints fixed:

`P(z) = scaler.transform(clip(scaler.inverse_transform(z), -1, 1))`.

`projection_control.py` uses the same installed sklearn float32 arithmetic, including its statistics cast to input dtype. CPU contracts compare it exactly to an independent scalar-operation oracle, test idempotence/executed-action equality up to explicit float32 roundoff, and recover the original initial/goal image bytes directly from HDF5 against the frozen pilot's hashes. No environment is constructed. The future GPU branch would first reproduce only the first8 original native costs per model to validate reconstructed inputs, then score64 projected candidates for each of the same identity/basis checkpoints (**144 candidate predictions total**). It reuses the original true-cost arrays, verifies unchanged weights/buffers and uses the original `ShootingCostEvaluator+GoalMSE(sum)`. No new observations are fed into the predictor; no new environment rollout, case, model or training is added.

This is a **planner-input control**, not a new algorithm or an already proved bug fix. The previous bank has24.31% of requested coordinates outside the observed training bounds, but range support does not establish joint action/trajectory support. A command-conditioned model could in principle learn saturation. If this control helps, every baseline still needs the same planner treatment with its own normalizer, and original-protocol results remain preserved. GPU execution requires a new root-reviewed frozen source; it has not been launched.

Candidate command after freeze/review only:

```sh
python <frozen source>/projection_control.py --score --device cuda \
  --output /efs/prescottsun/SPT-JEPA/runs/basis_projection_pilot_20261031
```

CPU evidence: `results/action-support.json` and `results/projection-cpu-contracts.json`; full records remain under `reports/basis-counterfactual-20261031/` on EFS.
