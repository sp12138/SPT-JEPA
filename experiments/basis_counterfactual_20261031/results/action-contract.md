# Requested versus executed actions: existing bank audit

The native model receives unbounded normalized candidates. WorldModelPolicy then inverses the train action scaler; TwoRoom clips raw actions to[-1,1]. This saved bank has clipping in63/64 candidates; mean clipped-coordinate fraction=24.31% (median24.0%, range0–44%). No CEM action-bound mapping is present.

| Selection | Index | Clipped coordinates | Raw max overshoot | Normalized action RMSE | Normalized max difference |
|---|---:|---:|---:|---:|---:|
| mean_candidate | 0 | 0% | 0.000 | 0.000 | 0.000 |
| identity | 6 | 44% | 1.637 | 0.520 | 1.887 |
| basis | 29 | 26% | 1.042 | 0.270 | 1.200 |
| oracle | 59 | 34% | 1.098 | 0.256 | 1.266 |

`CEMSolver.configure` uses Box only for shape/type (`cem.py:63–71`); sampling goes straight to cost (`cem.py:196–215`). `ShootingCostEvaluator` forwards candidates unchanged (`evaluator.py:156`); LeWM flattens them into action_encoder (`lewm.py:101–103`). Policy unblocks the25 micro-actions and inverse-normalizes (`policy.py:513–537`); TwoRoom applies the clipping (`env.py:262–267`). `scripts/plan/config/solver/cem.yaml:1–9` and `tworoom.yaml:14–17` contain no action_bounds setting. All source hashes match the completed pilot.

This confirms a numerical command/execution difference, not its causal contribution. Clipping fraction is not ordered with regret: identity44%, basis26%, oracle34%. One case cannot establish that clipping caused bad selection or the basis gap. Models could learn saturation as part of requested-command dynamics. A proposed explicit bounds experiment must change the common planner consistently for every baseline and preserve original-protocol results; it is not a new learning algorithm. No model forward, GPU or new environment rollout was performed.
