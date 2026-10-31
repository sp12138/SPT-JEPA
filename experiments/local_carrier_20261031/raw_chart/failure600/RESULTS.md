# Raw-chart600: stopped numerical failure

All four training jobs failed the same **inverse-adjoint residual certificate**, after the genuine100-step records had finite loss and positive predictor/renderer/chart gradients. The600 gate did not complete; no planning, generation or endpoint state-use comparison exists.

| Task / seed | Last completed step | Chart gradient norm at100 | Peak GiB | Training exit |
|---|---:|---:|---:|---:|
| TwoRoom / 1 | 112 | .002637 | 107.4276 | 1 |
| TwoRoom / 2 | 124 | .003579 | 107.4276 | 1 |
| Cube / 1 | 163 | .001025 | 107.4288 | 1 |
| Cube / 2 | 116 | .001208 | 107.4288 | 1 |

Exact shared exception: `Chart inverse adjoint failed contraction residual certificate after 64 iterations`, raised in `_ImplicitChartInverse.backward`. This is not a forward-inverse exception, OOM or observed nonfinite-loss failure. CPU nonlinear contracts and two-step production checks did not establish continued numerical reliability as training changed the chart and gradient distributions. The available logs do not distinguish TF32 arithmetic from a small-gradient relative-residual issue; no cause is asserted without captured failing operands.

The first scheduled checkpoint was at200, so these failed runs have no `last.ckpt` or intermediate portable model. Initial tensors, complete logs/metrics and the earlier two-step preflight fullstates remain preserved. No resumable112/124/163/116 checkpoint is claimed.

Source remains immutable commit `0c9ae74840cf28247ae74341d2656e2d9ddd457c`; neither tolerance nor model code was changed. After verifying dev13 GPUs0–3 empty and the dispatcher idle, owned dispatcher1040736 and monitor1040737 were closed and `B/runs/local_raw_chart600_20261031/HOLD_DISPATCH` written. All prior holds remained unchanged. Four training failures are recorded; the16 dependent endpoints never ran. No automatic restart or extension.

Full compact proof and four trace tails: `B/reports/local-raw-chart600-failure-20261031`. Any numerical reproduction or repair requires a separate reviewed action. These failures do not establish either task performance or failure of the coordinate-consistency hypothesis.
