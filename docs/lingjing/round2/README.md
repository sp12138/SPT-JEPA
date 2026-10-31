# Round-2 reference implementation

These files implement the model classes, the benchmark anchor rule and the contract
tests that the [round-2 spec](../as-jepa-round2-spec.md) asks for. Copy them into the
collaborator's tree unchanged:

| File here | Destination | What it is |
| --- | --- | --- |
| `as_jepa_lewm/forecast_scored.py` | `as_jepa_lewm/` | `ForecastScoredLeWM`: L2 whose renderer reads, per rendered frame with probability `lam`, the prior's forecast instead of the encoded state. Switches: `horizon` (one-step or rollout), `forecast_noise` (a priced `sigma0` sample, or the noiseless mean as a comparison) and `rho` (AR(1) posterior noise with the exact KL). These reproduce the collaborator's forecast-only, TF and temporal-noise arms inside one class (spec Sections 0.3 and 3.1) |
| `as_jepa_lewm/nce.py` | `as_jepa_lewm/` | `NCELeWM`: the reconstruction-free control, an InfoNCE critic in place of the renderer (spec Section 3.6) |
| `as_jepa_tools/rollout_gain.py` | `as_jepa_tools/` | The rollout diagnostics: do the prior's rollouts contract or lag? Spread ratio, carry ratio, motion gain and error relative to copying, per horizon and per quartile of dimension spread, for the planner's open-loop rollout and for one-step predictions; plus the one-step calibration slope on posterior samples (spec Section 5.3, Appendix A.8) |
| `as_jepa_tools/benchmark_gap.py` | `as_jepa_tools/` | `GapBenchmarkWindows`: TwoRoom and Cube windows whose renderer anchor is a past frame at least `gap` kept frames before the window; `make_benchmark_dataset`; and the CPU staleness table that chooses the gap (spec Sections 3.4 and 5.5) |
| `tests/as_jepa/test_round2.py` | `tests/as_jepa/` | 15 CPU contract tests for the model classes, including bitwise equality with `RobustLeWM` for `rho = 0.8` and with the collaborator's forecast loop (spec Section 10) |
| `tests/as_jepa/test_benchmark_gap.py` | `tests/as_jepa/` | 5 CPU contract tests for the anchor rule and the staleness table (spec Section 10) |
| `tests/as_jepa/test_rollout_gain.py` | `tests/as_jepa/` | 4 CPU contract tests for the rollout diagnostics, including a known centre pull (spec Section 10) |
| `as_jepa_lewm/learned_variance.py` | `as_jepa_lewm/` | `LearnedVarianceLeWM`, learned posterior and prior variances. No arm of this round uses it; copy it because `test_round2.py` imports it |

Register the classes and route the benchmark datasets as spec Section 2.1 describes.
Do not edit `model.py`, `robust.py` or `research.py`.

## How these were checked

All 24 tests passed on 2026-10-31 against the collaborator's branch
`research/ideas-20261028` at `051e82d` (files exported with `git archive`). They used
stable-worldmodel's LeWM modules at upstream `21446f1`, on CPU with Python 3.12.2,
torch 2.7.0, transformers 4.53.1 and h5py 3.11.0, with the `sdpa_math` backend. They
also passed at `2fdd905`, and the original 11 model tests at `094e4f0`; none of the
collaborator's files these tests depend on changed between those commits. In the collaborator's environment,
run from the repository root:

```bash
PYTHONPATH=.:runtime:<stable-worldmodel checkout> python -B -m unittest \
    tests/as_jepa/test_round2.py tests/as_jepa/test_benchmark_gap.py tests/as_jepa/test_rollout_gain.py -v
```

The model tests use tiny models (one encoder layer, a 64-wide one-layer renderer,
56-pixel encoder input), so they check contracts, not training behaviour. The anchor
tests use a small synthetic HDF5 file whose pixels encode their row index, and the
rollout-diagnostic test runs the command line end to end on such a file. The GPU
smokes in spec Section 10.3 check the full-size models and the real data.
