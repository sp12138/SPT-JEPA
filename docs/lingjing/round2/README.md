# Round-2 reference implementation

These files implement the model classes, the benchmark anchor rule and the contract
tests that the [round-2 spec](../as-jepa-round2-spec.md) asks for. Copy them into the
collaborator's tree unchanged:

| File here | Destination | What it is |
| --- | --- | --- |
| `as_jepa_lewm/forecast_scored.py` | `as_jepa_lewm/` | `ForecastScoredLeWM`: L2 whose renderer reads, per rendered frame with probability `lam`, a `sigma0`-noisy sample of the prior's forecast instead of the encoded state (spec Section 3.1) |
| `as_jepa_lewm/nce.py` | `as_jepa_lewm/` | `NCELeWM`: the reconstruction-free control, an InfoNCE critic in place of the renderer (spec Section 3.6) |
| `as_jepa_tools/benchmark_gap.py` | `as_jepa_tools/` | `GapBenchmarkWindows`: TwoRoom and Cube windows whose renderer anchor is a past frame at least `gap` kept frames before the window; `make_benchmark_dataset`; and the CPU staleness table that chooses the gap (spec Sections 3.4 and 5.6) |
| `tests/as_jepa/test_round2.py` | `tests/as_jepa/` | 11 CPU contract tests for the two model classes (spec Section 10) |
| `tests/as_jepa/test_benchmark_gap.py` | `tests/as_jepa/` | 5 CPU contract tests for the anchor rule and the staleness table (spec Section 10) |
| `as_jepa_lewm/learned_variance.py` | not used in this round | `LearnedVarianceLeWM`, learned posterior and prior variances; kept for a later round |

Register the classes and route the benchmark datasets as spec Section 2.1 describes.
Do not edit `model.py`, `robust.py` or `research.py`.

## How these were checked

All 16 tests passed on 2026-10-31 against the collaborator's branch
`research/ideas-20261028` at `2fdd905` (files exported with `git archive`). They used
stable-worldmodel's LeWM modules at upstream `21446f1`, on CPU with Python 3.12.2,
torch 2.7.0, transformers 4.53.1 and h5py 3.11.0, with the `sdpa_math` backend. The
11 model tests had also passed at `094e4f0`; the collaborator's model, trainer and
data code did not change between the two commits. In the collaborator's environment,
run from the repository root:

```bash
PYTHONPATH=.:runtime:<stable-worldmodel checkout> python -B -m unittest tests/as_jepa/test_round2.py tests/as_jepa/test_benchmark_gap.py -v
```

The model tests use tiny models (one encoder layer, a 64-wide one-layer renderer,
56-pixel encoder input), so they check contracts, not training behaviour. The anchor
tests use a small synthetic HDF5 file whose pixels encode their row index. The GPU
smokes in spec Section 10.3 check the full-size models and the real data.
