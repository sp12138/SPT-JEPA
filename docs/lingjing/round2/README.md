# Round-2 reference implementation

These files implement the three model classes and the contract tests that the
[round-2 spec](../as-jepa-round2-spec.md) asks for. Copy them into the
collaborator's tree unchanged:

| File here | Destination | What it is |
| --- | --- | --- |
| `as_jepa_lewm/forecast_scored.py` | `as_jepa_lewm/` | `ForecastScoredLeWM`: L2 whose renderer reads, per rendered frame with probability `lam`, a `sigma0`-noisy sample of the prior's forecast instead of the encoded state (spec Section 3.1) |
| `as_jepa_lewm/learned_variance.py` | `as_jepa_lewm/` | `LearnedVarianceLeWM`: the same model with learned posterior and prior variances (spec Section 3.4) |
| `as_jepa_lewm/nce.py` | `as_jepa_lewm/` | `NCELeWM`: the reconstruction-free control, an InfoNCE critic in place of the renderer (spec Section 3.6) |
| `tests/as_jepa/test_round2.py` | `tests/as_jepa/` | 11 CPU contract tests (spec Section 11) |

Register the classes as spec Section 2.1 describes (factory, `evaluate.load_model`,
planning whitelist). Do not edit `model.py`, `robust.py` or `research.py`.

## How these were checked

All 11 tests passed on 2026-10-31 against the collaborator's branch
`research/ideas-20261028` at `094e4f0` (files exported with `git archive`), with
stable-worldmodel's LeWM modules at upstream `21446f1`, on CPU with Python 3.12.2,
torch 2.7.0 and transformers 4.53.1, using the `sdpa_math` backend. In the
collaborator's environment, run from the repository root:

```bash
PYTHONPATH=.:runtime:<stable-worldmodel checkout> python -B -m unittest tests/as_jepa/test_round2.py -v
```

The tests use tiny models (one encoder layer, a 64-wide one-layer renderer, 56-pixel
encoder input), so they check contracts, not training behaviour. The GPU smoke in
spec Section 11.3 checks the full-size models.
