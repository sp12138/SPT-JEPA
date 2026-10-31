# Decoder-free AS-JEPA (exploratory branch `lingjing/decoder-free-as-jepa`)

AS-JEPA with the renderer replaced by the state's information beyond the anchor, computed in state space. With an optimal renderer, the round-1 objective depends on the encoder only through that quantity, the encoder's other entropies, and a constant. So at `lam = 0` this is the same objective with a different estimator for one term, and the theory carries over. It has no decoder, EMA, stop-gradient, augmentation or schedule, and every term is in nats.

| File | What it is |
| --- | --- |
| [story.md](story.md) | formulation, objective, theory, how the explored techniques (`lam`, noise, open loop, one step) behave in this form, compatibility with the renderer version, narrative, prior work, risks |
| [spec.md](spec.md) | experiment and implementation handoff: Phase 1 on PushT (price × prediction-weight grid; forecast structure × `lam`; prior input; pooled against per-window estimator; the transient threshold on the dot data; appearance and estimator gaming on the colour data; initial scale; an equivalence test) and TwoRoom; Phase 2 conditional |
| [as_jepa_lewm/decoder_free.py](as_jepa_lewm/decoder_free.py) | reference implementation of `DecoderFreeLeWM`; copy it into the collaborator's `as_jepa_lewm/` |
| [tests/as_jepa/test_decoder_free.py](tests/as_jepa/test_decoder_free.py) | 14 CPU contract tests; copy into `tests/as_jepa/` |
| [../discussions/2026-10-31-decoder-free-as-jepa.md](../discussions/2026-10-31-decoder-free-as-jepa.md) | how the idea was reached, with the synthetic toy and its results |

## How the reference files were checked

All 14 tests passed on 2026-10-31 against the collaborator's branch `research/ideas-20261028` at `051e82d`. The files were exported with `git archive`, with stable-worldmodel's LeWM modules at upstream `21446f1`, on CPU with Python 3.12.2, torch 2.7.0 and transformers 4.53.1, using the `sdpa_math` backend. In the collaborator's environment, run from the repository root:

```bash
PYTHONPATH=.:runtime:<stable-worldmodel checkout> python -B -m unittest tests/as_jepa/test_decoder_free.py -v
```

The tests use tiny models, so they check contracts, not training behaviour. The synthetic toy behind the predictions is scratch (`.tmp/latent-toy/` on the authoring machine) and is summarized in the discussion log.
