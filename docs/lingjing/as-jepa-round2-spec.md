# AS-JEPA round 2: the forecast-scored renderer (implementation and experiment spec, handoff)

Written 2026-10-31 for the collaborator who runs the L-line experiments. Follow it as written. Section 13 lists what you may change without asking; anything else needs our agreement first.

It builds on your branch `research/ideas-20261028` at `094e4f0`: its code, data, evaluation tools and the full-data PushT recipe of your `robustness20261029` and `stability20261029` runs. Where this spec is silent, the first-round spec ([as-jepa-experiment-spec.md](as-jepa-experiment-spec.md)) still applies.

Appendix A gives the theory this round tests. You do not need it to execute the spec, but every component below names the design reason it serves, so that implementation shortcuts do not silently change the method.

A reference implementation of every new model class, with 11 CPU contract tests that pass against `094e4f0`, is in [round2/](round2/). Copy it; do not reimplement it.

Context, if you want it: the review of your first-round results ([as-jepa-results-review-20261028.md](as-jepa-results-review-20261028.md)) and the plan that followed it ([as-jepa-status-and-next-steps-20261029.md](as-jepa-status-and-next-steps-20261029.md)). This spec supersedes that plan's Phases 1 and 2. Its blocks A to F reappear below, adapted to your later full-data PushT results; Appendix C lists what was narrowed or dropped.

## 0. Overview

### 0.1 The one change

In L2 the renderer always reads the encoded state of the frame it draws, `s_t = mu(x_t) + sigma0 * noise`.

In round 2, a coin is tossed for each rendered frame. With probability `lam`, the renderer instead reads a sample of the prior's forecast of that state, `s_hat_t = m_t + sigma0 * noise`. Here `m_t` is the prior's mean for `s_t`, computed from the earlier states and the actions. It is the same tensor the surprise term already uses.

Nothing else changes. Encoder, prior, rate (the one-step Gaussian KL priced at `beta`), renderer, flow loss, anchor, data and optimizer all stay L2's. At `lam = 0` the new class computes exactly L2's loss, with the same random draws.

```python
forecast = predicted + sigma0 * xi                 # a sample of the prior p(s_t | s_<t, a_<t); xi ~ N(0, I)
use = torch.rand(B, T, 1, 1) < lam                 # one coin per rendered frame
condition = torch.where(use, forecast, s[:, 1:])   # what the renderer reads
```

### 0.2 Why

1. **It rewards what predicts the future (sufficiency).** At `lam = 0` the renderer asks whether a frame's own state, together with the anchor, describes that frame. That rewards information about the current frame. At `lam = 1` it asks whether the earlier states and the actions, passed through the prior, describe the next frame. That rewards information about the future, beyond what the anchor shows: the textbook definition of a state.

   It also closes a gap in the `lam = 0` objective. Take a fixed property that the dynamics read and the anchor shows, such as block shape or size when they vary. At `lam = 0` such a property earns nothing, because the anchor already draws it. An exact code then leaves it for the prior to infer after the first contact, and a planner that starts before any contact never sees it. At `lam = 1` the prior, which never sees the anchor, cannot forecast contacts without it, so the property must enter the state.
2. **Unpredictable detail leaves without calibrating `beta`.** Take detail that is drawn fresh in every frame (the dot in the dot domain). It costs `beta` per nat, and it earns `1 - lam` per nat, because only the encoded-state branch can use it; no forecast can predict it. It is therefore excluded when `beta >= 1 - lam`, and at `lam = 1` that means at every `beta > 0`. At `lam = 0`, exclusion needed a true price near one, and the flow loss cannot certify that price because it is not measured in nats. (The renderer's condition dropout shifts these thresholds slightly; see Appendix A.3.)
3. **The renderer trains on the kind of input generation feeds it.** Generation renders prior rollouts, but at `lam = 0` the renderer has only ever seen encoded states. Your forecast-render runs already suggest that rendering forecasts during training helps planning (Section 0.3).

### 0.3 How this differs from your forecast-render and temporal-forecast arms

Your forecast-only (I05) and temporal-forecast (TF) arms are the closest existing evidence. They differ from `lam` in what the renderer reads.

| | Forecast-only (I05) and TF (`robust.py`) | Round-2 `lam` (`forecast_scored.py`) |
| --- | --- | --- |
| What the renderer reads, with probability 0.5 or `lam` | an open-loop rollout of prior **means** started at the noisy first state; no noise after frame 0 | **canonical:** the teacher-forced one-step prior mean plus `sigma0` noise; **variant:** an open-loop rollout of prior **samples**, with `sigma0` noise at every step |
| Capacity of that channel | unbounded: a noiseless continuous vector, never priced. The prior can pass action information to the renderer for free | the code's own resolution `sigma0`. What the forecast carries is priced by the surprise term |
| Posterior noise | i.i.d. (I05), or AR(1) with `rho = 0.8` (TF) | i.i.d. `sigma0`, as in L2 |
| PushT P0, 3 epochs, seeds 1/2 | I05 90 / 90; TF 94 / 94 (original `beta = 1/4`: 88 / 74.67; lower-beta: 86 / 84) | not yet run |
| Harder tasks | TF: TwoRoom state collapsed (2 to 6% success at 6,000 updates; an untrained model scores 4%); Cube 18 / 20% against native LeWM 26 / 26% at 12,000 updates (`hardtasks20261030`, `native_baseline_20261031`, `ideas10_20261030/results/relative600`) | not yet run |

The difference matters for the harder tasks. Suppose the anchor is the window's first frame, as in your TwoRoom and Cube runs, and the forecast is noiseless. Then an empty state plus a prior that integrates the actions can already draw the forecast frames: the anchor shows the start, and the forecast carries the displacement. Nothing forces information into the state (Appendix A.5).

That is one candidate cause of the TwoRoom collapse you measured: state spread 1e-4 to 1e-3, far below `sigma0`, even at `beta = 0`. The other candidate is the initialization scale: the state starts far below `sigma0`, so the renderer learns to ignore it. Block F tests the first cause on PushT, and Phase 3 tests both on TwoRoom.

### 0.4 What this round must answer

| Question | Block | Phase |
| --- | --- | --- |
| Q1. Does the principled `lam` keep the planning gain of your forecast-render arms on PushT, and does it reduce open-loop drift? | C | 1 |
| Q2. At `lam = 1`, does the dot leave the state at every `beta`, while it stays at `lam = 0`? | D | 1 |
| Q3. Is colour kept out of the state equally at every `lam`, and do LeWM's objective and L1 keep the colour that AS-JEPA leaves to the anchor? | E | 1 |
| Q4. Does the price matter at the working point, and what does pricing add over L1's SIGReg? | A | 1 |
| Q5. Does a smaller code noise (`sigma0 = 0.03`) make `beta = 1` work? | B | 1 |
| Q6. Does a first-frame anchor open the forecast shortcut, with noisy and with noiseless forecasts? | F | 1 |
| Q7. How does the selected configuration compare with LeWM, L1 and TF at the full 10-epoch budget, and is it stable across training seeds? | H | 2 |
| Q8. Does a fixed property that the dynamics read (block shape and size) enter the state at `lam = 1` and not in an exact `lam = 0` code? | G | 2 |
| Q9. What does the same model learn with a contrastive critic instead of a renderer? | I | 2 |
| Q10. Does the canonical recipe train a state on TwoRoom and Cube, where TF collapsed or trailed LeWM? | T | 3 |

Stopping points: after Phase 1, write the Phase-1 report and wait for our agreement. After Phase 2, write the PushT report and wait for our agreement before Phase 3.

## 1. Named quantities

New code reads these from the run config or from module constants in `as_jepa_lewm/forecast_scored.py`. Derived values are computed, not retyped.

### 1.1 Derived from your data and recipe

| Name | Definition | Value |
| --- | --- | --- |
| `BATCH` | windows per optimizer step | 128 |
| `EPOCH_UPDATES` | `floor(train windows / BATCH)` on `pusht-lance-full-v1` at `W = 8` (1,598,974 train windows) | 12,491 |
| `MAX_STEPS` | `10 * EPOCH_UPDATES`; the learning-rate schedule's horizon for every L2-family run, screens included | 124,910 |
| `SCREEN_STEPS` | `3 * EPOCH_UPDATES`; where Phase-1 screens stop | 37,473 |
| `EPOCH2_STEPS` | `2 * EPOCH_UPDATES` | 24,982 |
| `L0_EPOCH_UPDATES` | the same for LeWM's window of 4 (your `full-L0-10epoch-*` configs) | 15,265 |
| `STEP_S` | measured seconds per step of full L2 on one H200 | about 0.91: 11,328 s for epoch 2 of a new-seed replication run, peak 129 GiB (`hardtasks20261030/legacy-closure/newseed-epoch2/replication-results.json`) |
| `SCREEN_GPU_H` | `SCREEN_STEPS * STEP_S / 3600` | about 9.5 |
| `FULL_GPU_H` | `MAX_STEPS * STEP_S / 3600` | about 31.6 |
| `CONTINUE_GPU_H` | from `SCREEN_STEPS` to `MAX_STEPS` | about 22.1 |
| `EPOCH2_GPU_H` | `EPOCH2_STEPS * STEP_S / 3600` | about 6.3 |

### 1.2 Method knobs

| Name | Meaning | Value | Status |
| --- | --- | --- | --- |
| `lam` | probability that the renderer reads a forecast sample instead of the encoded sample, per rendered frame | {0, 0.5, 1} | **canonical value 1**: the principled setting (Appendix A), compared rather than searched. 0.5 is a hedge, and 0 is L2 |
| `horizon` | `one_step`: the teacher-forced one-step prior mean. `rollout`: an open-loop rollout of prior samples from the encoded first state | `one_step` | `rollout` is one arm of Block C |
| `forecast_grad` | `joint`: no stop-gradient anywhere. `encoder_only`: the forecast branch sends no gradient into the prior's parameters | `joint` | `encoder_only` only by the rule in Section 12 |
| `beta` | price of one nat of state, per rendered value (the unit your configs use) | {1/16, 1/4, 1}; 1e-3 as "almost no price" | the only searched knob |
| `sigma0` | code noise; the prior's spread is tied to it, and so is the forecast noise | 0.1 | frozen; 0.03 only in Block B |
| `gap` | anchor gap in kept frames; set the top-level `gap` **and** `model.anchor_gap` to the same value (Section 3.7) | 8 | frozen; 0 only in Block F |
| `W` | window, in kept frames | 8 | frozen |
| `rho` | AR(1) correlation of the posterior noise | 0 | frozen; only the TF reference uses 0.8 |
| `calibrate_state_std` | your one-time initialization of the state spread (`train.py`, `on_fit_start`) | absent, as in the reference runs | Phase 3 sets it to `sigma0`; never combine it with `LearnedVarianceLeWM` (Section 3.4) |

Why only `beta` is searched (unchanged from round 1): `beta` cannot be set from first principles while the flow loss is not in nats. `sigma0`, the gap and the window are frozen at values that favour keeping slow components. `lam` has a principled value, 1. It is compared against 0.5 and 0, and the selection rule (Section 8) prefers 1 when they tie.

### 1.3 Evaluation and decision constants

| Name | Value | Meaning |
| --- | --- | --- |
| `PLAN_SEEDS` | 42, 43, 44 | primary planning cases: 3 × 50 = 150 episodes per checkpoint |
| `FRESH_SEEDS` | 1042, 1043, 1044 | fresh planning cases, same protocol |
| `PAIRED_CI` | 95% episode-clustered bootstrap of paired P0 differences (`report_main_conclusions.cluster_ci`), pooled over training seeds 1 and 2 | the uncertainty from planning episodes. It is conditional on the trained weights and does **not** include training-seed variation, which is large (the original arm scores 88 and 74.67 on seeds 1 and 2). Hence the per-seed conditions below |
| `NONINF` | 5 points | arm A is **non-inferior** to arm B if the lower end of `PAIRED_CI(A - B)` is above -5 **and** each seed's mean difference is above -5 |
| better, worse | | A is **better** than B if `PAIRED_CI(A - B)` lies above 0 **and** both seeds' mean differences are positive; **worse** if the interval lies below 0 and both are negative; otherwise **tied** |
| `SLOPE_MIN` | 0.9 | lowest acceptable calibration slope of the prior, measured in eval mode (Section 5.3). The toys predict at most 3.5% overshoot with fixed variances, so 0.9 is about three times that |
| `VAR_RATIO_MAX` | 1.25 | highest acceptable ratio of the prior's squared miss to its variance (learned variances, Section 5.3) |
| `CHANCE_R2` | 0.05 | a probe `R^2` at or below this counts as chance. For the dot, decide on the **largest** of the five dot `R^2` values (position x and y, colour r, g, b), not their mean, so that one decodable component is not hidden |
| `COLOUR_TOL` | 0.10 | two colour-probe `R^2` values closer than this count as unchanged |
| `SHAPE_KEEP`, `SHAPE_DROP` | 0.8, 0.3 | balanced accuracy of the 7-way block-shape probe that counts as kept or dropped (chance 0.14) |
| `COLLAPSE_RATIO` | 0.9 | as in `collapse_audit.py` |
| `MOVE_PX` | 10 workspace pixels | as in your evaluation: a window moves the block if it moves more than this |
| `GALLERY_PASS_MIN` | 24 of 32 | fixed gallery cases that we mark as passing (round-1 gate) |

## 2. Starting point

### 2.1 Code

1. Create branch `as-jepa/round2` from your current head (`094e4f0` or later; record the hash in every report).
2. Copy, unchanged:
   - `docs/lingjing/round2/as_jepa_lewm/forecast_scored.py`, `learned_variance.py` and `nce.py` into `as_jepa_lewm/`;
   - `docs/lingjing/round2/tests/as_jepa/test_round2.py` into `tests/as_jepa/`.
3. Register the three classes in the three places that list model targets:
   - `as_jepa_lewm/factory.py`, `MODEL_TARGETS`: add `"forecast_scored": "as_jepa_lewm.forecast_scored.ForecastScoredLeWM"`, `"learned_variance": "as_jepa_lewm.learned_variance.LearnedVarianceLeWM"` and `"nce": "as_jepa_lewm.nce.NCELeWM"`.
   - `as_jepa_tools/evaluate.py`, `load_model`: add one `elif` per class before the final `else`, as for `RobustLeWM`. Without it, a new checkpoint falls through to `ASJepaLeWM` and fails on the new keyword arguments.
   - `as_jepa_tools/planning.py`: add the three targets to the set of allowed `_target_` values.
4. Add two small evaluation options and one tool (Sections 5.1 to 5.3): `--condition {posterior,forecast}` and `--fp32` for `collapse_audit.py`, `--condition` for `color_swaps.py`, and `as_jepa_tools/forecast_calibration.py`.
5. Do not edit `model.py`, `robust.py`, `research.py` or any frozen source; your earlier runs depend on them.
6. Freeze every campaign's source with your usual `git archive` plus SHA256 manifest, and dispatch with `as_jepa_tools/workflow.py`.

### 2.2 Data

- **`D-std`:** `/efs/prescottsun/SPT-JEPA/data/pusht-lance-full-v1` with `all_phase_data: true`. Evaluation data is its `phase0/` view, and planning uses `pusht_expert_train.h5` exactly as `planning.py` sets it up. Use the existing 934 test episodes.
- **`D-dot` and `D-colour`:** `D-std` with `model.domain` set to `"dot"` or `"color"`, applied in `ASJepaLeWM.prepare`. The dot is redrawn for every row (frame), and the anchor carries its own, different dot. The colour transform is a per-episode affine palette, shared by every frame and the anchor.
- **`D-shape`:** Phase 2 only; see Section 3.5.

### 2.3 Reference runs reused, not rerun

Use the 3-epoch checkpoints (`model-step0037473`) and their existing evaluations. The epoch-1 exports of these runs no longer exist, so compare against them only at epochs 2 and 3.

| Reference | Configs | Seeds and budget | P0 at 3 epochs, seeds 1/2 |
| --- | --- | --- | --- |
| original: L2, `beta = 1/4` | `experiments/stability20261029/configs/full-L2-beta0.25-stage3-seed{1..4}.json` | seeds 1 to 4 at 3 epochs; seeds 5 to 8 at 2 epochs | 88 / 74.67 |
| lower-beta: L2, `beta = 1/16` | `experiments/robustness20261029/configs/lower-beta-seed{1..4}.json` | seeds 1, 2 at 3 epochs; seeds 3 to 8 at 2 epochs | 86 / 84 |
| forecast-only (I05): `RobustLeWM`, `rho = 0`, `forecast_fraction = 0.5` | `experiments/robustness20261029/configs/forecast-render-seed{1,2}.json`; third epoch in `runs/forecast_confirm_20261030/forecast-render-seed{1,2}-forecast-epoch3` | seeds 1, 2 at 3 epochs | 90 / 90 |
| TF: `RobustLeWM`, `rho = 0.8`, `forecast_fraction = 0.5` | `experiments/robustness20261029/configs/temporal-forecast-seed{1..4}.json` | seeds 1, 2 at 3 epochs; seeds 3 to 8 at 2 epochs | 94 / 94 |

At a given training seed, `ASJepaLeWM`, `RobustLeWM` and `ForecastScoredLeWM` build identical initial tensors. The data order depends only on the seed. So the round-2 arms at seeds 1 and 2 are paired with these references. Section 11.2 checks this against the recorded initial weights of `lower-beta-seed{1,2}` (the original arm's configs did not record them).

## 3. Implementation

### 3.1 `ForecastScoredLeWM` (`as_jepa_lewm/forecast_scored.py`)

The class subclasses `ASJepaLeWM` and adds four constructor arguments: `lam`, `horizon`, `forecast_grad` and `forecast_seed`. It adds no parameters. Its loss is `ASJepaLeWM.loss` line for line, except for the renderer's condition:

```python
mu = flat.reshape(b, w, K, d)
s = mu + self.sigma0 * torch.randn_like(mu)                                   # posterior sample, as in L2
predicted = self.predict(s.flatten(2)[:, :-1], info["act_emb"][:, :-1]).reshape_as(mu[:, 1:])
entry = entry_nats(mu[:, 0], self.sigma0); surprise = surprise_nats(mu[:, 1:], predicted, self.sigma0)
condition = s[:, 1:]
if self.lam > 0:
    with torch.random.fork_rng(devices=devices):                             # separate random stream (Section 3.2)
        torch.manual_seed(FORECAST_STREAM + 1000003 * self.forecast_seed + int(step))
        use = torch.rand(b, w - 1, 1, 1, device=mu.device) < self.lam        # one coin per rendered frame
        xi = torch.randn(mu[:, 1:].shape, device=mu.device, dtype=mu.dtype)
        if self.horizon == "one_step" and self.forecast_grad == "joint":
            forecast = predicted + self.sigma0 * xi                           # reuses the surprise term's forward pass
        else:
            forecast = self._forecast(s, info["act_emb"], xi)                 # rollout, or encoder_only
    condition = torch.where(use, forecast, condition)
fm = flow_loss(self.renderer, y, condition.flatten(0, 1), a, self.training_shift)   # y, a: packed targets and anchors, as in L2
rate = (entry + surprise.sum(1)).mean() / ((w - 1) * FRAME_ELEMS)
loss = fm + beta * rate                                                       # unchanged
```

The forecasts, by setting:

- **`horizon = "one_step"` (canonical).** `m_t` is the teacher-forced prior mean from the encoded `s_<t` and `a_<t`, which is exactly the `predicted` tensor of the surprise term. The forecast is a sample of the distribution the rate already charges against.
- **`horizon = "rollout"` (Block C, arm C5).** An open-loop rollout of prior samples from the encoded first state: `h_0 = s_0`, then `h_t = m(h_<t, a_<t) + sigma0 * xi_t`, with the identity carry. It is the principled counterpart of your forecast-render rollout, which uses means and adds no noise after frame 0.

The forecast noise always equals the code noise `sigma0`, because the prior's spread is tied to it. Never render a noiseless prior mean during training: that is an unpriced channel of unbounded capacity (Appendix A.5).

Logged metrics (every 100 steps, as your other metrics), in addition to L2's:

- `forecast_share`: the fraction of rendered frames that read a forecast. It should equal `lam` on average.
- `calib_slope`: the calibration slope on the training batch, for plotting only. Predictor dropout pulls it slightly below 1; decisions use the eval-mode tool (Section 5.3).
- `active_dims`, computed as in `ASJepaLeWM`.

### 3.2 Random streams and pairing

The coins, the forecast noise and any extra predictor forward passes (including their dropout masks) run inside `torch.random.fork_rng`, seeded by `FORECAST_STREAM + 1000003 * forecast_seed + step`. The main random stream is therefore consumed exactly as in L2: posterior noise, flow-matching noise and timesteps, and the renderer's condition dropout.

Consequences:

- An arm at seed `k` sees the same data order and the same initial tensors as the reference arms at seed `k`.
- Its first update uses the same posterior noise and flow noise.
- Every later update draws from the same stream positions.

Comparisons across `lam` at the same seed are therefore paired. Set `forecast_seed` to the training seed. Pairing also requires one GPU per run with the same per-step batch (Section 13): `train.py` uses one device, and several ranks would change the data order and draw identical coins on every rank.

### 3.3 Gradient routing

With `forecast_grad = "joint"`, the default, there is no stop-gradient anywhere. The forecast branch's flow loss reaches:

- the prior, through `m_t`;
- the encoder, through the earlier states `s_<t`. This is how the encoder is rewarded for information that helps forecast later frames: the sufficiency reward.

At `lam = 1`, a frame's own encoding never enters its own render; it enters only the renders of later frames, because the prior's attention is causal. In particular, the last frame of a window reaches the loss only through the rate. The tests check the last-frame case.

Joint training has one known side effect, from Appendix A.5. The renderer values forecast information at `lam` per nat, while the prior pays only to second order in surprise. So the prior is rewarded for slightly exaggerating its predicted changes. The eval-mode calibration slope (Section 5.3) watches for this.

The fallback, `forecast_grad = "encoder_only"`, recomputes the forecast branch with the predictor's parameters detached (`torch.func.functional_call` with detached parameters) and the action embedding detached. States still receive the gradient; the prior's parameters and the action encoder do not. This matches the theory's assumption that the prior is trained only by the code length.

The recomputation runs in the separate random stream, so its dropout mask differs from the rate's; the forecast mean differs from the rate's `m_t` by a dropout-sized amount. It is a stop-gradient, so use it only when the rule in Section 12 fires.

### 3.4 `LearnedVarianceLeWM` (`as_jepa_lewm/learned_variance.py`; Phase 2, Block G)

Learned variances give the exact Gaussian code that the theory's fixed-property prediction is about (Appendix A.3). A fixed-variance prior cannot widen its uncertainty before a contact, so it keeps fixed properties for a side reason.

The same modules gain a log-variance half:

- **Posterior.** `pool.head` becomes `Linear(width, 2 d)`. The first `d` outputs are `mu`, initialized from the original head. The second `d` are `log var`, with zero weight and bias `log(sigma0^2)`. `encode` sets `info["emb"]` to the mean, so probes and planning are unchanged, and sets `info["logvar"]`.
- **Prior.** `predictor.transformer.output_proj` becomes `Linear(in, 2 * STATE_DIM)`. The first half is the mean change, zero-initialized, so the identity carry is unchanged. The second half is `log var`, with zero weight and bias `log(sigma0^2)`. `predict` returns means only, so LeWM's `rollout` and the planner work unchanged. `predict_dist` returns both.
- **Rate.** `entry = 0.5 * sum(exp(lv0) + mu0^2 - 1 - lv0)`, and `surprise = 0.5 * sum(lvp - lvq + (exp(lvq) + (mu - m)^2) / exp(lvp) - 1)`, in FP32.
- **Samples.** Posterior `s = mu + exp(lvq / 2) * xi`. Forecast `m + exp(lvp / 2) * xi`, one-step only, with `forecast_grad` `joint` or `encoder_only`.
- **Numerical guard.** Log variances are clamped to `[LOGVAR_MIN, LOGVAR_MAX] = [2 log 1e-3, 0]`, so the code spread stays between 0.001 and 1 prior unit.
- **No `calibrate_state_std`.** The trainer's one-time initialization rescales `pool.head` as a mean-only head; with the log-variance half it would fail or rescale the log variances.

At initialization both variances equal `sigma0^2`, so the loss equals the fixed-variance loss exactly; this is tested. Additional metrics:

- `prior_var_ratio`: the mean squared miss of a posterior sample, divided by the mean prior variance. It is 1 when the prior is calibrated.
- `posterior_sd` and `prior_sd`.

### 3.5 `D-shape` dataset (Phase 2, Block G)

Collect with stable-worldmodel's own API at your pinned commit, following `scripts/data/collect_pusht_fov.py`:

```python
world = swm.World("swm/PushT-v1", num_envs=16, max_episode_steps=SHAPE_EP_STEPS,
                  image_shape=(224, 224), render_mode="rgb_array")
world.set_policy(WeakPolicy(dist_constraint=100, seed=SHAPE_SEED))
world.collect(path, episodes=SHAPE_EPISODES, seed=SHAPE_SEED,
              options={"variation": ("block.shape", "block.scale",
                                     "agent.start_position", "block.start_position", "block.angle")})
```

Settings:

- **Collection constants.** `SHAPE_EP_STEPS = 125`, which gives about 25 kept frames at frameskip 5, the median of `D-std`. `SHAPE_EPISODES = 20,000`. `SHAPE_SEED = 20261031`.
- **Variation names.** Check them with `swm fovs PushT-v1`. `block.shape` draws from {L, T, Z, square, I, small_tee, +}, and `block.scale` from [20, 60]. The goal overlay uses the block's shape.
- **Preparation.** `prepare_lance_full.py` asserts 18,685 episodes, hard-codes its source revision and drops the `variation.*` columns. Write `as_jepa_tools/prepare_shape.py` as a copy that:
  - reads the episode count and source from the collected file;
  - writes the same arrays and splits (5% test episodes, seed 0);
  - also writes per-row `variation_block_shape.npy` and `variation_block_scale.npy`.

  Then `FullPushTWindows` reads `D-shape` like `D-std`.
- **Policy.** `WeakPolicy` makes random pushes near the block. It works for every shape, but its action distribution differs from `D-std`'s expert data. Compare arms only within `D-shape`.
- **Budget.** `SHAPE_EPOCH_UPDATES = floor(D-shape train windows / BATCH)`, computed by the preparation script. Block G configs use `max_steps = 10 * SHAPE_EPOCH_UPDATES`, stop at `3 * SHAPE_EPOCH_UPDATES` (Section 3.7), `checkpoint_every = SHAPE_EPOCH_UPDATES`, and retain epochs 1 to 3.
- **Dataset card.** Episode counts by shape, the fraction of windows with a contact, `SHAPE_EPOCH_UPDATES`, and the share of windows without an eligible anchor at `gap = 8`.

### 3.6 `NCELeWM` (`as_jepa_lewm/nce.py`; Phase 2, Block I)

This is the reconstruction-free control for the JEPA reader. It keeps L2's encoder, pooling, action encoder, prior and rate, and has no renderer.

The flow loss is replaced by an InfoNCE bound on the information that the condition carries about its frame. The condition is the encoded sample at `lam = 0`, or the one-step forecast sample at `lam = 1`. The critic is the encoder's own Gaussian log-likelihood, so it has no parameters: `score(t, j) = -||condition_t - mu(x_j)||^2 / (2 sigma0^2)`. The candidates are all `B * T` rendered frames of the batch, including the other frames of the same window, so the bound is capped at `log(B * T)`, about 6.8 nats per frame.

The loss per window is `T * InfoNCE + beta * rate`. Both terms are in nats, so here `beta` is a calibrated price, and `beta = 1` is the canonical value.

- **Gradient audit.** The class declares `audit_expected_gradients` without a renderer.
- **Evaluation.** Nothing renders, so `completion_evaluate` does not apply: it always runs the collapse audit, which renders. Evaluate with planning and probes only, using the commands in Section 5.1.

### 3.7 Config template and launch

Example: arm C1, seed 1. Other arms change only the fields listed in Section 6.

```json
{
  "output": "/efs/prescottsun/SPT-JEPA/runs/round2_20261101/C1-lam1-b0.25-seed1",
  "data": "/efs/prescottsun/SPT-JEPA/data/pusht-lance-full-v1",
  "evaluation_data": "/efs/prescottsun/SPT-JEPA/data/pusht-lance-full-v1/phase0",
  "pose_weights": "/efs/prescottsun/SPT-JEPA/runs/insights12h_20261027/pose-v2/weights.pt",
  "seed": 1, "batch_size": 128, "workers": 4, "cpu_threads": 2,
  "max_steps": 124910, "max_seconds": 31536000, "checkpoint_every": 12491,
  "all_phase_data": true, "stop_after": 37473, "gap": 8,
  "retain_steps": [12491, 24982, 37473, 99928, 124910],
  "model_class": "forecast_scored",
  "model": {"mode": "L2", "beta": 0.25, "sigma0": 0.1, "window": 8, "anchor_gap": 8,
            "renderer_width": 384, "renderer_depth": 12, "renderer_heads": 6,
            "lam": 1.0, "horizon": "one_step", "forecast_grad": "joint", "forecast_seed": 1},
  "record_initial_weights": true
}
```

- **Launch screens with your staged trainer:** `python -B -u -m as_jepa_tools.train_stage --config <config.json> --stop-step 37473`, and add `--resume` after an interruption. `stop_after` acts only on a fresh start (`train.py` line 155), so a screen resumed after a crash without `--stop-step` would run to 124,910 steps.
- **Set `gap` and `model.anchor_gap` together.** Training reads the top-level `gap`, and evaluation reads `model.anchor_gap`; nothing enforces equality. Add a startup assertion to `train.py` that they are equal for every round-2 run.
- **Keep `max_steps = 124910`** in every L2-family screen, so the learning-rate schedule matches the reference runs and a screen can be continued to 10 epochs (Phase 2) with `as_jepa_tools.resume_lewm`.

### 3.8 What does not change

The following are unchanged:

- the encoder, `QueryPool`, action encoder, LeWM predictor with the identity carry, and the renderer (384 / 12 / 6);
- the optimizer: AdamW, 5e-5 and 1e-4, 1% warmup, cosine to `MAX_STEPS`;
- batch 128, bf16 with the rate in FP32, gradient clip 1.0, and one GPU per run;
- data, anchors and evaluation tools;
- the planner, which reads encoder means and noiseless prior rollouts, with no renderer and no anchor.

No schedule acts on `lam`, `beta` or `sigma0`.

## 4. Design invariants for round 2

Each line is a property the method depends on. Items 4, 6 and 8 changed from round 1.

1. The encoder reads one frame and nothing else: no history, anchor or actions.
2. The anchor reaches only the renderer. The prior never reads frames, the anchor or renderer features.
3. Actions reach only the prior. The renderer never reads actions or history directly. A forecast can carry information about actions only through the prior's mean, which the surprise term prices, plus `sigma0` noise.
4. **Changed.** For each rendered frame, the renderer reads either the posterior sample `s_t` (probability `1 - lam`) or a noisy prior sample (probability `lam`), with noise `sigma0` (or the learned prior spread in Block G). It never reads `mu`, never a noiseless prior mean, and never anything else from the prior.
5. Window frame 0 is encoded and starts the chain, and is never rendered.
6. **Changed.** No augmentation, EMA, target network, SIGReg, VICReg, masking or stop-gradient in AS-JEPA arms. The only allowed exception is `forecast_grad = "encoder_only"` under the rule in Section 12.
7. Nothing normalizes `mu`, `s`, the prior's mean or the forecast before the renderer's linear projection.
8. **Changed.** Within a run, `sigma0`, `beta` and `lam` are constant, and the prior's spread and the forecast noise are tied to `sigma0` (fixed-variance code). The rate is the exact one-step Gaussian KL with one `beta` for entry and surprise. The `rollout` horizon changes only what the renderer reads, never the rate.
9. The renderer's target is fixed data (packed rendered frames).
10. The coins and forecast noise use their own random stream, so arms at the same seed are paired.

## 5. Evaluation

### 5.1 The package, and keeping comparisons paired

Every comparison with a reference run must use the evaluation settings the reference was evaluated with. The defaults of your tools are those settings:

- `completion_evaluate` runs probes in BF16 and FP32;
- the collapse audit runs under BF16 autocast.

Do not change those defaults for the comparisons. Because BF16 can make an empty state look full (status note, insight 7), the new `--fp32` option of `collapse_audit.py` adds a second, FP32 verdict for every round-2 arm and its references. Report both.

**At 2 epochs (short gates)**, run the same commands and flags as your `forecastconfirm20261030` gates, so the cases match:

```bash
python -B -u -m as_jepa_tools.planning --checkpoint <ckpt> --output <dir> --upstream <stable-worldmodel> \
    --data-root <data/planning> --test-episodes <pusht-lance-full-v1/phase0/test_episodes.npy> --seed 42 --episodes 50    # also 43, 44
python -B -u -m as_jepa_tools.collapse_audit --checkpoint <ckpt> --output <dir> --data <phase0> --pose <pose> --cases 32 --steps 20
python -B -u -m as_jepa_tools.quality_complete --checkpoint <ckpt> --output <dir> --data <phase0> --pose <pose> --i3d <i3d> \
    --cases 64 --samples 1 --steps 20 --selection-seed 20261029
python -B -u -m as_jepa_tools.counterfactual --checkpoint <ckpt> --output <dir> --data <phase0> --pose <pose> \
    --cases 16 --steps 20 --selection-seed 20261029
```

**At 3 epochs**, run the full package, `as_jepa_tools.completion_evaluate --mode L2 --expected-step 37473`. It runs:

- probes, the collapse audit and `quality_complete` (256 cases);
- `counterfactual` (64 cases), planning on `PLAN_SEEDS`, the swap gallery and `gallery_complete`.

Also run planning on `FRESH_SEEDS`, the forecast checks (5.2), the calibration tool (5.3) and the drift decomposition (5.6).

**Models without a renderer (L0 and `NCELeWM`).** Run `as_jepa_tools.planning` (as above) and `as_jepa_tools.probes --checkpoint <ckpt> --data <phase0> --output <dir> [--noisy] [--fp32]` directly. `completion_evaluate` renders, and its gallery stage reads `cases.json` before checking `--planning-only`, so it fails for these models.

**Comparisons.** P0 at 3 epochs on the primary cases, pooled over seeds 1 and 2 (300 paired episodes), with `PAIRED_CI` and the per-seed conditions of Section 1.3, against each reference in Section 2.3.

P0 is now primary, where the superseded plan made it secondary until three seeds existed. Three things changed: the seeds are paired with the references, the bootstrap is clustered by episode, and every decision also requires agreement in each training seed. The finals (Block H) add a third seed and a stability block.

**Also report:**

- `h{1,3,7,13}_{agent,block}_ratio` and `h7_moving_block_ratio`, for mean and sample rollouts;
- LPIPS and SSIM;
- `action_identification_accuracy`;
- the rate per window, `entry` and `surprise`.

### 5.2 Forecast-based checks for `lam > 0`

At `lam = 1` the renderer never sees encoded states during training, so any test that renders encoded states is out of distribution. Add a `--condition {posterior,forecast}` option, defaulting to `posterior`, to the two tools below. For `lam > 0` arms, report both conditions; the forecast version is the one that counts.

- **`collapse_audit.py`.** With `forecast`, render each held-out frame `t >= 1` from its one-step forecast sample, `m_t + sigma0 * xi`. Here `m_t` is computed from the encoded posterior samples of the earlier frames of the same window and their actions. Apply the existing rule: a run has collapsed if the ratio to the anchor-as-prediction baseline is at least `COLLAPSE_RATIO`, or if fewer than 2 dimensions are active.
- **`color_swaps.py` (Block E).** With `forecast`, the state swap renders episode B's one-step forecasts with episode A's anchor.

### 5.3 Calibration of the prior

Add the tool `as_jepa_tools/forecast_calibration.py`. It takes 32 held-out windows, the model in eval mode (no dropout) and FP32, and reports `calibration_slope` (from `forecast_scored.py`): the least-squares slope of the true change `mu_t - s_(t-1)` on the predicted change `m_t - s_(t-1)`, pooled over dimensions.

- A prior whose mean is the conditional expectation gives a slope of 1. A slope below 1 means the predicted changes are exaggerated.
- For learned variances, also report `prior_var_ratio`; 1 means the prior is calibrated.
- Evaluate at epochs 2 and 3 for every `lam > 0` arm. Also evaluate the `lam = 0` arm with the same `beta`, data and gap where one exists:
  - C1, C4 and C5 against original;
  - C2 against lower-beta;
  - D2 and D3 against D1;
  - E2 against E1;
  - B3 against B1, and B4 against B2.

  C3, D4, F1, F2 and A2 have no such reference and use the absolute threshold only.

### 5.4 Domain probes and colour invariance

- **Dot (Block D).** `probes.py --noisy --fp32` on the dot domain reports `R^2` for `dot_x`, `dot_y`, `dot_r`, `dot_g` and `dot_b`, and their mean `nuisance_r2`. Decide on the largest of the five (Section 1.3), and report all six numbers plus the dynamics probes.
- **Colour probe (Block E).** The same tool on the colour domain reports `R^2` for the three gains and three biases. Report them all.
- **Colour invariance (new tool, Block E).** For 512 held-out frames, encode each frame twice: once under its own palette A, and once under the palette B of a different random test episode. The palette transform is the same as in `ASJepaLeWM.prepare`, with the episode id substituted. Report:
  - the **invariance index**, `E||mu(x_i^A) - mu(x_i^B)||^2 / E||mu(x_i^A) - mu(x_j^A)||^2`, where `j != i` and both frames share palette A. Near 0 is colour-invariant; near 1 means palette changes move the state as much as pose changes do;
  - the **recoloured-goal retrieval** accuracy: the fraction of frames for which `mu(x_i^B)` is nearest to `mu(x_i^A)` among the 512 frames under palette A.

  For L0 use its embedding, and for L1 use `mu`.

### 5.5 `D-shape` metrics (Block G)

- **Pose estimator.** Retrain your pose estimator (`pose.py`, same recipe) on `D-shape` train frames. Its targets are agent and block position plus `sin` and `cos` of `(2 pi / sym) * angle`, where `sym` is the shape's symmetry angle from the environment (`square` and `+`: `pi/2`; `I` and `Z`: `pi`; others: `2 pi`). Report its floor on test frames.
- **Shape probes.** Logistic regression of `block.shape` on the window frame-0 `mu` (balanced accuracy over 7 classes), and ridge regression of `block.scale` (`R^2`). Fit on train episodes and score on test episodes.
- **First-contact test.** Take 256 test windows of 16 kept frames in which the block moves less than `MOVE_PX` over the 3 context frames and more than `MOVE_PX` within the next 7 frames. Roll out the prior from 1 context frame and from 3 context frames. Report:
  - block position and angle error at each horizon, relative to the copy baseline;
  - the same error at the first frame in which the block has moved more than `MOVE_PX`, plus 3 frames.
- **Planning.** Run planning on `D-shape` only if the upstream evaluation restores each episode's `block.shape` and `block.scale` at reset. Check this in a dry run, and say so either way.

### 5.6 Open-loop drift

Your `as_jepa_tools/decode_decomposition.py` separates the prior's forecast error from the renderer's error given the true future state. It is the measurement behind the superseded plan's "block error 5.1, 10.7, 18.9 px at h1, h7, h13 from forecasts, about 5 px from the frame's own state".

Run it at 3 epochs (`--cases 128 --steps 50`, with and without `--fp32`) for every Block C arm and its references. Report the block error from open-loop forecasts and from the frame's own encoded state at h1, h7 and h13, and their difference, the **drift**.

For `lam = 1` arms, the own-state render is out of distribution (Section 5.2), so read the forecast column as the primary number.

### 5.7 What every report records

For every run, record its config, source hash, `gap`, `lam`, `horizon`, `forecast_grad`, `beta`, `sigma0` and `seed`. Also record the evaluated checkpoint step, GPU-hours, the metrics above, the collapse verdicts (posterior and forecast, BF16 and FP32) and the calibration slope.

Write one JSON record per run. The report tables give the mean and spread over seeds, `PAIRED_CI` and both per-seed differences for every comparison.

## 6. Run plan

Run IDs follow `r2-<arm>-seed<k>`, with arm names as in the tables. Every Phase-1 L2-family run uses the Section 3.7 template, stopped at `SCREEN_STEPS`, with the changes listed. Seeds 1 and 2 pair with the references.

Why Phase-1 screens stop at 3 epochs, not the published 10:

- every reference arm exists at 3 epochs on the same seeds, so the comparisons are paired;
- planning is already 75 to 94% there;
- the learning-rate schedule spans 10 epochs, so the selected arms continue to the published budget in Phase 2 rather than restart.

### 6.1 Phase 1: `D-std` and its domains

**Block C (main): what the renderer reads.**

| Arm | `lam` | `horizon` | `beta` | Seeds | Compared with |
| --- | --- | --- | --- | --- | --- |
| C1 `lam1` | 1 | one_step | 1/4 | 1, 2 | original, forecast-only, TF |
| C2 `lam1-b0.0625` | 1 | one_step | 1/16 | 1, 2 | lower-beta |
| C3 `lam1-b1` | 1 | one_step | 1 | 1, 2 | C1; the full-data `beta = 1` run at `lam = 0` (seed 0: P0 8.67%, h7 agent ratio 0.838, action identification 35.94%, collapse test passed) |
| C4 `lam0.5` | 0.5 | one_step | 1/4 | 1, 2 | C1, original, forecast-only |
| C5 `lam0.5-roll` | 0.5 | rollout | 1/4 | 1, 2 | forecast-only: these differ only by the `sigma0` noise on every forecast step |

**Block D (purity): dot domain** (`model.domain = "dot"`, seed 1).

| Arm | `lam` | `beta` |
| --- | --- | --- |
| D1 `dot-lam0` | 0 | 1/4 |
| D2 `dot-lam0.5` | 0.5 | 1/4 |
| D3 `dot-lam1` | 1 | 1/4 |
| D4 `dot-lam1-b0.0625` | 1 | 1/16 |

**Block E (appearance): colour domain** (`model.domain = "color"`, seed 1, 3 epochs).

| Arm | Model | Notes |
| --- | --- | --- |
| E1 `color-lam0` | L2, `lam = 0`, `beta = 1/4` | |
| E2 `color-lam1` | L2, `lam = 1`, `beta = 1/4` | |
| E3 `color-L1` | your `full-L1-10epoch` config (`mode = "L1"`, `batchnorm_projector: true`) | `domain = "color"`, stopped at `SCREEN_STEPS` |
| E4 `color-L0` | your `full-L0-10epoch` config (`mode = "L0"`, `window = 4`) | `domain = "color"`, stopped at `3 * L0_EPOCH_UPDATES` |

**Block A (price): does the price matter at the working point, and what does it add over SIGReg?**

| Arm | Model | Seeds | Budget |
| --- | --- | --- | --- |
| A1 `lam0-b0.001` | L2, `lam = 0`, `beta = 1e-3` | 1, 2 | 3 epochs |
| A2 `lam1-b0.001` | L2, `lam = 1`, `beta = 1e-3` | 1, 2 | 3 epochs |
| A3 `L0-full` | your `full-L0-10epoch-seed{0,1,2}` configs, unchanged except the output directory and `retain_steps` (keep `L0_EPOCH_UPDATES` × 2, 3 and 10) | 0, 1, 2 | 10 LeWM epochs |
| A4 `L1-full` | your `full-L1-10epoch-seed{0,1,2}` configs, unchanged except the output directory and `retain_steps` (keep epochs 2, 3 and 10) | 0, 1, 2 | 10 epochs |

A3 and A4 are the missing PushT baselines: no full-data LeWM or L1 run has completed. They have no dependency, so start them first. Only seeds 0 to 2 of their configs exist; they are not paired with the L2 arms (different models), so their seeds need not match. Their 3-epoch checkpoints answer Q4; their 10-epoch checkpoints are the finals' baselines.

**Block B (entry price)** (`sigma0 = 0.03`, seed 1, 3 epochs).

| Arm | `lam` | `beta` |
| --- | --- | --- |
| B1 `s0.03-lam0` | 0 | 1/4 |
| B2 `s0.03-lam0-b1` | 0 | 1 |
| B3 `s0.03-lam1` | 1 | 1/4 |
| B4 `s0.03-lam1-b1` | 1 | 1 |

At `sigma0 = 0.1`, `beta = 1` did not collapse; it failed planning and action following (C3's comparison row). So Block B is judged on the gate items of Section 10 and P0, not on the collapse test.

**Block F (anchor)** (`gap = 0` and `anchor_gap = 0`, seeds 1 and 2, 3 epochs).

| Arm | Change | Question |
| --- | --- | --- |
| F1 `g0-lam1` | C1 with a first-frame anchor | with priced, noisy forecasts, does a first-frame anchor still open the shortcut? |
| F2 `g0-I05` | your forecast-only config (`RobustLeWM`, `rho = 0`, `forecast_fraction = 0.5`) with a first-frame anchor | with noiseless forecasts, as in your TwoRoom runs, does it? |

**Evaluation per Phase-1 run:** short gates at 2 epochs, and the full package at 3 epochs (Section 5). For Blocks D and E, also run the domain probes and colour tools of Section 5.4.

### 6.2 Phase 2: after the Phase-1 report and our agreement

**Block H (finals, `D-std`, 10 epochs).** The selected configuration `(lam*, horizon*, beta*)` from Section 8, plus two references (lower-beta and TF):

- **Continuations.** Continue seeds 1 and 2 of each of the three from their 3-epoch full states to `MAX_STEPS`, with `as_jepa_tools.resume_lewm`. If a full state is missing, rerun that seed from scratch.
- **Seed 3.** Train seed 3 of each of the three from scratch, to `MAX_STEPS`.
- **Baselines.** A3 (L0) and A4 (L1) at 10 epochs.
- **Evaluation.** At 10 epochs: the full package, plus planning on `PLAN_SEEDS` and `FRESH_SEEDS`.
- **Continue rule (round 1).** If the generation score of the selected configuration improves by more than 5% between epochs 8 and 10, run its seed 1 once to 20 epochs.

**Block H2 (stability, `D-std`, 2 epochs).** Train seeds 4 to 8 of the selected configuration to `EPOCH2_STEPS`, and run the 2-epoch short gates. Compare the per-seed P0 with your existing seeds 3 to 8 of original, lower-beta and TF at 2 epochs. The original arm failed on several new seeds at 2 epochs, so stability must be shown directly.

**Block G (fixed properties, `D-shape`, 3 epochs at `SHAPE_EPOCH_UPDATES`, seed 1).** All four arms use `beta*`.

| Arm | Code | `lam` |
| --- | --- | --- |
| G1 `shape-lam0-fixed` | `ForecastScoredLeWM` | 0 |
| G2 `shape-lam0-learned` | `LearnedVarianceLeWM` | 0 |
| G3 `shape-lam1-fixed` | `ForecastScoredLeWM` | 1 |
| G4 `shape-lam1-learned` | `LearnedVarianceLeWM` | 1 |

Add seed 2 to all four if any two shape-probe accuracies that the predictions distinguish differ by less than 0.1.

**Block I (reconstruction-free, `D-std`, 3 epochs, seed 1).**

| Arm | Model | `lam` | `beta` |
| --- | --- | --- | --- |
| I1 `nce-lam0-b1` | `NCELeWM` | 0 | 1 |
| I2 `nce-lam1-b1` | `NCELeWM` | 1 | 1 |
| I3 `nce-lam1-b0.1` | `NCELeWM` | 1 | 0.1 |

Here `beta` is in true nats (Section 3.6). Without a renderer these runs cost a small fraction of an L2 run. Evaluate them with planning and probes (Section 5.1).

### 6.3 Phase 3: harder tasks, after the PushT report and our agreement

Run TwoRoom first, then Cube, with your `BenchmarkWindows` data, budgets (TwoRoom 6,000 updates; Cube 12,000) and evaluation (primary and fresh cases, 50 each, two seeds). Your native LeWM and TF results are the references.

- **Anchor gap.** Implement the anchor-gap rule in `BenchmarkWindows`. Model it on `FullPushTWindows` (`as_jepa_tools/data.py`, lines 72 to 74), which, like `BenchmarkWindows`, indexes raw rows and keeps the anchor phase-aligned with the window. Set `gap_T` to the largest gap that leaves at most 20% of windows without an eligible anchor, and record the table in the dataset card.

| Arm (TwoRoom) | Recipe |
| --- | --- |
| T1 | `lam*`, `horizon*`; `beta` in {1/16, 1/4}; `gap = gap_T`; `calibrate_state_std = sigma0` |
| T2 | T1 with a first-frame anchor (`gap = 0`) |
| T3 | T1 without `calibrate_state_std` |
| T4 | T1 with `lam = 0` |

- **Diagnostics.** Track the FP32 state spread in units of `sigma0` (`feature_std / sigma0`) every 100 steps, and report the agent-position probe `R^2` from `mu`, the rate per window, the forecast collapse test and planning success.
- **Cube.** If any T arm trains a state, run Cube with the best T arm at both prices, under the same planner as the native baseline.

### 6.4 Counts, GPU-hours and priorities

| Block | Runs | Approximate GPU-hours |
| --- | --- | --- |
| C | 10 L2 screens | 95 |
| D | 4 L2 screens | 38 |
| E | 2 L2 screens, 1 L1 and 1 L0 at 3 epochs | 31 |
| A | 4 L2 screens; 3 full L0 runs; 3 full L1 runs | 38 + about 25 (L0 has no renderer; measure in the smoke) + 95 |
| B | 4 L2 screens | 38 |
| F | 4 screens | 38 |
| **Phase 1 total** | 30 screens, 6 full baseline runs | **about 400** |
| H | 6 continuations + 3 new 10-epoch runs | about 133 + 95 |
| H2 | 5 runs of 2 epochs | about 32 |
| G | 4 screens on `D-shape`, plus data and pose estimator | about 45 |
| I | 3 short runs without a renderer | under 10 |
| **Phase 2 total** | | **about 315** |

You decide the allocation. When compute is short, run in this order:

1. **A3** (`L0-full`) and **C** (C1 and C4 first, then C5, C2 and C3).
2. **D**, then **E**.
3. **A4** (`L1-full`), then **F**, then **A1** and **A2**, then **B**.

Everything without a dependency may start at once. `D-shape` collection (CPU) may start during Phase 1. Before launching, report the measured step times and the projected GPU-hours, and wait for our acknowledgement if they exceed this table by more than 25%.

## 7. Frozen predictions (record them, with the commit hash, before launch)

Each prediction states what the theory or the toys predict and how confident we are. A failed prediction is a result, not an error.

1. **C, planning (moderate confidence).**
   - C1 is non-inferior to forecast-only and better than original (Section 1.3 definitions; 3 epochs; primary cases).
   - C5 is non-inferior to forecast-only. These differ only by the forecast noise, which costs at most what an unpriced channel gains.
2. **C, drift (moderate confidence).** The drift at h7 and h13 (Section 5.6) is smaller for C1 and C4 than for original, and smallest for C5.
3. **C, calibration (high confidence).** Every Block C arm with `lam > 0` has an eval-mode calibration slope of at least `SLOPE_MIN` at epochs 2 and 3, and within 0.05 of its reference where one exists.
4. **C3, `beta = 1` (no prediction).** The toy is ambiguous: `lam` raises the collapse barrier slightly with a prior trained only by the code length, but lowers it under joint training. C3 measures it.
5. **D, dots (high confidence).**
   - The largest dot `R^2` is at most `CHANCE_R2` in D3 and D4.
   - In D1 it is above `CHANCE_R2`, because the working price is below the threshold of a true price of 0.9 that condition dropout sets at `lam = 0`.
   - D2 lies between them. If D2 reaches chance, the true price at code `beta = 1/4` is at least 0.45.
6. **E, colour (moderate confidence).**
   - The colour-probe `R^2` in E2 lies within `COLOUR_TOL` of E1.
   - E3 and E4 keep colour: `R^2` of at least 0.9, and an invariance index above E1's and E2's.
7. **A, price (high confidence).** The rate per window falls as `beta` rises, at both `lam` values. Planning, and L1 against L2: no prediction.
8. **F, anchor (moderate confidence).**
   - F2's state carries less agent information than forecast-only at gap 8: a lower noisy agent-probe `R^2` and a lower P0.
   - F1 loses less than F2 relative to its own gap-8 counterpart (C1), because its forecasts are priced.
9. **B, entry price (moderate confidence).** B2 passes the rollout and action-identification items of the Section 10 gate, and its P0 is well above the 8.67% of the `beta = 1`, `sigma0 = 0.1` run. Its rate per window is above that run's 37 nats. A smaller `sigma0` makes entry cheaper, so the state can afford more information at the same price.
10. **G, fixed properties (moderate confidence).**
    - **Shape probe.** G1, G3 and G4 are at least `SHAPE_KEEP`. G2 is at most `SHAPE_DROP`, or at least 0.2 below G4.
    - **First contact.** At the contact frame plus 3, the block error from one context frame is lower in G3 and G4 than in G2.
11. **H2, stability (no prediction).** H2 measures it; compare with TF's seeds 3 to 8 (86.67 to 92.67% at 2 epochs).
12. **I, contrastive critic (moderate confidence).** I2 keeps the agent (probe `R^2` of at least 0.8). Its block probe and P0 are below C1's: the bound saturates once the agent alone identifies the frame.

## 8. Selection rule (Phase 1 to Phase 2)

**Candidates** are the Block C arms (C1 to C5) and the two AS-JEPA references at `lam = 0` (original and lower-beta). Forecast-only and TF are comparisons only: they render unpriced means, so they cannot be the AS-JEPA configuration.

1. **Exclude failing arms.** Exclude any candidate that fails the collapse test that counts for it (the posterior version for `lam = 0`, the forecast version for `lam > 0`; BF16 verdict, as for the references). Also exclude any candidate that still violates the calibration rule after its fallback (Section 12).
2. **Generation filter.** Keep candidates whose mean rollout at 3 epochs has an `h7` agent ratio of at most 0.5 and an `h7_moving_block_ratio` below 1. This is the round-1 generation gate on the full 256-case evaluation.
3. **Best planner.** Find the candidate `A*` with the highest pooled P0 (primary cases, seeds 1 and 2).
4. **Tied set.** The tied set is every candidate non-inferior to `A*` (Section 1.3).
5. **Prefer the canonical design.** Within the tied set, prefer:
   1. `lam = 1, one_step` (C1, C2, C3);
   2. `lam = 0.5, one_step` (C4);
   3. `lam = 0.5, rollout` (C5);
   4. `lam = 0` (original, lower-beta).
6. **Lowest rate.** Among the remaining candidates in the same class, choose the lowest rate per window on held-out windows.

The selected `(lam*, horizon*, beta*)` goes to Phase 2. Then compare it with forecast-only and TF (Section 1.3 definitions), and apply Section 9.

## 9. Decision rules after Phase 1

| Result | Then |
| --- | --- |
| A `lam = 1, one_step` arm is selected | The canonical method uses `lam = 1`, and the story's empirical design states the selected `beta`. |
| C4 or C5 is selected | The theory keeps `lam` in the objective; the empirical design uses the selected value without emphasis. |
| A `lam = 0` reference is selected | Rendering forecasts does not help planning at this point. Phase 2 keeps `lam = 0` for H, and `lam`'s case rests on Blocks D and G. Report it before Phase 2. |
| The selected arm is non-inferior to forecast-only and to TF | The principled version keeps the gain. Proceed. |
| The selected arm is non-inferior to forecast-only but worse than TF | The remaining TF gain comes from its AR(1) posterior noise (`rho = 0.8`), which is outside the AS-JEPA objective as specified. Report it; we decide whether `rho` deserves a theory and a test. |
| The selected arm is worse than forecast-only | The noise or the pricing costs planning. Check the calibration slopes and B3. Do not adopt noiseless forecasts into AS-JEPA, because they are an unpriced channel. Report, and we decide with you before Phase 2. |
| C1 or C4 does not reduce drift (Section 5.6), but C5 does | Multi-step training of what the renderer reads is what matters for drift. Report it; it informs the choice between C4 and C5. |
| D3 or D4 keeps dot information | Check `forecast_share`, and check that the renderer really reads the forecast. If the implementation is correct, the mechanism does not operate, and `lam` loses its main theoretical support. Report it before Phase 2. |
| E2 differs from E1 by more than `COLOUR_TOL` | Appearance exclusion under `lam` is in doubt; report it. |
| E3 or E4 does not keep colour | The JEPA comparison for appearance does not hold on this domain; report it. |
| A1 and A2 plan as well as C1 and C2, with much higher rate | The price does not drive planning at this point. The story must say so; its role is purity and minimality, tested in D, E and G. |
| A4 (L1) plans as well as the selected arm at 3 epochs | Pricing adds nothing over SIGReg for planning at this budget; report it, and keep the comparison at 10 epochs in H. |
| F2 loses state information and F1 does not | The noiseless shortcut is real and priced forecasts close it: strong support for T1 over TF on TwoRoom. |
| F1 and F2 both equal their gap-8 counterparts | On PushT the first-frame shortcut does not matter. Phase 3 then focuses on the initialization scale (T3). |
| B2 or B4 passes the gate items where `sigma0 = 0.1` failed | Propose `sigma0 = 0.03` for the finals, with the evidence. |
| B2 and B4 fail as at `sigma0 = 0.1` | The `beta = 1` failure is not an entry-price effect. Report the entry and surprise split early in training; keep `sigma0 = 0.1`. |

## 10. Gate: when Phase 1 counts as working

Phase 2 needs our agreement after the Phase-1 report. Agreement requires the selected configuration (seeds 1 and 2) to meet all of the following:

- the round-1 generation gate: `h7` agent ratio of at most 0.5, and moving-block ratio below 1;
- action identification of at least 0.75;
- the collapse test that counts for it (Section 8, step 1) passed;
- P0 non-inferior to the original arm;
- at least `GALLERY_PASS_MIN` of the 32 fixed gallery cases of seed 1 marked as passing. We do this review; send us the gallery path.

Report everything else for understanding.

## 11. Tests and checks before any long run

1. **CPU contracts.** `tests/as_jepa/test_round2.py`, 11 tests. Run from the repository root:

   ```bash
   PYTHONPATH=.:runtime:<stable-worldmodel checkout> python -B -m unittest tests/as_jepa/test_round2.py -v
   ```

   They check:
   - `lam = 0` reproduces `ASJepaLeWM.loss` exactly, for every `horizon` and `forecast_grad`;
   - the main random stream is identical across `lam`;
   - `forecast_share` is about `lam`;
   - gradient routing: at `lam = 0` the flow loss never reaches the prior; under `joint` it reaches encoder and prior; under `encoder_only` it reaches the encoder but not the prior. This holds for both fixed and learned variances;
   - at `lam = 1` the last frame is not rendered from its own encoding;
   - the calibration slope equals 1 for an exact prior and `1/1.2` for a prior that overshoots by 20%;
   - the learned-variance KLs equal the fixed-variance ones at `sigma0` and torch's Normal KL in general, and the learned-variance loss at initialization equals the fixed-variance loss;
   - all three classes log the same core metrics, each convertible to a float as `train.py` does;
   - the contrastive variant has no renderer, its gradients reach the encoder and the prior, and its gradient audit lists no renderer.

   Also rerun your existing `tests/as_jepa` suite.
2. **Pairing check.** For C2 at seeds 1 and 2, the recorded initial weights (`record_initial_weights`) must equal those of `lower-beta-seed1` and `lower-beta-seed2`, tensor for tensor. Since every L2-family class builds identical tensors at a seed, this also covers the other arms. If they differ, stop and report: the pairing in Section 3.2 does not hold.
3. **GPU smoke (200 steps each, one GPU).** Run C1, C5, C1 with `encoder_only`, G4 (learned variances, on `D-std`) and I2. Record step time and peak memory. Check that:
   - the loss is finite;
   - every class logs `loss`, `rate`, `entry`, `surprise`, `feature_std`, `active_dims`, `forecast_share` (about `lam`) and `calib_slope`;
   - the renderer classes also log `fm` and `rate_share`, and `fm`, `entry` and `surprise` all move;
   - `LearnedVarianceLeWM` also logs `prior_var_ratio`, `posterior_sd` and `prior_sd`;
   - `NCELeWM` also logs `nce` and `info_bound`;
   - for the renderer classes, renders from noise change when the condition is replaced by zeros.

## 12. Failure responses

| Symptom | Likely cause | Response |
| --- | --- | --- |
| A `lam > 0` arm (fixed variance) has an eval-mode calibration slope below `SLOPE_MIN` at epochs 2 and 3, and, where a reference exists, more than 0.05 below it | joint training exaggerates the forecasts (Appendix A.5) | Rerun that arm, on every seed it has, with `forecast_grad = "encoder_only"`. Report both; the fallback runs replace the arm in selection. |
| G4 has `prior_var_ratio` above `VAR_RATIO_MAX` at epochs 2 and 3 | an overconfident learned prior; the toys predict this at `lam = 1` | Rerun G4 with `forecast_grad = "encoder_only"`, and report both. |
| A `lam > 0` arm fails the forecast collapse test | the price is above the collapse barrier at that `beta` | Exclude it; it marks the top of the useful range. No rescue runs. |
| Every `lam > 0` arm collapses | an implementation error, or `lam` lowers the barrier | Check `forecast_share`, the tests in Section 11 and the FP32 state spread. Then rerun C1 with `calibrate_state_std = sigma0`, once, and report. |
| The BF16 and FP32 collapse verdicts disagree | BF16 makes an empty state look full | Use the FP32 verdict for that arm, and re-evaluate its reference in FP32 too. |
| A screen is interrupted | `stop_after` does not act on resume | Resume with `train_stage --config <config> --stop-step 37473 --resume`. |
| NaN or loss spikes | numerics | Keep the rate in FP32 and the gradient clip. If spikes persist, halve the learning rate for that arm only, mark its comparisons as unpaired, and report it. |
| Good one-step forecasts but fast drift after a few steps | open-loop compounding | Report it; C5 (`rollout`) is the designed response. Do not add multi-step targets to the rate. |
| `D-shape` shape probe at chance in every arm | shape is not visible at the encoder's resolution, or it is never dynamics-relevant in the windows | Report the contact fraction and the pose-estimator floor. Do not tune. |

## 13. Latitude: what you may change without asking

- Compute allocation and the order of runs, within the priorities of Section 6.4.
- Evaluation sample counts, upward, applied to an arm and its references alike.
- Adding seeds to any arm, applied to its paired references too.
- Implementation mechanics not fixed here: launch wiring, report scripts, file layout of outputs.

Keep one GPU per run (as `train.py` does) and the reference runs' learning rates. Several GPUs per run would change the data order and the coins, and a different learning rate would unpair an arm from its references.

Everything in Sections 3, 4 and 8 needs our agreement before it changes: what the renderer reads, the noise, the pairing, the rate, the anchors and the selection rule. When unsure, stop and ask.

## 14. Order of work

1. Branch, copy the reference files and register them (Section 2.1). Run the CPU tests (Section 11.1) and the pairing check (11.2).
2. Add the evaluation options and the calibration tool (Sections 5.1 to 5.3). Run the GPU smokes (11.3). Report step times and projected GPU-hours.
3. Launch A3 and Block C, then D and E, then A4, F, A1 and A2, and B, as compute allows (Section 6.4). Start the `D-shape` collection on CPU.
4. Short gates at 2 epochs, and the full package, fresh planning, forecast checks, calibration and drift at 3 epochs (Section 5).
5. **Phase-1 report:** every metric of Section 5 per arm, the predictions of Section 7 marked held or failed, the selection of Section 8, and the decision of Section 9. Stop for our agreement.
6. Phase 2: H and H2, then G, then I. **PushT report**, then stop for our agreement.
7. Phase 3 only after agreement.

## Appendix A. The theory this round relies on

### A.1 The objective at `(beta, lam)`

For each rendered frame `t = 1 .. T`, a coin `c_t ~ Bernoulli(lam)` chooses the renderer's condition:

- the posterior sample `s_t = mu(x_t) + sigma0 xi_t`, when `c_t = 0`;
- a prior sample `s_hat_t = m_t + sigma0 xi'_t`, with `m_t = prior(s_<t, a_<t)`, when `c_t = 1`.

Training minimizes

```
J(beta, lam) = sum_t E[ -log p_renderer(x_t | condition_t, x_anchor) ]
             + beta * [ KL(q(s_0|x_0) || N(0, I)) + sum_t KL(q(s_t|x_t) || p_prior(s_t | s_<t, a_<t)) ]
```

At `beta = 1` this is the negative evidence lower bound of a generative model in which each frame is decoded from either its inferred state or the model's own forecast, chosen by a coin. So every `lam` is a code length, and `lam = 0` is round 1.

### A.2 What the objective rewards

With an optimal prior and renderer, and per nat:

```
J* = C + beta * Entry + sum_t [ beta * I(s_t; x_t | s_<t, a_<t)        cost: what the state adds beyond what the past predicts
                               - (1 - lam) * I(x_t; s_t | x_anchor)     reward: what the state says about its own frame beyond the anchor
                               - lam * I(x_t; s_hat_t | x_anchor) ]     reward: what the forecast says about the frame beyond the anchor
```

Here `C = sum_t H(x_t | x_anchor)` does not depend on the encoder.

### A.3 Consequences, block by block

| Information | Cost | Reward | Result |
| --- | --- | --- | --- |
| Detail fresh in each frame (the dot) | `beta` per nat | `1 - lam` per nat; no forecast can predict it | Excluded if and only if `beta >= 1 - lam`, so at every `beta > 0` when `lam = 1`. Proved for an anchor outside the encoded chain, as with `gap = 8`. The renderer's condition dropout `p = 0.1` shifts the threshold to `(1 - p)(1 - lam)`: 0.9 at `lam = 0`, 0.45 at `lam = 0.5`. |
| Persistent appearance (colour) | `beta` per nat added after the first state | 0 in both branches; the anchor supplies it | Excluded at every `beta > 0`. Proved for codes that keep appearance separate from dynamics; the general case at `lam > 0` is open. |
| Dynamics state (poses) | entry once, then surprise | both branches | Kept when it pays for its entry within the window (round-1 condition A5), at every `lam`. |
| Fixed property that the dynamics read and the anchor shows (block shape and size, when varied) | entry once | at `lam = 0`: 0, because the anchor shows it. At `lam = 1`: its value for forecasting contacts, because the prior cannot see the anchor | `lam = 0`, exact code (learned variances): not stored. The prior infers it after a contact, and the toy gain from storing it is 0.0000 nats at every `beta`. `lam = 0`, fixed-variance code: stored for a side reason, a toy gain of 11 to 15 nats, because a fixed-variance prior pays about 12.5 nats per mispredicted contact instead of about 1.6. `lam = 1`: stored, with toy gains of 2.96, 2.51, 2.06 and 1.26 nats at `beta` = 0.1, 0.3, 1 and 2. |

On the line `beta + lam = 1` the incentive to store detail unique to one frame is zero. Round 1's canonical point (`beta = 1, lam = 0`) and the sufficiency point (`lam = 1`, `beta` toward 0) are its two ends.

### A.4 What `lam = 1` optimizes

At `lam = 1` the objective selects the smallest state whose one-step forecast, made without seeing the anchor and rendered with the anchor, is most informative about the next frame. This is a sequential, anchor-conditional relative of the past-future information bottleneck (Creutzig, Globerson and Tishby 2009).

Under deterministic dynamics given the actions, as in PushT, maximizing it makes the prior's context sufficient for the future. Under stochastic dynamics one forecast sample guarantees less, and that weaker claim must be stated.

The sufficiency claim is about the pair (prior context, anchor), not about `s_t` alone:

- `s_t` is what one frame shows of the dynamics;
- the prior's context adds velocity and anything inferred from history;
- the anchor carries appearance.

### A.5 Known costs, and what watches them

- **Exaggerated forecasts under joint training.** The renderer values forecast information at `lam` per nat, while the prior pays for exaggeration only to second order in surprise. The toys give:
  - with fixed variances, an overshoot of 0.5 to 3.5%;
  - with learned variances, prior variances scaled by 0.78, 0.48 and 0.29 at `(beta, lam)` = (1, 1), (0.1, 1) and (0.01, 1), that is, variance ratios of about 1.3 to 3.4.

  Watched by the eval-mode calibration slope and the variance ratio; the fallback is the encoder-only forecast gradient.
- **The noiseless-mean shortcut.** A noiseless forecast mean is a channel of unbounded capacity that nothing prices, so the prior can pass information about actions to the renderer for free. With a first-frame anchor this admits an empty state plus a prior that integrates the actions. A `sigma0`-noisy sample limits the channel to the code's resolution, and the surprise term prices the mean's departure from the encoded state. A distant anchor also removes the start position that the shortcut needs. Block F tests both halves.
- **Out-of-distribution tests.** At `lam = 1` the renderer never sees encoded states, so tests that render encoded states are out of distribution; hence the forecast versions in Section 5.2.
- **No branch flag.** A renderer without a flag saying which branch it reads pays at most `ln 2` nats per frame when `0 < lam < 1`. That is negligible against a frame's 12,288 rendered values, so there is no flag.
- **Collapse barrier.** In the round-1 A5 toy, with a prior trained only by the code length, `lam` raises the collapse barrier slightly without changing which components are kept: from 0.34 to 0.40 nats with a first-frame anchor, and from 0.12 to 0.30 with a distant one. Under joint training at `lam = beta = 1` with a first-frame anchor, the barrier falls from 0.44 to 0.02.

## Appendix B. Design-to-plan trace

| Design point | Reason | Where | Checked by |
| --- | --- | --- | --- |
| Per-frame encoder | the state is a function of its frame; goal images encode without history | `ASJepaLeWM.encode`, unchanged | existing `test_contracts.py` |
| Anchor only to the renderer | appearance is free side information and must not be free in the state | unchanged | existing contracts |
| Actions only to the prior | the renderer must not do the dynamics | unchanged; forecasts carry actions only through the priced mean plus noise | `test_gradient_routing` |
| The renderer reads the posterior sample or a noisy prior sample, never a noiseless mean or `mu` | sufficiency (`lam`), and a priced channel | `ForecastScoredLeWM.loss`; `LearnedVarianceLeWM.loss` | `test_lambda_zero_reproduces_l2_loss`, `test_forecast_share`, `test_lambda_one_never_renders_a_frame_from_its_own_encoding` |
| Frame 0 never rendered | transient exclusion (Proposition 3) | unchanged | existing contracts |
| No stop-gradient by default | one objective, no heuristics | `forecast_grad = "joint"` | `test_gradient_routing`, `test_forecast_gradient_routing` |
| One exact rate, one `beta`, `sigma0` fixed in a run | the rate is a code length | unchanged | existing KL tests; `test_kl_functions` |
| `lam` constant | no schedules | constructor argument | Section 3.7 config |
| Paired random streams | clean comparisons across `lam` | `fork_rng` stream | `test_main_random_stream_is_paired_across_lambda`; pairing check 11.2 |
| Learned variances (Block G) | the exact code that the fixed-property prediction is about | `learned_variance.py` | `test_initial_loss_equals_fixed_variance`, `test_forecast_gradient_routing` |
| Contrastive critic (Block I) | a reconstruction-free control | `nce.py` | `test_no_renderer_and_gradients` |
| Comparable logging | one report format for every class | all three classes | `test_core_metrics` |

## Appendix C. Changed from the superseded plan, or not in this round

- **Narrowed.**
  - **Block F.** The old Block F swept anchor distance ({0, 8, the largest}) at the three highest prices that train, to see whether the surviving `beta` rises with the gap. With `lam`, purity no longer depends on reaching high prices, so that sweep lost its purpose. F now tests the first-frame shortcut that bears on TwoRoom.
  - **Block B** is now judged on gate items and P0, because the full-data `beta = 1` failure was not a collapse.
  - **P0 is now primary** (Section 5.1).
- **Not in this round.**
  - **Branch S (S2) and general video.** The S line returns when we move beyond PushT and the LeWM tasks.
  - **An explicit appearance encoder for the renderer.** It already exists implicitly as the renderer's anchor pathway; making it a separate module would be presentation only.
  - **Calibrating `beta` in nats** with an I-MMSE estimate from the renderer's anchor-only (condition-dropout) branch. This is our analysis and needs no training run.
  - **Story edits.** We make them after the Phase-1 report.
