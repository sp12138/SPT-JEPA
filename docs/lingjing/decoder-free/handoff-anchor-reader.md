# Handoff: the anchor reader (decoder-free AS-JEPA with a learned denoiser) and the anchor-gap diagnostic

Written 2026-11-16 on branch `lingjing/decoder-free-as-jepa`, for the collaborator who runs the experiments. It supersedes Phase 2 of [spec.md](spec.md); Phase 1 of that spec is closed by your `CLOSEOUT_20261103.md` (62 PushT recipes of the log-det objective, none qualified). Checked against your branch `as-jepa/lingjing-fullspec-20261102` at `3131efaa`.

> **Superseded in part (2026-11-18).** Parts P and R are replaced by [handoff-two-tracks.md](handoff-two-tracks.md), which adds AR(1) posterior noise, drops the initial-scale calibration and makes the reader's coordinates anchor-relative. Part G below stands.

Three parts, in this order. Parts G and P are independent and can run in parallel.

| Part | What | New training | Budget (estimate) |
| --- | --- | --- | --- |
| G | Anchor-gap diagnostic on existing checkpoints: PushT, Cube and your current RoboCasa Reach3D models | none | under 1 GPU-hour in total |
| P | The denoiser objective on PushT, in the LeWM harness, against your decoder-free and native LeWM results | 12 runs, 2 more if the fallback triggers | about 33 GPU-hours, 35 with the fallback |
| R | The same objective in your current Reach3D protocol | 2 runs of 3 hours on 4 GPUs | about 24 GPU-hours, only if Part P passes its gate |

Why now. The JEPA audience this line must convince rejects any pixel reconstruction loss, on the ground that reconstruction makes the state encode what could not be foreseen. The renderer line stays as it is. This handoff tests a version in which no loss touches pixels.

## 1. The method

### 1.1 Objective

Everything is L2 (encoder, noisy state `s_t = mu_t + sigma0 * xi_t`, prior with the identity carry, entry price, surprise) except the renderer, which is removed. In its place a small denoiser, the **anchor reader**, reads the noisy state and the anchor and predicts the injected noise `xi_t`.

```
reader:   minimise  | xi_hat(s_t, anchor) - xi_t |^2                       (states detached)
encoder:  minimise  beta * Entry + (beta + lam) * sum_t Surprise_t + sum_t  sg(score_t) . mu_t
          score_t = - xi_hat(s_t, anchor) / sigma0
```

- **The two parts work against each other.** The reader recovers the noise perfectly exactly when the anchor already determines the state, that is, when the state says nothing beyond the anchor. The encoder never receives the denoising loss. Its last term is the gradient of `-h(s_t | anchor)`: it moves each state away from the reader's best guess of it.
- **In words.** The state should be easy to predict for a reader who holds the past states and the actions (the prior, which the encoder helps), and hard to predict for a reader who holds only the anchor (the denoiser, which the encoder works against).
- **Same objective as before.** `h(s_t | anchor)` minus a constant is `I_A`, the information beyond the anchor that [story.md](story.md) derives from the renderer objective. Only the estimator changes: a learned score in place of a Gaussian log-det. All terms are in nats, so `beta` is calibrated, and the total weight on the prediction loss is `w = beta + lam`.
- **The detached score is not a target trick.** Prediction targets stay attached, as in L2. The score is detached because each of the two players takes the gradient of its own loss. There is no EMA, augmentation, decoder or schedule.
- **Sample-level.** Given the reader's parameters, every loss term is separable across windows. No batch covariance, log-det or negatives are used.

The construction follows EBReg (Kappiyath et al., arXiv 2611.11379), which maximises the entropy of the embedding marginal with a learned score. Three differences: the entropy here is conditional on the anchor, which is what excludes appearance; scale is set by the priced rate, not by normalisation; and the alignment term is the action-conditioned prediction loss, with no augmented views.

### 1.2 What the reader reads

- The noisy state, because that makes the reward an entropy. It grows like the log of the state's spread and so balances the quadratic rate. A predictor that saw only the anchor would give a reward quadratic in the scale.
- The anchor's own state mean, detached. The encoder is shared, so any appearance that leaks into `s_t` also sits in the anchor's state, where the reader finds it.
- The anchor image through the reader's own small CNN (`reader_anchor = "frame"`, canonical). This adds protection against codes that mix appearance with the changing state. `reader_anchor = "state"` drops the CNN and is the cheap comparison.

### 1.3 What we know and do not know

- **Synthetic check** (linear encoder, 8-dimensional state, 18 runs; job record in `../discussions/2026-11-15-latent-action-and-decoder-free-directions.md`, Section 6.2). The alternating training reproduces the closed-form Gaussian estimator. From a start at 0.001 `sigma0` the spread passed `sigma0` within 50 steps in 5 of 5 runs. Appearance probe R² was at most 0.002 in 15 of 15 runs with the anchor and at least 0.996 in 3 of 3 without it. The transient was dropped in 8 of 8 runs with `w >= 1`. With `w >= 1` the block was only partly kept on one of two seeds.
- **Not known.** The toy is linear and Gaussian, where the log-det estimator is already exact. It does not explain why all 62 log-det recipes failed on PushT. Two readings remain: the Gaussian estimator was the problem (it was precision-sensitive and can be inflated), or the objective itself does not select the task state on PushT. Part P is built to separate them.
- **Appearance depends on the reader's precision.** Appearance costs the state only its entry price, once per window, and earns nothing only if the reader cancels it. In a second vector toy (the one behind contract test 6), a reader given the anchor directly held appearance at 0.2 to 0.45 `sigma0` while the changing part reached 6 to 17 `sigma0`; a reader given only the anchor's state mean held it at 0.5 to 1.75 `sigma0`; a reader blind to the anchor let it grow to 5.7 `sigma0`. Standard PushT has one appearance, so it cannot show this. Arm P6 tests it on the colour data.
- **Contrastive control.** With the fixed-noise code, InfoNCE over frames of the same window, with the code's own Gaussian likelihood as critic, is the K-sample estimate of the same reward (`estimator = "nce"`). It is capped at `log(T + 1)` nats per frame. In the toy it kept the agent and dropped the block.

## 2. Named quantities

| Name | Value | Meaning or reason |
| --- | --- | --- |
| `BATCH`, `EPOCH_UPDATES`, `MAX_STEPS`, `SCREEN_STEPS` | 128, 12,491, 124,910, 37,473 | as in [spec.md](spec.md) Section 1.1 |
| `T_FRAMES` | 7 | rewarded frames per window, `W - 1` |
| `sigma0`, `W`, `gap`, `calibrate_state_std` | 0.1, 8, 8, 0.1 | unchanged from the decoder-free canonical configuration |
| `READER_WIDTH`, `READER_DEPTH` | 512, 3 | reader trunk; about 2 M parameters; not searched |
| `READER_DRAWS` | 4 | noise draws per state in the reader's loss; lowers its gradient variance at negligible cost |
| `READER_LR` | 5e-4 | 10 times the encoder's 5e-5. EBReg takes 16 reader steps per encoder step; your trainer has one optimizer, so the reader's rate is raised instead. Not searched |
| `READER_LOSS_WEIGHT` | 0.01 | both trainers clip the global gradient norm at 1.0; this keeps the reader's gradient out of that norm. Adam makes the reader's own update independent of it |
| `PLAN_SEEDS`, `FRESH_SEEDS`, `PAIRED_CI`, `NONINF` | as in [spec.md](spec.md) Section 1.3 | PushT planning cases and comparison rules |
| `TRAP_R2`, `BLOCK_LOST`, `SPREAD_MIN` | 0.5, 0.5, 0.3 | as in [spec.md](spec.md) Section 1.3 |
| `READER_DEAD` | 0.05 | `reader_mse` below this means the reader recovers the noise almost fully: the state carries almost nothing beyond the anchor |

`reader_mse` is the reader's squared error per dimension on the training posterior sample. It is 0 when the noise is fully recovered and approaches the share of state dimensions in use when the anchor cannot separate signal from noise. It is the first number to read in every run.

**The only searched knob is the prediction weight `w`,** through three preset points (Section 5.1). Nothing else is tuned.

## 3. Code

Copy these files from this branch unchanged. Do not reimplement them.

| File here | Destination | What it is |
| --- | --- | --- |
| `as_jepa_lewm/anchor_reader.py` | `as_jepa_lewm/` | `AnchorReader` (the denoiser), two plain functions for the encoder-side term and the reader's loss, and `AnchorReaderLeWM(DecoderFreeLeWM)` with `estimator in {"denoiser", "nce", "logdet", "logdet_window"}` |
| `tests/as_jepa/test_anchor_reader.py` | `tests/as_jepa/` | CPU contract tests |
| `experiments/anchor_reader_spt.py` | `experiments/` | adapter for `AnalyticResidualSPT`: `attach_anchor_reader` and `anchor_reader_spt_loss` (Part R) |
| `tests/as_jepa/test_anchor_reader_spt.py` | `tests/as_jepa/` | its CPU contract tests |
| `../round2/as_jepa_tools/gap_buckets.py` | `as_jepa_tools/` | the anchor-gap diagnostic (Part G) |
| `../round2/tests/as_jepa/test_gap_buckets.py` | `tests/as_jepa/` | its CPU contract tests |

Register the class in `as_jepa_lewm/factory.py`:

```python
    "anchor_reader": "as_jepa_lewm.anchor_reader.AnchorReaderLeWM",
```

`decoder_free.py` is unchanged and still byte-identical to your copy. With `estimator = "logdet"` or `"logdet_window"` the new class is bitwise equal to `DecoderFreeLeWM`.

Other places in your tree that select by class or by `model_class` and need the new name:

- `as_jepa_tools/evaluate.py::load_model` has an allowlist; an unknown target falls back to `ASJepaLeWM(**cfg)`, which fails on the reader arguments.
- `as_jepa_tools/planning.py` raises on an unknown `_target_`.
- `as_jepa_tools/df_state_check.py` asserts `model_class == "decoder_free"`. This handoff does not use it.
- Your queue workers that pick tools by `model_class` (`full_worker.py`, `hard_worker.py`), if you reuse them.

Two logged values change meaning for `estimator = "denoiser"`. `info`, and `information_nats` in `latent_eval.py`, are the pooled log-det computed as a monitor: the trained term has a gradient but no value. The logged `loss` contains that surrogate, so it is not comparable across estimators and need not fall. Judge runs by `reader_mse`, the spread, the probes and planning.

How it fits your trainer:

- **One optimizer, one backward pass.** The denoising loss is computed on detached states and added to the total, so it reaches only reader parameters. The encoder-side term uses the reader's output under `no_grad`, so it reaches no reader parameter. A contract test asserts both.
- **Parameter groups.** The class defines `training_optimizer_config`, which reproduces your default groups and schedule and adds one group for `reader.*` (`READER_LR`, weight decay 0, betas 0.9 and 0.95).
- **Checkpoints and DDP.** The reader is an ordinary submodule: it is in the state dict, in the optimizer state and synchronised across ranks.
- **Inference.** `encode`, `predict`, rollouts, probes and planning never touch the reader.

Config template (Part P, arm P2, seed 1). Renderer fields are kept only because the parent constructor reads them; no renderer is built.

```json
{
  "output": "/efs/prescottsun/SPT-JEPA/runs/anchor_reader_20261117/ar-P2-b0.3-l0-seed1",
  "data": "/efs/prescottsun/SPT-JEPA/data/pusht-lance-full-v1",
  "evaluation_data": "/efs/prescottsun/SPT-JEPA/data/pusht-lance-full-v1/phase0",
  "pose_weights": "/efs/prescottsun/SPT-JEPA/runs/insights12h_20261027/pose-v2/weights.pt",
  "seed": 1, "batch_size": 128, "workers": 4, "cpu_threads": 2,
  "max_steps": 124910, "max_seconds": 31536000, "checkpoint_every": 12491,
  "all_phase_data": true, "gap": 8, "calibrate_state_std": 0.1,
  "retain_steps": [12491, 24982, 37473, 124910],
  "model_class": "anchor_reader",
  "model": {"mode": "L2", "beta": 0.3, "sigma0": 0.1, "window": 8, "anchor_gap": 8,
            "renderer_width": 384, "renderer_depth": 12, "renderer_heads": 6,
            "lam": 0.0, "horizon": "one_step", "forecast_noise": false, "prior_input": "sample",
            "estimator": "denoiser", "conditioning": "anchor", "forecast_seed": 1,
            "reader_anchor": "frame", "reader_width": 512, "reader_depth": 3, "reader_draws": 4,
            "reader_lr": 5e-4, "reader_loss_weight": 0.01, "reader_seed": 1, "fp32_pool": true},
  "record_initial_weights": true
}
```

Launch as before: `python -B -u -m as_jepa_tools.train_stage --config <config> --stop-step 12491`, then `--stop-step 37473 --resume` for arms that pass the gate. `fp32_pool` is your float32 pooling option, which your precision audit motivated; the reader itself always runs in float32.

## 4. Part G: the anchor-gap diagnostic

**Question.** The renderer draws each frame from its state and an anchor frame, and the distance between the two varies. Does the model's behaviour depend on that distance? We change the design (a bounded gap, or a renderer told the gap) only if it does.

**Correction to our earlier notes.** On PushT the anchor is a frame at least 8 kept frames from the window, on either side, so the gap is large and random. On Cube, TwoRoom and RoboCasa the default anchor is the window's own first frame, so the gap is the frame's index in the window, 1 to 7. It is not the elapsed time in the episode.

**Tool.** `as_jepa_tools/gap_buckets.py`, read-only, fixed seeds, no weight changes. For every rendered frame it records:

| Quantity | Meaning |
| --- | --- |
| `gap` | distance in kept frames between the anchor and the frame |
| `render_loss` | the model's own distortion for that frame, with noise draws shared across all conditions |
| `state_reliance` | increase of that loss when the frame's state is swapped for another frame's (another window, and another time in the same episode) |
| `anchor_reliance` | increase of that loss when the anchor is zeroed |
| `surprise_nats`, `entry_nats` | the rate per frame |
| `anchor_pixel_mse` | pixel distance between anchor and frame: how stale the anchor is, without any model |
| probe error | where physical labels exist: ridge probe from the noisy state, fit on a disjoint half |

It reports each quantity per gap bucket with standard errors clustered by window, and its Spearman correlation with the gap, raw and after removing `anchor_pixel_mse`. Labels are passed with `--label`; the Reach3D loader exposes none, so that family has no probe rows.

**One limit.** With a first-frame anchor the gap is the frame's index, so a trend there cannot be told apart from a window-position trend (later frames have more context behind them, and a rollout has more steps). PushT separates the two, because its gap is drawn independently of the frame's index; the records keep `frame` for that comparison.

**Checkpoints, in this order.**

1. PushT temporal-forecast, 37,473 updates, seeds 1 and 2 (primary 94.00%): the working model, with large random gaps.
2. Cube temporal-forecast, 23,342 updates, both seeds: first-frame anchor, state in use, labels for the cube and the effector.
3. Your current Reach3D models: the encoder-heavy shared-mean reference at its latest full hour, and one of the EMA or action-transport branches at the same hour. For this family the tool calls the model's own `distortion` one frame at a time, so it follows whichever pixel objective the run uses, and it reports the posterior condition and the clean-forecast condition separately.

TwoRoom is skipped: its states are trapped near zero spread, so every reliance is zero by construction.

Commands: see Section 11.

**Reading the result.** Call a quantity gap-dependent when its lowest-gap and highest-gap buckets differ by more than 3 standard errors and by more than 10% of its mean, on both seeds. The 10% floor keeps small but significant trends from triggering work.

| Outcome | Reading | Action |
| --- | --- | --- |
| Nothing depends on the gap | the variable gap is not a problem at these ranges | none; close the question |
| `render_loss` and `state_reliance` both rise with the gap | the state does more of the work when the anchor is stale, as intended | none |
| `render_loss` rises, `state_reliance` does not, or probe error rises | the renderer degrades with distance and the state does not compensate | one paired arm with a bounded gap on that task (anchor at most `W` kept frames from the frame), same seed and budget |
| The dependence disappears once `anchor_pixel_mse` is removed | staleness in pixels explains it, not time | none |
| On Reach3D or Cube, a trend appears that PushT does not show at matching gaps | position in the window, not distance from the anchor | none for the anchor; report it |
| On Reach3D, `forecast_excess` rises with the index | the rollout, not the anchor, loses accuracy with horizon | report it with your existing 1, 3 and 7 step decode errors; no anchor change |

We do not ask for a renderer conditioned on the gap. With a first-frame anchor the gap is the frame's index, which the renderer can already infer from the forecast horizon; on PushT it would add an input that a planner never has.

## 5. Part P: PushT

### 5.1 Arms

Standard PushT (`D-std`), seeds 1 and 2, all else as the config template.

| Arm | `estimator` | `beta` | `lam` | `w` | Why |
| --- | --- | --- | --- | --- | --- |
| P1 | denoiser | 0.3 | 1 | 1.3 | the reference point of the old spec: a swap of the estimator only, against your `df-A-b0.3-l1` |
| P2 | denoiser | 0.3 | 0 | 0.3 | the reward dominates; against your `lam = 0` decoder-free arms (noisy agent R² 0.011 and 0.006) |
| P3 | denoiser | 1 | 0 | 1 | exact nats: the round-1 objective with `I_A` estimated by the reader |
| P4 | denoiser, `reader_anchor = "state"` | as the best of P1 to P3 at the gate | | | does the reader's own view of the anchor image matter |
| P5 | nce | 0.3 | 0 | 0.3 | the contrastive control |
| P6 | denoiser on the colour data (`model.domain = "color"`), `reader_anchor = "frame"` and `"state"`, seed 1 each | as the best of P1 to P3 at the gate | | | is appearance excluded, and does the reader need its own view of the anchor image for that |

P4 and P6 start after the gate of P1 to P3. "Best" is the arm with the highest mean noisy agent probe R² at the gate; ties go to the higher `w`. P6 runs to one epoch only and is read with `colour_invariance.py` and the noisy colour probe: appearance counts as dropped at a mean colour probe R² of at most 0.1 and as kept at 0.8 or more (`COLOUR_DROP`, `COLOUR_KEEP` of [spec.md](spec.md)).

### 5.2 Gate at one epoch (12,491 updates)

Run `latent_eval.py --fp32 --windows 64` (your existing tool) and read the last 500 training rows.

- **Alive:** validation `spread_over_sigma0 >= SPREAD_MIN` and, for the denoiser arms, `reader_mse >= READER_DEAD` (P5 has no reader).
- **Task content:** noisy agent probe R² at least `TRAP_R2`.

An arm continues to `SCREEN_STEPS` only if both seeds pass both conditions. Arms that fail are recorded as first-epoch failures, with their checkpoints kept.

### 5.3 At three epochs (37,473 updates)

The full package of [spec.md](spec.md) Section 4.2: planning on `PLAN_SEEDS` and `FRESH_SEEDS`, both probes for agent and block, latent rollouts at h1, h3, h7 and h13 against copy, the calibration slope, and the information spectrum. Add the mean of `reader_mse` over the last 500 rows.

References, all yours:

| Model | Budget | Primary / fresh |
| --- | --- | --- |
| Native LeWM (L0), 3 seeds | 3 epochs | 84.89 / 78.67 |
| Native LeWM (L0), 3 seeds | 10 epochs | 90.89 / 87.33 |
| Temporal-forecast (renderer), 2 seeds | 3 epochs | 94.00 / 92.33 |
| Decoder-free, window log-det, `lam = 0`, 2 seeds | 3 epochs | 4 / 2 |

L0 uses a window of 4 and the arms here a window of 8, so image computation per update differs; report updates and GPU-hours next to every number.

### 5.4 Decisions

1. **Estimator or objective.** If at least one of P1 to P3 passes the gate on both seeds, the Gaussian estimator was what failed, and the line continues. If none passes, run the fallback of Section 5.5 once. If that also fails, the objective in this form does not select the task state on PushT: stop the decoder-free line here and report. Do not tune further.
2. **Control.** At three epochs, the best passing arm is compared with L0 at three epochs. If it is non-inferior (`NONINF`) on primary and fresh, continue that arm's two seeds to `MAX_STEPS` and start Part R. If it passes the gate but trails L0 by more than `NONINF`, report the probes and rollouts and stop for our agreement before spending more.
3. **Appearance.** If P6 drops colour with the frame reader, the central claim of the method holds on real images. If it drops colour only with the frame reader and not with the state reader, the frame reader is required and stays canonical. If neither drops it, the reader is not precise enough at `READER_LR`: report the colour R² next to `reader_mse` and stop for our agreement; do not raise the reader's capacity or rate on your own.
4. **Block.** Report block probe R² per arm. The synthetic check predicts that the block suffers as `w` rises. If the block is below `BLOCK_LOST` in every passing arm, say so in the first line of the report: it is the known weakness of this objective, not a detail.

### 5.5 Fallback, only if no arm passes the gate

Read `reader_mse` and the spread at the gate to see which failure it is.

| Signature | Reading | One response |
| --- | --- | --- |
| `reader_mse < READER_DEAD`, spread falling | the prediction cost outweighs the reward from the start | F1: `beta = 0.1`, `lam = 0` (`w = 0.1`), 2 seeds, to the gate |
| `reader_mse` high, spread high, agent R² low | the state fills with content that is not the task state | F2: rerun the gate evaluation with probes for block pose and frame index, and the information spectrum, on the P2 checkpoints; no new training |
| loss not finite, or reader gradient absent | an implementation fault | fix it; this is not a result |

## 6. Part R: Reach3D, in your current protocol

Start only after decision 2 of Section 5.4 says to.

- **Model.** Your encoder-heavy configuration (E 384 wide, 12 layers; the same pool and transition) with no pixel decoder. `attach_anchor_reader(model, ...)` adds the reader; `anchor_reader_spt_loss` replaces the loss. It repeats `AnalyticResidualSPT.loss` through the rate terms, with the same random draws, drops the distortion, and returns

  ```
  loss = (beta_nats * (entry + sum_t surprise_t) + sum_t sg(score_t) . mu_t) / (W - 1) + READER_LOSS_WEIGHT * reader_loss
  ```

  The reader reads the noisy future codes and is conditioned on the detached initial code and its own CNN on the window's first frame. `sigma0`, `rho`, the data (the fixed 12,006 TRAIN logs), the optimizer settings for the body, and the frozen action heads (17,920 and 1,536 current-action labels, 64 DEV cases) are yours, unchanged. `beta_nats` is in true nats; it is not your per-rendered-value `beta`.
- **Three changes your Reach3D trainer needs** (`experiments/robocasa_stable_scaling/train.py`), since the adapter cannot make them from outside:
  1. It asserts `norms["renderer."] > 0` from step 2. Nothing trains a renderer here, so skip that assertion for these runs, and expect `reader.` in its place.
  2. `make_optimizer` would put `reader.*` in the 5e-5, weight-decay 1e-3 group. Give it its own group: `READER_LR`, weight decay 0, betas 0.9 and 0.95, same schedule factor.
  3. `joint_pixel_loss` and `pixel_likelihood.py` read `metrics["fm"]`, which this loss does not produce. Do not combine them with the adapter.

  The adapter supports separately encoded views, the Gaussian coding family and no ramp; it raises on joint-view encoders, the Laplace family and `ramp_steps`. Your encoder-heavy configuration satisfies all three.
- **Arms.** R1: `beta_nats = 0.3`. R2: `beta_nats = 1`. One run each on 4 GPUs, from random initial weights, evaluated at each full hour for 3 hours with both of your readouts (the frozen-encoder mean head and the 384-parameter code head).
- **References.** Your encoder-heavy shared-mean run at the same hours (81.83% and 42.23 mm at 1 h), and the zero-update head on the same initial weights.
- **Disclose.** Removing the decoder removes about 18.6 M of 41.6 M parameters and most of the per-update computation. Report parameters, updates and GPU-hours per arm; equal hours are not equal updates.
- **Decision.** Your reference moves by 1 to 3 points between adjacent hours, so 5 points is the smallest difference worth acting on. If either arm is within 5 points of the reference at 3 h, continue it to 6 h and report. If both are more than 10 points below at 3 h with `reader_mse >= READER_DEAD`, the state carries information that the action head cannot use; report the code-head and mean-head gap and stop. If `reader_mse < READER_DEAD`, treat it as the first row of the Section 5.5 table.

## 7. Predictions, to record before launch

| # | Prediction | Basis | Confidence |
| --- | --- | --- | --- |
| 1 | P2 passes the gate on both seeds | toy; reward dominates at `w = 0.3` | moderate: every log-det recipe failed here |
| 2 | Block probe R² orders as P2 >= P3 >= P1 | toy; barrier for components the actions do not determine | moderate |
| 3 | P5 keeps the agent and loses the block | toy; the reward caps at `log 8` nats | high |
| 4 | P4 matches its paired arm within 0.05 on both probes | appearance is constant on standard PushT | high |
| 5 | P6: colour dropped with the frame reader; with the state reader colour R² is higher than with the frame reader | second toy: appearance leaks more when the reader sees only the anchor's state | moderate |
| 6 | Part G: no dependence on Reach3D and Cube beyond what `anchor_pixel_mse` and the frame index explain | gaps are 1 to 7 frames | moderate |

A failed prediction is a result. Report it as such.

## 8. Reports

One report per part, in your usual place, each opening with the decision it supports.

- **Part G:** the bucket tables and the two correlations per checkpoint, and which row of the Section 4 table applies.
- **Part P gate:** per arm and seed: spread, `reader_mse`, noisy agent and block R², pass or fail.
- **Part P at three epochs:** the table of Section 5.3 with the references, the decision of Section 5.4, GPU-hours and seconds per update.
- **Part R:** your hourly table for both readouts, with parameters, updates and GPU-hours.

## 9. Before any long run

1. The contract tests pass in your environment (Section 11).
2. One two-update start per `estimator`, with your usual restore check, confirming that `reader.*` has its own parameter group and a finite non-zero gradient, and that encoder and pool gradients are finite and non-zero.
3. Measure seconds per update for P2 and correct the budget in your report.

Nothing else is required. Do not add compatibility sweeps for this handoff.

## 10. What you may change without asking

- Machine and GPU allocation, launch order within a part, checkpoint cadence.
- `workers`, `cpu_threads` and other throughput settings that leave the numbers unchanged.
- Skipping Part G checkpoints that no longer exist; say which.

Anything that changes an objective, a named quantity, an arm or a decision rule needs our agreement first.

## 11. Commands and verification record

### 11.1 Part G commands

Copy the two files, then run from the repository root. Bracketed parts are yours to fill; the paths are read off your documents and were not opened from our side.

```bash
# (a) PushT temporal-forecast, seeds 1 and 2 (anchor at least 8 kept frames away, either side)
python -m as_jepa_tools.gap_buckets --dataset pusht_full \
  --checkpoint /efs/prescottsun/SPT-JEPA/runs/promising_20261029/temporal-forecast-seed1-selected-epoch3/<folder with weights.pt and config.json> \
  --data /efs/prescottsun/SPT-JEPA/data/pusht-lance-full-v1 \
  --cases 256 --label agent:0-2 --label block:2-4 --output <out>/gap-pusht-tf-seed1

# (b) Cube temporal-forecast, both seeds (first-frame anchor)
python -m as_jepa_tools.gap_buckets --dataset benchmark \
  --checkpoint /efs/prescottsun/SPT-JEPA/runs/cube_maturity23342_20261031/cube-temporal-forecast-seed1/<checkpoint folder> \
  --data /efs/prescottsun/SPT-JEPA/data/hardtasks-v1/cube-metadata \
  --cases 256 --label cube:privileged_block_0_pos --label effector:proprio_effector_pos \
  --output <out>/gap-cube-tf-seed1

# (c) a current Reach3D checkpoint
python -m as_jepa_tools.gap_buckets --dataset robocasa \
  --checkpoint <root>/training/<run_name>/<label> --data <root>/data.json --split validation \
  --cases 128 --output <out>/gap-reach3d-<run_name>
```

- Each run writes `gap_buckets.json` and prints the bucket table. Add `--fp32` when a reliance value is below about 1e-4 of the loss.
- The Cube label keys are the ones `benchmark_spatial_probe.py` reads; confirm they exist in the Cube HDF5.
- `--dataset robocasa` loads through your own `build` and `Windows`. Its checkpoint layout (`<root>/training/<run>/<label>/weights.pt`, `<run>/config.json`) is read off `campaign.py` and `diagnostics.py`. That loader is the one part we could not exercise on a real campaign; if the layout differs, adapt `load_robocasa` only.
- Labels are read only by the probes. They never reach a model.

### 11.2 Tests

From your repository root, after copying the files:

```bash
PYTHONPATH=.:runtime:<stable-worldmodel checkout> python -B -m unittest \
  tests/as_jepa/test_anchor_reader.py tests/as_jepa/test_anchor_reader_spt.py tests/as_jepa/test_decoder_free.py \
  tests/as_jepa/test_gap_buckets.py -v
```

What we ran, on 2026-11-16, on CPU (Python 3.12, torch 2.7.0), against an export of your branch at `3131efaa` with stable-worldmodel's LeWM modules:

| Tests | Result |
| --- | --- |
| `test_anchor_reader.py` (16), `test_anchor_reader_spt.py` (6), `test_decoder_free.py` (14) | 36 of 36 pass |
| `test_gap_buckets.py` (20), with the three round-2 test files (24) | 44 of 44 pass |

What the tests establish:

- **Parity.** With the log-det estimators the new class equals `DecoderFreeLeWM` bitwise. With the new estimators, `entry`, `surprise` and `forecast` equal the parent's bitwise at a seed, for all five forecast variants. The adapter's rate terms equal `AnalyticResidualSPT.loss` bitwise.
- **Gradient routing.** The denoising loss sends exact zeros to encoder, pool, predictor and action encoder; the other terms send exact zeros to the reader.
- **Direction.** With an analytic Gaussian reader, the encoder-side gradient matches the gradient of the closed-form conditional entropy and increases the spread. With states the anchor determines, the expected gradient is zero and `reader_mse` is 0.
- **Mechanism on a vector problem.** From 0.01 `sigma0`, the part the anchor does not determine grows past 3 `sigma0`, appearance stays below 0.6 `sigma0`, and with a reader blind to the anchor appearance grows past 3 `sigma0`.
- **Separability.** Changing another window leaves a window's terms bitwise unchanged, for the denoiser and for `nce`.
- **Diagnostic tool.** Per-frame values average to the model's own loss terms for the LeWM classes and for four Reach3D variants built with your patch functions; the tool leaves weights and random generators unchanged; planted gap trends are recovered.

What they do not establish:

- Nothing ran on a GPU, on real data or on a real checkpoint. Bf16 autocast on CUDA was only approximated on CPU.
- Lightning is absent from our test Python, so the optimizer test compares against a copy of your default path, not against `TrainModule.configure_optimizers` itself. Step 2 of Section 9 covers this.
- Raw-pixel, shared-visual-anchor, stereo and EMA variants of the Reach3D family were read but not run through the diagnostic.
- Under multi-GPU training every rank draws the same reader noise at a step (the convention of the forecast stream). The noise is independent of the data, so this correlates ranks slightly and biases nothing.
- Training behaviour on images is what Parts P and R measure. The tests check contracts on tiny models.
