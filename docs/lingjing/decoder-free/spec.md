# Decoder-free AS-JEPA: experiment and implementation spec (handoff)

Written 2026-10-31 on branch `lingjing/decoder-free-as-jepa` for the collaborator who runs the L-line experiments. Follow it as written. Section 12 lists what you may change without asking; anything else needs our agreement first.

This is a separate, exploratory track next to the round-2 spec.

- **Self-contained.** Every constant and tool definition this spec needs is copied below.
- **Pinned to revision 2.** Where it refers to the round-2 spec, it means revision 2 as committed in `8d19e26` ([../as-jepa-round2-spec.md](../as-jepa-round2-spec.md) on this branch). Later revisions of the round-2 spec do not change this spec.
- **Theory** is in [story.md](story.md). Each component below names the design reason it serves.
- **Reference implementation.** [as_jepa_lewm/decoder_free.py](as_jepa_lewm/decoder_free.py) with 14 CPU contract tests in [tests/](tests/), all passing against your head `051e82d`. Copy it; do not reimplement it.

## 0. Overview

### 0.1 The model

AS-JEPA without a renderer. Encoder, prior, noise, rate and identity carry are L2's. The renderer's flow loss is replaced by the state's information beyond the anchor, computed in state space:

```
J_DF = beta * Entry + beta * sum_t Surprise_t + lam * sum_t Forecast_t - T * I_A          (all in nats)
I_A  = 0.5 * logdet(I + Sigma_A / sigma0^2),   Sigma_A = 0.5 * E[(mu_t - mu_A)(mu_t - mu_A)^T]  (pooled over the batch)
```

- `mu_A` is the anchor frame encoded by the same encoder. It enters nothing else.
- With the canonical one-step mean forecast, `Forecast_t = Surprise_t`, so the prediction loss has weight `beta + lam`.
- There is no decoder, EMA, stop-gradient, augmentation or schedule.
- Planning is unchanged: CEM on encoder means and prior rollouts.

Why (story Sections 1 and 7): with an optimal renderer, the round-1 objective depends on the encoder only through `I_A`, the encoder's other entropies, and a constant. So at `lam = 0` this objective is the round-1 objective up to a constant, with the renderer replaced by a state-space estimate. Everything is in nats, so `beta` is calibrated, and exclusion of unpredictable detail becomes a sharp prediction: iff `beta + kappa * lam >= 1` (story Section 5.4).

The known costs, which the plan measures:

- **The Gaussian estimate of `I_A` can be gamed** by codes that are not Gaussian, or whose distribution differs across episodes (story Section 3.4).
- **Components the actions do not reveal can be trapped** when the prediction weight is high (story Section 5.6).

### 0.2 What the factors mean here

| Factor | Values | What it changes |
| --- | --- | --- |
| `beta` | 0.1, 0.3, 1 | price per nat of state (entry and one-step surprise) |
| `lam` | 0, 0.5, 1 | weight of the forecast-information reward |
| forecast `horizon` | `one_step` (teacher-forced), `rollout` (open loop from the encoded first state) | which forecast's information is rewarded; the rate always uses one step |
| `forecast_noise` | false (prior mean), true (a `sigma0` sample) | a sample halves the effective weight (`kappa = 1/2`) with fixed-variance decoders |
| `prior_input` | `sample` (canonical), `mean` | whether the prior reads the transmitted noisy states or their means (LeWM-like) |
| `estimator` | `logdet` (pooled, canonical), `logdet_window` (per window) | the per-window estimate cannot be raised by episode-level keys, but under-credits components that barely change within a window |
| `conditioning` | `anchor` (canonical), `batch` | spread measured relative to the anchor's state, or across the batch |
| anchor `gap` | 8 (canonical, two-sided on PushT as in the references), 0 | how far the reference for change is |
| `calibrate_state_std` | 0.1 = `sigma0` (canonical), absent, 0.3 | the state's initial spread |
| `sigma0` | 0.1; 0.3 only in Block H | only through the entry price `pi = beta * sigma0^2` (story Section 3.3) |

In noise units the one-step objective has two effective numbers: the entry price `pi = beta * sigma0^2` and the prediction weight `w = beta + kappa * lam`. Block A spans that plane; the other blocks change one structural factor at a time.

### 0.3 What this round answers

| Question | Block |
| --- | --- |
| Q1. How do price and prediction weight shape planning, latent rollouts and the state's content? Where does purity (`w >= 1`) start to cost completeness for the block? | A |
| Q2. How do open loop versus one step and noisy versus noiseless forecasts interact with `lam`? | B |
| Q3. Does letting the prior read noiseless means change content or planning? | C |
| Q4. Does the per-window estimator behave differently from the pooled one? Does the anchor reference matter on standard data? | D |
| Q5. Is the dot excluded exactly when `beta + kappa * lam >= 1`? | E |
| Q6. Is colour excluded by the anchor reference, does LeWM keep it, and is the pooled estimator gamed where an episode-level key exists? | F |
| Q7. Does the initial scale trap components, and does a near reference lose the block? | G |
| Q8. Are runs with equal entry price and prediction weight equivalent? | H |
| Q9. Does the decoder-free objective escape the TwoRoom trap, compared with the calibrated renderer arm? | T |

Stopping point: after Phase 1, write the Phase-1 report and wait for our agreement. Phase 2 (Section 5.3) needs it.

### 0.4 Relation to the round-2 runs

The round-2 runs are references only; this spec reruns none of them:

- revision 2's P1 (L0 on full PushT) and P2 (renderer `lam = 1`);
- its T4 (renderer `lam = 1`, calibrated, past anchor at `gap_T`) on TwoRoom;
- your first-round PushT arms and your native LeWM results.

Where this spec needs a reference that the round-2 plan may not run, it runs its own: Block F's L0 on the colour domain (F4). If GPUs are short, the round-2 plan's runs take precedence, except this spec's first priority tier (Section 5.2).

## 1. Named quantities

### 1.1 Derived and budget

| Name | Definition | Value |
| --- | --- | --- |
| `BATCH` | windows per optimizer step | 128 |
| `EPOCH_UPDATES` | `floor(train windows / BATCH)` on `pusht-lance-full-v1` at `W = 8` (1,598,974 windows) | 12,491 |
| `MAX_STEPS` | `10 * EPOCH_UPDATES`, the learning-rate horizon of every PushT run | 124,910 |
| `SCREEN_STEPS` | `3 * EPOCH_UPDATES`, where Phase-1 PushT runs stop | 37,473 |
| `EPOCH2_STEPS` | `2 * EPOCH_UPDATES` | 24,982 |
| `L0_EPOCH_UPDATES` | the same at LeWM's window of 4 (your `full-L0-10epoch-*` configs) | 15,265 |
| `TWOROOM_GATE`, `TWOROOM_FULL` | TwoRoom updates at the state check and at the end | 2,000 and 6,000 |
| `STEP_S_DF` | seconds per update, decoder-free, one H200; measure in the smoke | budget 0.3 (revision 2's estimate for the renderer-free `NCELeWM`) |
| `SCREEN_GPU_H_DF` | `SCREEN_STEPS * STEP_S_DF / 3600` | about 3.1 |
| `T_FRAMES` | rewarded frames per window, `W - 1` | 7 |
| `ACTIVE_NATS` | a direction or dimension counts as informative above this many nats | 0.1 |

### 1.2 Canonical configuration and reference point

- **Canonical:** `lam = 1`, `horizon = one_step`, `forecast_noise = false`, `prior_input = sample`, `estimator = logdet`, `conditioning = anchor`, `gap = 8`, `calibrate_state_std = 0.1`, `sigma0 = 0.1`, `W = 8`.
- **The only searched knob is `beta`** (Block A).
- **The reference point of the one-factor blocks is `beta = 0.3`, `lam = 1`, so `w = 1.3`.** It is above the transient threshold. In the synthetic toy it kept the agent and the push gain `K` from every initial scale tried, and the block only partly. Block A decides whether a lower `w` serves better.

### 1.3 Evaluation and decision constants

| Name | Value | Meaning |
| --- | --- | --- |
| `PLAN_SEEDS`, `FRESH_SEEDS` | 42, 43, 44 and 1042, 1043, 1044 | PushT primary and fresh planning cases, 3 × 50 episodes each, with `planning.py` |
| `PAIRED_CI` | 95% episode-clustered bootstrap of paired P0 differences, pooled over training seeds 1 and 2 (`report_main_conclusions.cluster_ci`) | episode uncertainty only; it does not include training-seed variation |
| `NONINF` | 5 points | A is non-inferior to B if the lower end of `PAIRED_CI(A - B)` is above -5 and each seed's mean difference is above -5 |
| better, worse | | A is better than B if `PAIRED_CI(A - B)` lies above 0 and both seeds' differences are positive; worse if below 0 and both negative; otherwise tied |
| `SLOPE_MIN` | 0.9 | lowest acceptable eval-mode calibration slope of the prior (Section 4.2) |
| `CHANCE_R2` | 0.05 | a probe R² at or below this is chance; for the dot, decide on the largest of the five dot R² values |
| `TRAP_R2` | 0.5 | an arm is trapped if its agent probe R² (noisy state) is below this, or fewer than 2 directions carry `ACTIVE_NATS` |
| `BLOCK_LOST` | 0.5 | the block is lost if the mean block-position probe R² from the noisy state is below this |
| `COLOUR_KEEP`, `COLOUR_DROP` | 0.8, 0.1 | mean colour probe R² (6 gains and biases, noisy state) that counts as kept or dropped |
| `SPREAD_MIN` | 0.3 | lowest validation `feature_std / sigma0` for a usable state (revision 2) |
| `HARD_TIE` | 6 points | TwoRoom success differences below this, averaged over 2 seeds and both case sets, are ties (revision 2) |
| `CASES`, `WHITE_CEM`, `CORR_CEM` | revision 2, Section 1.2 | TwoRoom case sets and controllers: your 50-case primary and fresh sets; CEM 300 × 30, top 30, horizon 5, receding 5, history 3, budget 50, GoalMSE on `mu`; and the same in your fixed training-action covariance |
| `GAME_GAP` | 2 | gaming flag: the pooled `I_A` estimate is more than twice the per-window estimate, and the ratio grew between epochs 1 and 3 |
| `GAME_PR` | 2 | gaming flag: the participation ratio of `Sigma_A` exceeds twice the number of probe targets with noisy-state R² above 0.5 |

Equivalence (Blocks B and H) is judged on metrics that do not depend on `sigma0`, each with its own tolerance:

- P0: non-inferior both ways;
- agent and block probe R² (noisy): within 0.05;
- latent h7 ratios: within 0.05;
- `I_A` per frame and surprise per step: within 10%;
- `info_dirs`: within 1;
- `spread_over_sigma0`: within 10%.

## 2. Starting point

### 2.1 Code

1. **Branch.** Create branch `as-jepa/decoder-free` from your current head (`051e82d` or later). Record the hash in every report.
2. **Copy, unchanged:**
   - `docs/lingjing/decoder-free/as_jepa_lewm/decoder_free.py` into `as_jepa_lewm/`;
   - `docs/lingjing/decoder-free/tests/as_jepa/test_decoder_free.py` into `tests/as_jepa/`;
   - for Block T: `docs/lingjing/round2/as_jepa_tools/benchmark_gap.py` and `docs/lingjing/round2/tests/as_jepa/test_benchmark_gap.py` from this branch (revision 2), with revision 2's dataset routing (its Section 2.1, item 4). Without it, `BenchmarkWindows` returns window frame 0 as `anchor224`, and TwoRoom would silently train against a near reference.
3. **Register `DecoderFreeLeWM`:**
   - `factory.MODEL_TARGETS["decoder_free"] = "as_jepa_lewm.decoder_free.DecoderFreeLeWM"`;
   - one `elif` in `evaluate.load_model`, before the final `else`; without it, loading fails on the new arguments;
   - the target in `planning.py`'s allowed set.

   `benchmark_evaluate.py` instantiates `_target_` through `load_pretrained`; confirm it in the smoke. Tools that render do not apply to this class, which raises a clear error if asked to render: `completion_evaluate`, `collapse_audit`, `quality_complete`, `counterfactual`'s pose path, `color_swaps`, `benchmark_health`, and `benchmark_information_audit.main`.
4. **Write three tools** (Section 4): `as_jepa_tools/latent_eval.py`, `as_jepa_tools/df_state_check.py` and `as_jepa_tools/colour_invariance.py`.
5. **Do not edit** `model.py`, `robust.py`, `research.py` or any frozen source.

### 2.2 Data

- **`D-std`:** `/efs/prescottsun/SPT-JEPA/data/pusht-lance-full-v1` with `all_phase_data: true`, evaluation on its `phase0/` view, planning on `pusht_expert_train.h5` as `planning.py` sets it up. The anchor is the two-sided gap-8 rule of the references.
- **`D-dot` and `D-colour`:** `D-std` with `model.domain = "dot"` or `"color"`. The decoder-free class applies the domain transform to the anchor exactly as L2 does to the window; this is tested.
- **TwoRoom:** `/efs/prescottsun/SPT-JEPA/data/hardtasks-v1/tworoom-metadata`, `dataset_kind: benchmark_hdf5`, `action_dim: 10`. The past-only anchor is at `gap_T`: the largest gap in {1, 2, 4, 8, 16} whose share of windows without an eligible past anchor is at most 0.2, computed by `python -B -m as_jepa_tools.benchmark_gap --data <tworoom-metadata> --gaps 0 1 2 4 8 16 --label agent:pos_agent:position:<AGENT_TOL> --output tworoom-anchor-table.json`. If revision 2 has already run this, reuse its `gap_T`.

### 2.3 References

- **PushT P0 at 3 epochs, seeds 1 / 2:** original (L2, `beta = 1/4`) 88 / 74.67; lower-beta (L2, `beta = 1/16`) 86 / 84; forecast-only 90 / 90; TF 94 / 94.
- **Revision 2, when they exist:** P1 (L0 at 3 and 10 epochs) and P2 (renderer `lam = 1`).
- **Pairing.** At a given seed, `DecoderFreeLeWM` builds the same encoder, pool, prior and action-encoder tensors as L2. The renderer is built last and then dropped; this is tested. The data order is the seed's. Calibration rescales the pool head once, so the canonical arms are paired with the references in data order, and in initial tensors up to that rescaling.
- **TwoRoom:**
  - at 2,000 updates, seeds 1 / 2: your AS-JEPA arms 8 / 2, 2 / 6, 2 / 6, all uncalibrated with a first-frame anchor, and all trapped (spread about 1.5e-4);
  - revision 2's T4 (calibrated renderer `lam = 1`, past anchor), when it exists;
  - at 6,000 updates: native LeWM 80 / 72 and your fixed carrier 90 / 94.

## 3. Implementation

### 3.1 `DecoderFreeLeWM` (`as_jepa_lewm/decoder_free.py`)

It subclasses `ASJepaLeWM`, adds the constructor arguments of Section 0.2, and drops the renderer after construction. Its loss per window:

1. **Frames.** `frames(batch)` stacks the window frames and the anchor (`anchor224`), applies the domain transform, and normalizes. One encoder pass gives `mu` for window frames 0 to 7 and `mu_A` for the anchor.
2. **Prior.** Posterior samples are `s = mu + sigma0 * xi`. The prior reads `s` (`prior_input = sample`) or `mu` (`mean`), and gives one-step means `m_t` with the identity carry.
3. **Rate.** `Entry = KL(N(mu_0, sigma0^2) || N(0, I))` and `Surprise_t = ||mu_t - m_t||^2 / (2 sigma0^2)`, as in L2.
4. **`Forecast_t`**, the decoder cross-entropy of the target state minus a constant:

   | `horizon`, `forecast_noise` | `Forecast_t` |
   | --- | --- |
   | `one_step`, false | `Surprise_t` (decoder `N(m_t, sigma0^2)`) |
   | `one_step`, true | `||s_t - (m_t + sigma0 xi')||^2 / (4 sigma0^2)` (decoder variance `2 sigma0^2`) |
   | `rollout`, false | `||mu_t - h_t||^2 / (2 sigma0^2)`, with `h` the open-loop rollout of prior means from `s_0` (or `mu_0`) |
   | `rollout`, true | `||s_t - h_t||^2 / (4 sigma0^2)`, with `h` the open-loop rollout of prior samples |

   Forecast noise and extra predictor passes run in a separate random stream, so the main stream is paired across variants (tested).
5. **`I_A`**, in float64, over frames 1 to 7:
   - **`logdet` (pooled):** `0.5 * logdet(I + Sigma / sigma0^2)`, with `Sigma` the outer products of `mu_t - mu_A` over the batch, divided by `2N`.
   - **`logdet_window`:** the same per window, from its 7 changes (a 7 × 7 determinant by Sylvester's identity), averaged.
   - **`batch` conditioning:** the batch covariance of `mu_t`.
6. **Loss.** `beta * mean(Entry) + beta * mean(sum_t Surprise_t) + lam * mean(sum_t Forecast_t) - T_FRAMES * I_A`.

Logged every 100 steps:

- **Content:** `info` (the trained estimate, nats per frame), `entry`, `surprise`, `forecast`, `rate` (nats per window), `feature_std`, `spread_over_sigma0`, `active_dims`, `calib_slope`.
- **Gaming diagnostics on the anchor-relative changes:** `info_pooled`, `info_window`, `info_pairwise` (a trend diagnostic, biased upward on minibatches), `participation` (the participation ratio of `Sigma_A`) and `info_dirs`.

### 3.2 Config template and launch

Example: Block A, `beta = 0.3`, `lam = 1`, seed 1. Other arms change only the fields listed in Section 5. The renderer fields are the references': the renderer is built last and dropped, so they affect nothing but construction time.

```json
{
  "output": "/efs/prescottsun/SPT-JEPA/runs/decoder_free_20261101/df-A-b0.3-l1-seed1",
  "data": "/efs/prescottsun/SPT-JEPA/data/pusht-lance-full-v1",
  "evaluation_data": "/efs/prescottsun/SPT-JEPA/data/pusht-lance-full-v1/phase0",
  "pose_weights": "/efs/prescottsun/SPT-JEPA/runs/insights12h_20261027/pose-v2/weights.pt",
  "seed": 1, "batch_size": 128, "workers": 4, "cpu_threads": 2,
  "max_steps": 124910, "max_seconds": 31536000, "checkpoint_every": 12491,
  "all_phase_data": true, "gap": 8, "calibrate_state_std": 0.1,
  "retain_steps": [12491, 24982, 37473, 99928, 124910],
  "model_class": "decoder_free",
  "model": {"mode": "L2", "beta": 0.3, "sigma0": 0.1, "window": 8, "anchor_gap": 8,
            "renderer_width": 384, "renderer_depth": 12, "renderer_heads": 6,
            "lam": 1.0, "horizon": "one_step", "forecast_noise": false, "prior_input": "sample",
            "estimator": "logdet", "conditioning": "anchor", "forecast_seed": 1},
  "record_initial_weights": true
}
```

- **Launch** with `python -B -u -m as_jepa_tools.train_stage --config <config> --stop-step 37473`, and `--resume` after an interruption. `stop_after` acts only on a fresh start.
- **Keep `gap` equal to `model.anchor_gap`.**
- **TwoRoom configs:** your TwoRoom recipe (`dataset_kind: benchmark_hdf5`, `action_dim: 10`, `max_steps: 124910`), with:
  - `model_class: decoder_free`, `calibrate_state_std: 0.1` unless stated, and the past anchor at `gap_T`;
  - `--stop-step 2000`, then `--stop-step 6000 --resume` for arms that pass the state check.

### 3.3 Design invariants

1. The encoder reads one frame.
2. The prior never reads frames or the anchor.
3. Actions reach only the prior.
4. The anchor enters only `I_A`.
5. Window frame 0 starts the chain and is never rewarded: `I_A` and `Forecast` cover frames 1 to 7.
6. No EMA, stop-gradient, augmentation, masking or schedule; the class rejects `ramp_steps`.
7. Nothing normalizes `mu`.
8. `beta`, `lam` and `sigma0` are constant within a run.
9. The rate is always the one-step code length of the noisy states. Forecast variants change only the reward.

### 3.4 What does not change, and one numerical note

The optimizer, learning rates, warmup and cosine schedule, batch, bf16 with float64 `I_A`, and gradient clip 1.0 are the references'.

The loss is in nats per window, so pre-clip gradient norms are large, and the clip binds on most steps. Adam's per-parameter normalization makes the step size about the learning rate either way, so we keep the references' settings. Log the pre-clip gradient norm.

## 4. Evaluation

Nothing renders, so PushT generation is measured in state space through linear probes. Pixel-based metrics wait for the Phase-2 post-hoc decoder.

### 4.1 Reused tools

- **Planning.** `planning.py`, unchanged.
  - At 2 epochs: `PLAN_SEEDS` 42 and 43 (100 episodes), as for the references.
  - At 3 epochs: `PLAN_SEEDS` and `FRESH_SEEDS`.
- **Probes.** `probes.py --fp32`, both mean and `--noisy`.
  - Every decision on the dot, colour, block or agent uses the `--noisy` probe: the mean probe standardizes features, so a fading leftover signal reads as kept.
  - The probe targets are agent, block, angle, velocity, and the dot or colour nuisance targets.
- **TwoRoom planning.** `benchmark_evaluate.py` under `WHITE_CEM` and `CORR_CEM` on the primary and fresh cases.

### 4.2 New tool: `as_jepa_tools/latent_eval.py`

Input: a checkpoint, the evaluation data (`phase0`), `--fp32`, and `--windows` (64 for the short gate, 256 otherwise). The model runs in eval mode. Output: one JSON file.

1. **Two ridge probes** (`RidgeCV`, alphas `logspace(-4, 4, 9)`), fit on 20,000 training frames. Targets: agent x and y, block x and y, and the sine and cosine of the block angle, in the pose estimator's units.
   - The **readout probe** is fit on clean `mu`. It decodes predicted states and the copy baseline, since prior means are clean.
   - The **information probe** is fit on noisy `mu + sigma0 * xi`. It reports the information at the code's resolution.

   Report both on 5,000 test frames; the readout probe's error is the floor.
2. **Latent rollouts** on the `quality_complete` test windows (selection seed `20261029`, half with a moving block):
   - Use `evaluate.predict_states` with the same context (3 frames), horizon (13) and history cap as `quality_complete`.
   - Decode each predicted state with the readout probe, and compare it with the true simulator state.
   - The copy baseline is the readout of the last context frame, held constant.
   - Report agent and block errors at h1, h3, h7 and h13, absolute and relative to copy, plus `h7_moving_block_ratio`.
3. **Latent action identification**, `D-std` only, because `counterfactual.py` supports only the standard domain. Use `counterfactual.py`'s cases and true futures: 64 cases, or 16 for the short gate, with `--selection-seed 20261029`. Roll the model out under each of the 4 action sequences, and decode with the readout probe. Assign each rollout to the true future with the smallest summed agent and block error. Report accuracy (chance 0.25).
4. **Information spectrum** on at least 128 held-out windows, with the run's own anchor rule (gap 0 for G-gap0). Report:
   - the eigenvalues of `Sigma_A / sigma0^2`, and the per-direction nats `0.5 * log1p(eigenvalue)`;
   - `info_dirs`;
   - `info_pooled`, `info_window`, `info_pairwise` and `participation`;
   - the rate per window, split into entry and surprise.
5. **Calibration slope**, in eval mode (no dropout) and FP32, on 128 windows: the least-squares slope of the true change `mu_t - s_(t-1)` on the predicted change `m_t - s_(t-1)`, pooled over dimensions. It is 1 for a prior that predicts the conditional mean, and below 1 when predicted changes are exaggerated.
6. **Latent drift:** `E||mu_t - h_t||^2 / E||mu_t - mu_last_context||^2` per horizon, with `h` the open-loop prior means from the 3 context frames.

**Short gate at 2 epochs:** planning on 42 and 43, `latent_eval.py --windows 64` with 16 counterfactual cases, and the noisy agent and block probes. **Full package at 3 epochs:** everything above, plus fresh planning.

### 4.3 New tool: `as_jepa_tools/df_state_check.py` (TwoRoom)

Load the model and its config. Call `benchmark_information_audit.run_probes(model, cfg, count, seed)` directly, which reads encoder features only; do not run that script's `main`, which renders. From the same validation features, compute `feature_std / sigma0`. Report:

- the clean and `noisy_sigma0.1` agent-position R²;
- `feature_std / sigma0`;
- `I_A`, `info_dirs` and the gaming diagnostics from a forward pass on 128 validation windows.

**State gate:** both seeds have `feature_std / sigma0 >= SPREAD_MIN` and noisy agent R² of at least `TRAP_R2`. Arms that pass continue to `TWOROOM_FULL`. Also plot the training-log `spread_over_sigma0` every 100 steps.

### 4.4 New tool: `as_jepa_tools/colour_invariance.py` (Block F)

For 512 held-out frames, encode each frame twice: under its own palette A, and under the palette B of a different random test episode. Use `DecoderFreeLeWM.frames`' colour transform with the episode id substituted. Report:

- the **invariance index**, `E||mu(x_i^A) - mu(x_i^B)||^2 / E||mu(x_i^A) - mu(x_j^A)||^2`, with `j != i` and both frames under palette A. Near 0 means colour-invariant. Near 1 means palette changes move the state as much as pose changes do. A colour-keyed rotation, the gaming case, shows up here even when a linear colour probe misses it;
- the **recoloured-goal retrieval** accuracy: the fraction of frames for which `mu(x_i^B)` is nearest to `mu(x_i^A)` among the 512 frames under palette A.

For L0 (F4), use its embedding.

### 4.5 What every report records

For every run: its config, source hash, `beta`, `lam`, `horizon`, `forecast_noise`, `prior_input`, `estimator`, `conditioning`, `gap`, `calibrate_state_std`, `sigma0`, `pi`, `w`, and `seed`; the evaluated step and GPU-hours; every Section 4 output; and the two gaming flags (`GAME_GAP`, `GAME_PR`).

## 5. Run plan

Run IDs follow `df-<arm>-seed<k>`. Every PushT run uses the Section 3.2 template and stops at `SCREEN_STEPS`, which is 3 epochs inside the 10-epoch schedule, as the references. Seeds 1 and 2 unless stated.

### 5.1 Phase 1

**Block A: price and prediction weight** (one-step mean forecast; everything else canonical).

| Arm | `beta` | `lam` | `pi = beta * sigma0^2` | `w = beta + lam` |
| --- | --- | --- | --- | --- |
| A-b0.1-l0 | 0.1 | 0 | 0.001 | 0.1 |
| A-b0.1-l0.5 | 0.1 | 0.5 | 0.001 | 0.6 |
| A-b0.1-l1 | 0.1 | 1 | 0.001 | 1.1 |
| A-b0.3-l0 | 0.3 | 0 | 0.003 | 0.3 |
| A-b0.3-l0.5 | 0.3 | 0.5 | 0.003 | 0.8 |
| **A-b0.3-l1** (reference point) | 0.3 | 1 | 0.003 | 1.3 |
| A-b1-l0 | 1 | 0 | 0.01 | 1.0 (the code-length point) |
| A-b1-l0.5 | 1 | 0.5 | 0.01 | 1.5 |
| A-b1-l1 | 1 | 1 | 0.01 | 2.0 |

**Block B: forecast structure × `lam`** (at `beta = 0.3`). The one-step mean arms at `lam = 0.5` and 1 are in Block A.

| Arm | `lam` | `horizon` | `forecast_noise` | `w` (implemented `kappa`) |
| --- | --- | --- | --- | --- |
| B-l0.5-1s | 0.5 | one_step | true | 0.55 |
| B-l0.5-om | 0.5 | rollout | false | 0.8 |
| B-l0.5-os | 0.5 | rollout | true | 0.55 |
| B-l1-1s | 1 | one_step | true | 0.8 |
| B-l1-om | 1 | rollout | false | 1.3 |
| B-l1-os | 1 | rollout | true | 0.8 |

**Block C: what the prior reads** (`prior_input = mean`, at `beta = 0.3`).

| Arm | `lam` | Compared with |
| --- | --- | --- |
| C-mean-l1 | 1 | A-b0.3-l1 |
| C-mean-l0 | 0 | A-b0.3-l0 |

**Block D: estimator and reference** (at `beta = 0.3`).

| Arm | Change | Compared with |
| --- | --- | --- |
| D-win-l1 | `estimator = logdet_window`, `lam = 1` | A-b0.3-l1 |
| D-win-l0 | `estimator = logdet_window`, `lam = 0` | A-b0.3-l0 |
| D-batch | `conditioning = batch`, `lam = 1` | A-b0.3-l1 |

**Block E: the transient threshold** (`model.domain = "dot"`, seed 1). `w` uses the implemented `kappa`. The upper bound on the kept dot is from story Section 5.4.

| Arm | `beta` | `lam` | Forecast | `w` | Predicted largest dot R² (noisy probe) |
| --- | --- | --- | --- | --- | --- |
| E1 | 0.3 | 0 | none | 0.3 | above 0.3, at most about 0.7 (positive control) |
| E2 | 0.3 | 0.5 | one_step mean | 0.8 | at most about 0.2 |
| E3 | 0.3 | 1 | one_step mean | 1.3 | at most `CHANCE_R2` |
| E4 | 0.3 | 1 | one_step sample | 0.8 | at most about 0.2, and above E3 |
| E5 | 0.3 | 1 | rollout mean | 1.3 | at most `CHANCE_R2` |
| E6 | 1 | 0 | none | 1.0 | at most `CHANCE_R2` (boundary) |
| E7 | 0.6 | 0.5 | one_step mean | 1.1 | at most `CHANCE_R2` |
| E8 | 0.3 | 1 | rollout sample | 0.8 | at most about 0.2, and above E5 |

**Block F: appearance** (`model.domain = "color"`, seed 1).

| Arm | Change | Compared with |
| --- | --- | --- |
| F1 | `beta = 0.3`, `lam = 1` | F3, F4 |
| F2 | `beta = 0.3`, `lam = 0` | F1 |
| F3 | F1 with `conditioning = batch` | F1 |
| F4 | your `full-L0-10epoch` config with `domain = "color"`, to `3 * L0_EPOCH_UPDATES` | F1 |
| F5 | F1 with `estimator = logdet_window` | F1 (the gaming check where an episode-level key exists) |

**Block G: initial scale and reference distance.**

| Arm | Change from A-b0.3-l1 (unless stated) | Compared with |
| --- | --- | --- |
| G-noinit | no `calibrate_state_std` (default initialization) | A-b0.3-l1 |
| G-init3 | `calibrate_state_std = 0.3` | A-b0.3-l1 |
| G-gap0 | `gap = 0`, `anchor_gap = 0` (window frame 0 as the reference) | A-b0.3-l1 |
| G-noinit-b1 | A-b1-l1 without `calibrate_state_std` | A-b1-l1 (where the toy was trapped) |

**Block H: equivalence.**

| Arm | Settings | `pi` | `w` | Compared with |
| --- | --- | --- | --- | --- |
| H-s0.3 | `sigma0 = 0.3`, `calibrate_state_std = 0.3`, `beta = 1/30`, `lam = 0.7667` | 0.003 | 0.8 | A-b0.3-l0.5 (predicted equivalent) |

**Block T: TwoRoom** (seeds 1 and 2; past anchor at `gap_T`; to `TWOROOM_GATE`, then to `TWOROOM_FULL` if the state gate passes).

| Arm | Change | Compared with |
| --- | --- | --- |
| T-df | canonical, `beta = 0.3`, `lam = 1` | revision 2's T4 (calibrated renderer `lam = 1`, same anchor): the objective's effect. Native LeWM |
| T-df-b0.1 | `beta = 0.1` | T-df |
| T-df-noinit | no `calibrate_state_std` | your uncalibrated arms (trapped): the initialization's effect under this objective |
| T-df-l0 | `lam = 0` | T-df |

### 5.2 Counts, GPU-hours and priorities

| Block | Runs | GPU-hours at the budget step time |
| --- | --- | --- |
| A | 18 | 56 |
| B | 12 | 37 |
| C | 4 | 12 |
| D | 6 | 19 |
| E | 8 | 25 |
| F | 4 decoder-free + 1 L0 | 12 + about 6 (L0 at about 0.5 s per update) |
| G | 8 | 25 |
| H | 2 | 6 |
| T | 8 to 2,000 updates; those passing continue to 6,000 | about 1.3 + up to 2.7 |
| **Phase 1** | 62 PushT decoder-free screens, 1 L0 run, 8 TwoRoom runs | **about 200**, plus evaluation (planning dominates; measure and report it) |

Everything without a dependency may start at once. When compute is short, run in three tiers.

1. **About 55 GPU-hours:**
   - A-b0.3-l1, A-b0.3-l0, A-b1-l0 and A-b0.1-l1 (seeds 1 and 2);
   - E1, E3 and E4;
   - F1, F3, F4 and F5;
   - T-df and T-df-noinit.
2. The rest of A, then E, then B.
3. G, D, C, H, and the rest of T.

Before launching, report the measured `STEP_S_DF` and the projected GPU-hours. Wait for our acknowledgement if they exceed this table by more than 25%.

### 5.3 Phase 2 (after the Phase-1 report and our agreement)

- **DF-final.** The selected configuration `DF*` (Section 7) on `D-std`: seeds 1 and 2 continued to `MAX_STEPS` with `resume_lewm`, and seed 3 from scratch to `MAX_STEPS`. Compare with L0 at 10 epochs.
- **DF-stability.** Seeds 4 to 8 of `DF*` to `EPOCH2_STEPS`, with short gates.
- **DF-viz.** A post-hoc decoder for `DF*` seed 1: revision 2's renderer architecture trained on the frozen state's posterior samples and the anchor, with encoder and prior frozen, for 1 epoch. Then run the PushT generation package on it (pose-based rollouts, action identification, swaps, FVD). This is evaluation only and does not touch the representation.
- **DF-Cube.** `DF*` on Cube with the past anchor at `gap_K`, seeds 1 and 2, to 12,000 updates, against revision 2's K arms and native LeWM.
- **Later.** A shape-varied PushT with a learned-variance decoder-free class, for fixed properties (story Section 5.5).

## 6. Frozen predictions (record them, with the commit hash, before launch)

1. **E, transient threshold (high confidence).** The Block E table's predictions hold:
   - E3, E5, E6 and E7 at or below `CHANCE_R2`;
   - E1 above 0.3;
   - E2, E4 and E8 between the excluded arms and E1.
2. **B and E, noise halves the forecast weight (moderate).** Each pair has the same `w = 0.8` and the same horizon:
   - B-l1-1s matches A-b0.3-l0.5 (both one-step);
   - B-l1-os matches B-l0.5-om (both open loop);

   both under the Section 1.3 equivalence tolerances. And E4 and E8 keep more dot information than E3 and E5.
3. **F, appearance (high).**
   - F1 and F2 colour R² at most `COLOUR_DROP`, with invariance index at most 0.1.
   - F3 and F4 colour R² at least `COLOUR_KEEP`.
4. **F and D, gaming (moderate).** F1 raises neither gaming flag, and F1's invariance index is within 0.05 of F5's.
5. **A, price (high).** The rate per window and `info_dirs` fall as `beta` rises, at every `lam`.
6. **A, purity against completeness (moderate).**
   - The agent's noisy probe R² is at least 0.9 in every A arm (the actions reveal it).
   - At each `beta`, the block's noisy probe R² falls as `lam` rises, and the block is lost in A-b1-l1 on at least one seed (story Section 5.6).
7. **B, open loop (moderate).** B-l1-om has lower latent drift at h7 and h13 than A-b0.3-l1, and P0 non-inferior to it.
8. **C, noiseless prior input (moderate).** C-mean arms have a higher rate per window than their sample-input counterparts.
9. **D (moderate).** D-win-l1 has a lower block probe R² than A-b0.3-l1, as in the toy. D-batch and A-b0.3-l1 are equivalent in content on `D-std`, where appearance does not vary.
10. **G (moderate).**
    - G-noinit-b1 is trapped or loses the block on at least one seed.
    - G-gap0 has a lower block probe R² than A-b0.3-l1.
    - G-init3 is no worse than A-b0.3-l1.
11. **H, equivalence (moderate).** H-s0.3 matches A-b0.3-l0.5 under the Section 1.3 tolerances.
12. **T (moderate).**
    - T-df passes the state gate on both seeds at 2,000 updates.
    - T-df-noinit also passes: the information term reads the state's spread directly, so no reader has to learn first. Our confidence in this is low.
    - At 6,000 updates, T-df is within `HARD_TIE` of native LeWM or better.
13. **Against the renderer (low; the open question).** The best decoder-free arm is non-inferior to lower-beta on P0 at 3 epochs.

## 7. Selection rule (Phase 1 to Phase 2)

**Candidates** are the Block A and Block B arms, all of which have the canonical structure: `prior_input = sample`, pooled `logdet`, anchor reference, gap 8, calibrated initialization.

1. **Exclude failing arms:** trapped arms (`TRAP_R2`), arms that fail `SLOPE_MIN` in eval mode, and arms that raise both gaming flags.
2. **Generation filter, in state space:** keep arms with a latent `h7` agent ratio of at most 0.5 and an `h7_moving_block_ratio` below 1 (Section 4.2).
3. **No viable configuration.** If no candidate passes steps 1 and 2, there is none. Report, do not start Phase 2, and diagnose with Blocks D, G and the gaming diagnostics.
4. **Best planner:** the candidate `A*` with the highest pooled P0 (primary cases, seeds 1 and 2).
5. **Tied set:** every candidate non-inferior to `A*`.
6. **Prefer the canonical design.** Within the tied set, prefer, in order:
   1. `lam = 1`, one-step mean;
   2. `lam = 1`, rollout mean;
   3. `lam = 0.5`, mean forecasts;
   4. `lam = 0`;
   5. sampled-forecast variants.

   Mean forecasts are canonical because noise only weakens the decoder (story Section 3.2).
7. **Lowest rate:** among the rest, choose the lowest rate per window.

## 8. Decision rules after Phase 1

**`DF*` against L0 and the best renderer arm** (lower-beta; revision 2's P2 if it exists). Both comparisons use P0 at 3 epochs.

| | Non-inferior to the renderer arm | Worse than the renderer arm |
| --- | --- | --- |
| **Non-inferior to L0** | propose the decoder-free form as the canonical AS-JEPA; run Phase 2; we revise the story | keep it as the JEPA-compatible variant; run Phase 2; the story keeps the renderer canonical |
| **Worse than L0** | report the unusual ordering; run Phase 2's DF-final only, to see whether 10 epochs change it | diagnose before Phase 2 (trapped components, gaming diagnostics, calibration, drift); rerun the best A arm once with `calibrate_state_std = 0.3` |

**Other results.**

| Result | Then |
| --- | --- |
| Block E matches prediction 1 | The transient theory holds at calibrated prices; this is a headline result |
| The dot is kept in E3, E5, E6 or E7 | Check the forecast term's weight, `info_dirs` and the gaming flags; rerun E3 with `estimator = logdet_window` |
| E1 is at chance | Something other than the threshold prunes the dot (the probe floor, or the encoder's resolution at 224 pixels); the threshold test is inconclusive; report |
| F1 keeps colour, or F1's invariance index is far above F5's | The pooled estimator is being gamed, or the anchor difference does not cancel this appearance. Adopt `logdet_window` for colour-varying data, and report |
| B's sampled arms do not match A-b0.3-l0.5 | `kappa = 1/2` fails in practice; report the measured effective weight |
| B-l1-om plans better than A-b0.3-l1 | Adopt `horizon = rollout` in the empirical design |
| C-mean arms plan better | Report the gap; the canonical prior input stays `sample` (the rate must be a code length) unless we decide otherwise |
| The block is lost in the high-`w` arms and kept at low `w` | Purity costs completeness; the selection (Section 7) decides by planning, and we report the trade-off as a finding |
| G-noinit or G-noinit-b1 is trapped | Calibrated initialization is required; G-init3 decides its value |
| H matches | `sigma0` is not a separate knob; report results in (`pi`, `w`) |
| H does not match | Optimizer scale effects matter; report |
| T-df passes the state gate | Continue to 6,000. Compare with T4 (same initialization and anchor) for the objective's effect, and T-df-noinit with your uncalibrated arms for the initialization's effect |
| T-df is trapped | Report with T-df-noinit and T-df-b0.1; no further TwoRoom runs without agreement |

## 9. Reports

The Phase-1 report contains:

- one table per block;
- the (`pi`, `w`) plane for Block A, with P0, latent h7 ratios, agent and block noisy probe R², rate, `info_dirs`, the gaming diagnostics and drift;
- the dot threshold table for Block E against the predictions;
- the colour table for Block F with the invariance index;
- the TwoRoom trajectories of `spread_over_sigma0`;
- the predictions of Section 6, each marked held or failed;
- the selection and the decision;
- configs, commit hashes and GPU-hours (training and evaluation).

## 10. Tests and checks before any long run

1. **CPU contracts.** Run from the repository root:

   ```bash
   PYTHONPATH=.:runtime:<stable-worldmodel checkout> python -B -m unittest tests/as_jepa/test_decoder_free.py -v
   ```

   The 14 tests check that:
   - the class has no renderer, and its other initial tensors equal L2's at the same seed;
   - the objective's weights are as stated (at `lam = 0` and 1);
   - the anchor reference cancels per-episode constants for both estimators, while the batch reference does not;
   - the per-window estimator is unchanged by an episode-keyed rotation that inflates the pooled one;
   - the pooled log-det and the pairwise value agree for Gaussian states, up to the stated constant;
   - the canonical forecast equals the surprise, and an open-loop rollout's first step equals the one-step forecast (eval mode);
   - the main random stream is paired across the forecast variants;
   - gradients reach the encoder and the prior in all 5 forecast variants;
   - the anchor enters only through `I_A`;
   - the colour and dot transforms of the window and the anchor match L2's;
   - the prior-input and estimator variants are finite, and invalid combinations are rejected;
   - every logged metric is finite;
   - `predict` returns state-width means, which LeWM's rollout appends;
   - the class rejects schedules;
   - the logged `info` is the trained estimate.

   Also rerun your existing suites, and the round-2 `test_benchmark_gap.py` for Block T.
2. **Registration.**
   - `evaluate.load_model` and `planning.py` load a `decoder_free` checkpoint, and a one-episode PushT planning dry run succeeds.
   - `benchmark_evaluate.py` runs one TwoRoom case.
   - For T-df, print 16 training samples' anchor rows and window start rows, and confirm the anchor is at least `gap_T` kept frames before the window, as revision 2's smoke does.
3. **GPU smoke** (200 updates each: A-b0.3-l1, B-l1-om, D-win-l1 and T-df). Check that:
   - the loss is finite;
   - `info`, `surprise` and `entry` move, and `spread_over_sigma0` starts near 1 with calibration;
   - step time, peak memory and the pre-clip gradient norm are recorded.

## 11. Failure responses

| Symptom | Likely cause | Response |
| --- | --- | --- |
| `spread_over_sigma0` below `SPREAD_MIN` after the first epoch (PushT) or 2,000 updates (TwoRoom) | the trap (story Section 5.6) | Finish the arm as a result; rerun it once with `calibrate_state_std = 0.3` |
| The block is lost while the agent is kept | the barrier for components the actions do not reveal, at high `w` | Report the (`pi`, `w`) point; it informs the selection; no rescue |
| A gaming flag rises during training | the pooled estimate is being inflated (story Section 3.4) | Finish the arm; rerun it once with `estimator = logdet_window`, and compare content probes and the invariance index |
| `spread_over_sigma0` above 300, or `info` growing without bound | a price too small for a perfectly predictable component | Report; no change to the objective |
| NaN | numerics | `I_A` is float64 already. Check `mu` for non-finite values; halve the learning rate for that arm only, and mark it unpaired |
| Planning far below L0 while probes and latent rollouts are good | the planner's GoalMSE reads a poorly scaled latent | Report `spread_over_sigma0` and the probe floor. Do not change the planner without agreement |

## 12. Latitude: what you may change without asking

- Compute allocation and the order of runs, within Section 5.2's tiers.
- Evaluation sample counts, upward, for an arm and its comparisons alike.
- Adding seeds to any arm, together with its comparisons.
- Implementation mechanics of the three new tools, launch wiring and reports.

Keep one GPU per run, and the references' optimizer settings. Everything in Sections 3, 7 and 8 needs our agreement before it changes: the objective's terms, the estimator, the reference, the noise, the initialization rule and the selection rule.

## 13. Order of work

1. Branch, copy and register (Section 2.1). Run the CPU tests and registration checks (Section 10).
2. Write the three tools. Sanity-check `latent_eval.py` on one existing L2 checkpoint: its readout probe on encoded frames should match the noisy-probe numbers your earlier reports give.
3. GPU smokes. Report `STEP_S_DF` and the projected GPU-hours.
4. Launch Phase 1 in tier order (Section 5.2).
5. Phase-1 report (Section 9). Stop for our agreement.
