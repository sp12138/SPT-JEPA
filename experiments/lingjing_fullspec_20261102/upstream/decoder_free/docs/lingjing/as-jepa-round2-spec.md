# AS-JEPA round 2: PushT closure and the harder tasks (implementation and experiment spec, handoff)

Written 2026-10-31 for the collaborator who runs the L-line experiments; this is revision 2, from the same day. Follow it as written. Section 12 lists what you may change without asking; anything else needs our agreement first.

Revision 1 (commit `937707e`) planned about 715 GPU-hours on PushT before any harder-task work. This revision replaces it:

- **PushT** shrinks to a closing set of about 100 GPU-hours. It settles six questions and then stops.
- **TwoRoom and Cube** start now, in parallel, at about 70 GPU-hours. They are designed around what your runs revealed there.
- **Section 0.7 divides the work with your campaign**, so that the two plans complement each other instead of repeating runs.

The spec builds on your branch `research/ideas-20261028` at `2fdd905`: its code, data, evaluation tools and recipes. A reference implementation of everything new is in [round2/](round2/): the two model classes, the past-only anchor for benchmark windows, and 16 CPU contract tests. All 16 tests pass against `2fdd905`. Copy the files; do not reimplement them. Appendix A gives the theory; you do not need it to execute the spec, but each component names the design reason it serves.

## 0. Overview

### 0.1 What changed in this revision, and why

Two things changed our view of the plan.

- **PushT cannot carry the performance claim.** It is simple, and published LeWM already reaches 96% on it. What PushT can still do cheaply is settle a few questions that need its controlled variants (the dot and colour domains) and its paired multi-seed references. Then we leave it.
- **Your TwoRoom and Cube runs exposed two failures that our theory speaks to directly, and both are cheap to test.**
  - On TwoRoom, the state never left its initial scale.
  - On Cube, the state is used, but the planning cost follows the arm rather than the cube.

  Section 0.5 gives the numbers and our reading.

Unchanged from revision 1: the λ model class, the contrastive control, their tests and the PushT evaluation tools. New in this revision:

- the past-only distant anchor for benchmark windows (Section 3.4);
- the initial state scale on the harder tasks (Section 3.5);
- the dataset checks that choose the anchor gap (Section 5.6);
- a use gate for the state (Section 5.7).

Appendix C lists everything dropped from revision 1.

### 0.2 The method as it runs on TwoRoom and Cube

| Component | Setting | Reason |
| --- | --- | --- |
| Encoder | your L2 encoder: LeWM ViT on 224×224 frames, `QueryPool` to 4 slots × 16 = 64 dimensions, one frame at a time | The state is a function of its frame, so goal images encode without history. Kept fixed so that objective effects are attributable (Section 0.7). |
| Initial state scale | `calibrate_state_std = sigma0`: the pool head is rescaled once at step 0 so that `mu` has spread `sigma0` | Start where the renderer can read the state. At the default initial spread (about 1e-3) each dimension carries about 5e-5 nats through the renderer's noise; on TwoRoom the state then shrank and never recovered (Section 0.5, Appendix A.6). |
| Code noise | `sigma0 = 0.1`, fixed | The code's resolution. |
| Prior | LeWM predictor with the identity carry, fixed variance `sigma0^2`; it reads states and actions only | Unchanged from L2. |
| Rate | the exact one-step Gaussian KL, priced at `beta` ∈ {1/16, 1/4} per rendered value | The code length. `beta` is the only searched knob. |
| Renderer | DiT 384 / 12 / 6, flow matching on 64×64 targets, condition dropout 0.1 | Unchanged from L2. |
| What the renderer reads | for each rendered frame, with probability `lam = 1`, the one-step forecast sample `m_t + sigma0 * xi` | It rewards what predicts the future (Section 0.3, Appendix A.2). |
| Anchor | a past frame at least `gap` kept frames before the window (TwoRoom, Cube). Cube also tests a constant (zero) anchor. Never a future frame | The anchor should carry appearance, not the current value of anything that moves (Appendix A.7). |
| Window | 8 kept frames at frameskip 5 | Unchanged. |
| Planning | unchanged: CEM on the encoder means with GoalMSE. The renderer and the anchor are not used | Unchanged. |

### 0.3 The model change, `lam`

In L2 the renderer always reads the encoded state of the frame it draws, `s_t = mu(x_t) + sigma0 * noise`. In round 2, a coin is tossed for each rendered frame. With probability `lam`, the renderer instead reads a sample of the prior's forecast of that state, `s_hat_t = m_t + sigma0 * noise`. Here `m_t` is the prior's mean for `s_t`, computed from the earlier states and the actions; it is the tensor the surprise term already uses.

```python
forecast = predicted + sigma0 * xi                 # a sample of the prior p(s_t | s_<t, a_<t); xi ~ N(0, I)
use = torch.rand(B, T, 1, 1) < lam                 # one coin per rendered frame
condition = torch.where(use, forecast, s[:, 1:])   # what the renderer reads
```

Nothing else changes. At `lam = 0` the class computes exactly L2's loss, with the same random draws. Why this change:

1. **It rewards what predicts the future.** At `lam = 0` the renderer asks whether a frame's own state, with the anchor, describes that frame. At `lam = 1` it asks whether the earlier states and actions, passed through the prior, describe the next frame. The first rewards information about the current frame; the second rewards information about the future beyond what the anchor shows, which is the textbook definition of a state.
2. **Unpredictable detail leaves at every price.** Detail drawn fresh in each frame costs `beta` per nat and earns `1 - lam` per nat, because no forecast can predict it. At `lam = 1` it is therefore excluded at every `beta > 0`, without calibrating `beta`.
3. **The renderer trains on the input generation feeds it.** At generation time the renderer draws prior forecasts, but at `lam = 0` it has only ever seen encoded states.

### 0.4 What the forecasts are: one-step and open-loop, noisy and noiseless

Take one window: frames `x_0 .. x_7`, actions `a_0 .. a_6`, encoded states `s_0 .. s_7`. Consider the forecast of frame 5.

- **One-step (teacher-forced).** The prior reads the encoded states of the real frames 0 to 4 and the actions: `m_5 = prior(s_0 .. s_4, a_0 .. a_4)`. Every forecast is one step beyond real information. The rate term always uses one-step forecasts, and so does canonical `lam`.
- **Open-loop (free-running rollout).** The prior reads only the encoded state of frame 0, then feeds back its own forecasts: `h_1 = prior(s_0, a_0)`, `h_2 = prior(s_0, h_1, a_0, a_1)`, and so on to `h_5`. The forecast of frame 5 is five steps beyond real information, so errors compound. This is what the planner does at test time.

Either kind can be **noisy** (a sample, with `sigma0` noise at every step) or **noiseless** (the prior's means):

| Forecast | What the prior reads for frame t | Noise | Used by |
| --- | --- | --- | --- |
| one-step | encoded states of frames 0 .. t-1 | `sigma0` sample | the rate term (all models); canonical `lam` |
| open-loop, noisy | encoded `s_0`, then its own samples | `sigma0` at every step | our `horizon = "rollout"` (conditional arm C5) |
| open-loop, noiseless | encoded `s_0`, then its own means | none after frame 0 | your forecast-only and TF arms; the planner at test time |

The noise matters because a noiseless mean is a continuous vector that nothing prices. The prior can pass information about the actions through it to the renderer for free (Appendix A.5). A `sigma0` sample limits that channel to the code's resolution, and the surprise term prices the mean's departure from the encoded state.

### 0.5 What your harder-task runs showed, and how this spec reads them

**TwoRoom: the state never left its initial scale.**

- **Every AS-JEPA arm stayed near its initial scale.** That covers original, lower-beta, TF, the high-noise variants, observed and constant anchors, shared prior, rate off and noise removal. The validation spread of `mu` was 0.9 to 2.1e-4 at every evaluated checkpoint (600 to 2,600 updates), about 1/600 of `sigma0`. The training-batch spread fell from about 1e-3 at step 100 to 5e-4 at 6,000 updates.
- **The renderer ignored the state.** At 2,000 updates, swapping in another video's state changed the renderer's error by at most 0.0003% at t = 0.95, against 53 to 110% on Cube at the same step; no swap tested on TwoRoom exceeded 0.005%. Linear probes of the agent's position give R² 0.11 to 0.29 from clean `mu` and about -0.06 through the `sigma0` noise. Planning stayed at 0 to 16%.
- **Two other models plan well on the same 50 cases.** Natively retrained LeWM scores 80 / 72% at 6,000 updates (82 / 78% with recalibrated BatchNorm statistics). Your frozen random ViT with a fixed spatial projection scores 90 / 94% at 6,000 updates with the free chart.
- **The information was available from the start.** A fixed random projection of the same ViT's patch features reads the agent's position at R² 0.89 to 0.94 at initialization (0.96 at 2,000 updates).

**This is not the collapse of the JEPA literature.** In a JEPA, a constant encoder makes the latent prediction loss exactly zero. The collapsed solution is the best one, so EMA, VICReg or SIGReg must rule it out. Here the renderer's loss would be lower with the agent in the state: 84 to 91% of its remaining error sits in the 9.4% of pixels that move. So the objective's best solution uses the state, and TwoRoom is stuck at a stationary point of training, not at the objective's optimum (Appendix A.6). The mechanism has three steps:

1. The state starts at a spread of about 1e-3, because the `QueryPool` head is initialized with standard deviation 0.02. Through `sigma0 = 0.1` noise, each dimension then carries about 5e-5 nats.
2. In the first 600 updates the encoder did receive sizeable gradients (norm up to 0.045 to 0.236), but the state moved toward the prior mean instead of spreading out. The entry term's excess fell from 2.8 to 7.5 nats to 0.1, and the validation spread settled near 1.5e-4, about 1e-6 nats per dimension.
3. A renderer cannot use what it cannot read, so it learned to ignore the state. The encoder's gradient fell to 0.008 to 0.027 over updates 600 to 2,000 and to 0.002 to 0.005 over 2,000 to 2,600 (Cube over the same span: 0.15 to 0.27), and the state stayed small.

Your linear toy (`hardtasks20261030/THEORY-AFTER-INFORMATION-AUDIT-20261030.md`) puts the escape rate at about `2 eta^2 / sigma0^2` for state scale `eta`. At the default initial scale (`eta / sigma0 ≈ 1e-2`) that is a ten-thousandth of its value at `eta = sigma0`, and at the trapped scale (1.5e-3) about two millionths. Removing the rate (`beta = 0`) did not free the state in your runs, so the rate is not the only cause. It is, however, a reason a calibrated state could shrink back; T4 against T6, and a `beta = 0` fallback (Section 11), test that.

**Why Cube escaped from the same initial scale (our inference).** More of the picture moves: 31.6% of Cube render pixels against 9.4%. So the renderer gains more from the state at any scale. Cube's state grew 24 to 50 times by 2,000 updates.

**Two contributing factors, which this spec also tests:**

- **The pooled state carries little agent information even before noise**, while a fixed spatial projection carries a lot.
- **Your TF arms combine a first-frame anchor with noiseless open-loop forecasts, which lets the renderer place the agent from the actions alone.** Their wrong-action effect reached 72% while no state swap exceeded 0.005%.

The forecast shortcut cannot explain the `lam = 0` arms, which have no forecast path to the renderer. So this spec treats the initial scale as the primary cause (T1 against your original arm) and tests the shortcut separately (T5).

**Cube: the state is used, but it holds the arm better than the cube.**

- **Planning trails matched LeWM.** Original, lower-beta and TF reach 20 / 24, 22 / 26 and 24 / 26% at 11,671 updates. Native LeWM reaches 26 / 26% at 12,000 updates (30 / 28% recalibrated).
- **The planning cost follows the arm.** On your fixed candidate bank, TF's cost has partial rank correlation 0.93 with arm displacement and 0.03 with cube displacement.
- **The cube is read less accurately.** Cube position is read at 4.7 to 5.1 cm from TF against 3.7 to 3.9 cm from native LeWM.

**Our reading.** With the first-frame anchor, the renderer sees the cube's pose at window frame 0. A resting cube keeps that pose through the window, and a carried cube follows the gripper, which the state already holds. In most windows, then, storing the cube's own pose earns nothing (Appendix A.7), and the planner compares states that barely differ in cube pose.

Two anchor changes remove that free information, and `lam = 1` adds a separate reason to store the cube:

- **A past anchor far enough back** that the cube has often moved since.
- **No informative anchor at all**, a constant one. Cube's appearance does not vary across episodes, so the renderer can learn the scene itself.
- **`lam = 1`**: the prior must forecast whether the gripper closes on the cube, and for that it needs the cube's pose.

Two other Cube findings are in your campaign and are not repeated here:

- **The planner's candidates don't look like the training actions.** Correlated CEM fixes part of this for every model.
- **12,000 updates is only about one loader epoch**, with the learning rate still at 98% of peak.

### 0.6 What this round answers

| Question | Where |
| --- | --- |
| P-Q1. How does matched LeWM (L0) do on full-data PushT at 3 and 10 epochs? No full-data L0 run exists | P1 |
| P-Q2. Does canonical `lam = 1` keep the planning gain of your forecast-render arms, over four training seeds? | P2 |
| P-Q3. Is the original arm's seed instability at 2 epochs (P0 10 to 84% across seeds 1 to 8) the same trapped state, removed by the initial scale? | P3 |
| P-Q4. At `lam = 1`, does the dot leave the state, while it stays at `lam = 0`? | P4 |
| P-Q5. Does LeWM keep colour in its embedding while AS-JEPA leaves it to the anchor? | P5 |
| P-Q6. What does a reconstruction-free version (a contrastive critic in place of the renderer) keep? | P6 |
| T-Q1. Does starting the state at `sigma0` let TwoRoom train a state the renderer uses? | T1 against your original arm |
| T-Q2. Which of initial scale, anchor rule and `lam` matters on TwoRoom? | T1 to T4 |
| T-Q3. Was TF's TwoRoom failure the initial scale or the noiseless forecast shortcut? | T5 against your TF arm |
| T-Q4. Does the canonical recipe plan near native LeWM and your fixed carrier at 6,000 updates? | T, continuation |
| K-Q1. How often does each anchor rule show the cube where it no longer is? | K-pre (CPU) |
| K-Q2. Does removing the anchor's free cube pose move the cube into the state and into the planning cost? | K1, K2, K3 |
| K-Q3. Does `lam = 1` add to that? | K1 against K4 |
| K-Q4. Does the canonical recipe reach native LeWM at matched updates and planner? | K-main, K-mature |

### 0.7 Division of work with your campaign

| Topic | Your campaign | This spec |
| --- | --- | --- |
| Native LeWM baselines: TwoRoom 600 to 6,000 updates; Cube 600 to 12,000, with 23,342 running | owns them | reuses your checkpoints, cases and evaluations; reruns nothing |
| PushT LeWM (L0) on full data | not run | P1 |
| State architecture: fixed and local carriers, charts, readouts, patch tokens, predictor output span | owns it | keeps L2's encoder and `QueryPool` fixed, so that objective effects stay attributable |
| Objective: `lam`, `beta`, initial state scale | | owns it |
| Anchor rule | first frame (your default); constant anchor as an adaptation from TF at 2,000 | past-only distant anchor; constant anchor trained from scratch with calibrated scale (K3) |
| Controllers: correlated CEM, NGC | owns them | evaluates every checkpoint under white CEM (primary) and your fixed correlated CEM (secondary). No new controller work |
| Training maturity | owns it (the six 23,342-update continuations) | uses your result to decide whether K continues (Section 8) |
| Already settled on TwoRoom and Cube by your runs | rate off (`beta = 0`), posterior-noise removal, training at pure target noise, high-noise logit shift, shared forecast prior, constant-anchor adaptation, quadrant pooling | cited as evidence; not repeated |
| TF and original arms on the harder tasks | your runs at 2,000, 6,000, 11,671 and 12,000 updates | reused as references. T5 is your TF config with only the initial scale changed |

Where the two plans meet. Each meeting point is run once, by whoever has capacity, after both sides agree:

- **(a) Combining recipes.** If T or K shows that the canonical objective trains a useful state, the next step combines it with your best state architecture.
- **(b) Readout bottleneck.** If every T arm fails the use gate (Section 5.7), the pooled readout is the bottleneck. We then decide together whether AS-JEPA adopts a spatial state, such as your local carrier made learnable.
- **(c) Anchor rule.** If K shows that the first-frame anchor hides the cube, your renderer-based variants (TF, carriers with a renderer) should switch to the winning anchor rule. We suggest you own that change.

Your six maturity runs occupy dev2, dev13 and dev15 GPUs 0 and 1 until about 16:30 UTC on 10-31. This spec's runs use other GPUs; do not stop or reuse those slots.

### 0.8 Stopping points

There are three reports: PushT closure (Part P), TwoRoom (Part T) and Cube (Part K). Write each when its part finishes, then stop for our agreement once all three exist.

- **Launch automatically when the trigger fires:** the TwoRoom continuation from 2,000 to 6,000 updates for arms that pass the use gate (Section 8.2), and the PushT conditional arms C5, C2 and E1 (Section 8.1; at most about 60 GPU-hours together).
- **Wait for our agreement:** the C1 continuation to 10 epochs (Section 8.1) and K-mature (Section 8.3).

## 1. Named quantities

New code reads these from the run config or from module constants. Derived values are computed, not retyped.

### 1.1 PushT

| Name | Definition | Value |
| --- | --- | --- |
| `BATCH` | windows per optimizer step | 128 |
| `EPOCH_UPDATES` | `floor(train windows / BATCH)` on `pusht-lance-full-v1` at `W = 8` (1,598,974 windows) | 12,491 |
| `MAX_STEPS` | `10 * EPOCH_UPDATES`, the learning-rate horizon of every L2-family run | 124,910 |
| `SCREEN_STEPS` | `3 * EPOCH_UPDATES` | 37,473 |
| `EPOCH2_STEPS` | `2 * EPOCH_UPDATES` | 24,982 |
| `L0_EPOCH_UPDATES` | the same for LeWM's window of 4 (your `full-L0-10epoch-*` configs) | 15,265 |
| `STEP_S` | measured seconds per step of full L2 on one H200 | about 0.91 (11,328 s for 12,491 steps, `legacy-closure/newseed-epoch2/replication-results.json`) |
| `SCREEN_GPU_H`, `EPOCH2_GPU_H`, `CONTINUE_GPU_H` | `STEP_S` times 37,473, 24,982 and 87,437 steps | about 9.5, 6.3 and 22.1 |

### 1.2 Harder tasks

| Name | Value | Why |
| --- | --- | --- |
| `TWOROOM_GATE` | 2,000 updates | past the 1,249-update warmup; your TwoRoom references exist here |
| `TWOROOM_FULL` | 6,000 updates | native LeWM (80 / 72%) and your fixed carrier (90 / 94%) were evaluated here |
| `CUBE_FULL` | 12,000 updates | native LeWM (26 / 26%) and TF (18 / 20%) exist here; one loader epoch is 11,671 |
| `CUBE_MATURE` | 23,342 updates | two loader epochs, as in your running maturity continuations |
| `MAX_STEPS` | 124,910 | the learning-rate horizon of all your harder-task runs |
| `STEP_S_HARD` | budget 1.5 s per update for L2 on one H200 | your Cube startups measured 1.11 to 1.34 s; measure in the smoke (Section 10.3) |
| `CASES` | your fixed 50-case strict-validation primary set and 50-case fresh set, per task | the test split stays untouched |
| `WHITE_CEM` | 300 samples × 30 iterations, top 30, horizon 5, receding horizon 5, history 3, budget 50, GoalMSE on `mu` | your native protocol |
| `CORR_CEM` | `WHITE_CEM` sampling in your fixed training-action covariance (`action_covariance_20261031`, same frozen calibration files) | your common controller |

### 1.3 Method knobs

| Name | Meaning | Values in this round |
| --- | --- | --- |
| `lam` | probability that the renderer reads a forecast sample instead of the encoded sample, per rendered frame | 1 (canonical) and 0 (L2). 0.5 only in conditional arm C5 |
| `horizon` | `one_step` or `rollout` (Section 0.4) | `one_step`; `rollout` only in C5 |
| `forecast_grad` | `joint`: no stop-gradient. `encoder_only`: the forecast branch sends no gradient into the prior's parameters | `joint`; `encoder_only` only by the rule in Section 11 |
| `beta` | price of one nat of state, per rendered value (the unit of your configs) | 1/4 everywhere; 1/16 in T6 and conditional C2. The contrastive arm uses `beta = 1` in true nats |
| `sigma0` | code noise; the prior's spread and the forecast noise are tied to it | 0.1, frozen |
| `gap` | anchor distance in kept frames | PushT: 8, two-sided, as in your references (frozen for pairing). TwoRoom `gap_T` and Cube `gap_K`: past only, chosen by the rule in Section 5.6. Absent means the first-frame anchor |
| `renderer_anchor_mode` | `observed`, or `zero` for a constant anchor (existing option) | `observed`; `zero` only in K3 |
| `W` | window, in kept frames | 8, frozen |
| `rho` | AR(1) correlation of the posterior noise | 0; only the TF reference and T5 use 0.8 |
| `calibrate_state_std` | your one-time initialization of the state spread (`train.py`, `on_fit_start`) | `sigma0` on TwoRoom and Cube, and in P3. Absent in the other PushT arms, to stay paired with your references |

Why only `beta` is searched: it cannot be set from first principles while the flow loss is not in nats. `sigma0` and the window are frozen at values that keep slow components. `lam` has a principled value, 1. The anchor gap and the initial scale are chosen by stated rules, not by results.

### 1.4 Evaluation and decision constants

| Name | Value | Meaning |
| --- | --- | --- |
| `PLAN_SEEDS`, `FRESH_SEEDS` | 42, 43, 44 and 1042, 1043, 1044 | PushT primary and fresh planning cases, 3 × 50 episodes each |
| `PAIRED_CI` | 95% episode-clustered bootstrap of paired P0 differences, pooled over training seeds 1 and 2 (`report_main_conclusions.cluster_ci`) | the uncertainty from planning episodes. It does not include training-seed variation, hence the per-seed conditions below |
| `NONINF` | 5 points | A is non-inferior to B if the lower end of `PAIRED_CI(A - B)` is above -5 and each seed's mean difference is above -5 |
| better, worse | | A is better than B if `PAIRED_CI(A - B)` lies above 0 and both seeds' differences are positive; worse if below 0 and both negative; otherwise tied |
| `SLOPE_MIN` | 0.9 | lowest acceptable eval-mode calibration slope of the prior (Section 5.3); the toys predict at most 3.5% overshoot |
| `CHANCE_R2` | 0.05 | a probe R² at or below this is chance. For the dot, decide on the largest of the five dot R² values |
| `COLLAPSE_RATIO` | 0.9 | as in `collapse_audit.py` |
| `SPREAD_MIN` | 0.3 | lowest validation `feature_std / sigma0` for a used state. At 0.3 each dimension carries up to `0.5 * ln(1.09)` ≈ 0.04 nats, about 2.8 nats per frame over 64 dimensions; the trapped TwoRoom value is 0.0015, and Cube passed 0.3 by 2,000 updates |
| `USE_MIN` | 5% | lowest relative increase of the renderer's error at t = 0.95 when the state is swapped for another video's (Section 5.7). At 2,000 updates under the posterior swap: trapped TwoRoom at most 0.0003%, Cube 53 to 110% |
| `HARD_TIE` | 6 points | harder-task success differences below this, averaged over 2 seeds and both case sets, are ties. Your measured noise: re-evaluating one checkpoint moves 1 to 2 of 50 cases, and TF continued from one 2,000-update checkpoint scored 24 / 26% at 11,671 updates and 18 / 20% at 12,000 |
| `FALLBACK_MAX` | 0.2 | largest share of windows allowed without an eligible past anchor (Section 5.6) |
| `CUBE_TOL` | 3 cm and 15° (yaw modulo the cube's quarter-turn symmetry) | the strict success tolerance; used for anchor staleness |
| `AGENT_TOL` | the strict success distance of your TwoRoom manifests (8.0, in `pos_agent` units; confirm the units) | used for anchor staleness |
| `RANK_HIGH`, `RANK_LOW` | 0.3, 0.1 | the cube partial rank (`position_encoded_partial_rank`, Section 5.8): how well the latent distance from a candidate's reached endpoint to the goal tracks the cube's distance to its goal, holding the effector's distance and the angle fixed. Values at or above 0.3 count as cube-driven, at or below 0.1 as not. TF: 0.03 for the cube, 0.93 for the effector |

## 2. Starting point

### 2.1 Code

1. Create branch `as-jepa/round2` from your current head (`2fdd905` or later), and record the hash in every report.
2. Copy unchanged from `docs/lingjing/round2/`:
   - `as_jepa_lewm/forecast_scored.py` and `as_jepa_lewm/nce.py` into `as_jepa_lewm/`;
   - `as_jepa_tools/benchmark_gap.py` into `as_jepa_tools/`;
   - `tests/as_jepa/test_round2.py` and `tests/as_jepa/test_benchmark_gap.py` into `tests/as_jepa/`.

   `learned_variance.py` is not used in this round.
3. Register `forecast_scored` and `nce` in the three places that list model targets:
   - `as_jepa_lewm/factory.py`, `MODEL_TARGETS`: add `"forecast_scored": "as_jepa_lewm.forecast_scored.ForecastScoredLeWM"` and `"nce": "as_jepa_lewm.nce.NCELeWM"`. `train.py` and `benchmark_health.py` build models through this table.
   - `as_jepa_tools/evaluate.py`, `load_model`: one `elif` per class before the final `else`, as for `RobustLeWM`.
   - `as_jepa_tools/planning.py`: add both targets to the allowed `_target_` values.

   `benchmark_evaluate.py` needs no change, because it instantiates the `_target_` in each checkpoint's `config.json` through `load_pretrained`. Confirm this in the smoke.
4. Route benchmark datasets through `make_benchmark_dataset(cfg, split)` from `benchmark_gap.py`, at the places that build `BenchmarkWindows` for training or for renderer diagnostics:
   - `benchmark_data.make_training_dataset` (the benchmark branch). Import `benchmark_gap` inside the function, because `benchmark_gap` imports `benchmark_data`;
   - `benchmark_health.py` (line 41);
   - `benchmark_information_audit.py` (line 257, the renderer substitutions; line 175 never reads the anchor);
   - `gradient_preflight.py` (line 35). It also builds `RobustLeWM(**cfg["model"])` directly (line 33), which fails on a `forecast_scored` config; switch it to `build_model(cfg)`.

   With no `gap` in a config, `make_benchmark_dataset` returns `BenchmarkWindows` exactly as before; a test checks this, and none of your 75 existing benchmark configs sets a top-level `gap`. Old configs and frozen snapshots are therefore unaffected. `benchmark_spatial_probe.py` builds `ASJepaLeWM` directly and does not use the anchor; it is not used for round-2 runs.
5. Tool additions:
   - **PushT:** `--condition {posterior,forecast}` and `--fp32` for `collapse_audit.py`; `--condition` for `color_swaps.py`; the tool `as_jepa_tools/forecast_calibration.py` (Sections 5.2 and 5.3); and the colour-invariance tool (Section 5.4).
   - **Harder tasks:** two conditions in `benchmark_health.py`, `one_step` and `one_step_shuffled` (Section 5.7); and a checkpoint argument for your Cube readout probe, `experiments/carrier_progress_20261031/tf_pose.py`, which currently hard-codes the TF runs (Section 5.8).
6. Do not edit `model.py`, `robust.py`, `research.py` or any frozen source.
7. Freeze every campaign's source with your usual `git archive` plus SHA256 manifest, and dispatch with your workflow tools.

### 2.2 Data

- **`D-std`:** `/efs/prescottsun/SPT-JEPA/data/pusht-lance-full-v1` with `all_phase_data: true`. Evaluation uses its `phase0/` view, and planning uses `pusht_expert_train.h5` as `planning.py` sets it up.
- **`D-dot` and `D-colour`:** `D-std` with `model.domain` set to `"dot"` or `"color"`, applied in `ASJepaLeWM.prepare`.
  - The dot is redrawn for every frame, and the anchor carries its own, different dot.
  - The colour transform is a per-episode palette, shared by every frame and the anchor.
- **TwoRoom:** `/efs/prescottsun/SPT-JEPA/data/hardtasks-v1/tworoom-metadata`, `dataset_kind: benchmark_hdf5`, `action_dim: 10`.
- **Cube:** `/efs/prescottsun/SPT-JEPA/data/hardtasks-v1/cube-metadata`, `dataset_kind: benchmark_hdf5`, `action_dim: 25`.

Splits and action normalization are those of your runs: training episodes only.

### 2.3 Reference runs reused, not rerun

**PushT** (P0 %, by training seed):

| Reference | Configs | 3 epochs, seeds 1 / 2 | 2 epochs, seeds 1 to 8 |
| --- | --- | --- | --- |
| original: L2, `beta = 1/4` | `experiments/stability20261029/configs/full-L2-beta0.25-stage3-seed*.json` | 88 / 74.67 (seeds 0 to 4: mean 85.1) | 73, 43, 22, 76, 10, 15.33, 15.33, 84 |
| lower-beta: L2, `beta = 1/16` | `experiments/robustness20261029/configs/lower-beta-seed*.json` | 86 / 84 | 87, 85, 86.67, 70.67, 84, 80.67, 84, 82.67 |
| forecast-only: `RobustLeWM`, `rho = 0`, `forecast_fraction = 0.5` | `robustness20261029/configs/forecast-render-seed{1,2}.json`; epoch 3 in `runs/forecast_confirm_20261030/` | 90 / 90 | 90, 90 (seeds 1, 2) |
| TF: `RobustLeWM`, `rho = 0.8`, `forecast_fraction = 0.5` | `robustness20261029/configs/temporal-forecast-seed*.json` | 94 / 94 | 88, 90, 92.67, 88, 90, 91.33, 86.67, 91.33 |

At a given training seed, `ASJepaLeWM`, `RobustLeWM` and `ForecastScoredLeWM` build identical initial tensors, and the data order depends only on the seed. So PushT arms at the same seed are paired with these references. All four PushT references use the two-sided anchor at gap 8 and no initial-scale calibration. Their 2-epoch values for seeds 1 and 2 were evaluated on planning seeds 42 and 43 only (100 episodes); compare C1's 2-epoch gates on the same 100.

**TwoRoom** (success %, seeds 1 / 2, primary cases):

- **AS-JEPA arms at 2,000 updates:** original 8 / 2, lower-beta 2 / 6, TF 2 / 6. All have a first-frame anchor and no calibration, and their spread is about 1.5e-4.
- **TF at 6,000 updates:** 2 / 4.
- **Native LeWM at 6,000 updates:** 80 / 72 (82 / 78 recalibrated).
- **Your fixed carrier with the free chart, at 6,000 updates:** 90 / 94.
- **Released LeWM:** 72. **Random:** 0.

**Cube** (success %, seeds 1 / 2, primary cases):

- **AS-JEPA arms at 11,671 updates:** original 20 / 24, lower-beta 22 / 26, TF 24 / 26.
- **TF at 12,000 updates:** 18 / 20.
- **Native LeWM at 12,000 updates:** 26 / 26 (30 / 28 recalibrated); 23,342 updates are running.
- **Under `CORR_CEM`, at 12,000 updates:** TF 33, native 36 (means).
- **Released LeWM:** 30. **Random:** 2.

## 3. Implementation

### 3.1 `ForecastScoredLeWM` (`as_jepa_lewm/forecast_scored.py`)

The class subclasses `ASJepaLeWM` and adds four constructor arguments: `lam`, `horizon`, `forecast_grad` and `forecast_seed`. Other keyword arguments pass through, including `action_dim`, `renderer_anchor_mode` and `flow_logit_shift`, so the class trains on the benchmark data unchanged. It adds no parameters. Its loss is `ASJepaLeWM.loss` line for line, except for the renderer's condition:

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
fm = flow_loss(self.renderer, y, condition.flatten(0, 1), a, self.training_shift)
rate = (entry + surprise.sum(1)).mean() / ((w - 1) * FRAME_ELEMS)
loss = fm + beta * rate                                                       # unchanged
```

With `horizon = "rollout"`, `_forecast` runs the open loop of Section 0.4 with a `sigma0` sample at every step: `h_0 = s_0`, then `h_t = m(h_<t, a_<t) + sigma0 * xi_t`. Never render a noiseless prior mean during training (Appendix A.5).

Logged metrics, in addition to L2's:

- `forecast_share`: the fraction of rendered frames that read a forecast. It should equal `lam`.
- `calib_slope`: the calibration slope on the training batch, for plotting.
- `active_dims`, as in `ASJepaLeWM`.

### 3.2 Random streams and pairing

The coins, the forecast noise and any extra predictor passes, including their dropout masks, run inside `torch.random.fork_rng`. That stream is seeded by `FORECAST_STREAM + 1000003 * forecast_seed + step`. The main stream is therefore consumed exactly as in L2: posterior noise, flow-matching noise and timesteps, and the renderer's condition dropout.

An arm at seed `k` sees the same data order and the same initial tensors as the references at seed `k`, and the same draws at every update. Set `forecast_seed` to the training seed. Keep one GPU per run with the same per-step batch: several ranks would change the data order and draw identical coins on every rank.

### 3.3 Gradient routing

With `forecast_grad = "joint"`, the default, there is no stop-gradient anywhere. The forecast branch's flow loss reaches:

- the prior, through `m_t`;
- the encoder, through the earlier states `s_<t`. This is how the encoder is rewarded for information that helps forecast later frames.

At `lam = 1`, a frame's own encoding enters only the renders of later frames, because the prior's attention is causal. The last frame of a window reaches the loss only through the rate; a test checks this.

Joint training has one known side effect (Appendix A.5): the prior is rewarded for slightly exaggerating its predicted changes. The eval-mode calibration slope (Section 5.3) watches for this. The fallback, `forecast_grad = "encoder_only"`, recomputes the forecast branch with the predictor's parameters and the action embedding detached. It is a stop-gradient, so use it only under the rule in Section 11.

### 3.4 Past-only distant anchors for benchmark windows (`as_jepa_tools/benchmark_gap.py`)

`GapBenchmarkWindows(root, window, seed, split, gap)` is `BenchmarkWindows` with one change: the renderer's anchor (`anchor`, `anchor224`, `anchor_row`) is a frame of the same episode, phase-aligned with the window, at least `gap` kept frames before window frame 0.

- **The draw.** It is uniform over the eligible past frames, using a generator seeded by `(episode, start, epoch)`, as `FullPushTWindows` does. So it is fixed within an epoch and redrawn across epochs.
- **Fallback.** When no past frame is far enough back, the earliest phase-aligned frame before the window is used, or window frame 0 when the window starts the episode, and `anchor_fallback` is True.
- **Everything else is unchanged.** Pixels, render targets, actions, rows, episode and start are byte-identical to `BenchmarkWindows`. `gap = 0` reproduces `BenchmarkWindows` exactly.
- **Transforms.** The anchor frame goes through the transforms that window frames get: resizing to 224×224 if needed, then area downsampling to 64×64.

`make_benchmark_dataset(cfg, split)` returns `GapBenchmarkWindows` when the config sets a top-level `gap`, and `BenchmarkWindows` otherwise.

Why past only:

- **A future frame would leak.** It would show where slow objects end up, such as the cube's resting place after the window, which is information about the future given to the renderer for free.
- **Generation never has one.** It only ever has past frames.
- **It matches your convention.** Your data contract already rules out future anchors.

PushT keeps its references' two-sided rule, for pairing.

### 3.5 The initial state scale on the harder tasks

Set `"calibrate_state_std": 0.1` (that is, `sigma0`) at the top level of every TwoRoom and Cube config, and in P3. `train.py`'s `on_fit_start` then rescales the `QueryPool` head once, from 128 training windows, so that at step 0 the average per-dimension spread of `mu` is `sigma0`; it also subtracts the head's mean output, averaged over the four slots. Nothing normalizes the state after that, and the objective is unchanged. The step is skipped on resume.

It writes `initial_scale.json` with `target_std`, `original_std`, `scale` and `mean_before`. Expect `original_std` of about 1e-3 to 2e-3 (your PushT calibrations measured 1.6 to 1.9e-3, with scales of 52 to 64). Read the spread after initialization from the first row of `metrics.jsonl` (step 100).

Why `sigma0`: it is the code's resolution. At spread `sigma0`, each dimension can carry up to `0.5 * ln 2` ≈ 0.35 nats through the renderer's noisy channel from the first update, against about 5e-5 at the default spread (Appendix A.6).

This is an initialization, so it fits your constraint of sample-level objectives. The other PushT arms stay uncalibrated to remain paired with your references; P3 tests calibration on PushT directly.

### 3.6 `NCELeWM` (`as_jepa_lewm/nce.py`; P6)

This is the reconstruction-free control. It keeps L2's encoder, pooling, action encoder, prior and rate, and has no renderer.

- **The bound.** The flow loss is replaced by an InfoNCE bound on the information the condition carries about its frame. The condition is the encoded sample at `lam = 0`, or the one-step forecast sample at `lam = 1`.
- **The critic** is the encoder's own Gaussian log-likelihood, `score(t, j) = -||condition_t - mu(x_j)||^2 / (2 sigma0^2)`, so it has no parameters.
- **The candidates** are all `B * T` rendered frames of the batch, including the other frames of the same window, so the bound is capped at `log(B * T)` ≈ 6.8 nats per frame.
- **The loss** per window is `T * InfoNCE + beta * rate`. Both terms are in nats, so `beta = 1` is a calibrated price here.
- **Evaluation.** Nothing renders, so evaluate with planning and probes only (Section 5.1).

### 3.7 Configs and launch

**PushT, arm C1, seed 1.** Other PushT arms change only the fields listed in Section 6.1.

```json
{
  "output": "/efs/prescottsun/SPT-JEPA/runs/round2_20261101/C1-lam1-b0.25-seed1",
  "data": "/efs/prescottsun/SPT-JEPA/data/pusht-lance-full-v1",
  "evaluation_data": "/efs/prescottsun/SPT-JEPA/data/pusht-lance-full-v1/phase0",
  "pose_weights": "/efs/prescottsun/SPT-JEPA/runs/insights12h_20261027/pose-v2/weights.pt",
  "seed": 1, "batch_size": 128, "workers": 4, "cpu_threads": 2,
  "max_steps": 124910, "max_seconds": 31536000, "checkpoint_every": 12491,
  "all_phase_data": true, "gap": 8,
  "retain_steps": [12491, 24982, 37473, 124910],
  "model_class": "forecast_scored",
  "model": {"mode": "L2", "beta": 0.25, "sigma0": 0.1, "window": 8, "anchor_gap": 8,
            "renderer_width": 384, "renderer_depth": 12, "renderer_heads": 6,
            "lam": 1.0, "horizon": "one_step", "forecast_grad": "joint", "forecast_seed": 1},
  "record_initial_weights": true
}
```

**TwoRoom, arm T4, seed 1.** Other T arms change only the fields listed in Section 6.2.

```json
{
  "output": "/efs/prescottsun/SPT-JEPA/runs/round2_hard_20261101/T4-lam1-gapT-seed1",
  "data": "/efs/prescottsun/SPT-JEPA/data/hardtasks-v1/tworoom-metadata",
  "dataset_kind": "benchmark_hdf5",
  "seed": 1, "batch_size": 128, "workers": 4, "cpu_threads": 2,
  "max_steps": 124910, "max_seconds": 31536000, "checkpoint_every": 500,
  "retain_steps": [2000, 6000],
  "calibrate_state_std": 0.1,
  "gap": "<gap_T from Section 5.6>",
  "model_class": "forecast_scored",
  "model": {"mode": "L2", "beta": 0.25, "sigma0": 0.1, "window": 8, "action_dim": 10,
            "renderer_width": 384, "renderer_depth": 12, "renderer_heads": 6,
            "lam": 1.0, "horizon": "one_step", "forecast_grad": "joint", "forecast_seed": 1},
  "record_initial_weights": true
}
```

**Cube** uses the same template with `cube-metadata`, `"action_dim": 25`, `"gap": <gap_K>` and `"retain_steps": [6000, 12000]`. Arms with a first-frame anchor omit `gap`. K3 omits `gap` and sets `"renderer_anchor_mode": "zero"` in `model`.

**Launching:**

- Launch with your staged trainer: `python -B -u -m as_jepa_tools.train_stage --config <config.json> --stop-step <N>`. Continue a stopped run with `--stop-step <new N> --resume`, which is also how to resume after an interruption. `stop_after` acts only on a fresh start.
- PushT: set top-level `gap` and `model.anchor_gap` to the same value; training reads the first and evaluation the second.
- Keep `max_steps = 124910` everywhere, so the learning-rate schedule matches all references.

### 3.8 What does not change

- **Model:** the encoder, `QueryPool`, action encoder, LeWM predictor with the identity carry, and the renderer (384 / 12 / 6).
- **Optimization:** AdamW at 5e-5 and 1e-4, 1% warmup, cosine to `MAX_STEPS`; batch 128; bf16 with the rate in FP32; gradient clip 1.0; one GPU per run.
- **Harder-task data and evaluation:** your `BenchmarkWindows` data, splits, action normalization, renderer targets (64×64 from 224×224 frames), case manifests and evaluators.
- **Planning:** the planner reads encoder means and noiseless prior rollouts, with no renderer and no anchor.

No schedule acts on `lam`, `beta` or `sigma0`.

## 4. Design invariants

Each line is a property the method depends on.

1. The encoder reads one frame and nothing else: no history, anchor or actions.
2. The anchor reaches only the renderer. The prior never reads frames, the anchor or renderer features.
3. Actions reach only the prior. A forecast can carry information about actions only through the prior's mean, which the surprise term prices, plus `sigma0` noise.
4. For each rendered frame, the renderer reads either the posterior sample (probability `1 - lam`) or a `sigma0`-noisy prior sample (probability `lam`). It never reads `mu` or a noiseless prior mean.
5. Window frame 0 is encoded and starts the chain, and is never rendered.
6. AS-JEPA arms use no augmentation, EMA, target network, SIGReg, VICReg, masking or stop-gradient. The only exception is `forecast_grad = "encoder_only"` under the rule in Section 11.
7. Nothing normalizes `mu`, `s`, the prior's mean or the forecast during training. **New:** the initial state scale is set once, at step 0 (Section 3.5).
8. Within a run, `sigma0`, `beta` and `lam` are constant. The rate is the exact one-step Gaussian KL, with one `beta` for entry and surprise. The `rollout` horizon changes only what the renderer reads, never the rate.
9. The renderer's target is fixed data.
10. The coins and forecast noise use their own random stream.
11. **New:** on the harder tasks, the anchor is never a future frame (Section 3.4).

## 5. Evaluation

### 5.1 PushT package

Every comparison with a reference uses the settings the reference was evaluated with. Those are your tools' defaults: probes in BF16 and FP32, and the collapse audit under BF16 autocast. The new `--fp32` option of `collapse_audit.py` adds a second, FP32 verdict for every round-2 arm and its references; report both.

- **At 2 epochs**, run the commands and flags of your `forecastconfirm20261030` gates:

  ```bash
  python -B -u -m as_jepa_tools.planning --checkpoint <ckpt> --output <dir> --upstream <stable-worldmodel> \
      --data-root <data/planning> --test-episodes <pusht-lance-full-v1/phase0/test_episodes.npy> --seed 42 --episodes 50    # also 43, 44
  python -B -u -m as_jepa_tools.collapse_audit --checkpoint <ckpt> --output <dir> --data <phase0> --pose <pose> --cases 32 --steps 20
  python -B -u -m as_jepa_tools.quality_complete --checkpoint <ckpt> --output <dir> --data <phase0> --pose <pose> --i3d <i3d> \
      --cases 64 --samples 1 --steps 20 --selection-seed 20261029
  ```

- **At 3 epochs**, run `as_jepa_tools.completion_evaluate --mode L2 --expected-step 37473`, plus planning on `FRESH_SEEDS`, the forecast checks (5.2), the calibration tool (5.3) and the drift decomposition (5.5).
- **Models without a renderer** (L0, `NCELeWM`): run `as_jepa_tools.planning` and `as_jepa_tools.probes --checkpoint <ckpt> --data <phase0> --output <dir> [--noisy] [--fp32]` directly. `completion_evaluate` renders, so it fails for these models.
  - For L0, also evaluate after your train-only BatchNorm recalibration, as in `native_baseline_20261031`, and report both.
- **Comparisons.** P0 at 3 epochs on the primary cases, pooled over seeds 1 and 2 (300 paired episodes), with `PAIRED_CI` and the per-seed conditions of Section 1.4. At 2 epochs, compare seed by seed with your references.
- **Also report:**
  - `h{1,3,7,13}_{agent,block}_ratio` and `h7_moving_block_ratio`, for mean and sample rollouts;
  - LPIPS and SSIM;
  - `action_identification_accuracy`;
  - the rate per window, `entry` and `surprise`.

  Send us the swap gallery of C1, seed 1, at 3 epochs; it is no longer a gate.

### 5.2 Forecast-based checks for `lam > 0`

At `lam = 1` the renderer never sees encoded states in training, so any test that renders encoded states is out of distribution. Add `--condition {posterior,forecast}`, defaulting to `posterior`, to the two tools below. For `lam > 0` arms, report both conditions; the forecast version is the one that counts.

- **`collapse_audit.py`.** With `forecast`, render each held-out frame `t >= 1` from its one-step forecast sample `m_t + sigma0 * xi`. Here `m_t` is computed from the encoded posterior samples of the earlier frames of the same window and their actions. The existing collapse rule applies.
- **`color_swaps.py` (P5).** With `forecast`, the state swap renders episode B's one-step forecasts with episode A's anchor.

### 5.3 Calibration of the prior

`as_jepa_tools/forecast_calibration.py` takes 32 held-out windows, the model in eval mode and FP32. It reports `calibration_slope`: the least-squares slope of the true change `mu_t - s_(t-1)` on the predicted change `m_t - s_(t-1)`, pooled over dimensions.

- A slope of 1 means calibrated; below 1 means the predicted changes are exaggerated.
- Evaluate C1, D3 and E2 at epochs 2 and 3, and C1's reference (original) with the same tool.

### 5.4 Dot and colour

- **Dot (P4).** `probes.py --noisy --fp32` on the dot domain reports R² for `dot_x`, `dot_y`, `dot_r`, `dot_g` and `dot_b`. Decide on the largest of the five, and report all of them plus the dynamics probes.
- **Colour probe (P5).** The same tool on the colour domain reports R² for the three gains and three biases.
- **Colour invariance (new tool, P5).** For 512 held-out frames, encode each frame under its own palette A and under the palette B of a different random test episode. Report:
  - the **invariance index**, `E||mu(x_i^A) - mu(x_i^B)||^2 / E||mu(x_i^A) - mu(x_j^A)||^2`, with `j != i`. Near 0 means colour-invariant; near 1 means a palette change moves the state as much as a pose change does;
  - the **recoloured-goal retrieval** accuracy: the fraction of frames whose `mu(x_i^B)` is nearest to `mu(x_i^A)` among the 512 frames under palette A.

  For L0, use its embedding.

### 5.5 Open-loop drift (PushT)

Run `as_jepa_tools/decode_decomposition.py` (`--cases 128 --steps 50`, with and without `--fp32`) at 3 epochs for C1 and its references (original, forecast-only, TF). Report the block error from open-loop forecasts and from the frame's own encoded state at h1, h7 and h13, and their difference, the **drift**. For C1, the own-state render is out of distribution, so the forecast column is primary.

### 5.6 Dataset checks before harder-task training (CPU)

The gap rule needs the share of windows without an eligible past anchor. The Cube hypothesis needs to know how often an anchor shows a stale cube. `benchmark_gap.py` computes both from the stored physical labels, which never reach a model:

```bash
python -B -m as_jepa_tools.benchmark_gap --data <hardtasks-v1/tworoom-metadata> --gaps 0 1 2 4 8 16 \
    --label agent:pos_agent:position:<AGENT_TOL> --output tworoom-anchor-table.json
python -B -m as_jepa_tools.benchmark_gap --data <hardtasks-v1/cube-metadata> --gaps 0 1 2 4 8 16 \
    --label cube:privileged_block_0_pos:position:0.03 --label cube_yaw:privileged_block_0_yaw:yaw90:15 \
    --label effector:proprio_effector_pos:position:0.03 --output cube-anchor-table.json
```

For each gap (0 is the first-frame anchor), the table reports two things:

- **`fallback_share`:** the share of windows without an eligible past anchor.
- **Staleness, per label:** the share of rendered frames (window frames 1 to 7) in which the anchor's value differs from the frame's by more than the tolerance.

A constant anchor counts as stale for every label.

**Rule.** `gap_T` and `gap_K` are the largest gap in {1, 2, 4, 8, 16} with `fallback_share <= FALLBACK_MAX`. Before running, confirm that `privileged_block_0_pos` is in metres and the yaw in radians, and confirm the units of `AGENT_TOL`. Record both tables in the dataset cards and the reports. They are evidence in their own right: the Cube table tests the reading of Section 0.5 before any training.

### 5.7 Harder-task health and the use gate

**Health.** Run `benchmark_health.py --cases 16` at every evaluated checkpoint. It evaluates the step in `last.ckpt`, so run the 2,000-update health check before any `--resume`. Add two conditions:

- **`one_step`:** the one-step forecast samples `m_t + sigma0 * xi`, from the window's own encoded posterior samples and actions;
- **`one_step_shuffled`:** the same, from the other video's encoded states, with the window's own actions and the same draw of `xi`.

Report, for every arm:

- the validation `feature_signal_to_posterior_noise` (that is, `feature_std / sigma0`), in FP32;
- every swap effect: the relative increase of the renderer's error at t = 0.95 for `shuffled_posterior` against `posterior`, `one_step_shuffled` against `one_step`, and `wrong_state` against `forecast`;
- the wrong-action and zero-state effects; and, from `benchmark_information_audit.py`, the wrong-anchor effect (`forecast_wrong_anchor`).

The **use effect** of an arm is the swap effect on the condition it trains on: `posterior` for `lam = 0`, `one_step` for `lam = 1`, `forecast` for T5. Compare T5 with T3 on the shared `one_step` condition as well. Also plot the training-log `feature_std / sigma0` every 100 steps.

**Use gate (TwoRoom, at `TWOROOM_GATE`).** An arm passes if both seeds have `feature_std / sigma0 >= SPREAD_MIN` and a use effect of at least `USE_MIN`. Arms that pass continue to `TWOROOM_FULL`. Arms that fail stop there; they are diagnostic results.

### 5.8 Harder-task planning and content

- **Planning:** `benchmark_evaluate.py` on the primary and fresh manifests under `WHITE_CEM`; then `CORR_CEM` with your `action_covariance_20261031/evaluate.py` and the frozen calibration files.
  - `CORR_CEM` needs per-checkpoint preparation: its hash-pinned `--spec` must carry the same checkpoint's white-CEM result as `reference_metrics`, and `prepare.py` hard-codes its model list. So run white CEM first and build one spec per checkpoint.
  - TwoRoom: every arm at 2,000 updates, and the continuing arms at 6,000. No correlated-CEM reference exists for AS-JEPA TwoRoom arms at 2,000 updates; compare those across our arms only.
  - Cube: every arm at 12,000.
  - Report success per seed and case set; call differences below `HARD_TIE` ties.
- **Content probes.** `benchmark_information_audit.py`, with clean and `sigma0`-noisy linear probes of the agent's position (TwoRoom), and of the cube's and effector's positions (Cube). For the Cube readout in centimetres and degrees, use your probe in `experiments/carrier_progress_20261031/tf_pose.py`, given a checkpoint argument, with the same 128 training and 64 validation clips, ridge and codebook. Compare clean with clean (TF 4.7 to 5.1 cm, native 3.7 to 3.9 cm). Add the noisy readout for the K arms and, by rerunning the probe, for TF and native.
- **Cube goal geometry.** Score each K checkpoint on the candidate pool of your TF audit (`CUBE-ORACLE-AUDIT.md`), then summarize against TF:

  ```bash
  python -B -m as_jepa_tools.cube_oracle_audit score --manifest <same manifest> --data <cube-metadata> \
      --pool <your TF audit pool> --checkpoint <ckpt> --label K1-seed1 --output <dir>
  python -B -m as_jepa_tools.cube_oracle_summary --scores K1-seed1=<dir> --scores TF-seed1=<TF scores> \
      --reference TF-seed1 --data <cube-metadata> --output <dir>
  ```

  Report `position_encoded_partial_rank` (the cube) and `effector_encoded_partial_rank` (the arm) for the non-oracle candidates. Both are computed on the encoded-goal cost: the latent distance from each candidate's reached endpoint to the goal.

### 5.9 What every report records

For every run, record:

- its config and source hash;
- `gap`, `lam`, `horizon`, `forecast_grad`, `beta`, `sigma0`, `calibrate_state_std` and `seed`;
- the evaluated checkpoint step and GPU-hours;
- the metrics above;
- PushT only: the collapse verdicts (posterior and forecast, BF16 and FP32) and the calibration slope;
- harder tasks only: `initial_scale.json`, the health report and `GapBenchmarkWindows.fallback_share()` for the run's gap.

Write one JSON record per run. Tables give per-seed values, the mean over seeds, and for PushT `PAIRED_CI` with both per-seed differences.

## 6. Run plan

Run IDs follow `r2-<arm>-seed<k>`.

### 6.1 Part P: PushT closure

All L2-family arms use the PushT template (Section 3.7), uncalibrated except P3, with gap 8 as in the references.

| Block | Arm | Change from the template | Seeds and budget | Compared with |
| --- | --- | --- | --- | --- |
| P1 | `L0-full` | your `full-L0-10epoch-seed{0,1,2}` configs, unchanged except output and `retain_steps` (`L0_EPOCH_UPDATES` × 2, 3 and 10) | 0, 1, 2; 10 LeWM epochs | C1 and the references, at 3 epochs and at 10 |
| P2 | C1 | none (`lam = 1`, `one_step`, `beta = 1/4`) | 1, 2 to `SCREEN_STEPS`; 3, 4 to `EPOCH2_STEPS` | original, forecast-only, TF (3 epochs); original, lower-beta, TF seeds 3, 4 (2 epochs) |
| P3 | `cal` | `lam = 0` (the same class computes L2's loss exactly), plus `"calibrate_state_std": 0.1` | 3, 5 to `EPOCH2_STEPS` | original seeds 3, 5 at 2 epochs (P0 22, 10) |
| P4 | D1, D3 | `model.domain = "dot"`; D1 `lam = 0`, D3 `lam = 1` | 1; `SCREEN_STEPS` | each other |
| P5 | E2 | `model.domain = "color"`, `lam = 1` | 1; `SCREEN_STEPS` | E4 |
| P5 | E4 | your `full-L0-10epoch` config with `domain = "color"` | 1; `3 * L0_EPOCH_UPDATES` | E2 |
| P6 | I2 | `model_class: nce`, `lam = 1`, `beta = 1` (nats) | 1; `SCREEN_STEPS` | C1 |

**P3, first step, free.** Before P3 runs, read `feature_std / sigma0` from the training logs (`metrics.jsonl`) of the original arm's seeds 1 to 8 over their first two epochs. If the failing seeds (3, 5, 6, 7) reach `SPREAD_MIN` later than the working seeds (1, 4, 8), or never, the PushT instability looks like the TwoRoom trap, and P3 tests the remedy. Run P3 either way; the log check sharpens the reading.

**Conditional arms** (triggers in Section 8.1):

- **C5:** `lam = 0.5`, `horizon = "rollout"`, seeds 1 and 2, `SCREEN_STEPS`. It differs from forecast-only only by the `sigma0` noise on each forecast step.
- **C2:** `lam = 1`, `beta = 1/16`, seeds 1 and 2 to `SCREEN_STEPS` and seeds 3 and 4 to `EPOCH2_STEPS`.
- **E1:** colour domain at `lam = 0`, seed 1, `SCREEN_STEPS`.
- **C1 continuation:** seeds 1 and 2 to `MAX_STEPS`, with `as_jepa_tools.resume_lewm`.

### 6.2 Part T: TwoRoom

**T-pre (CPU).** The anchor table and `gap_T` (Section 5.6).

**T-gate.** Seeds 1 and 2, to `TWOROOM_GATE`. Every arm uses `calibrate_state_std = 0.1` and `beta = 1/4` unless stated.

| Arm | Class | `lam` | Anchor | Other | Differs from |
| --- | --- | --- | --- | --- | --- |
| T1 | `forecast_scored` | 0 | first frame | | your original arm at 2,000: only the initial scale |
| T2 | `forecast_scored` | 0 | past, `gap_T` | | T1: only the anchor |
| T3 | `forecast_scored` | 1 | first frame | | T1: only `lam` |
| T4 | `forecast_scored` | 1 | past, `gap_T` | | canonical; T2: only `lam`; T3: only the anchor |
| T5 | your TF config (`RobustLeWM`, `rho = 0.8`, `forecast_fraction = 0.5`) | | first frame | | your TF arm at 2,000: only the initial scale |
| T6 | `forecast_scored` | 1 | past, `gap_T` | `beta = 1/16` | T4: only the price |

**Evaluation at 2,000 updates:** health and the use gate (Section 5.7), probes, and planning under both controllers.

**T-continuation.** Every arm that passes the use gate continues to `TWOROOM_FULL` (`--stop-step 6000 --resume`). Evaluate it with health, probes and planning under both controllers.

### 6.3 Part K: Cube

**K-pre (CPU).** The anchor table and `gap_K` (Section 5.6).

**K-main.** Seeds 1 and 2, to `CUBE_FULL`. Every arm uses `forecast_scored`, `calibrate_state_std = 0.1` and `beta = 1/4`.

| Arm | `lam` | Anchor | Differs from |
| --- | --- | --- | --- |
| K1 | 1 | past, `gap_K` | canonical |
| K2 | 1 | first frame | K1: only the anchor (the rule of your runs) |
| K3 | 1 | constant (`renderer_anchor_mode: "zero"`) | K1: only the anchor (none) |
| K4 | 0 | past, `gap_K` | K1: only `lam` |

**Evaluation at 12,000 updates:** health, content probes, the oracle audit, and planning under both controllers (Section 5.8).

**K-mature (conditional, Section 8.3).** The best K arm, both seeds, from 12,000 to `CUBE_MATURE`.

### 6.4 Counts, GPU-hours and priorities

| Part | Runs | Approximate GPU-hours (training) |
| --- | --- | --- |
| P1 | 3 L0 runs of 10 LeWM epochs | 25 to 50 (no renderer; native LeWM ran 0.46 to 0.56 s per update on Cube; measure in the smoke) |
| P2 | 2 screens + 2 two-epoch runs | 19 + 12.6 |
| P3 | 2 two-epoch runs | 12.6 |
| P4 | 2 screens | 19 |
| P5 | 1 screen + 1 short L0 run | 9.5 + 2.5 |
| P6 | 1 short run without a renderer | about 3 |
| **Part P** | | **about 103 to 128** |
| T-gate | 12 runs to 2,000 updates | about 10 |
| T-continuation | up to 12 runs from 2,000 to 6,000 | up to 20 |
| K-main | 8 runs to 12,000 updates | about 40 |
| **Parts T and K** | | **about 70** |
| Conditional: C5, C2, E1, C1 continuation, K-mature | | 19, 31.6, 9.5, 44, about 10 |

Evaluation GPU time is extra; measure it and report it. Everything without a dependency may start at once. When compute is short, run in this order:

1. T-pre and K-pre (CPU), the P3 log check, T-gate, P1, and P2 seeds 1 and 2.
2. K-main and P4.
3. P3, P2 seeds 3 and 4, P5 and P6.

Before launching, report the measured step times and projected GPU-hours. Wait for our acknowledgement if they exceed this table by more than 25%.

## 7. Frozen predictions (record them, with the commit hash, before launch)

Each prediction states what the theory predicts and how confident we are. A failed prediction is a result, not an error.

**PushT**

1. **P1 (moderate).** L0 at 10 epochs reaches at least 90% P0 on the primary cases (published: 96 ± 2.8). No prediction at 3 epochs.
2. **P2 (moderate).** C1 is non-inferior to forecast-only and better than original at 3 epochs. At 2 epochs, C1 seeds 3 and 4 each reach at least 70%, where original scored 22 and 76.
3. **Calibration (high).** C1, D3 and E2 have an eval-mode slope of at least `SLOPE_MIN` at epochs 2 and 3.
4. **P3 (moderate).** Both calibrated seeds reach at least 70% P0 at 2 epochs, where uncalibrated original scored 22 and 10. If they do, the original arm's instability was a delayed escape from the trapped state.
5. **P4 (high).** The largest dot R² is at most `CHANCE_R2` in D3 and above it in D1.
6. **P5 (moderate).** E4 keeps colour (R² of at least 0.9) and has a higher invariance index than E2. E2's colour R² is at most 0.6; a round-1 anchor run measured per-channel R² of 0.23 to 0.61 (seed 0, FP32).
7. **P6 (moderate).** I2 keeps the agent (probe R² of at least 0.8). Its block probe and P0 fall below C1's, because the bound saturates once the agent alone identifies the frame.

**TwoRoom**

8. **The initial scale (moderate to high).** T1, T2, T3, T4 and T6 pass the use gate at 2,000 updates. Their uncalibrated counterparts had `feature_std / sigma0` of 0.0015 and posterior-swap effects of at most 0.0003% at t = 0.95.
9. **T5 (low).** T5 passes the spread criterion, but its swap effect on the shared `one_step` condition is below T3's, because the noiseless open-loop forecast still offers the renderer an action path.
10. **Anchor on TwoRoom (moderate, conditional on T-pre).** If the agent is stale relative to the first-frame anchor in more than 80% of rendered frames, T1 and T2, and likewise T3 and T4, differ by less than 0.1 in noisy agent-probe R². The anchor matters for slow components, and the agent is not slow.
11. **Planning (moderate).** At 6,000 updates, the best continuing arm reaches at least 60% on the primary cases, averaged over seeds. For reference: native LeWM 80 / 72%, fixed carrier 90 / 94%, uncalibrated AS-JEPA at most 16%.

**Cube**

12. **K-pre (moderate).** With the first-frame anchor, the cube is stale in fewer than 30% of rendered frames, and the effector in more than 60%. With `gap_K`, the cube's stale share rises but stays below 50%.
13. **Cube content (moderate).** The cube partial rank orders K3 ≥ K1 > K2. K2 stays at or below `RANK_LOW`, near TF's 0.03, and K3 reaches at least `RANK_HIGH`. The clean cube-position readout error of K1 and K3, from the same probe and clips, falls below TF's 4.7 to 5.1 cm.
14. **`lam` on Cube (moderate).** At the same anchor, K1 has higher cube content than K4, by cube rank and probe, because the prior must forecast contacts.
15. **Cube planning (low).** The best K arm is within `HARD_TIE` of native LeWM at 12,000 updates under the same controller.

## 8. Decision rules

### 8.1 PushT

| Result | Then |
| --- | --- |
| C1 is non-inferior to forecast-only and better than original, and seeds 3 and 4 hold | The canonical `lam = 1, one_step` stands on PushT, and the harder tasks keep it. |
| C1 is worse than forecast-only | Run C5. If C5 is non-inferior to forecast-only, the open-loop horizon carries the gain: report it, and switch the `lam = 1` arms of any later harder-task run to C5's setting. If C5 is also worse, the noise or the pricing costs planning; report before deciding anything else. Never adopt noiseless forecasts into AS-JEPA. |
| C1 holds on seeds 1 and 2 but one of seeds 3 and 4 falls below 70% | Run C2. Lower-beta was the stable `lam = 0` arm across eight seeds. |
| L0 at 3 epochs is non-inferior to C1 | AS-JEPA has no planning advantage on PushT at matched data. The story's PushT claims rest on P3 to P6. |
| C1 at 3 epochs trails L0 at 10 epochs by more than 5 points | Propose the C1 continuation to 10 epochs, to compare at matched epochs, and wait for our agreement. |
| P3 reaches 70% on both seeds | Calibration becomes part of the recipe on every dataset. A later PushT table uses calibrated arms. |
| P3 does not help | The PushT instability has another cause. Report the log check. |
| D3 keeps dot information | Check `forecast_share` and that the renderer really reads the forecast. If the implementation is correct, `lam` loses its main theoretical support; report before anything else. |
| E2's colour R² is at least 0.9 | Run E1, to see whether `lam` or the anchor fails to exclude appearance. |
| E4 does not keep colour | The appearance comparison with LeWM does not hold on this domain; report it. |

### 8.2 TwoRoom

| Result | Then |
| --- | --- |
| T1 passes the use gate and plans at least 30% by 6,000 updates | The initial scale explains your TwoRoom failure. Calibration becomes part of the recipe; tell your campaign. |
| T1 fails, but T3 or T4 passes | The forecast reward is needed in addition to the scale. |
| T5 passes and plans like T3 | TF's failure was the initial scale; the noiseless shortcut was not decisive. |
| T5 fails while T3 passes | The noiseless forecast shortcut is real once the state can carry information, and priced forecasts close it. |
| T1, T2, T3, T4 and T6 all fail the use gate | The initial scale is not enough, and the pooled readout is the bottleneck (your quadrant and carrier evidence). Stop Part T and report; this is meeting point (b). |
| An arm passes the use gate but plans below 30% at 6,000 updates | The state carries the agent but planning fails. Report the agent probe, the health table and the planning logs, and stop for a joint diagnosis before any further training. (No TwoRoom candidate audit exists; `cube_oracle_audit.py` is Cube-only.) |

### 8.3 Cube

| Result | Then |
| --- | --- |
| K-pre: the cube is stale in at most 50% of rendered frames at `gap_K` | A past anchor can only partly remove the free cube pose, so K3 is the decisive arm. Run all four arms anyway. |
| K1 or K3 reaches `RANK_HIGH`, and K2 stays at or below `RANK_LOW` | The anchor was hiding the cube from the state. Adopt the better rule as the Cube recipe; this is meeting point (c). |
| K1, K2 and K3 all stay at or below `RANK_LOW` | The anchor is not the cause. Report the content probes. Candidates outside this round: the cube's pixel footprint at 64×64, and the orientation that no model reads. No rescue runs. |
| The best K arm is within `HARD_TIE` of native LeWM at 12,000 updates, under either controller | If your maturity runs show native LeWM gaining more than `HARD_TIE` between 12,000 and 23,342 updates, propose K-mature and wait for our agreement; then compare at 23,342. |
| The best K arm trails native LeWM by more than `HARD_TIE` | Report it with the oracle audit. The Cube performance gap remains open; no automatic extension. |

## 9. Reports

- **PushT closure report:** every Section 5.1 to 5.5 metric per arm, predictions 1 to 7 marked held or failed, the Section 8.1 decisions, and any conditional arm that ran.
- **TwoRoom report:** the anchor table, `gap_T`, `initial_scale.json` values, spread curves for every arm, the health table, the use-gate verdicts, probes, planning at 2,000 and 6,000 updates under both controllers, predictions 8 to 11, and the Section 8.2 decisions.
- **Cube report:** the anchor table, `gap_K`, health, content probes, oracle-audit ranks, planning under both controllers, predictions 12 to 15, and the Section 8.3 decisions.

Each report lists the references it compares against, with their paths.

## 10. Tests and checks before any long run

1. **CPU contracts, 16 tests.** `tests/as_jepa/test_round2.py` (11) and `tests/as_jepa/test_benchmark_gap.py` (5). Run from the repository root:

   ```bash
   PYTHONPATH=.:runtime:<stable-worldmodel checkout> python -B -m unittest tests/as_jepa/test_round2.py tests/as_jepa/test_benchmark_gap.py -v
   ```

   They check:
   - `lam = 0` reproduces `ASJepaLeWM.loss` exactly; the main random stream is identical across `lam`; `forecast_share` is about `lam`;
   - gradient routing: at `lam = 0` the flow loss never reaches the prior; under `joint` it reaches encoder and prior; under `encoder_only`, the encoder only;
   - at `lam = 1` the last frame is not rendered from its own encoding; the calibration slope is exact for a known overshoot;
   - the contrastive variant has no renderer, and its gradients reach encoder and prior;
   - `gap = 0` reproduces `BenchmarkWindows` exactly;
   - the gap anchor is a past, phase-aligned frame of the same episode, read with the window frames' transforms, deterministic within an epoch and redrawn across epochs, with fallbacks flagged;
   - the dataset dispatch follows the config;
   - the staleness table counts moving and static labels correctly.

   Also rerun your existing `tests/as_jepa` suite.
2. **Pairing check.** C1's recorded initial weights at seeds 1 and 2 must equal those of `lower-beta-seed1` and `lower-beta-seed2`, tensor for tensor. If they differ, stop and report.
3. **GPU smokes (200 steps each, one GPU).** Run C1 (PushT), L0 (PushT, P1), T4 (TwoRoom, with calibration and `gap_T`), K1 (Cube, gap anchor: the anchor adds one HDF5 read per window), K3 (Cube, constant anchor) and I2. Record step time and peak memory. Check that:
   - the loss is finite;
   - `forecast_share` is about `lam`;
   - `initial_scale.json` shows `original_std` of about 1e-3 to 2e-3, and the first `metrics.jsonl` row (step 100) shows `feature_std` near `sigma0`;
   - for 64 training windows of T4 and K1, either `anchor_row <= rows[0] - gap * 5` or `anchor_fallback` is True, and `fallback_share()` matches the table of Section 5.6;
   - `benchmark_evaluate.py` loads the T4 checkpoint through its `_target_`.

## 11. Failure responses

| Symptom | Likely cause | Response |
| --- | --- | --- |
| A `lam > 0` arm has an eval-mode calibration slope below `SLOPE_MIN` at epochs 2 and 3 | joint training exaggerates the forecasts | Rerun that arm on every seed it has with `forecast_grad = "encoder_only"`, and report both. |
| A T arm's `feature_std / sigma0` falls back below `SPREAD_MIN` after starting near 1 | the rate pulls the state down faster than the renderer rewards it | Rerun T4 once with `beta = 0` from the same initialization. If the state survives, the price is above TwoRoom's barrier at `beta = 1/4`; report. |
| Every T arm fails the use gate | Section 8.2 | Stop Part T; report the spread curves and probes. |
| The anchor table gives `gap_T` or `gap_K` = 0 (no gap meets `FALLBACK_MAX`) | short episodes | Use gap 1 and report its fallback share; do not lower the standard silently. |
| A PushT `lam > 0` arm fails the forecast collapse test | the price is above the collapse barrier | Exclude it from comparisons and report. No rescue runs. |
| The BF16 and FP32 collapse verdicts disagree | BF16 makes an empty state look full | Use the FP32 verdict for that arm and its reference. |
| NaN or loss spikes | numerics | Keep the rate in FP32 and the gradient clip. If spikes persist, halve the learning rate for that arm only, mark its comparisons as unpaired, and report. |
| A run is interrupted | `stop_after` does not act on resume | Resume with `train_stage --config <config> --stop-step <N> --resume`. |

## 12. Latitude: what you may change without asking

- Compute allocation and the order of runs, within the priorities of Section 6.4.
- Evaluation sample counts, upward, applied to an arm and its references alike.
- Adding seeds to any arm, applied to its paired references too.
- Implementation mechanics not fixed here: launch wiring, report scripts, file layout of outputs.

Keep one GPU per run and the references' learning rates. Everything in Sections 3, 4 and 8 needs our agreement before it changes: what the renderer reads, the noise, the pairing, the rate, the anchors, the initial scale and the decision rules. When unsure, stop and ask.

## 13. Order of work

1. Branch, copy the reference files and register them (Section 2.1). Run the 16 CPU tests and the pairing check (Section 10).
2. Run T-pre and K-pre (CPU) and the P3 log check, then fix `gap_T` and `gap_K`.
3. Add the evaluation options (Sections 5.2, 5.3 and 5.7). Run the GPU smokes, then report step times and projected GPU-hours.
4. Launch by the priorities of Section 6.4: T-gate, P1 and P2 first. Parts P, T and K run in parallel.
5. Evaluate as Section 5 describes. Continue the T arms that pass the use gate to 6,000 updates.
6. Write the three reports (Section 9), and stop for our agreement.

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

At `beta = 1` this is the negative evidence lower bound of a generative model in which each frame is decoded either from its inferred state or from the model's own forecast, chosen by a coin. So every `lam` is a code length, and `lam = 0` is round 1.

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
| Detail fresh in each frame (the dot) | `beta` per nat | `1 - lam` per nat; no forecast can predict it | Excluded if and only if `beta >= 1 - lam`, so at every `beta > 0` when `lam = 1`. Proved for an anchor outside the encoded chain. Condition dropout `p = 0.1` shifts the threshold to `(1 - p)(1 - lam)`: 0.9 at `lam = 0`. |
| Persistent appearance (colour) | `beta` per nat added after the first state | 0 in both branches; the anchor supplies it | Excluded at every `beta > 0`. Proved for codes that keep appearance separate from dynamics; the general case at `lam > 0` is open. |
| Dynamics state (poses) | entry once, then surprise | both branches, but only for what the anchor does not show (A.7) | Kept when it pays for its entry within the window (round-1 condition A5), at every `lam`. |
| Fixed property that the dynamics read and the anchor shows (block shape, when varied) | entry once | at `lam = 0`: 0. At `lam = 1`: its value for forecasting contacts, because the prior cannot see the anchor | Not stored in an exact `lam = 0` code; stored at `lam = 1` (toy gains of 1.3 to 3.0 nats). Not tested in this round (Appendix C). |

On the line `beta + lam = 1`, the incentive to store detail unique to one frame is zero. Round 1's canonical point (`beta = 1, lam = 0`) and the sufficiency point (`lam = 1`, `beta` toward 0) are its two ends.

### A.4 What `lam = 1` optimizes

At `lam = 1`, the objective selects the smallest state whose one-step forecast, made without seeing the anchor and rendered with the anchor, is most informative about the next frame. This is a sequential, anchor-conditional relative of the past-future information bottleneck (Creutzig, Globerson and Tishby 2009). Under deterministic dynamics given the actions, maximizing it makes the prior's context sufficient for the future; under stochastic dynamics, one forecast sample guarantees less.

The sufficiency claim is about the pair (prior context, anchor), not about `s_t` alone:

- `s_t` is what one frame shows of the dynamics;
- the prior's context adds velocity and anything inferred from history;
- the anchor carries appearance.

### A.5 Known costs, and what watches them

- **Exaggerated forecasts under joint training.** The renderer values forecast information at `lam` per nat, while the prior pays for exaggeration only to second order in surprise. The toys give a 0.5 to 3.5% overshoot with fixed variances. The eval-mode calibration slope watches for it; the fallback is the encoder-only forecast gradient.
- **The noiseless-mean shortcut.** A noiseless forecast mean is a channel of unbounded capacity that nothing prices, so the prior can pass information about the actions to the renderer for free. With a first-frame anchor, this admits an empty state plus a prior that integrates the actions. Your TF arms on TwoRoom show the signature: action effect up to 72%, while no state swap exceeded 0.005%. A `sigma0`-noisy sample limits the channel to the code's resolution, and the surprise term prices the mean's departure from the encoded state.
- **Out-of-distribution tests.** At `lam = 1` the renderer never sees encoded states; hence the forecast versions of the renderer diagnostics (Sections 5.2 and 5.7).
- **No branch flag.** Without a flag saying which branch it reads, the renderer pays at most `ln 2` nats per frame when `0 < lam < 1`, which is negligible against a frame's 12,288 rendered values.

### A.6 Why a reconstruction objective can still leave the state unused

**Collapse in a JEPA.** The latent prediction loss `E||g(f(x_t), a_t) - f(x_t+1)||^2` is exactly zero for a constant encoder `f`. The collapsed solution is a global minimum, so something outside the prediction loss (EMA with stop-gradient, VICReg, SIGReg) must exclude it.

**AS-JEPA's objective has no such minimum.** A constant state leaves the renderer only the anchor. When the frames contain changes the anchor does not show, a state that encodes them lowers `J` (A.2). The trapped TwoRoom state is instead a stationary point of training:

1. The renderer reads `c = mu + sigma0 xi`. Per dimension, `c` carries at most `0.5 ln(1 + Var(mu) / sigma0^2)` nats about `mu`. At the default initial spread of about 1e-3 that is about 5e-5 nats per dimension; at the trapped spread of 1.5e-4, about 1e-6 nats per dimension, or 7e-5 nats per frame over 64 dimensions.
2. A renderer cannot use what it cannot read, so it learns to ignore `c`, and the gradient of the loss with respect to `c` is about 0.
3. The encoder's gradient from the renderer passes through that same derivative, so `mu` does not grow. With `beta > 0`, the rate also pulls `mu` toward the prior mean. On TwoRoom the encoder's gradient was sizeable only in the first 600 updates, when the entry term's excess fell from 2.8 to 7.5 nats to about 0.1; by 2,000 to 2,600 updates it had fallen to 0.002 to 0.005.

The point is unstable, but it escapes slowly. In your linear toy at `beta = 0`, the escaping direction grows at about `2 eta^2 / sigma0^2` for state scale `eta`. At the default initial scale (`eta / sigma0 ≈ 1e-2`) that is a ten-thousandth of its value at `eta = sigma0`, and at the trapped scale about two millionths. More moving pixels make the escape faster at any scale, which is our reading of why Cube (31.6% moving pixels) escaped and TwoRoom (9.4%) did not. That reading is an inference: the toy has not been computed with TwoRoom's numbers.

So the objective's optimum is not the issue; the starting point is. SIGReg acts at every step by holding each embedding dimension at unit variance across the batch, so LeWM's embedding never sits at this scale. AS-JEPA's counterpart is an initialization: start `mu` at the code's resolution, a spread of `sigma0`, where each dimension can carry `0.5 ln 2` ≈ 0.35 nats from the first update. It adds no term to the loss and nothing that acts during training.

The general phenomenon is known for VAEs as posterior collapse, or an inactive latent: a decoder that has other inputs (here the anchor and the noisy target) never learns to read a latent whose signal starts below its noise (Bowman et al. 2016; He et al. 2019). It also explains why the first-frame anchor and the forecast shortcut are secondary on TwoRoom. They account for your TF arms learning to read actions, but not for the `lam = 0` arms, which have no forecast path, leaving the state unused.

### A.7 What an anchor gives away

At `lam = 0`, the reward for storing a component in `s_t` is `I(x_t; s_t | x_anchor)` (A.2): only what the frame shows beyond the anchor. A component whose value in frame `t` equals its value in the anchor earns nothing in that frame. A component that is a function of others already in the state, such as a carried cube following the gripper, earns nothing either. Round 1's condition A5 gives the consequence: a slow component enters the state only if it changes often enough, between the anchor and the rendered frames, to pay its entry. On PushT, the gap of 8 kept frames was chosen for the block.

Three anchor rules differ in how much of a slow component they give away:

- **First frame:** everything that does not change within the window. On Cube that includes the cube in most windows, and the oracle audit (cube rank 0.03, arm rank 0.93) is what this looks like at planning time.
- **A past frame at gap `g`:** everything that has not changed since the anchor time. The K-pre table measures how often that is the cube.
- **Constant (no anchor information):** nothing. This suits datasets whose appearance does not vary across episodes, such as TwoRoom and Cube, because the renderer can learn the constant scene.

The design principle: the anchor should share the episode's appearance and be as uninformative as possible about the current value of anything that moves. A future anchor violates this differently, by showing where slow objects end up, and generation never has one.

At `lam = 1`, a component also earns its value for the forecast. The prior cannot see the anchor, so to forecast whether the gripper closes on the cube, it needs the cube's pose in the state.

## Appendix B. Design-to-plan trace

| Design point | Reason | Where | Checked by |
| --- | --- | --- | --- |
| Per-frame encoder | the state is a function of its frame; goal images encode without history | `ASJepaLeWM.encode`, unchanged | existing `test_contracts.py` |
| Anchor only to the renderer | appearance is free side information and must not be free in the state | unchanged | existing contracts |
| Past-only anchor on the harder tasks, chosen by a stated rule | the anchor must not show the current value of slow components (A.7), nor the future | `benchmark_gap.py`; Section 5.6 | `test_benchmark_gap.py`; T2 against T1, K1 to K3 |
| Actions only to the prior | the renderer must not do the dynamics | unchanged; forecasts carry actions only through the priced mean plus noise | `test_gradient_routing` |
| The renderer reads the posterior sample or a noisy prior sample, never a noiseless mean or `mu` | sufficiency (`lam`), and a priced channel | `ForecastScoredLeWM.loss` | `test_lambda_zero_reproduces_l2_loss`, `test_forecast_share`, `test_lambda_one_never_renders_a_frame_from_its_own_encoding` |
| Initial state spread `sigma0`, set once | start where the renderer can read the state (A.6) | `calibrate_state_std` in `train.py` | `initial_scale.json` in the smoke; T1 against your original arm; P3 |
| Frame 0 never rendered | transient exclusion | unchanged | existing contracts |
| No stop-gradient by default | one objective, no heuristics | `forecast_grad = "joint"` | `test_gradient_routing`, `test_forecast_gradient_routing` |
| One exact rate, one `beta`, `sigma0` fixed in a run | the rate is a code length | unchanged | `test_kl_functions` |
| Paired random streams | clean comparisons across `lam` | `fork_rng` stream | `test_main_random_stream_is_paired_across_lambda`; pairing check |
| Contrastive critic (P6) | a reconstruction-free control | `nce.py` | `test_no_renderer_and_gradients` |

## Appendix C. Changed from revision 1

- **Dropped from PushT:**
  - **Block A1 and A2** (the price at the working point): the harder tasks test `beta` ∈ {1/16, 1/4} directly.
  - **A4** (full L1).
  - **Block B** (`sigma0 = 0.03`).
  - **Block F** (the first-frame anchor on PushT): now tested where it matters, in Parts T and K.
  - **Block G** (block-shape variants): it needed new data and a new pose estimator. The fixed-property prediction waits for a dataset where such properties vary naturally.
  - **Blocks H and H2** (finals and stability): replaced by C1's four seeds and P3.
  - **Arms C3 and C4, E1, E3, I1 and I3.** C2, C5 and E1 remain as conditional arms.
  - **The gallery gate:** the gallery is still sent.
- **Kept unchanged:** `ForecastScoredLeWM`, `NCELeWM`, their 11 tests, and the PushT evaluation tools. `learned_variance.py` stays in `round2/` for a later round and is not used.
- **Phase 3 replaced by Parts T and K**, which start now:
  - the first-frame anchor is replaced by a past-only anchor, plus a constant anchor on Cube;
  - the initial state scale moves from one arm to the recipe, with T1, T5 and P3 testing it;
  - the arms are designed from your TwoRoom and Cube evidence.
- **The theory gains A.6 and A.7.** Both come from your harder-task evidence.
