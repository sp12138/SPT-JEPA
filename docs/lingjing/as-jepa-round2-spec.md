# AS-JEPA round 2: design exploration on PushT, TwoRoom and Cube (implementation and experiment spec, handoff)

Written 2026-10-31 for the collaborator who runs the L-line experiments; this is revision 3, from the same day. Follow it as written. Section 12 lists what you may change without asking; anything else needs our agreement first.

- Revision 1 (`937707e`) planned a long PushT program.
- Revision 2 (`8d19e26`) reduced PushT to a closing set and added TwoRoom and Cube parts.
- Revision 3 (this one) defers the dot and colour domains to a later round and completes the design exploration. It covers what the renderer reads (`lam`, forecast noise, one-step or open-loop), the posterior noise structure (`rho`) and the code resolution (`sigma0`), on PushT and on the two new datasets. It also starts with a free diagnostic, on your existing checkpoints, of a mechanism that may contribute to your PushT results (Section 0.6, Appendix A.8).

The spec builds on your branch `research/ideas-20261028` at `051e82d`. The reference implementation in [round2/](round2/) includes:

- the model classes, with the switches below;
- the past-only anchor for benchmark windows;
- the rollout-gain diagnostic;
- 24 CPU contract tests, all passing against `051e82d`.

Copy the files; do not reimplement them. Appendix A gives the theory. You do not need it to execute the spec, but every component names the design reason it serves.

## 0. Overview

### 0.1 What changed in this revision, and why

- **The dot and colour domains wait.** They test purity (unpredictable detail, appearance). The open questions now concern how the design's switches shape the dynamics model and planning, and they need the budget.
- **The forecast design space becomes a factorial.**
  - **Revision 2:** one principled point (`lam = 1`, one-step, noisy forecasts), compared against your forecast-only and TF arms. Those arms differ from it in three ways at once.
  - **Revision 3:** a small factorial over `lam`, forecast noise, horizon and `rho` (Part P2). Every switch is measured in at least four paired contexts.
- **`sigma0` is explored for the first time.** All 355 of your configs use 0.1, and round 1 froze it by design. Part P3 varies it on PushT; parts T and K vary it on TwoRoom and Cube.
- **One candidate mechanism connects your PushT results.**
  - **The calculation:** your `robustness20261029/THEORY.md` shows that with independent posterior noise, the rate charges the prior for carrying its noisy input forward.
  - **The consequence:** the prior pulls the state toward a centre, so open-loop rollouts contract.
  - **The fit:** lower `beta`, correlated noise and forecast rendering each counter this, and each improved the 7-step rollout and planning over the original arm.
  - **The size is uncertain:** at your runs' average state spread the pull is 1 to 2% per step, so it can matter only through low-spread dimensions or early in training.
  - **The test:** Part D0 measures contraction on your existing checkpoints before any training.
- **The model class gains two switches** that reproduce your arms inside one class with paired random draws:
  - `rho`: AR(1) posterior noise with your exact KL. With `lam = 0` it matches `RobustLeWM` bit for bit, and a test checks this.
  - `forecast_noise = "mean"`: noiseless forecasts. The rollout matches your forecast loop exactly, and a test checks this.

Unchanged from revision 2: the harder-task diagnosis (Section 0.7), the past-only anchor, the initial state scale, the division of work with your campaign (Section 0.9), and the TwoRoom and Cube core arms. Appendix C lists every change.

### 0.2 The method as it runs on TwoRoom and Cube

| Component | Setting | Reason |
| --- | --- | --- |
| Encoder | your L2 encoder: LeWM ViT on 224×224 frames, `QueryPool` to 4 slots × 16 = 64 dimensions, one frame at a time | The state is a function of its frame, so goal images encode without history. Kept fixed so that objective effects are attributable (Section 0.9). |
| Initial state scale | `calibrate_state_std = sigma0`: the pool head is rescaled once at step 0 so that `mu` has spread `sigma0` | Start where the renderer can read the state. At the default initial spread (about 1e-3), each dimension carries about 5e-5 nats through the renderer's noise; on TwoRoom the state then shrank and never recovered (Section 0.7, Appendix A.6). |
| Code noise | `sigma0 = 0.1`; 0.05 and 0.2 explored | The code's resolution (Appendix A.9). |
| Posterior noise | independent across frames (`rho = 0`); `rho = 0.8` explored | The simplest posterior; `rho` is a structured posterior with an exact KL (Appendix A.8). |
| Prior | LeWM predictor with the identity carry, fixed variance `sigma0^2`; it reads states and actions only | Unchanged from L2. |
| Rate | the exact one-step Gaussian KL, priced at `beta` ∈ {1/16, 1/4} per rendered value | The code length. `beta` is the only searched knob. |
| Renderer | DiT 384 / 12 / 6, flow matching on 64×64 targets, condition dropout 0.1 | Unchanged from L2. |
| What the renderer reads | for each rendered frame, with probability `lam = 1`, the one-step forecast sample `m_t + sigma0 * xi`; alternatives explored (Section 0.3) | It rewards what predicts the future (Appendix A.2). |
| Anchor | a past frame at least `gap` kept frames before the window (TwoRoom, Cube); Cube also tests a constant (zero) anchor; never a future frame | The anchor should carry appearance, not the current value of anything that moves (Appendix A.7). |
| Window | 8 kept frames at frameskip 5 | Unchanged. |
| Planning | unchanged: CEM on the encoder means with GoalMSE; the renderer and the anchor are not used | Unchanged. |

### 0.3 The design space

All arms of this round use one class, `ForecastScoredLeWM`, with five switches. The canonical method fixes each switch by a principle. The exploration measures what each principled choice costs or buys against the alternative the evidence favours.

| Switch | Values tested | Canonical | What it changes |
| --- | --- | --- | --- |
| `lam` | 0, 0.5, 1 | 1 | **The renderer's reward.** At 0 it rewards information about the frame the state came from. At 1 it rewards what the past tells, through the prior, about the next frame: sufficiency. At 1 a frame's own state never enters its own render, so the encoder learns only through the prior, one frame later. 0.5 keeps half of the direct reward and matches your forecast-only and TF arms (Appendix A.2, A.8). |
| `horizon` | `one_step`, `rollout` | `one_step` | **What the forecast is built from.** A one-step forecast reads the encoded past: it is the exact code length, and it trains the one-step map. A rollout starts from the encoded first frame and feeds back the prior's own outputs: it is a multi-step predictive bound, and it trains what the planner uses. At `lam = 1` a rollout gives renderer reward only to frame 0's state (Section 0.4). |
| `forecast_noise` | `sample`, `mean` | `sample` | **Whether the forecast is priced.** A sample's information is limited by the code resolution, and the surprise term charges the mean's departure from the encoded state. A mean is not priced; with a first-frame anchor it opens the action shortcut (Appendix A.5). `mean` cells are comparisons only and cannot be selected as the method. |
| `rho` | 0, 0.8 | 0 | **The posterior's noise structure.** AR(1) noise with the exact KL. Each frame keeps marginal noise `sigma0`, but the noise on frame-to-frame differences falls from `2 sigma0^2` to `2 (1 - rho) sigma0^2`. Carrying the state forward then costs the prior almost nothing in the rate (Appendix A.8). |
| `sigma0` | 0.05, 0.1, 0.2 | 0.1 | **The code's resolution against the unit entry prior.** It sets how many nats a dimension holds per unit of entry cost, the state's signal-to-noise ratio, and the forecast channel's precision (Appendix A.9). |

```python
s = mu + sigma0 * ar1(randn_like(mu), rho)               # posterior sample; rho = 0 gives L2's i.i.d. noise
m = prior(s[:, :-1], actions)                             # one-step prior means, also used by the rate
forecast = rollout_or_one_step(m) + (sigma0 * xi if forecast_noise == "sample" else 0)
condition = where(rand() < lam, forecast, s[:, 1:])       # what the renderer reads, per frame
```

### 0.4 One-step and open-loop forecasts, noisy and noiseless

Take one window: frames `x_0 .. x_7`, actions `a_0 .. a_6`, encoded states `s_0 .. s_7`. Consider the forecast of frame 5.

- **One-step (teacher-forced).** The prior reads the encoded states of real frames 0 to 4 and the actions: `m_5 = prior(s_0 .. s_4, a_0 .. a_4)`. Every forecast is one step beyond real information. The rate term always uses one-step forecasts.
- **Open-loop (rollout).** The prior reads only frame 0's encoded state, then feeds back its own forecasts: `h_1 = prior(s_0, a_0)`, `h_2 = prior(s_0, h_1, a_0, a_1)`, and so on to `h_5`. The forecast of frame 5 is five steps beyond real information, so errors compound. This is what the planner does at test time, from 3 context frames.
- **Noisy or noiseless.** Either kind can be noisy (a sample, with `sigma0` noise at every step) or noiseless (the prior's means). A noisy rollout spreads with the horizon: with a unit carry its noise reaches about `sqrt(k + 1) * sigma0` after `k` steps (the rollout starts from a sample), and less if the prior pulls toward a centre. A noiseless mean is a continuous vector that nothing prices.

| Forecast | What the prior reads for frame t | Noise | Your existing arms |
| --- | --- | --- | --- |
| one-step, sample | encoded states of frames 0 .. t-1 | `sigma0` | none |
| one-step, mean | encoded states of frames 0 .. t-1 | none | none |
| rollout, sample | encoded `s_0`, then its own samples | `sigma0` at every step | none |
| rollout, mean | encoded `s_0`, then its own means | none after frame 0 | forecast-only (`rho = 0`) and TF (`rho = 0.8`), both at `lam = 0.5` |

### 0.5 Four different noises

The word "noise" covers four separate things in this project. They act through different routes:

1. **Code noise `sigma0`.** It is added to the state that the renderer and the prior read, and it is part of the code length.
2. **Its correlation across frames, `rho`.** This is a choice of posterior family. Your TF arm uses 0.8, with the exact KL.
3. **Forecast noise.** It decides whether the forecast branch reads a sample or a mean, and so whether that channel is priced.
4. **Flow-matching corruption.** Your high-noise arms shifted the distribution of noise levels the renderer trains on. This is a renderer training choice, not part of the code. It gave no consistent gain on TwoRoom or Cube, and it is not in this round.

### 0.6 What your PushT results already say, and the mechanism that may connect them

At 2 epochs (24,982 updates), seeds 1 / 2, planning seeds 42 and 43. `h7` is agent error at 7 steps divided by the error of copying the last frame; lower is better:

| Arm | `lam` | Forecast | `rho` | `beta` | P0 % | `h7` agent | `h7` moving block |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original | 0 | none | 0 | 1/4 | 73 / 43 | 0.185 / 0.376 | 0.318 / 0.341 |
| lower-beta | 0 | none | 0 | 1/16 | 87 / 85 | 0.117 / 0.125 | 0.244 / 0.204 |
| temporal-noise (TN) | 0 | none | 0.8 | 1/4 | 84 / 63 | 0.135 / 0.210 | 0.250 / 0.240 |
| forecast-only (FO) | 0.5 | rollout, mean | 0 | 1/4 | 90 / 90 | 0.118 / 0.141 | 0.260 / 0.325 |
| TF | 0.5 | rollout, mean | 0.8 | 1/4 | 88 / 90 | 0.120 / 0.119 | 0.254 / 0.304 |

Over eight seeds at 2 epochs, original averages 42% P0 (four seeds at 10 to 22%), lower-beta 83% and TF 90%. At 3 epochs: original 88 / 74.67 (5-seed mean 85), lower-beta 86 / 84, FO 90 / 90, TF 94 / 94.

**Three changes rescued the original arm: a lower price, correlated noise, and rendering open-loop forecasts.** Each improves the 7-step rollout, and planning follows. One candidate mechanism connects them; Appendix A.8 gives it in full.

- **The cost.** With independent posterior noise, the expected surprise of a one-step prior `m = A s_prev` contains `0.5 ||A||_F^2`. Carrying the previous sample faithfully also carries its noise, which the next sample does not share, so the rate charges about half a nat per dimension per step for faithful carrying.
- **The prior's response.** For a component whose spread is `r` times `sigma0`, the rate-optimal carry is `a = r^2 / (1 + r^2)`: the prior pulls the state toward a centre, as a regression on a noisy input does.
- **The consequence at planning time.** The planner feeds the prior clean states, but the pull stays. Rollouts contract and forget where they started, so GoalMSE ranks candidate actions in a compressed geometry.

Each rescue counters this:

- **Lower `beta`** lets the encoder afford a wider state, which raises `r`.
- **`rho = 0.8`** replaces the cost with `0.5 ||A - rho I||^2`, so the optimal carry becomes `(r^2 + rho) / (r^2 + 1)`.
- **Forecast rendering** sends the renderer's loss into the prior and rewards rollouts that keep up.

The same calculation predicts a fourth lever that has never been tried: **a smaller `sigma0`**, which raises `r` for the same state spread.

**How large the effect can be is the open question.** Your runs' average state spread (`feature_std`, with `sigma0 = 0.1`) is 7 to 11.5 times `sigma0`:

- at 2 epochs: lower-beta 1.08 to 1.15, TN 0.76 to 0.89, FO and TF about 0.70;
- the original arm at 3 epochs: 0.80 to 0.96.

At those spreads the single-frame carry is 0.98 to 0.99, a pull of 1 to 2% per step for an average dimension. So the mechanism can matter only through dimensions with a much smaller spread, or by shaping the prior early in training, when the spread is about 2e-3. It also cannot be how FO and TF were rescued: they have the smallest average spread and the best rollouts, which fits the third lever, not the first.

This is a hypothesis consistent with your numbers, not yet a test. Part D0 measures the contraction directly on your checkpoints, by dimension group, at no training cost.

### 0.7 What your harder-task runs showed

**TwoRoom: the state never left its initial scale.**

- **Every AS-JEPA arm stayed small.** That includes `lam = 0` and `beta = 0`. The validation spread of `mu` was 0.9 to 2.1e-4, about 1/600 of `sigma0`, at every evaluated checkpoint.
- **The renderer ignored the state.** At 2,000 updates, swapping in another video's state changed the renderer's error by at most 0.0003% at t = 0.95, against 53 to 110% on Cube.
- **The information was available.** A fixed random projection of the same ViT's patch features reads the agent's position at R² 0.89 to 0.94 at initialization.
- **Other models plan well on the same cases.** Native LeWM scores 80 / 72% and your frozen-carrier model 90 / 94%, both at 6,000 updates.

This is not JEPA-style collapse. A constant state is not the objective's optimum here: 84 to 91% of the renderer's remaining error sits on the 9.4% of pixels that move. The state is trapped at a stationary point of training:

1. It starts at a spread of about 1e-3.
2. It shrank toward the prior mean in the first 600 updates.
3. The renderer, unable to read it, learned to ignore it.

Appendix A.6 gives the details. Cube escaped from the same initial scale; our reading is that its moving region (31.6% of render pixels) pulls harder. The first-frame anchor combined with noiseless forecasts is a secondary shortcut. It explains your TF arms' action use (wrong-action effect up to 72%), but not the trapped `lam = 0` arms.

**Cube: the state is used, but it holds the arm better than the cube.**

- **Planning trails matched LeWM.** TF reaches 24 / 26% at 11,671 updates, against native LeWM's 26 / 26% at 12,000 (30 / 28% recalibrated).
- **The goal distance follows the arm.** Its cube partial rank is 0.03, against 0.93 for the effector.

Our reading: the first-frame anchor shows a resting cube, and a carried cube follows the gripper, so the state seldom earns anything for storing the cube's own pose (Appendix A.7). Your campaign covers the planner and training-maturity findings; they are not repeated here.

### 0.8 What this round answers

| Question | Where |
| --- | --- |
| D-Q1. Do your existing priors contract their rollouts, in which dimensions, and does the contraction track the arms and seeds that plan badly? | D0 (free) |
| P-Q1. How does matched LeWM (L0) do on full-data PushT at 3 and 10 epochs? | P1 |
| P-Q2. What does pricing the forecast cost (sample against mean), with each horizon and each `rho`? | P2 |
| P-Q3. What does the horizon change (one-step against rollout)? | P2 |
| P-Q4. What does `rho = 0.8` add, with and without forecasts, priced or not? | P2, with your TN and TF arms |
| P-Q5. `lam = 1` against `lam = 0.5`? | P2 |
| P-Q6. Which priced configuration is the method on PushT, and is it stable over four seeds? | P2 and its continuation |
| P-Q7. Is the original arm's seed instability the trapped state, and what does `sigma0` change at `lam = 0`? | P3 |
| P-Q8. What does a reconstruction-free version keep? | P4 |
| T-Q1 to T-Q4. As in revision 2: the initial scale, the anchor, TF's failure, planning at 6,000 updates | T1 to T6 |
| T-Q5. With a first-frame anchor and a live state, does pricing the forecast close the action shortcut? | T5 against T9 |
| T-Q6. What does `sigma0` change on TwoRoom? | T7, T8 against T4 |
| K-Q1 to K-Q4. As in revision 2: anchor staleness, the anchor rules, `lam`, planning against native LeWM | K-pre, K1 to K4 |
| K-Q5. Does a smaller `sigma0` make the slow cube cheap enough to store? | K5, K6 against K1 |

### 0.9 Division of work with your campaign

| Topic | Your campaign | This spec |
| --- | --- | --- |
| Native LeWM baselines: TwoRoom 600 to 6,000 updates; Cube 600 to 12,000, with 23,342 running | owns them | reuses your checkpoints, cases and evaluations; reruns nothing |
| PushT LeWM (L0) on full data | not run | P1 |
| State architecture: fixed and local carriers, charts, readouts, patch tokens, predictor output span | owns it | keeps L2's encoder and `QueryPool` fixed, so that objective effects stay attributable |
| Objective and code: `lam`, forecast noise, horizon, `rho`, `sigma0`, `beta`, initial state scale | | owns it |
| Anchor rule | first frame (your default); constant anchor as an adaptation | past-only distant anchor; constant anchor trained from scratch with calibrated scale (K3) |
| Controllers: correlated CEM, NGC | owns them | evaluates every checkpoint under white CEM (primary) and your fixed correlated CEM (secondary); no new controller work |
| Training maturity | owns it (the six 23,342-update continuations) | uses your result to decide whether K continues (Section 8.3) |
| Already settled on TwoRoom and Cube by your runs | rate off, posterior-noise removal, training at pure target noise, high-noise logit shift, shared forecast prior, constant-anchor adaptation, quadrant pooling | cited as evidence; not repeated |
| TF, forecast-only, temporal-noise and original arms | your runs on PushT, TwoRoom and Cube | reused as references; the factorial recreates the forecast-only and TF settings inside the new class only where a paired comparison needs them (T5) |

Where the two plans meet. Each meeting point is run once, by whoever has capacity, after both sides agree:

- **(a) Combining recipes.** If T or K shows that the canonical objective trains a useful state, combine it with your best state architecture.
- **(b) Readout bottleneck.** If every T arm fails the use gate, the pooled readout is the bottleneck, and we decide together whether AS-JEPA adopts a spatial state.
- **(c) Anchor rule.** If K shows that the first-frame anchor hides the cube, your renderer-based variants should switch to the winning anchor rule.
- **(d) Shrinkage.** If D0 and P2 confirm the dynamics-shrinkage mechanism, its remedies (`rho`, `sigma0`, forecast rendering) apply to your renderer-based variants too.

Your six maturity runs occupy dev2, dev13 and dev15 GPUs 0 and 1 until about 16:30 UTC on 10-31; do not stop or reuse those slots.

### 0.10 Stopping points

There are three reports, written when their parts finish: PushT (Parts D0 and P), TwoRoom (Part T) and Cube (Part K). Stop for our agreement once all three exist.

- **Launch automatically when the trigger fires:**
  - the TwoRoom continuation from 2,000 to 6,000 updates, for arms that pass the use gate (Section 8.2);
  - the continuation of the PushT cell selected by Section 8.1 (P2c);
  - the conditional PushT arm stated in Section 8.1;
  - the failure reruns of Section 11, within their caps: `encoder_only` reruns of at most two priced candidate cells (4 runs), and one `beta = 0` rerun of T4 (2 runs).
- **Wait for our agreement:**
  - Stage 2, the selected forecast cells on TwoRoom and Cube (Section 6.4);
  - the continuation of the selected cell to 10 epochs;
  - K-mature.

## 1. Named quantities

New code reads these from the run config or from module constants. Derived values are computed, not retyped.

### 1.1 PushT

| Name | Definition | Value |
| --- | --- | --- |
| `BATCH` | windows per optimizer step | 128 |
| `EPOCH_UPDATES` | `floor(train windows / BATCH)` on `pusht-lance-full-v1` at `W = 8` (1,598,974 windows) | 12,491 |
| `MAX_STEPS` | `10 * EPOCH_UPDATES`, the learning-rate horizon of every L2-family run | 124,910 |
| `EPOCH2_STEPS` | `2 * EPOCH_UPDATES`, where the factorial screens stop | 24,982 |
| `SCREEN_STEPS` | `3 * EPOCH_UPDATES`, where the selected cell is compared with L0 | 37,473 |
| `L0_EPOCH_UPDATES` | the same for LeWM's window of 4 (your `full-L0-10epoch-*` configs) | 15,265 |
| `STEP_S` | measured seconds per step of full L2 on one H200 | about 0.91 (11,328 s for 12,491 steps); rollout cells add up to 7 sequential predictor passes per window, so measure them in the smoke |
| `EPOCH2_GPU_H`, `EPOCH_GPU_H` | `STEP_S` times 24,982 and 12,491 steps | about 6.3 and 3.2 |

Why the factorial screens stop at 2 epochs:

- Every reference arm exists there for seeds 1 and 2 (original, lower-beta, TN, FO, TF).
- The forecast arms already plan at 88 to 90% there.
- The original arm's instability, which the mechanism question is about, is visible there.

The selected cell continues to 3 epochs for the comparison with L0.

### 1.2 Harder tasks

| Name | Value | Why |
| --- | --- | --- |
| `TWOROOM_GATE` | 2,000 updates | past the 1,249-update warmup; your TwoRoom references exist here |
| `TWOROOM_FULL` | 6,000 updates | native LeWM (80 / 72%) and your fixed carrier (90 / 94%) were evaluated here |
| `CUBE_FULL` | 12,000 updates | native LeWM (26 / 26%) and TF (18 / 20%) exist here; one loader epoch is 11,671 |
| `CUBE_MATURE` | 23,342 updates | two loader epochs, as in your running maturity continuations |
| `MAX_STEPS` | 124,910 | the learning-rate horizon of all your harder-task runs |
| `STEP_S_HARD` | budget 1.5 s per update for L2 on one H200 | your Cube startups measured 1.11 to 1.34 s; measure in the smoke |
| `CASES` | your fixed 50-case strict-validation primary set and 50-case fresh set, per task | the test split stays untouched |
| `WHITE_CEM` | 300 samples × 30 iterations, top 30, horizon 5, receding horizon 5, history 3, budget 50, GoalMSE on `mu` | your native protocol |
| `CORR_CEM` | `WHITE_CEM` sampling in your fixed training-action covariance (`action_covariance_20261031`, same frozen calibration files) | your common controller |

### 1.3 Method knobs

| Name | Meaning | Values in this round |
| --- | --- | --- |
| `lam` | probability that the renderer reads a forecast instead of the encoded sample, per rendered frame | 0, 0.5, 1 (Section 0.3) |
| `horizon` | `one_step` or `rollout` (Section 0.4) | both, in P2; `one_step` elsewhere unless stated |
| `forecast_noise` | `sample` (priced) or `mean` (unpriced comparison) | both, in P2 and T5; `sample` elsewhere |
| `rho` | AR(1) correlation of the posterior noise, with the exact KL | 0 and 0.8 in P2, T5 and T9; 0 elsewhere |
| `sigma0` | code noise; the prior's spread and the forecast noise are tied to it | 0.1; 0.05 and 0.2 in P3, T7, T8, K5 and K6 |
| `forecast_grad` | `joint`: no stop-gradient. `encoder_only`: the forecast branch sends no gradient into the prior's parameters | `joint`; `encoder_only` only by the rule in Section 11 |
| `beta` | price of one nat of state, per rendered value (the unit of your configs) | 1/4; 1/16 in T6 and the conditional arm of Section 8.1. The contrastive arm uses `beta = 1` in true nats |
| `gap` | anchor distance in kept frames | PushT: 8, two-sided, as in your references (frozen for pairing). TwoRoom `gap_T` and Cube `gap_K`: past only, chosen by the rule in Section 5.5. Absent means the first-frame anchor |
| `renderer_anchor_mode` | `observed`, or `zero` for a constant anchor | `observed`; `zero` only in K3 |
| `W` | window, in kept frames | 8, frozen |
| `calibrate_state_std` | your one-time initialization of the state spread (`train.py`, `on_fit_start`) | `sigma0` on TwoRoom and Cube, and in P3. Absent in the other PushT arms, to stay paired with your references |

Why only `beta` is searched: it cannot be set from first principles while the flow loss is not in nats. Every other switch is either fixed by a principle (the canonical column of Section 0.3) or compared at two or three predeclared values. No value is tuned on results.

### 1.4 Evaluation and decision constants

| Name | Value | Meaning |
| --- | --- | --- |
| `PLAN_SEEDS`, `FRESH_SEEDS` | 42, 43, 44 and 1042, 1043, 1044 | PushT primary and fresh planning cases, 50 episodes each. Compare with a reference on the episodes it was evaluated on: 42 and 43 (100 episodes) for your 2-epoch seeds 1 and 2; 42, 43 and 44 (150) for seeds 3 to 8 and at 3 epochs. Compare new cells with each other on all six (300 episodes per training seed) |
| `PAIRED_CI` | 95% episode-clustered bootstrap of paired P0 differences, pooled over training seeds 1 and 2 (`report_main_conclusions.cluster_ci`) | the uncertainty from planning episodes. It does not include training-seed variation, hence the per-seed conditions below |
| `NONINF` | 5 points | A is non-inferior to B if the lower end of `PAIRED_CI(A - B)` is above -5 and each seed's mean difference is above -5 |
| better, worse | | A is better than B if `PAIRED_CI(A - B)` lies above 0 and both seeds' differences are positive; worse if below 0 and both negative; otherwise tied |
| `SLOPE_MIN` | 0.9 | lowest acceptable one-step gain of the prior on posterior samples (`one_step_noisy_gain`, Section 5.3). Below it, the prior exaggerates; the toys predict at most 3.5% overshoot under joint training |
| `ROLLOUT_TOL` | 0.1 | a spread ratio, carry ratio or motion gain within 0.1 of 1 counts as calibrated (Section 5.3). Spread and carry ratios below 0.9 mean contraction; a motion gain below 0.9 means lag, above 1.1 exaggeration |
| `COLLAPSE_RATIO` | 0.9 | as in `collapse_audit.py` |
| `SPREAD_MIN` | 0.3 | lowest validation `feature_std / sigma0` for a used state. At 0.3 each dimension carries up to `0.5 * ln(1.09)` ≈ 0.04 nats, about 2.8 nats per frame over 64 dimensions. The trapped TwoRoom value is 0.0015; Cube passed 0.3 by 2,000 updates |
| `USE_MIN` | 5% | lowest relative increase of the renderer's error at t = 0.95 when the state is swapped for another video's (Section 5.6). At 2,000 updates under the posterior swap: trapped TwoRoom at most 0.0003%, Cube 53 to 110% |
| `HARD_TIE` | 6 points | harder-task success differences below this, averaged over 2 seeds and both case sets, are ties. Your measured noise: re-evaluating one checkpoint moves 1 to 2 of 50 cases, and TF continued from one 2,000-update checkpoint scored 24 / 26% at 11,671 updates and 18 / 20% at 12,000 |
| `FALLBACK_MAX` | 0.2 | largest share of windows allowed without an eligible past anchor (Section 5.5) |
| `CUBE_TOL` | 3 cm and 15° (yaw modulo the cube's quarter-turn symmetry) | the strict success tolerance; used for anchor staleness |
| `AGENT_TOL` | the strict success distance of your TwoRoom manifests (8.0, in `pos_agent` units; confirm the units) | used for anchor staleness |
| `RANK_HIGH`, `RANK_LOW` | 0.3, 0.1 | the cube partial rank (`position_encoded_partial_rank`, Section 5.7): how well the latent distance from a candidate's reached endpoint to the goal tracks the cube's distance to its goal, holding the effector's distance and the angle fixed. At or above 0.3 counts as cube-driven, at or below 0.1 as not. TF: 0.03 for the cube, 0.93 for the effector |

## 2. Starting point

### 2.1 Code

1. Create branch `as-jepa/round2` from your current head (`051e82d` or later), and record the hash in every report.
2. Copy unchanged from `docs/lingjing/round2/`:
   - `as_jepa_lewm/forecast_scored.py`, `nce.py` and `learned_variance.py` into `as_jepa_lewm/`;
   - `as_jepa_tools/benchmark_gap.py` and `as_jepa_tools/rollout_gain.py` into `as_jepa_tools/`;
   - `tests/as_jepa/test_round2.py`, `test_benchmark_gap.py` and `test_rollout_gain.py` into `tests/as_jepa/`.

   No arm uses `learned_variance.py`; copy it because `test_round2.py` imports it.
3. Register `forecast_scored` and `nce` in the three places that list model targets:
   - `as_jepa_lewm/factory.py`, `MODEL_TARGETS`: add `"forecast_scored": "as_jepa_lewm.forecast_scored.ForecastScoredLeWM"` and `"nce": "as_jepa_lewm.nce.NCELeWM"`. `train.py` and `benchmark_health.py` build models through this table.
   - `as_jepa_tools/evaluate.py`, `load_model`: one `elif` per class before the final `else`, as for `RobustLeWM`. `rollout_gain.py` and `decode_decomposition.py` load models through it.
   - `as_jepa_tools/planning.py`: add both targets to the allowed `_target_` values.

   `benchmark_evaluate.py` needs no change, because it instantiates the `_target_` in each checkpoint's `config.json` through `load_pretrained`. Confirm this in the smoke.
4. Route benchmark datasets through `make_benchmark_dataset(cfg, split)` from `benchmark_gap.py`:
   - `benchmark_data.make_training_dataset` (the benchmark branch). Import `benchmark_gap` inside the function, because `benchmark_gap` imports `benchmark_data`.
   - `benchmark_health.py` (line 41).
   - `benchmark_information_audit.py` (line 257, the renderer substitutions).
   - `gradient_preflight.py` (line 35). It also builds `RobustLeWM(**cfg["model"])` directly (line 33), which fails on a `forecast_scored` config; switch it to `build_model(cfg)`.

   With no `gap` in a config, `make_benchmark_dataset` returns `BenchmarkWindows` exactly as before (tested), and none of your 75 existing benchmark configs sets a top-level `gap`. `benchmark_spatial_probe.py` is not used for round-2 runs.
5. Tool additions:
   - `--condition {posterior,forecast}` and `--fp32` for `collapse_audit.py` (Section 5.2).
   - Two conditions in `benchmark_health.py`, `one_step` and `one_step_shuffled` (Section 5.6).
   - A checkpoint argument for your Cube readout probe, `experiments/carrier_progress_20261031/tf_pose.py`, which currently hard-codes the TF runs (Section 5.7).

   `rollout_gain.py` replaces the calibration tool of revision 2.
6. Do not edit `model.py`, `robust.py`, `research.py` or any frozen source.
7. Freeze every campaign's source with your usual `git archive` plus SHA256 manifest, and dispatch with your workflow tools.

### 2.2 Data

- **`D-std`:** `/efs/prescottsun/SPT-JEPA/data/pusht-lance-full-v1` with `all_phase_data: true`. Evaluation uses its `phase0/` view, and planning uses `pusht_expert_train.h5` as `planning.py` sets it up.
- **TwoRoom:** `/efs/prescottsun/SPT-JEPA/data/hardtasks-v1/tworoom-metadata`, `dataset_kind: benchmark_hdf5`, `action_dim: 10`.
- **Cube:** `/efs/prescottsun/SPT-JEPA/data/hardtasks-v1/cube-metadata`, `dataset_kind: benchmark_hdf5`, `action_dim: 25`.

Splits and action normalization are those of your runs: training episodes only.

### 2.3 Reference runs reused, not rerun

**PushT** (P0 %, by training seed). At a given training seed, `ASJepaLeWM`, `RobustLeWM` and `ForecastScoredLeWM` build identical initial tensors, the data order depends only on the seed, and the posterior noise is drawn identically. So every PushT cell at seeds 1 and 2 is paired with these references. Only the forecast branch's coins and predictor dropout come from a different random stream than in `RobustLeWM`.

| Reference | Settings | 2 epochs, seeds 1 / 2 | 3 epochs, seeds 1 / 2 | Configs |
| --- | --- | --- | --- | --- |
| original | `lam 0`, `rho 0`, `beta 1/4` | 73 / 43 (seeds 1 to 8: 73, 43, 22, 76, 10, 15.33, 15.33, 84) | 88 / 74.67 | `stability20261029/configs/full-L2-beta0.25-stage3-seed*.json` |
| lower-beta | `lam 0`, `rho 0`, `beta 1/16` | 87 / 85 (seeds 1 to 8 mean 82.6) | 86 / 84 | `robustness20261029/configs/lower-beta-seed*.json` |
| temporal-noise (TN) | `lam 0`, `rho 0.8` | 84 / 63 | not run | `promising20261029` components |
| forecast-only (FO) | `lam 0.5`, rollout, mean, `rho 0` | 90 / 90 | 90 / 90 | `robustness20261029/configs/forecast-render-seed{1,2}.json` |
| TF | `lam 0.5`, rollout, mean, `rho 0.8` | 88 / 90 (seeds 1 to 8 mean 89.75) | 94 / 94 | `robustness20261029/configs/temporal-forecast-seed*.json` |

All PushT references use the two-sided anchor at gap 8 and no initial-scale calibration. D0 (Section 6.1) also reads exports from `promisingseed34epoch2_20261029` (seeds 3 and 4 at 2 epochs), `promisingnewseedepoch2_20261030` (seeds 5 to 8 at 2 epochs) and `completion20261028` (the original arm's seed 0).

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

The class subclasses `ASJepaLeWM` and adds six constructor arguments: `lam`, `horizon`, `forecast_noise`, `rho`, `forecast_grad` and `forecast_seed`. Other keyword arguments pass through, including `sigma0`, `action_dim` and `renderer_anchor_mode`. It adds no parameters. Its loss is `ASJepaLeWM.loss`, except for three lines: the posterior noise, the surprise term's target, and the renderer's condition.

```python
mu = flat.reshape(b, w, K, d)
s = mu + self.sigma0 * ar1(torch.randn_like(mu), self.rho)                     # rho = 0: L2's i.i.d. noise, same draw
predicted = self.predict(s.flatten(2)[:, :-1], info["act_emb"][:, :-1]).reshape_as(mu[:, 1:])
entry = entry_nats(mu[:, 0], self.sigma0)
if self.rho:                                                                   # exact KL of the AR(1) posterior
    target = mu[:, 1:] + self.rho * (s[:, :-1] - mu[:, :-1])
    surprise = surprise_nats(target, predicted, self.sigma0) + self.state_dim * .5 * (-rho**2 - log1p(-rho**2))
else:
    surprise = surprise_nats(mu[:, 1:], predicted, self.sigma0)
condition = s[:, 1:]
if self.lam > 0:
    scale = self.sigma0 if self.forecast_noise == "sample" else 0.
    with torch.random.fork_rng(devices=devices):                              # separate random stream (Section 3.2)
        torch.manual_seed(FORECAST_STREAM + 1000003 * self.forecast_seed + int(step))
        use = torch.rand(b, w - 1, 1, 1, device=mu.device) < self.lam         # one coin per rendered frame
        xi = torch.randn(mu[:, 1:].shape, device=mu.device, dtype=mu.dtype)   # always drawn: same coins for both noise settings
        if self.horizon == "one_step" and self.forecast_grad == "joint":
            forecast = predicted + scale * xi                                  # reuses the surprise term's pass
        else:
            forecast = self._forecast(s, info["act_emb"], xi, scale)           # rollout, or encoder_only
    condition = torch.where(use, forecast, condition)
fm = flow_loss(self.renderer, y, condition.flatten(0, 1), a, self.training_shift)
loss = fm + beta * (entry + surprise.sum(1)).mean() / ((w - 1) * FRAME_ELEMS)
```

With `horizon = "rollout"`, `_forecast` runs the open loop of Section 0.4: `h_0 = s_0`, then `h_t = m(h_<t, a_<t) + scale * xi_t`. With `scale = 0`, it is your forecast loop verbatim; a test checks this.

Settings that reproduce your arms:

- `lam = 0.5`, `horizon = "rollout"`, `forecast_noise = "mean"`, `rho = 0` is forecast-only;
- the same with `rho = 0.8` is TF;
- `lam = 0` with `rho = 0.8` is temporal-noise.

These match up to the forecast branch's random stream.

Logged metrics, in addition to L2's: `forecast_share` (about `lam`), `calib_slope` (on the training batch, for plotting) and `active_dims`.

`NCELeWM` and `LearnedVarianceLeWM` override the loss, so they refuse `rho > 0` and `forecast_noise = "mean"`.

### 3.2 Random streams and pairing

The coins, the forecast noise and any extra predictor passes, including their dropout masks, run inside `torch.random.fork_rng`. That stream is seeded by `FORECAST_STREAM + 1000003 * forecast_seed + step`. Consequences:

- **The main stream is consumed exactly as in L2:** posterior noise (before the AR(1) transform, which draws nothing), flow-matching noise and timesteps, and the renderer's condition dropout.
- **Pairing across cells:** a cell at seed `k` sees the same data order, initial tensors and main-stream draws as every reference at seed `k`.
- **Pairing across noise settings:** two cells of the new class that differ only in `forecast_noise` also share their coins. Contrasts with your FO and TF arms (F1 against FO, F2 against TF, F3 against FO, F4 against TF) also differ in the forecast branch's coins and dropout draws, which are statistically equivalent but not identical.

Set `forecast_seed` to the training seed. Keep one GPU per run with the same per-step batch.

### 3.3 Gradient routing

With `forecast_grad = "joint"`, the default, there is no stop-gradient anywhere. The forecast branch's flow loss reaches:

- the prior, through the forecast;
- the encoder, through the earlier states.

At `lam = 1` with one-step forecasts, a frame's own encoding enters only the renders of later frames. With rollouts, only frame 0's encoding enters the forecast branch at all; the encoder still receives the posterior branch's gradient at `lam < 1`.

Joint training has one known side effect (Appendix A.5): the prior is rewarded for slightly exaggerating its predicted changes. The one-step noisy gain (Section 5.3) watches for this. The fallback, `forecast_grad = "encoder_only"`, recomputes the forecast branch with the predictor's parameters and the action embedding detached. Use it only under the rule in Section 11.

### 3.4 Past-only distant anchors for benchmark windows (`as_jepa_tools/benchmark_gap.py`)

`GapBenchmarkWindows(root, window, seed, split, gap)` is `BenchmarkWindows` with one change: the renderer's anchor (`anchor`, `anchor224`, `anchor_row`) is a frame of the same episode, phase-aligned with the window, at least `gap` kept frames before window frame 0.

- **The draw.** It is uniform over the eligible past frames, using a generator seeded by `(episode, start, epoch)`, as `FullPushTWindows` does.
- **Fallback.** When no past frame is far enough back, the earliest phase-aligned frame before the window is used, or window frame 0 when the window starts the episode, and `anchor_fallback` is True.
- **Everything else is unchanged.** All other fields are byte-identical to `BenchmarkWindows`, and `gap = 0` reproduces `BenchmarkWindows` exactly.
- **Fallback share.** `fallback_share()` returns the exact share of windows without an eligible anchor.
- **Dispatch.** `make_benchmark_dataset(cfg, split)` returns the gap dataset when the config sets a top-level `gap`.

Why past only:

- **A future frame would leak.** It would show where slow objects end up.
- **Generation never has one.** It only ever has past frames.
- **It matches your convention.** Your data contract already rules out future anchors.

PushT keeps its references' two-sided rule, for pairing.

### 3.5 The initial state scale on the harder tasks

Set `"calibrate_state_std": <sigma0 of the run>` at the top level of every TwoRoom and Cube config, and in P3. `train.py`'s `on_fit_start` then rescales the `QueryPool` head once, from 128 training windows, so that at step 0 the average per-dimension spread of `mu` is `sigma0`. It also subtracts the head's mean output, averaged over the four slots. Nothing normalizes the state after that, the objective is unchanged, and the step is skipped on resume.

It writes `initial_scale.json` with `target_std`, `original_std`, `scale` and `mean_before`. Expect `original_std` of about 1e-3 to 2e-3; your PushT calibrations measured 1.6 to 1.9e-3, with scales of 52 to 64. Read the spread after initialization from the first row of `metrics.jsonl` (step 100).

Why `sigma0`: at spread `sigma0`, each dimension can carry up to `0.5 * ln 2` ≈ 0.35 nats through the renderer's noisy channel from the first update, whatever the value of `sigma0`. That also removes `sigma0`'s interaction with the trap at step 0, so the `sigma0` arms compare code resolutions, not starting points.

### 3.6 `NCELeWM` (`as_jepa_lewm/nce.py`; P4)

This is the reconstruction-free control. It keeps L2's encoder, pooling, action encoder, prior and rate, and has no renderer.

- **The bound.** The flow loss is replaced by an InfoNCE bound on the information the condition carries about its frame. The condition is the encoded sample at `lam = 0`, or the one-step forecast sample at `lam = 1`.
- **The critic** is the encoder's own Gaussian log-likelihood, `score(t, j) = -||condition_t - mu(x_j)||^2 / (2 sigma0^2)`.
- **The candidates** are all `B * T` rendered frames of the batch, so the bound is capped at `log(B * T)` ≈ 6.8 nats per frame.
- **The loss** per window is `T * InfoNCE + beta * rate`, both in nats, so `beta = 1` is a calibrated price.
- **Evaluation:** planning and probes only.

### 3.7 Configs and launch

**PushT factorial cell F7, seed 1.** Other cells change only the four switches in Section 6.1, and `retain_steps` stays.

```json
{
  "output": "/efs/prescottsun/SPT-JEPA/runs/round2_20261101/F7-lam1-one-sample-rho0-seed1",
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
            "lam": 1.0, "horizon": "one_step", "forecast_noise": "sample", "rho": 0.0,
            "forecast_grad": "joint", "forecast_seed": 1},
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
  "gap": "<gap_T from Section 5.5>",
  "model_class": "forecast_scored",
  "model": {"mode": "L2", "beta": 0.25, "sigma0": 0.1, "window": 8, "action_dim": 10,
            "renderer_width": 384, "renderer_depth": 12, "renderer_heads": 6,
            "lam": 1.0, "horizon": "one_step", "forecast_noise": "sample", "rho": 0.0,
            "forecast_grad": "joint", "forecast_seed": 1},
  "record_initial_weights": true
}
```

**Cube** uses the same template with `cube-metadata`, `"action_dim": 25`, `"gap": <gap_K>` and `"retain_steps": [6000, 12000]`. Arms with a first-frame anchor omit `gap`. K3 omits `gap` and sets `"renderer_anchor_mode": "zero"` in `model`. Arms with another `sigma0` set `model.sigma0` and `calibrate_state_std` to the same value.

**Launching:**

- Launch with your staged trainer: `python -B -u -m as_jepa_tools.train_stage --config <config.json> --stop-step <N>`. Continue a stopped run with `--stop-step <new N> --resume`, which is also how to resume after an interruption.
- PushT: set top-level `gap` and `model.anchor_gap` to the same value.
- Keep `max_steps = 124910` everywhere.

### 3.8 What does not change

- **Model:** the encoder, `QueryPool`, action encoder, LeWM predictor with the identity carry, and the renderer (384 / 12 / 6).
- **Optimization:** AdamW at 5e-5 and 1e-4, 1% warmup, cosine to `MAX_STEPS`; batch 128; bf16 with the rate in FP32; gradient clip 1.0; one GPU per run.
- **Harder-task data and evaluation:** your `BenchmarkWindows` data, splits, action normalization, renderer targets (64×64 from 224×224 frames), case manifests and evaluators.
- **Planning:** the planner reads encoder means and noiseless prior rollouts, with no renderer and no anchor.

No schedule acts on `lam`, `beta`, `sigma0` or `rho`.

## 4. Design invariants

1. The encoder reads one frame and nothing else.
2. The anchor reaches only the renderer. The prior never reads frames, the anchor or renderer features.
3. Actions reach only the prior. A priced forecast can carry information about actions only through the prior's mean, which the surprise term prices, plus `sigma0` noise.
4. For each rendered frame, the renderer reads the posterior sample (probability `1 - lam`) or the prior's forecast (probability `lam`). It never reads `mu`. A noiseless prior mean is read only in comparison cells (`forecast_noise = "mean"`), which cannot be selected as the method.
5. Window frame 0 is encoded and starts the chain, and is never rendered.
6. AS-JEPA arms use no augmentation, EMA, target network, SIGReg, VICReg, masking or stop-gradient. The only exception is `forecast_grad = "encoder_only"` under the rule in Section 11.
7. Nothing normalizes `mu`, `s`, the prior's mean or the forecast during training. The initial state scale is set once, at step 0.
8. Within a run, `sigma0`, `beta`, `rho` and `lam` are constant. The rate is the exact one-step KL of the posterior family in use (i.i.d. or AR(1)), with one `beta` for entry and surprise. The horizon and forecast noise change only what the renderer reads, never the rate.
9. The renderer's target is fixed data.
10. The coins and forecast noise use their own random stream.
11. On the harder tasks, the anchor is never a future frame.

## 5. Evaluation

### 5.1 PushT package

Every comparison with a reference uses the settings the reference was evaluated with. Those are your tools' defaults: probes in BF16 and FP32, and the collapse audit under BF16 autocast. The new `--fp32` option of `collapse_audit.py` adds an FP32 verdict; report both.

- **At 2 epochs (every factorial cell, P3, and the references where an item is missing),** run the evaluation of your `promising20261029` component follow-up:
  - planning on `PLAN_SEEDS` and `FRESH_SEEDS` (Section 1.4 says which episodes each comparison uses);
  - `quality_complete` (`h{1,3,7,13}` agent and block ratios, `h7_moving_block_ratio`, LPIPS, SSIM);
  - `counterfactual` (action identification);
  - the collapse audit (Section 5.2);
  - the rollout diagnostics (Section 5.3);
  - `as_jepa_tools.probes --noisy --fp32` (agent and block probe R²);
  - the drift decomposition (Section 5.4);
  - the rate per window, `entry` and `surprise`.

  ```bash
  python -B -u -m as_jepa_tools.planning --checkpoint <ckpt> --output <dir> --upstream <stable-worldmodel> \
      --data-root <data/planning> --test-episodes <pusht-lance-full-v1/phase0/test_episodes.npy> --seed 42 --episodes 50    # also 43, 44
  python -B -u -m as_jepa_tools.quality_complete --checkpoint <ckpt> --output <dir> --data <phase0> --pose <pose> --i3d <i3d> \
      --cases 64 --samples 1 --steps 20 --selection-seed 20261029
  python -B -u -m as_jepa_tools.counterfactual --checkpoint <ckpt> --output <dir> --data <phase0> --pose <pose> \
      --cases 16 --steps 20 --selection-seed 20261029
  ```

- **At 3 epochs (the selected cell),** run `as_jepa_tools.completion_evaluate --mode L2 --expected-step 37473`, plus planning on `FRESH_SEEDS`, the rollout diagnostics and the drift decomposition (Section 5.4). Send us the swap gallery of the selected cell, seed 1.
- **Models without a renderer** (L0, `NCELeWM`): run `as_jepa_tools.planning` and `as_jepa_tools.probes --checkpoint <ckpt> --data <phase0> --output <dir> [--noisy] [--fp32]` directly. For L0, also evaluate after your train-only BatchNorm recalibration, and report both.

### 5.2 Forecast-based collapse check for `lam > 0`

At `lam = 1` the renderer never sees encoded states in training. Add `--condition {posterior,forecast}`, defaulting to `posterior`, to `collapse_audit.py`. With `forecast`, render each held-out frame `t >= 1` from the forecast exactly as the cell renders it during training: one-step or rollout (from the encoded first frame), a sample or the mean, with the cell's `rho` in the posterior samples. The existing collapse rule applies. For `lam > 0` cells, the forecast verdict is the one that counts.

### 5.3 Rollout diagnostics: do rollouts contract or lag? (`as_jepa_tools/rollout_gain.py`)

```bash
python -B -u -m as_jepa_tools.rollout_gain --checkpoint <ckpt> --data <phase0> --output <dir> --cases 128 --window 16 --context 3
python -B -u -m as_jepa_tools.rollout_gain --checkpoint <ckpt> --data <hardtasks-v1/cube-metadata> --dataset benchmark --output <dir> \
    --cases 128 --window 16 --context 3                                           # also with --fp32
```

Over the same held-out windows as `decode_decomposition.py` (PushT), or 128 validation windows (benchmark), it reports, without the renderer:

- **The planner's open-loop rollout** (`predict_states` from 3 context frames, noiseless means; it keeps up to 7 frames of history as it goes), at every horizon:
  - **spread ratio:** the across-window variance of the rollout over that of the encoded truth;
  - **carry ratio:** how strongly the rollout still depends on where it started (the slope of the centred state on the centred last context state), relative to the same slope for the truth;
  - **motion gain:** the slope of the predicted displacement on the true displacement;
  - **error relative to copying** the last context state;
  - the spread and carry ratios within quartiles of dimensions ranked by spread, and the spread of every dimension.
- **The same metrics for teacher-forced one-step predictions** from clean encoder means (`one_step_clean`).
- **`one_step_noisy_gain`:** the slope of the true change on the predicted change, from posterior samples drawn with the model's own `sigma0` and `rho`. This is the training condition, and `SLOPE_MIN` applies to it.
- **`state_spread_over_sigma0`** and **`noisy_state_information_upper_bound_nats`**, the sum over dimensions of `0.5 ln(1 + Var(mu) / sigma0^2)`.

How to read it:

- **Contraction**, the pull toward a centre of Appendix A.8, shows as spread and carry ratios below 1 that fall with the horizon, most in the lowest-spread quartile.
- **Lag** shows as a motion gain below 1: the rollout moves less than the truth. **Exaggeration** shows as a motion gain above 1, or as a one-step noisy gain below `SLOPE_MIN`.
- **A calibrated prior** has all three ratios within `ROLLOUT_TOL` of 1.

Do not read the slope of the true displacement on the predicted displacement as a shrinkage measure. A pull toward a centre adds predicted displacement unrelated to the motion, which drives that slope below 1, not above it.

### 5.4 Open-loop drift (PushT)

Run `as_jepa_tools/decode_decomposition.py` (`--cases 128 --steps 50`, with and without `--fp32`) at 2 epochs for every P2 cell and its references, and at 3 epochs for the selected cell. Report the block error from open-loop forecasts and from the frame's own encoded state at h1, h7 and h13, and their difference, the **drift**. For `lam = 1` cells, the own-state render is out of distribution, so the forecast column is primary.

### 5.5 Dataset checks before harder-task training (CPU)

```bash
python -B -m as_jepa_tools.benchmark_gap --data <hardtasks-v1/tworoom-metadata> --gaps 0 1 2 4 8 16 \
    --label agent:pos_agent:position:<AGENT_TOL> --output tworoom-anchor-table.json
python -B -m as_jepa_tools.benchmark_gap --data <hardtasks-v1/cube-metadata> --gaps 0 1 2 4 8 16 \
    --label cube:privileged_block_0_pos:position:0.03 --label cube_yaw:privileged_block_0_yaw:yaw90:15 \
    --label effector:proprio_effector_pos:position:0.03 --output cube-anchor-table.json
```

For each gap (0 is the first-frame anchor), the table reports two things:

- **`fallback_share`:** the share of windows without an eligible past anchor.
- **Staleness, per label:** the share of rendered frames in which the anchor's value differs from the frame's by more than the tolerance.

A constant anchor counts as stale for every label.

**Rule.** `gap_T` and `gap_K` are the largest gap in {1, 2, 4, 8, 16} with `fallback_share <= FALLBACK_MAX`. Confirm before running that `privileged_block_0_pos` is in metres, the yaw in radians, and the units of `AGENT_TOL`. Record both tables in the dataset cards and the reports.

### 5.6 Harder-task health and the use gate

**Health.** Run `benchmark_health.py --cases 16` at every evaluated checkpoint. It evaluates the step in `last.ckpt`, so run the 2,000-update check before any `--resume`. Add two conditions:

- **`one_step`:** the one-step forecast samples `m_t + sigma0 * xi`, from the window's own encoded posterior samples and actions;
- **`one_step_shuffled`:** the same, from the other video's encoded states, with the window's own actions and the same draw of `xi`.

Report, for every arm:

- `feature_signal_to_posterior_noise` (`feature_std / sigma0`), in FP32;
- every swap effect at t = 0.95: `shuffled_posterior` against `posterior`, `one_step_shuffled` against `one_step`, and `wrong_state` against `forecast`;
- the wrong-action and zero-state effects; and, from `benchmark_information_audit.py`, the wrong-anchor effect (`forecast_wrong_anchor`).

The **use effect** of an arm is the swap effect on the condition it trains on: `posterior` for `lam = 0`, `one_step` for `lam = 1`, and `forecast` for the rollout arms T5 and T9. Compare T5 and T9 with T3 on the shared `one_step` condition as well. Also plot the training-log `feature_std / sigma0` every 100 steps, and run the rollout diagnostics (benchmark mode) at every evaluated checkpoint.

**Use gate (TwoRoom, at `TWOROOM_GATE`).** An arm passes if both seeds have `feature_std / sigma0 >= SPREAD_MIN` and a use effect of at least `USE_MIN`. Arms that pass continue to `TWOROOM_FULL`; arms that fail stop there as diagnostic results.

### 5.7 Harder-task planning and content

- **Planning:** `benchmark_evaluate.py` on the primary and fresh manifests under `WHITE_CEM`; then `CORR_CEM` with your `action_covariance_20261031/evaluate.py` and the frozen calibration files.
  - `CORR_CEM` needs per-checkpoint preparation: its hash-pinned `--spec` must carry the same checkpoint's white-CEM result as `reference_metrics`, and `prepare.py` hard-codes its model list. So run white CEM first and build one spec per checkpoint.
  - TwoRoom: every arm at 2,000 updates, and the continuing arms at 6,000. No correlated-CEM reference exists for AS-JEPA TwoRoom arms at 2,000 updates.
  - Cube: every arm at 12,000.
  - Report success per seed and case set; call differences below `HARD_TIE` ties.
- **Content probes:** `benchmark_information_audit.py`, with clean and `sigma0`-noisy linear probes of the agent's position (TwoRoom), and of the cube's and effector's positions (Cube).
  - For the Cube readout in centimetres and degrees, use your probe in `experiments/carrier_progress_20261031/tf_pose.py`, given a checkpoint argument, with the same 128 training and 64 validation clips, ridge and codebook.
  - Compare clean with clean (TF 4.7 to 5.1 cm, native 3.7 to 3.9 cm). Add the noisy readout for the K arms and, by rerunning the probe, for TF and native.
- **Cube goal geometry.** Score each K checkpoint on the candidate pool of your TF audit (`CUBE-ORACLE-AUDIT.md`), then summarize against TF:

  ```bash
  python -B -m as_jepa_tools.cube_oracle_audit score --manifest <same manifest> --data <cube-metadata> \
      --pool <your TF audit pool> --checkpoint <ckpt> --label K1-seed1 --output <dir>
  python -B -m as_jepa_tools.cube_oracle_summary --scores K1-seed1=<dir> --scores TF-seed1=<TF scores> \
      --reference TF-seed1 --data <cube-metadata> --output <dir>
  ```

  Report `position_encoded_partial_rank` (the cube) and `effector_encoded_partial_rank` (the arm) for the non-oracle candidates.

### 5.8 What every report records

For every run, record:

- its config and source hash;
- `lam`, `horizon`, `forecast_noise`, `rho`, `sigma0`, `beta`, `gap`, `calibrate_state_std`, `forecast_grad` and `seed`;
- the evaluated checkpoint step and GPU-hours;
- the metrics above, including the rollout-gain report;
- PushT only: the collapse verdicts (posterior and forecast, BF16 and FP32);
- harder tasks only: `initial_scale.json`, the health report and `GapBenchmarkWindows.fallback_share()`.

Write one JSON record per run. Tables give per-seed values and the mean over seeds, and for PushT `PAIRED_CI` with both per-seed differences.

## 6. Run plan

Run IDs follow `r2-<arm>-seed<k>`.

### 6.1 Part D0 and Part P: PushT

**D0. Diagnostics on existing checkpoints (free; run first).** Run `rollout_gain.py` (Section 5.3; PushT mode, with and without `--fp32`) on every existing 2- and 3-epoch export of:

- original, seeds 0 to 8;
- lower-beta, seeds 1 to 8;
- temporal-noise, seeds 1 and 2;
- forecast-only, seeds 1 and 2;
- TF, seeds 1 to 8.

Run it in benchmark mode on your Cube original, lower-beta and TF checkpoints at 11,671 updates. No training. The result tests the mechanism of Section 0.6 before P2 trains anything. The decision rules in Section 8.1 read it.

**P1. Matched LeWM.** Your `full-L0-10epoch-seed{0,1,2}` configs, unchanged except output and `retain_steps` (`L0_EPOCH_UPDATES` × 2, 3 and 10). Evaluate at 3 and 10 LeWM epochs (Section 5.1).

**P2. The forecast factorial.** Each cell is the PushT template (Section 3.7) with the switches below. Every cell runs seeds 1 and 2 to `EPOCH2_STEPS`, uncalibrated, at `beta = 1/4`, `sigma0 = 0.1` and gap 8. The `lam = 0.5` cells form a complete 2 × 2 × 2 over forecast noise, horizon and `rho`; your FO and TF arms fill two of its corners. The `lam = 1` cells are priced only.

| Cell | `lam` | `horizon` | `forecast_noise` | `rho` | Corresponds to |
| --- | --- | --- | --- | --- | --- |
| (FO) | 0.5 | rollout | mean | 0 | your forecast-only arm, reused |
| (TF) | 0.5 | rollout | mean | 0.8 | your TF arm, reused |
| F1 | 0.5 | rollout | sample | 0 | priced forecast-only; revision 2's conditional C5 |
| F2 | 0.5 | rollout | sample | 0.8 | priced TF |
| F3 | 0.5 | one_step | mean | 0 | |
| F4 | 0.5 | one_step | mean | 0.8 | |
| F5 | 0.5 | one_step | sample | 0 | revision 1's C4 |
| F6 | 0.5 | one_step | sample | 0.8 | |
| F7 | 1 | one_step | sample | 0 | canonical; C1 of revisions 1 and 2 |
| F8 | 1 | one_step | sample | 0.8 | |
| F9 | 1 | rollout | sample | 0 | |
| F10 | 1 | rollout | sample | 0.8 | |

Each switch is then measured in paired contexts:

- **Forecast noise**, in 4 contexts: F1 against FO, F2 against TF, F5 against F3, F6 against F4.
- **Horizon**, in 6 contexts: F1 against F5, F2 against F6, FO against F3, TF against F4, F9 against F7, F10 against F8.
- **`rho`**, in 6 contexts plus `lam = 0`: F2 against F1, TF against FO, F4 against F3, F6 against F5, F8 against F7, F10 against F9, and TN against original.
- **`lam` (1 against 0.5)**, in 4 priced contexts: F7 against F5, F8 against F6, F9 against F1, F10 against F2.

**P2c. Continuation of the selected cell (automatic).** The cell selected by Section 8.1 continues seeds 1 and 2 to `SCREEN_STEPS` and adds seeds 3 and 4 to `EPOCH2_STEPS`. It is compared with L0 at 3 epochs and with the references' seeds 3 and 4 at 2 epochs, on planning seeds 42 to 44 as those references were: original 22 / 76, lower-beta 86.67 / 70.67, TF 92.67 / 88.

**P3. The original arm's instability: initial scale and code resolution.** Every arm uses `lam = 0`, `rho = 0` and `beta = 1/4`, runs to `EPOCH2_STEPS`, and uses seeds where the uncalibrated original arm failed. At 2 epochs those seeds scored 22% (seed 3), 10% (seed 5) and 15.33% (seeds 6 and 7).

| Arm | `sigma0` | `calibrate_state_std` | Seeds | Question |
| --- | --- | --- | --- | --- |
| P3a | 0.1 | 0.1 | 3, 5, 6, 7 | Does starting the state at the code's resolution rescue the failing seeds? |
| P3s (sham) | 0.1 | each seed's own `original_std`, read from P3a's `initial_scale.json` | 3, 5 | The same one-time mean shift, without rescaling. Any change to the start re-rolls a seed's outcome, and 3 of 8 original seeds reached 70% at 2 epochs, so a rescue must beat this sham. |
| P3b | 0.05 | 0.05 | 3, 5 | Code resolution, at a matched starting signal-to-noise ratio |
| P3c | 0.2 | 0.2 | 3, 5 | Code resolution, at a matched starting signal-to-noise ratio |

- **Before P3 runs**, read `feature_std / sigma0` from the original arm's training logs, seeds 1 to 8, over their first two epochs. If the failing seeds (3, 5, 6, 7) reach `SPREAD_MIN` later than the working seeds (1, 4, 8), or never, the instability looks like the TwoRoom trap.

**P4. Reconstruction-free control.** I2: `model_class: nce`, `lam = 1`, `beta = 1` (nats), seed 1, `SCREEN_STEPS`.

### 6.2 Part T: TwoRoom

**T-pre (CPU).** The anchor table and `gap_T` (Section 5.5).

**T-gate.** Seeds 1 and 2, to `TWOROOM_GATE`. Every arm uses `forecast_scored`, `calibrate_state_std = sigma0`, `beta = 1/4`, `sigma0 = 0.1`, `rho = 0`, `forecast_noise = "sample"` and `one_step`, unless stated.

| Arm | `lam` | Anchor | Other | Differs from |
| --- | --- | --- | --- | --- |
| T1 | 0 | first frame | | your original arm at 2,000: only the initial scale |
| T2 | 0 | past, `gap_T` | | T1: only the anchor |
| T3 | 1 | first frame | | T1: only `lam` |
| T4 | 1 | past, `gap_T` | | canonical; T2: only `lam`; T3: only the anchor |
| T5 | 0.5 | first frame | rollout, `mean`, `rho = 0.8` | your TF arm at 2,000: only the initial scale (and the coin stream) |
| T6 | 1 | past, `gap_T` | `beta = 1/16` | T4: only the price |
| T7 | 1 | past, `gap_T` | `sigma0 = 0.05` | T4: only the code resolution (`sigma0`, and with it the initial scale's target) |
| T8 | 1 | past, `gap_T` | `sigma0 = 0.2` | T4: only the code resolution (`sigma0`, and with it the initial scale's target) |
| T9 | 0.5 | first frame | rollout, `sample`, `rho = 0.8` | T5: only the forecast noise |

**Evaluation at 2,000 updates:** health, the use gate and the rollout diagnostics (Section 5.6), probes, and planning under both controllers.

**T-continuation (automatic).** Every arm that passes the use gate continues to `TWOROOM_FULL` (`--stop-step 6000 --resume`) and is evaluated the same way.

### 6.3 Part K: Cube

**K-pre (CPU).** The anchor table and `gap_K` (Section 5.5).

**K-main.** Seeds 1 and 2, to `CUBE_FULL`. Every arm uses `forecast_scored`, `calibrate_state_std = sigma0`, `beta = 1/4`, `rho = 0`, `forecast_noise = "sample"` and `one_step`.

| Arm | `lam` | Anchor | `sigma0` | Differs from |
| --- | --- | --- | --- | --- |
| K1 | 1 | past, `gap_K` | 0.1 | canonical |
| K2 | 1 | first frame | 0.1 | K1: only the anchor (the rule of your runs) |
| K3 | 1 | constant (`renderer_anchor_mode: "zero"`) | 0.1 | K1: only the anchor (none) |
| K4 | 0 | past, `gap_K` | 0.1 | K1: only `lam` |
| K5 | 1 | past, `gap_K` | 0.05 | K1: only the code resolution (`sigma0`, and with it the initial scale's target) |
| K6 | 1 | past, `gap_K` | 0.2 | K1: only the code resolution (`sigma0`, and with it the initial scale's target) |

**Evaluation at 12,000 updates:** health, the rollout diagnostics, content probes, the oracle audit, and planning under both controllers (Section 5.7).

**K-mature (needs our agreement; Section 8.3).** The best K arm, both seeds, from 12,000 to `CUBE_MATURE`.

### 6.4 Stage 2: the forecast switches on TwoRoom and Cube (needs our agreement)

After the PushT report and the T and K reports, we agree which two P2 cells to carry to the harder tasks:

- the selected cell (Section 8.1);
- the priced cell whose open-loop spread ratio, carry ratio and motion gain lie closest to 1, if it is different.

They run on TwoRoom to `TWOROOM_FULL` and on Cube to `CUBE_FULL`, seeds 1 and 2, at the anchor rule and `sigma0` that T and K select. That is about 30 GPU-hours. No Stage-2 run starts before agreement.

### 6.5 Counts, GPU-hours and priorities

| Part | Runs | Approximate GPU-hours (training) |
| --- | --- | --- |
| D0 | none (evaluation of existing checkpoints) | under 5 |
| P1 | 3 L0 runs of 10 LeWM epochs | about 58 to 71 (3 × 152,650 updates at the 0.46 to 0.56 s per update native LeWM ran on Cube; measure) |
| P2 | 10 cells × 2 seeds to 2 epochs | about 126 (rollout cells cost more per step; measure) |
| P2c | 1 cell: 2 continuations to 3 epochs, 2 new seeds to 2 epochs | about 19 |
| P3 | 10 runs to 2 epochs | about 63 |
| P4 | 1 short run without a renderer | about 3 |
| **Parts D0 and P** | | **about 275 to 290** |
| T-gate | 9 arms × 2 seeds to 2,000 updates | about 15 |
| T-continuation | up to 18 runs from 2,000 to 6,000 | up to 30 |
| K-main | 6 arms × 2 seeds to 12,000 updates | about 60 |
| **Parts T and K** | | **about 105** |
| The conditional PushT arm (automatic, Section 8.1) | 2 two-epoch runs | about 13 |
| Failure reruns within the caps of Section 11 (automatic) | up to 6 runs | up to about 27 |
| Stage 2 (with agreement) | 2 cells × 2 seeds on TwoRoom and Cube | about 30 |

Evaluation GPU time is extra; measure and report it. Everything without a dependency may start at once. When compute is short, run by tier:

- **Tier 1 (about 200 to 215 GPU-hours):**
  - D0, the CPU tables and the P3 log check;
  - P1;
  - P2 cells F1, F2, F5, F7 and F8 (noise at your FO setting, priced TF, horizon, the canonical cell, and the canonical cell with `rho`);
  - T1 to T6 and T9;
  - K1 to K4.
- **Tier 2:** P2 cells F3, F4, F6, F9 and F10; P3; T7 and T8; K5 and K6; P4.

Before launching, report the measured step times and projected GPU-hours. Wait for our acknowledgement if they exceed this table by more than 25%.

## 7. Frozen predictions (record them, with the commit hash, before launch)

Each prediction states what the theory predicts and how confident we are. A failed prediction is a result, not an error.

**D0 and PushT**

1. **D0, contraction (low to moderate).**
   - In the 3-frame open-loop rollout at horizon 5, original has the smallest spread ratio and carry ratio of the five PushT arms. Lower-beta, TN, FO and TF all lie closer to 1.
   - The gap is largest in the lowest-spread quartile of dimensions.
   - Among original's eight seeds at 2 epochs, the four that failed planning (3, 5, 6, 7) have smaller horizon-5 spread ratios than the three that worked (1, 4, 8).
   - At the measured average spread the pull per step is small (Section 0.6), so a clear ordering with small magnitudes counts as held.
2. **P1 (moderate).** L0 at 10 epochs reaches at least 90% P0 on the primary cases (published: 96 ± 2.8). No prediction at 3 epochs.
3. **Forecast noise (moderate).**
   - With one-step forecasts, pricing costs little: F5 is non-inferior to F3, and F6 to F4.
   - With rollouts the noise accumulates, so F1 may trail FO, by at most 5 points. F2 is non-inferior to TF.
   - The open-loop spread ratio is farther below 1 for priced than for noiseless cells at `rho = 0`, and the gap shrinks at `rho = 0.8`.
4. **Horizon (moderate).**
   - Rollout cells have spread ratios and motion gains closer to 1 at horizons 7 to 13, and smaller drift, than their one-step counterparts.
   - The advantage is smaller at `rho = 0.8`.
5. **`rho` (moderate to high).**
   - `rho = 0.8` moves the open-loop spread and carry ratios toward 1 in every paired context.
   - The largest effect is in one-step cells, which have no rollout training to counter shrinkage.
6. **`lam` (low).**
   - F7 is non-inferior to F5.
   - F9 has the lowest agent and block probe R² of the priced cells, because at `lam = 1` with rollouts only frame 0's state receives renderer reward.
7. **The method against the evidence (moderate).** The best priced cell is non-inferior to FO. If that cell has `rho = 0.8`, it is also non-inferior to TF.
8. **Calibration (high).** Every `lam > 0` cell has a one-step noisy gain of at least `SLOPE_MIN`.
9. **P2c (moderate).** The selected cell's seeds 3 and 4 each reach at least 70% P0 at 2 epochs (planning seeds 42 to 44), where original scored 22 and 76.
10. **P3 (moderate).**
    - At least three of P3a's four seeds reach 70% P0 at 2 epochs, while at most one of the two sham seeds does.
    - At `sigma0 = 0.05`, `state_spread_over_sigma0` and the noisy-state information bound are larger, and the open-loop spread ratio is closer to 1, than at 0.2. That is, the encoder does not simply rescale its state with `sigma0`. Compare rates only after removing the `sigma0`-dependent constants (Appendix A.9).
    - P0 orders 0.05 ≥ 0.1 ≥ 0.2 (low confidence).
11. **P4 (moderate).** I2 keeps the agent (probe R² of at least 0.8). Its block probe and P0 fall below the selected cell's.

**TwoRoom**

12. **The initial scale (moderate to high).** T1 to T4 and T6 to T9 pass the use gate at 2,000 updates; T5 has no prediction here (see prediction 13). The uncalibrated counterparts had `feature_std / sigma0` of 0.0015 and posterior-swap effects of at most 0.0003% at t = 0.95.
13. **The noiseless shortcut (moderate).** With a first-frame anchor and a live state, T5 relies more on actions than T9. T5's wrong-action effect, relative to its swap effect on the shared `one_step` condition, exceeds T9's.
14. **Anchor on TwoRoom (moderate, conditional on T-pre).** If the agent is stale relative to the first-frame anchor in more than 80% of rendered frames, T1 and T2, and likewise T3 and T4, differ by less than 0.1 in noisy agent-probe R².
15. **Planning (moderate).** At 6,000 updates, the best continuing arm reaches at least 60% on the primary cases, averaged over seeds.
16. **`sigma0` on TwoRoom (low).** T7 has a larger `state_spread_over_sigma0` and information bound than T4, and T8 smaller ones: the encoder does not rescale its state with `sigma0`. No planning prediction.

**Cube**

17. **K-pre (moderate).** With the first-frame anchor, the cube is stale in fewer than 30% of rendered frames, and the effector in more than 60%. With `gap_K`, the cube's stale share rises but stays below 50%.
18. **Cube content (moderate).** The cube partial rank orders K3 ≥ K1 > K2. K2 stays at or below `RANK_LOW`, and K3 reaches at least `RANK_HIGH`. The clean cube readout error of K1 and K3 falls below TF's 4.7 to 5.1 cm.
19. **`lam` on Cube (moderate).** At the same anchor, K1 has higher cube content than K4.
20. **`sigma0` on Cube (moderate).** K5 has a higher cube partial rank than K1, and K6 a lower one. A smaller `sigma0` makes slow components cheaper to store (Appendix A.9).
21. **Cube planning (low).** The best K arm is within `HARD_TIE` of native LeWM at 12,000 updates under the same controller.

## 8. Decision rules

### 8.1 PushT

**Selection (P2 to P2c).**

1. **Candidates** are the priced cells F1, F2 and F5 to F10. The noiseless cells and your arms are comparisons only.
2. **Exclude** any candidate that fails its forecast collapse test, or whose one-step noisy gain stays below `SLOPE_MIN` after the fallback of Section 11.
3. **The best cell** is the one with the highest pooled P0 over seeds 1 and 2, on all six planning seeds (300 episodes per training seed). The tied set is every candidate that is not worse than it (Section 1.4).
4. **Within the tied set, prefer** `lam = 1` over 0.5, then one-step over rollout, then `rho = 0` over 0.8: the simpler code wins ties.
5. **Timing.** Select once every scheduled P2 cell is evaluated: the Tier-1 cells alone if Tier 2 is not run, otherwise all ten.

The selected cell goes to P2c.

| Result | Then |
| --- | --- |
| The selected cell is non-inferior to TF | The principled version keeps your best result; the canonical method uses the selected switches. |
| It is worse than TF | Attribute the gap from the paired contrasts at the selected settings: its `rho` twin (F8 against F7, F6 against F5, F2 against F1, F10 against F9), F2 against TF for forecast noise, and the horizon and `lam` contrasts. Report the attribution; the selected cell still continues to P2c. If a `rho = 0.8` cell is selected, `rho` enters the empirical design as a posterior family with an exact KL, not a heuristic. |
| It is worse than FO | Pricing or the horizon costs planning. Read F1 against FO (noise) and F5 against F1 (horizon), and report before Stage 2. Never adopt noiseless forecasts into the method. |
| Pricing costs more than 5 points only in rollout cells | The cost is the noise accumulating over the rollout (Section 0.4); one-step is the right priced form. |
| D0 shows original's horizon-5 spread and carry ratios within 0.05 of TF's, and no relation to the failing seeds | The contraction explanation fails. Interpret `rho` and the forecast effects through P0, drift and the motion gain, and report it. |
| D0 confirms contraction, and P3's `sigma0 = 0.05` reduces it relative to 0.2 | The state's signal-to-noise ratio is a design lever. Propose `sigma0` for the harder tasks from T7, T8, K5 and K6. |
| P2c: one of seeds 3 and 4 falls below 70% while TF holds | Run the selected cell at `beta = 1/16`, seeds 3 and 4 to 2 epochs (the conditional arm). Lower-beta was the stable `lam = 0` arm over eight seeds. |
| L0 at 3 epochs is non-inferior to the selected cell at 3 epochs | AS-JEPA has no planning advantage on PushT at matched data. The PushT claims rest on the mechanism results. |
| The selected cell at 3 epochs trails L0 at 10 epochs by more than 5 points | Propose its continuation to 10 epochs, and wait for our agreement. |
| At least three of P3a's four seeds reach 70%, and at most one sham seed does | The PushT instability was the trapped state. Calibration becomes part of the recipe on every dataset, and a later PushT table uses calibrated arms. |
| P3a does no better than the sham | The PushT instability has another cause, or four seeds cannot show it. Report it with the log check and D0. On the harder tasks, calibration is decided by T1 against your original arm. |

### 8.2 TwoRoom

| Result | Then |
| --- | --- |
| T1 passes the use gate and plans at least 30% by 6,000 updates | The initial scale explains your TwoRoom failure. Calibration becomes part of the recipe; tell your campaign. |
| T1 fails, but T3 or T4 passes | The forecast reward is needed in addition to the scale. |
| T5 passes and plans like T9 | With a live state, the noiseless shortcut is not decisive on TwoRoom. |
| T5 fails or leans on actions (prediction 13) while T9 does not | Pricing closes the shortcut. That is evidence for the priced forecast on any dataset with a first-frame anchor. |
| T1 to T9 all fail the use gate | The initial scale is not enough, and the pooled readout is the bottleneck. Stop Part T and report; this is meeting point (b). |
| An arm passes the use gate but plans below 30% at 6,000 updates | The state carries the agent but planning fails. Report the agent probe, the rollout diagnostics, the health table and the planning logs, and stop for a joint diagnosis. |
| T7 or T8 differs from T4 by more than `HARD_TIE` in planning at 6,000 updates | Report it; Stage 2 uses the better `sigma0`. |

### 8.3 Cube

| Result | Then |
| --- | --- |
| K-pre: the cube is stale in at most 50% of rendered frames at `gap_K` | A past anchor can only partly remove the free cube pose, so K3 is the decisive arm. Run all arms anyway. |
| K1 or K3 reaches `RANK_HIGH`, and K2 stays at or below `RANK_LOW` | The anchor was hiding the cube from the state. Adopt the better rule as the Cube recipe; this is meeting point (c). |
| K1, K2 and K3 all stay at or below `RANK_LOW` | The anchor is not the cause. Report the content probes, and K5 against K1. No rescue runs. |
| K5 raises the cube rank above K1 by more than 0.1 | The entry price was holding the cube out. Stage 2 uses `sigma0 = 0.05` on Cube. |
| The best K arm is within `HARD_TIE` of native LeWM at 12,000 updates, under either controller | If your maturity runs show native LeWM gaining more than `HARD_TIE` between 12,000 and 23,342 updates, propose K-mature and wait for our agreement. |
| The best K arm trails native LeWM by more than `HARD_TIE` | Report it with the oracle audit and the rollout diagnostics. The Cube gap remains open; no automatic extension. |

## 9. Reports

- **PushT report:**
  - D0, every rollout-gain report;
  - P1 to P4, every Section 5.1 to 5.4 metric per cell;
  - main-effect tables per switch, with each paired contrast of Section 6.1 listed separately;
  - predictions 1 to 11, marked held or failed;
  - the Section 8.1 selection and decisions.
- **TwoRoom report:**
  - the anchor table, `gap_T` and `initial_scale.json` values;
  - spread curves, the health table, use-gate verdicts and the rollout diagnostics for every arm;
  - probes, and planning at 2,000 and 6,000 updates under both controllers;
  - predictions 12 to 16, and the Section 8.2 decisions.
- **Cube report:**
  - the anchor table and `gap_K`;
  - health, the rollout diagnostics, content probes and oracle-audit ranks;
  - planning under both controllers;
  - predictions 17 to 21, and the Section 8.3 decisions.

Each report lists the references it compares against, with their paths.

## 10. Tests and checks before any long run

1. **CPU contracts, 24 tests:** `tests/as_jepa/test_round2.py` (15), `test_benchmark_gap.py` (5) and `test_rollout_gain.py` (4). Run from the repository root:

   ```bash
   PYTHONPATH=.:runtime:<stable-worldmodel checkout> python -B -m unittest \
       tests/as_jepa/test_round2.py tests/as_jepa/test_benchmark_gap.py tests/as_jepa/test_rollout_gain.py -v
   ```

   They check:
   - **Equivalence with your arms:**
     - `lam = 0` reproduces `ASJepaLeWM.loss` exactly, for every horizon, forecast noise and `forecast_grad`;
     - `lam = 0` with `rho = 0.8` reproduces `RobustLeWM(rho = 0.8)` bit for bit;
     - the noiseless rollout equals your forecast loop.
   - **Pairing:**
     - the main random stream is identical across `lam`, horizon, forecast noise and `rho`;
     - noiseless and sampled forecasts share their coins, and differ by exactly `sigma0 * xi`;
     - `forecast_share` is about `lam`.
   - **Gradient routing:** the routing of each setting, including noiseless and correlated cells.
   - **Model guards:**
     - at `lam = 1` the last frame is not rendered from its own encoding;
     - the calibration slope is exact for a known overshoot;
     - the subclasses refuse switches they do not implement;
     - the contrastive variant has no renderer.
   - **Anchor rule and staleness table:** as in revision 2, plus the exact fallback share.
   - **Rollout diagnostics:** they recover a known motion attenuation and a known centre pull, per quartile of dimension spread; compute the AR(1)-noisy one-step gain correctly; and run end to end on benchmark windows.

   Also rerun your existing `tests/as_jepa` suite.
2. **Pairing check.** F7's recorded initial weights at seeds 1 and 2 must equal those of `lower-beta-seed1` and `lower-beta-seed2`, tensor for tensor. If they differ, stop and report.
3. **GPU smokes (200 steps each, one GPU).** Run F7, F2 (rollout with `rho`), F3 (noiseless one-step), L0, T4 (TwoRoom, calibrated, `gap_T`), K1 (Cube, gap anchor), K3 (Cube, constant anchor), K5 (`sigma0 = 0.05`) and I2. Record step time and peak memory. Check that:
   - the loss is finite;
   - `forecast_share` is about `lam`;
   - `initial_scale.json` shows `original_std` of about 1e-3 to 2e-3, and the first `metrics.jsonl` row shows `feature_std` near the run's `sigma0`;
   - for 64 training windows of T4 and K1, either `anchor_row <= rows[0] - gap * 5` or `anchor_fallback` is True, and `fallback_share()` matches the table;
   - `benchmark_evaluate.py` loads the T4 checkpoint through its `_target_`, and `rollout_gain.py` runs on the F7 and K1 smoke checkpoints.

## 11. Failure responses

| Symptom | Likely cause | Response |
| --- | --- | --- |
| A priced candidate cell (F1, F2, F5 to F10) has a one-step noisy gain below `SLOPE_MIN` | joint training exaggerates the forecasts | Rerun it, on both seeds, with `forecast_grad = "encoder_only"`, and report both. At most two cells (4 runs) without asking; noiseless cells and your arms are not rerun. |
| A T arm's `feature_std / sigma0` falls back below `SPREAD_MIN` after starting near 1 | the rate pulls the state down faster than the renderer rewards it | Rerun T4 once, on both seeds, with `beta = 0` from the same initialization (2 runs to 2,000 updates). If the state survives, the price is above TwoRoom's barrier at `beta = 1/4`; report. |
| Every T arm fails the use gate | Section 8.2 | Stop Part T; report the spread curves and probes. |
| The anchor table gives no gap meeting `FALLBACK_MAX` | short episodes | Use gap 1 and report its fallback share. |
| A rollout cell's step time exceeds `STEP_S` by more than 50% | sequential predictor passes | Report it. Run the Tier-1 rollout cells first, and ask before the Tier-2 rollout cells. |
| A PushT `lam > 0` cell fails the forecast collapse test | the price is above the collapse barrier | Exclude it from selection and report. No rescue runs. |
| The BF16 and FP32 verdicts disagree (collapse or rollout diagnostics) | BF16 hides small state differences | Use the FP32 verdict for that cell and its reference. |
| NaN or loss spikes | numerics | Keep the rate in FP32 and the gradient clip. If spikes persist, halve the learning rate for that cell only, mark its comparisons as unpaired, and report. |
| A run is interrupted | `stop_after` does not act on resume | Resume with `train_stage --config <config> --stop-step <N> --resume`. |

## 12. Latitude: what you may change without asking

- Compute allocation and the order of runs, within the tiers of Section 6.5.
- Evaluation sample counts, upward, applied to an arm and its references alike.
- Adding seeds to any arm, applied to its paired references too.
- Implementation mechanics not fixed here: launch wiring, report scripts, file layout of outputs.

Keep one GPU per run and the references' learning rates. Everything in Sections 3, 4 and 8 needs our agreement before it changes: what the renderer reads, the noise structure, the pairing, the rate, the anchors, the initial scale and the decision rules. When unsure, stop and ask.

## 13. Order of work

1. Branch, copy the reference files and register them (Section 2.1). Run the 24 CPU tests and the pairing check (Section 10).
2. Run D0, T-pre, K-pre and the P3 log check (no training). Fix `gap_T` and `gap_K`.
3. Add the evaluation options (Sections 5.2 and 5.6). Run the GPU smokes, then report step times and projected GPU-hours.
4. Launch Tier 1, then Tier 2 (Section 6.5). Parts P, T and K run in parallel.
5. Evaluate as Section 5 describes. Continue the T arms that pass the use gate, and the P2 cell selected by Section 8.1.
6. Write the three reports (Section 9), and stop for our agreement before Stage 2.

## Appendix A. The theory this round relies on

### A.1 The objective at `(beta, lam)`

For each rendered frame `t = 1 .. T`, a coin `c_t ~ Bernoulli(lam)` chooses the renderer's condition:

- the posterior sample `s_t`, when `c_t = 0`;
- a prior sample `s_hat_t = m_t + sigma0 xi'_t`, with `m_t = prior(s_<t, a_<t)`, when `c_t = 1`.

Training minimizes

```
J(beta, lam) = sum_t E[ -log p_renderer(x_t | condition_t, x_anchor) ]
             + beta * [ KL(q(s_0|x_0) || N(0, I)) + sum_t KL(q(s_t | s_<t, x) || p_prior(s_t | s_<t, a_<t)) ]
```

At `beta = 1` this is the negative evidence lower bound of a generative model in which each frame is decoded either from its inferred state or from the model's own forecast, chosen by a coin. So every `lam` is a code length, and `lam = 0` is round 1.

How the switches enter:

- **`rho`** changes only the posterior family `q`. Its conditional is `N(mu_t + rho (s_t-1 - mu_t-1), sigma0^2 (1 - rho^2))`, and the KL stays exact.
- **A rollout** replaces the one-step prior sample by a sample of the multi-step forecast. The renderer term becomes `E[-log p(x_t | h_t)]` with `h_t` sampled from the prior given `s_0` and the actions. By Jensen's inequality this is an upper bound on the multi-step predictive code length, the same kind of term as PlaNet's latent overshooting, scored in pixels.
- **A noiseless forecast** is not a sample of the model, so with it `J` is no longer a code length.

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
| Detail fresh in each frame | `beta` per nat | `1 - lam` per nat; no forecast can predict it | Excluded if and only if `beta >= (1 - p)(1 - lam)`, with condition dropout `p = 0.1`; so at every `beta > 0` when `lam = 1`. Tested in a later round (the dot domain). |
| Persistent appearance | `beta` per nat added after the first state | 0 in both branches; the anchor supplies it | Excluded at every `beta > 0`, for codes that keep appearance separate from dynamics. Tested in a later round (the colour domain). |
| Dynamics state (poses) | entry once, then surprise | both branches, for what the anchor does not show (A.7) | Kept when it pays for its entry within the window (round-1 condition A5), at every `lam`. |
| Fixed property that the dynamics read and the anchor shows | entry once | at `lam = 0`: 0. At `lam = 1`: its value for forecasting contacts | Stored at `lam = 1`, not in an exact `lam = 0` code. Not tested in this round. |

### A.4 What `lam = 1` optimizes

At `lam = 1` the objective selects the smallest state whose one-step forecast, made without seeing the anchor and rendered with the anchor, is most informative about the next frame. This is a sequential, anchor-conditional relative of the past-future information bottleneck (Creutzig, Globerson and Tishby 2009). Under deterministic dynamics given the actions, maximizing it makes the prior's context sufficient for the future. The sufficiency claim is about the pair (prior context, anchor), not about `s_t` alone.

### A.5 Known costs of forecast rendering, and what watches them

- **Exaggerated forecasts under joint training.** The renderer values forecast information at `lam` per nat, while the prior pays for exaggeration only to second order in surprise. The toys give a 0.5 to 3.5% overshoot. The one-step noisy gain watches it; the fallback is the encoder-only forecast gradient.
- **The noiseless-mean shortcut.** A noiseless forecast mean is a channel of unbounded capacity that nothing prices. In exact arithmetic it can carry any amount of information about the actions at no surprise cost; in BF16 only the mantissa limits it. With a first-frame anchor, this admits an empty state plus a prior that integrates the actions. Your TF arms on TwoRoom show the signature: action effect up to 72%, while no state swap exceeded 0.005%. A `sigma0` sample limits the channel to the code's resolution, and the surprise term prices the mean's departure from the encoded state. T5 against T9 tests this with one variable.
- **Noise accumulation in priced rollouts.** With a unit carry, a sampled rollout's noise grows as `sqrt(k + 1) * sigma0`, because the rollout starts from a sample: about `2.8 sigma0` at step 7. If the prior pulls toward a centre with carry `a`, it saturates at `sigma0 / sqrt(1 - a^2)`. The renderer therefore learns blurrier long-horizon frames. That is the honest predictive uncertainty, but it weakens the training signal at long horizons.
- **Out-of-distribution tests.** At `lam = 1` the renderer never sees encoded states; hence the forecast versions of the renderer diagnostics.

### A.6 Why a reconstruction objective can still leave the state unused

**Collapse in a JEPA.** The latent prediction loss `E||g(f(x_t), a_t) - f(x_t+1)||^2` is exactly zero for a constant encoder `f`. The collapsed solution is a global minimum, so something outside the prediction loss must exclude it. AS-JEPA's objective has no such minimum: a constant state leaves the renderer only the anchor, and a state that encodes what the anchor does not show lowers `J`. The trapped TwoRoom state is instead a stationary point of training:

1. The renderer reads `c = mu + sigma0 xi`. Per dimension, `c` carries at most `0.5 ln(1 + Var(mu) / sigma0^2)` nats. That is about 5e-5 at the default initial spread of 1e-3, and about 1e-6 at the trapped 1.5e-4.
2. A renderer cannot use what it cannot read, so it learns to ignore `c`.
3. The encoder's gradient from the renderer passes through that derivative, so `mu` does not grow. With `beta > 0` the rate also pulls `mu` toward the prior mean.

In your linear toy at `beta = 0`, the escaping direction grows at about `2 eta^2 / sigma0^2` for state scale `eta`. At the default initial scale that is a ten-thousandth of its value at `eta = sigma0`.

SIGReg holds each embedding dimension at unit variance at every step, so LeWM never sits at this scale. AS-JEPA's counterpart is an initialization at the code's resolution. The phenomenon is known for VAEs as posterior collapse (Bowman et al. 2016; He et al. 2019).

### A.7 What an anchor gives away

At `lam = 0`, the reward for storing a component in `s_t` is only what the frame shows beyond the anchor. A component whose value in frame `t` equals its value in the anchor earns nothing there, and neither does one that is a function of others already in the state, such as a carried cube. Three anchor rules differ in how much they give away:

- **First frame:** everything that does not change within the window. On Cube that includes the cube in most windows.
- **A past frame at gap `g`:** everything that has not changed since the anchor time.
- **Constant:** nothing. This suits datasets whose appearance does not vary across episodes.

The principle: the anchor should share the episode's appearance and be as uninformative as possible about the current value of anything that moves. At `lam = 1`, a component also earns its value for the forecast. To forecast whether the gripper closes on the cube, the prior needs the cube's pose.

### A.8 Dynamics shrinkage: why the prior can contract its rollouts, and the four levers

**The calculation** (your `robustness20261029/THEORY.md`, exact for a linear one-step prior). Take a prior `m = A s_prev` and posterior samples `s = mu + sigma0 eps` with independent `eps`. The expected surprise is

```
||mu_next - A mu_prev||^2 / (2 sigma0^2) + 0.5 ||A||_F^2.
```

The second term is the price of carrying the previous sample's noise into a prediction that the next sample does not share. Consider a component that persists (`mu_next = mu_prev`), whose spread across windows is `r sigma0`, and a prior that carries it toward a centre `c` with a scalar carry `a`: `m = c + a (s - c)`. The cost per dimension is `(1 - a)^2 r^2 / 2 + a^2 / 2`, minimized at

```
a* = r^2 / (1 + r^2).
```

This is the attenuation of a regression on a noisy regressor:

- **at `r = 1`** the prior halves the component's distance from the centre at every step;
- **at `r = 3`**, `a* = 0.9`, so 59% of the distance remains after 5 steps;
- **at `r = 10`**, `a* = 0.99`, so 95% remains.

If the prior averages `k` frames of history, the noise term falls to `a^2 / (2k)` and `a* = r^2 / (r^2 + 1/k)`. The planner starts from 3 context frames, and `predict_states` keeps up to 7 as the rollout proceeds, so averaging softens the effect without removing it.

**How large it can be in your runs.** The average state spread is 7 to 11.5 times `sigma0` (Section 0.6), where the single-frame carry is 0.98 to 0.99. So the mechanism can matter only in two ways:

- **through dimensions with a much smaller spread** than the average, which the per-quartile report of `rollout_gain.py` exposes;
- **through training dynamics.** Early on, the spread is about 2e-3 (`r` ≈ 0.02), so the rate-optimal prior carries almost nothing, and that may shape what the prior learns.

**Why it would matter for planning.** At `lam = 0` the rate is the prior's only training signal, so the prior settles at this pull. At planning time it is fed clean encoder means, with no noise left to discount, but it keeps the pull. Rollouts then contract toward the centre and forget where they started, and GoalMSE ranks candidate actions in a compressed geometry. `rollout_gain.py` measures this as spread and carry ratios below 1 (Section 5.3), without assuming where the centre is; the motion gain, which a centre pull does not bias, measures lag separately.

**The four levers:**

1. **Raise `r`.** A lower `beta` lets the encoder afford a wider state; a smaller `sigma0` makes the same spread worth more (A.9).
2. **Correlate the noise.** With AR(1) noise the penalty becomes `0.5 ||A - rho I||_F^2`, and the optimal carry is `(r^2 + rho) / (r^2 + 1)`: 0.9 at `r = 1` and `rho = 0.8`.
3. **Render forecasts.** At `lam > 0` the renderer's loss reaches the prior and rewards accurate forecasts, against the rate. Rollouts do so over several steps. Noiseless forecasts give the sharper signal, because a `sigma0` sample hides a pull smaller than `sigma0`.
4. **Give the prior longer history** (architecture; not varied here).

**Evidence so far.** The ordering of 7-step errors and P0 at 2 epochs (Section 0.6) is consistent with this mechanism, but it is not a test. FO and TF, which have the smallest average spread, must owe their rescue to the third lever. D0 measures the contraction directly; P2 and P3 vary the levers.

### A.9 The code resolution `sigma0`, and how it differs from `beta` and `rho`

**Entry.** Per dimension, entry costs `0.5 (mu^2 + sigma0^2 - 1 - ln sigma0^2)`. The part that depends on the encoder, `0.5 mu^2`, does not involve `sigma0`; only the constant does. A dimension's information through the renderer's channel is `0.5 ln(1 + Var(mu) / sigma0^2)`. So `sigma0` sets how many nats a unit of entry cost buys. A dimension that fills the prior holds about `ln(1/sigma0)` nats: 3.0 at `sigma0 = 0.05`, 2.3 at 0.1 and 1.6 at 0.2.

**Surprise.** In nats, surprise depends only on the ratio of an unpredicted change to `sigma0`, so changes cost the same at any `sigma0` once measured in nats. Therefore `sigma0` shifts the relative price of slow information, paid once at entry, against fast information, paid at every step. A smaller `sigma0` makes slow components cheaper. That is round 1's entry-price argument, and why it matters for Cube's slow cube (K5, K6).

**Signal-to-noise.** In a simple model, the encoder sets a dimension's spread `v` by balancing the entry cost against the renderer's reward. Then `v ≈ 1/beta' - sigma0^2`, nearly independent of `sigma0`, so `r^2 = v / sigma0^2` grows as `sigma0` shrinks. That means less shrinkage (A.8), a more precise forecast channel, and more nats stored at the same `beta`. With calibrated initialization the starting signal-to-noise ratio is 1 at any `sigma0`, so `sigma0` no longer interacts with the trap at step 0.

**Comparing rates across `sigma0`.** The logged rate includes entry's constant, `64 * 0.5 (sigma0^2 - 1 - ln sigma0^2)` per window: 159.8, 115.7 and 72.3 nats at `sigma0` = 0.05, 0.1 and 0.2. With `rho = 0.8` it also includes 12.2 nats per transition. A fixed state change also costs `1 / sigma0^2` more surprise at a smaller `sigma0`. So compare information through `noisy_state_information_upper_bound_nats` and `state_spread_over_sigma0`, not through the raw rate.

**`beta` and `rho` are different levers:**

- `beta` scales every price uniformly.
- `sigma0` changes entry against surprise, and the state's signal-to-noise ratio.
- `rho` changes only the noise on frame-to-frame differences, `2 (1 - rho) sigma0^2`, and leaves the renderer's per-frame noise at `sigma0`. It acts like a smaller `sigma0` for the dynamics alone.

## Appendix B. Design-to-plan trace

| Design point | Reason | Where | Checked by |
| --- | --- | --- | --- |
| Per-frame encoder | the state is a function of its frame; goal images encode without history | `ASJepaLeWM.encode`, unchanged | existing `test_contracts.py` |
| Anchor only to the renderer | appearance is free side information and must not be free in the state | unchanged | existing contracts |
| Past-only anchor on the harder tasks, chosen by a stated rule | the anchor must not show the current value of slow components (A.7), nor the future | `benchmark_gap.py`; Section 5.5 | `test_benchmark_gap.py`; T2 against T1, K1 to K3 |
| Actions only to the prior | the renderer must not do the dynamics | unchanged; forecasts carry actions only through the priced mean plus noise | `test_gradient_routing` |
| The renderer reads the posterior sample or a forecast; the method reads only priced forecasts | sufficiency (`lam`), and a priced channel | `ForecastScoredLeWM.loss`; Section 8.1 step 1 | `test_lambda_zero_reproduces_l2_loss`, `test_noiseless_forecasts_read_the_prior_mean_with_the_same_coins` |
| Correlated posterior noise with the exact KL | a posterior family, compared at `rho` ∈ {0, 0.8} (A.8) | `ar1`, the surprise target | `test_rho_reproduces_the_collaborators_correlated_posterior` |
| Code resolution `sigma0`, compared at three values | entry price, signal-to-noise ratio (A.9) | P3, T7, T8, K5, K6 | Section 7, predictions 10, 16, 20 |
| Initial state spread `sigma0`, set once | start where the renderer can read the state (A.6) | `calibrate_state_std` in `train.py` | `initial_scale.json` in the smoke; T1 against your original arm; P3 |
| Frame 0 never rendered | transient exclusion | unchanged | existing contracts |
| No stop-gradient by default | one objective, no heuristics | `forecast_grad = "joint"` | `test_gradient_routing` |
| One exact rate, one `beta`, `sigma0` and `rho` fixed in a run | the rate is a code length | unchanged | `test_kl_functions` |
| Paired random streams | clean comparisons across every switch | `fork_rng` stream | `test_main_random_stream_is_paired_across_lambda`; pairing check |
| Shrinkage measured, not assumed | the mechanism behind the levers (A.8) | `rollout_gain.py` | `test_rollout_gain.py`; D0 |
| Contrastive critic (P4) | a reconstruction-free control | `nce.py` | `test_no_renderer_and_gradients` |

## Appendix C. Changed from earlier revisions

**From revision 2:**

- **Deferred to a later round:** the dot domain (D1, D3) and the colour domain (E2, E4), with their tools.
- **The single canonical arm C1 became the factorial P2:**
  - it covers `lam`, forecast noise, horizon and `rho` at 2 epochs;
  - C1 is now cell F7, revision 1's C4 is F5, and revision 2's conditional C5 is F1;
  - the selected cell continues to 3 epochs and to seeds 3 and 4 (P2c).
- **New switches in the model class:** `rho` and `forecast_noise`, with 4 new tests.
- **New tool:** `rollout_gain.py`, with 4 tests. It replaces the calibration tool, and measures contraction (spread and carry ratios) and lag (motion gain) separately.
- **New Part D0:** diagnostics on your existing checkpoints.
- **P3 grew:** calibration on four failing seeds, a sham control on two, and `sigma0` ∈ {0.05, 0.2} on two.
- **New harder-task arms:**
  - TwoRoom T7 and T8 (`sigma0`), and T9 (priced forecasts against T5's noiseless ones);
  - T5 now runs in the new class, so that T9 differs from it in one switch;
  - Cube K5 and K6 (`sigma0`).
- **Stage 2 added:** two forecast cells on TwoRoom and Cube, after agreement.
- **New theory:** Appendix A.8 (dynamics shrinkage) and A.9 (the code resolution).

**From revision 1** (still applies): the price sweep at the working point, full L1, `sigma0 = 0.03` at `beta = 1`, the first-frame anchor test on PushT, block-shape variants, and the 10-epoch finals were dropped or replaced as revision 2 described. `learned_variance.py` stays in `round2/` for a later round and is not used.
