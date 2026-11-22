# Handoff: making the decoder-free AS-JEPA work, in two parallel tracks

Written 2026-11-18 on branch `lingjing/decoder-free-as-jepa`, for the collaborator and the agent that runs the experiments. Checked against your branch `as-jepa/lingjing-fullspec-20261102` at `318cbb10`. **Please reply in GitHub issue [sp12138/SPT-JEPA#1](https://github.com/sp12138/SPT-JEPA/issues/1).**

It replaces Parts P and R of [handoff-anchor-reader.md](handoff-anchor-reader.md). Part G of that handoff (the anchor-gap diagnostic, under 1 GPU-hour, no training) is unchanged and still requested.

## 0. For the agent reading this

- **What this asks.** Track 1 runs the decoder-free objective, with three corrections learned from your own results, on PushT, the dataset it was designed for. Track 2 runs a windowed, stale-anchor version on your Reach3D mainline. Track 1 decides whether the method works. Track 2 checks that it is compatible with your scaling protocol.
- **What needs your user's approval.** GPU time. Track 1 needs one 4-GPU slot for the gate (about 10 to 15 GPU-hours) and more only if an arm passes. Track 2 needs about 36 GPU-hours. PushT is outside your current Reach3D directive, so please bring this request to your user at your next hourly check and post the decision in the issue. If Track 1 cannot be scheduled, say so there, and we will look for compute on our side.
- **Priority.** Track 1, then Part G, then Track 2.
- **Your project rules hold throughout.** Every learner starts from random weights, reads only RGB and executed actions, and uses sample-level losses (no batch statistics across physical samples). There are no future or optimal action labels; in Track 2 the frozen 384-parameter head is trained only on the current executed action.
- **What we need back** is listed in Section 8. The first item costs nothing: the `reader_mse` values (training and, if available, held-out) of your earlier Reach3D anchor-reader runs, from `analysis/anchor-reader-first-hour-local-response-readout.json` and the `relative-anchor-reader-20261117` root.

## 1. What your results taught us, and what we changed

| Your result | What it means for the design | Change |
| --- | --- | --- |
| Round-2 factorial (`CLOSEOUT_20261103.md`): changing only `rho` from 0 to 0.8 improved all five matched pairs by 19 to 79 points; F7 to F8 went from 8.00 / 7.33% to 87.33 / 83.00% | Independent frame-to-frame posterior noise is what sank the renderer arms. Every one of the 62 decoder-free PushT recipes, and the PushT anchor reader, also used independent noise: `decoder_free.py` has no `rho` | `rho = 0.8` is canonical, with the exact AR(1) KL of `ForecastScoredLeWM`, the code path behind F8 |
| P3: initial-scale calibration at `sigma0` = 0.05, 0.1, 0.2 gave 4.3 to 5.2%; the sham (mean shift only) gave 40.3 / 30.3% | Rescaling the initial state to `sigma0` hurts | No initial-scale calibration (`calibration = None`, as in F8) |
| Your relative-anchor reader (`CONCLUSIONS.md`) cut the entropy term's gradient on the code's origin from 30.40 to 2.3e-6 | Conditional entropy is unchanged by translating with the anchor's code, but a finite reader still drags the origin | `reader_relative = True` is canonical. This is your fix, adopted as is |
| Your identity for the current-frame conditional entropy: under `z' = z + Q a` with `Q` orthonormal, `H(Y | X)` is fixed by the action distribution | With the anchor at the current frame, the reward cannot rank encoders. The design needs a stale anchor, so that the reward measures content the actions do not determine | Gap 8 on PushT; the window's first frame in Track 2; no current-frame arm |
| A frozen reader fit longer on 64 TRAIN windows: training noise error .402 to .191, held-out .426 to .745 | A reader can overfit, and then the encoder follows a wrong score on new states | Held-out `reader_mse` is reported at every gate, next to the training value (Section 4.3) |
| Your isometric action map pins the action directions | On those directions the reward is fixed by the action distribution; any useful reward lives in the orthogonal complement | Track 2 tests a reward restricted to the complement |
| Decoder-free window `lam = 0`: 1.91 / 1.65 nats beyond the anchor, but noisy agent R² .011 / .006 | Information beyond the anchor does not by itself select the task state | The gate reads probes, not the information value; the fallback identifies what fills the state |
| F8 against F10 (87.33 against 22.00%), and your early-log check (successful runs had SNR 0.009 at 2,000 steps) | One-step forecasts beat open-loop rollouts, and early spread is not a verdict | One-step only; the first gate is at one full epoch |

## 2. The method after these changes

```
posterior   s_t = mu_t + sigma0 * xi_t,   xi_t = rho * xi_{t-1} + sqrt(1 - rho^2) * eps_t,   rho = 0.8
reader      minimise  | xi_hat(mu_t - mu_A + sigma0 * xi', anchor) - xi' |^2,   fresh xi' ~ N(0, I), states detached
encoder     minimise  beta * Entry + (beta + lam) * sum_t Surprise_t + sum_t sg(score_t) . (mu_t - mu_A)
            Surprise_t = | mu_t + rho (s_{t-1} - mu_{t-1}) - m_t |^2 / (2 sigma0^2) + D * (-rho^2 - log(1 - rho^2)) / 2
            score_t    = -xi_hat(s_t - mu_A, anchor) / sigma0,   mu_A attached
```

- `m_t` is the prior's one-step prediction from the posterior samples and actions, exactly as in L2. `Surprise_t` is the KL of the AR(1) posterior's conditional `N(mu_t + rho (s_{t-1} - mu_{t-1}), sigma0^2 (1 - rho^2))` against the prior `N(m_t, sigma0^2)`, the same formula as `ForecastScoredLeWM` and `RobustLeWM`.
- **The reward is unchanged by `rho`.** Each frame's noisy state is still a draw from `N(mu_t, sigma0^2)`, so the entropy-gradient identity and Tweedie's formula hold per frame. The reader therefore trains on fresh independent draws, as before.
- **Why `mu_A` is attached.** A shift `c` of every code changes `mu_t - mu_A` by `c - c = 0`, so the encoder term gives a common shift exactly zero gradient. In the population the extra gradient on `mu_A` is the expected score of the conditional density, which is zero, so this changes nothing an optimal reader would do.
- The rest is as in [handoff-anchor-reader.md](handoff-anchor-reader.md) Section 1: two players, no EMA, no augmentation, no decoder, and the score detached because each player takes the gradient of its own loss.

## 3. Code

`as_jepa_lewm/anchor_reader.py` gains two arguments; everything else is as in your copy of 2026-11-16.

- `rho` (default 0): AR(1) posterior noise and the exact KL above. It reuses `ar1` from your `as_jepa_lewm/forecast_scored.py`. It needs `estimator` in {denoiser, nce}, `prior_input = "sample"`, and either `lam = 0` or the one-step noiseless forecast; other combinations raise `ValueError`. Logged as `temporal_noise_rho`.
- `reader_relative` (default False): the denoiser reads `s_t - mu_A` and the encoder term is `sg(score) . (mu_t - mu_A)`. Denoiser only; nce is translation invariant already. Logged as `reader_relative`.
- **Defaults reproduce every tensor bitwise**, so existing configs and checkpoints are unaffected.

Five new tests in `tests/as_jepa/test_anchor_reader.py`:

| Test | What it establishes |
| --- | --- |
| `test_rho_argument_validation` | the combinations above raise; the canonical ones build |
| `test_rho_zero_is_bitwise_the_default` | `rho = 0` equals the default for both estimators and all five forecast variants, including the main random stream afterwards |
| `test_rho_gives_the_ar1_posterior_and_its_exact_kl` | unit marginal variance and lag-one correlation 0.8; the posterior sample is the AR(1) transform of the same draws; the surprise equals the formula above exactly; `Forecast_t = Surprise_t` for the one-step noiseless forecast; gradients reach encoder and pool |
| `test_reader_relative_gives_a_common_shift_exactly_zero_gradient` | with `beta = lam = 0`, the gradient on a constant added to every code is at most 1e-5 of the absolute version's |
| `test_reader_relative_denoises_the_change_from_the_anchor` | the loss is exactly the rate, the relative encoder term and the weighted reader loss |

**Verified on 2026-11-18,** on CPU (Python 3.12, torch 2.7.0), against a `git archive` export of `318cbb10`: `test_anchor_reader.py` (21), `test_decoder_free.py` (14), `test_anchor_reader_spt.py` (6) and `test_round2.py` (15), 56 of 56 pass. Nothing ran on a GPU or on real data.

Copy the two files from this directory, then run from your repository root:

```bash
cp <this branch>/docs/lingjing/decoder-free/as_jepa_lewm/anchor_reader.py as_jepa_lewm/
cp <this branch>/docs/lingjing/decoder-free/tests/as_jepa/test_anchor_reader.py tests/as_jepa/
PYTHONPATH=.:runtime:<stable-worldmodel checkout> python -B -m unittest \
  tests/as_jepa/test_anchor_reader.py tests/as_jepa/test_decoder_free.py tests/as_jepa/test_anchor_reader_spt.py -v
```

## 4. Track 1: the PushT gate

### 4.1 Configuration

Standard PushT (`D-std`), seeds 1 and 2, class `AnchorReaderLeWM`, everything else as your decoder-free template (`experiments/lingjing_fullspec_20261102/catalog.py`, the `decoder_free` base), with these model arguments:

```python
dict(mode="L2", sigma0=0.1, window=8, anchor_gap=8,
     horizon="one_step", forecast_noise=False, prior_input="sample", conditioning="anchor",
     estimator="denoiser", reader_anchor="frame", reader_relative=True, rho=0.8,
     reader_width=512, reader_depth=3, reader_draws=4, reader_lr=5e-4, reader_loss_weight=0.01)
# recipe level: calibration=None (no initial-scale calibration, as in F8)
```

The reader's optimizer group comes from `model.training_optimizer_config` (handoff-anchor-reader.md Section 2). Before any long run, do the two-update start of that handoff's Section 9: the reader has its own parameter group, and every gradient group is finite and non-zero.

### 4.2 Arms

| Arm | `estimator` | `beta` | `lam` | `w = beta + lam` | `rho` | `reader_relative` | Question |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Q1 | denoiser | 0.3 | 1 | 1.3 | 0.8 | True | the old P1 point with all three corrections |
| Q2 | denoiser | 0.3 | 0 | 0.3 | 0.8 | True | the reward dominates |
| Q3 | denoiser | 1 | 0 | 1 | 0.8 | True | exact nats |
| Q0 | denoiser | 0.3 | 1 | 1.3 | **0** | True | Q1 with independent noise: was `rho` what failed? |
| Q5 | nce | 0.3 | 0 | 0.3 | 0.8 | (not applicable) | contrastive control |

That is 10 runs. `w` is the only searched knob, through these preset points; nothing else is tuned.

### 4.3 Gate at one epoch (12,491 updates)

Run `latent_eval.py --fp32 --windows 64` and read the last 500 training rows. An arm passes only if both seeds pass all three:

1. **Alive:** validation `spread_over_sigma0 >= 0.3` and, for the denoiser arms, training `reader_mse >= 0.05` (below that, the reader recovers the noise almost fully, so the state carries almost nothing beyond the anchor).
2. **Task content:** noisy agent probe R² at least 0.5.
3. **Reported, not gating:** held-out `reader_mse`, the mean of the logged `reader_mse` from `model.loss` on the same 64 validation windows, under `torch.no_grad()` and `model.eval()`, with no optimizer step. Flag the arm if the held-out value exceeds the training value by more than 0.1. That is a quarter of the values seen so far (about 0.4), and your frozen-fit gap was 0.55.

Also report the noisy block probe R² per arm.

**Budget.** The old Part P estimate was about 33 GPU-hours for 12 runs, most of them to the gate and the passing ones to 3 epochs. That scales to roughly 1 to 1.5 GPU-hours per run to the gate, so 10 to 15 GPU-hours for these 10. Please measure seconds per update in the two-update start and correct the figure in your first report.

### 4.4 After the gate

Arms that pass on both seeds continue to 37,473 updates (3 epochs) with the full PushT package: planning on primary seeds 42, 43, 44 and fresh seeds 1042, 1043, 1044, both probes, latent rollouts at h1, h3, h7 and h13 against copy, the calibration slope, and training and held-out `reader_mse`. References, all yours:

| Model | Updates | Primary / fresh |
| --- | --- | --- |
| Native LeWM (L0), 3 seeds | 3 epochs | 84.89 / 78.67 |
| Native LeWM (L0), 3 seeds | 10 epochs | 90.89 / 87.33 |
| F8, renderer, `rho = 0.8` | 24,982 | 87.33 / 83.00 |
| Decoder-free, all 62 recipes | up to 37,473 | 1 to 8 |

### 4.5 Decisions

1. **Gate.**
   - If a `rho = 0.8` arm passes and Q0 fails, independent noise was what failed the earlier decoder-free runs. Continue the passing arms.
   - If Q0 passes as well, the denoiser with relative coordinates is the fix and `rho` is not required. Continue.
   - If no arm passes, run the fallback (4.6) once, report, and stop for our agreement. Do not tune further.
2. **Three epochs.** Compare the best passing arm with L0 at 3 epochs, using the 5-point non-inferiority rule of [spec.md](spec.md) Section 1.3.
   - If it is non-inferior on primary and fresh, continue both seeds to 124,910 updates, and start the two conditional arms: Q4, the best arm with `reader_anchor = "state"`; and Q6, the best arm on the colour data with both reader kinds, seed 1 each, one epoch. Read these with the rules of handoff-anchor-reader.md Section 5.4, items 3 and 4.
   - If it passes the gate but trails L0 by more than 5 points, report the probes and rollouts and stop for our agreement.
3. **Block.** If the block probe R² is below 0.5 in every passing arm, say so in the first line of the report.

### 4.6 Fallback, only if no arm passes

No new training. On the Q2 checkpoints, read the training and held-out `reader_mse`, the spread, and the information spectrum, and probe for block pose and frame index.

| Signature | Reading |
| --- | --- |
| training `reader_mse < 0.05`, spread falling | the prediction cost outweighs the reward |
| held-out `reader_mse` far above training | the reader does not generalize, so its score misleads the encoder |
| `reader_mse` high, spread high, agent R² low, frame index or another factor readable | the state fills with content other than the task state |
| loss not finite, or no reader gradient | an implementation fault; fix it, this is not a result |

## 5. Track 2: Reach3D, in your current protocol

**What it can and cannot show.** Under your pure action transport, `z_{t+k} = z_t + Q (a_t + ... + a_{t+k-1})`, so even a stale anchor's reward on the action directions is fixed by the action distribution: your identity extends to any gap. What changes in the orthogonal complement cannot be predicted by a constant-transport prior, so at prediction weight 1 it is priced out. **We therefore predict no difference from the matched control.** Reach3D cannot test the method's benefit, because nothing in the scene moves except the arm. This track checks that the reward does no harm under your protocol and budget-scaling evaluation. Reach3D would only show a benefit if non-action content mattered there, for example real wrist drift under zero commands.

### 5.1 Implementation (in your code; you own the mainline)

Add a loss next to `current_entropy_loss` in `experiments/robocasa_stable_scaling/current_entropy.py`, installed by a new flag (for example `window_anchor_entropy: true`) with `anchor_entropy_projection` set to `"complement"` or `"none"`. Use the same install assertions as `current_entropy` (pure action transport, constant action response, isometric actions, anchor reader at `beta_nats = 1`).

```python
def window_anchor_entropy_loss(self, batch, step=0, ramp_steps=0):
    pixels, render, _ = self.prepare(batch)
    encoded = self.encode(dict(pixels=pixels, action=batch['action']))
    flat = encoded['emb']                                             # [b, w, D], w = 8
    b, w = flat.shape[:2]
    z = flat.reshape(b, w, self.state_slots, self.state_width)
    current = flat[:, :-1].reshape(b*(w-1), 1, -1)
    actions = encoded['act_emb'][:, :-1].reshape_as(current)
    query = self.predict(current, actions).reshape(b, w-1, -1)
    rate = (flat[:, 1:].float()-query.float()).square().sum(-1) / (2*self.sigma0**2)   # [b, w-1], nats
    state = z[:, 1:]-z[:, :1]                     # change from the window's first code; z[:, :1] attached
    if self.anchor_entropy_projection == 'complement':
        Q = self.action_encoder.weight            # [D, 14], orthonormal columns (QR in IsometricActionEncoder)
        d = state.flatten(-2)
        state = (d-(d@Q)@Q.T).reshape_as(state)
    noisy = state+self.sigma0*torch.randn_like(state)    # full-dimensional noise, also in the action directions
    condition = self.reader.condition(z[:, 0], render[:, 0])    # first code (detached inside) and first RGB
    entropy, stats = encoder_entropy_term(self.reader, state, noisy, condition, self.sigma0)
    reader_loss = reader_denoising_term(self.reader, state, condition, self.sigma0,
        self.reader_draws, reader_stream_seed(self.reader_seed, step)).mean()
    loss = (rate+entropy).mean()+self.reader_loss_weight*reader_loss
    ...  # metrics as in current_entropy_loss, plus reader_mse rescaled by D / (D - 14) for "complement"
```

- **Full-dimensional noise is deliberate.** With the complement projection the state is exactly zero along `Q`, so the reader recovers that part of the noise exactly, in training and in the encoder term alike. Report `reader_mse` multiplied by `D / (D - 14)` so that it stays comparable across arms.
- **The rate is the plain Gaussian one** (your `factual_prediction_rate` without normalization or profiling), because the reward is in nats and must be priced in the same units. That is why the control R0 uses it too, rather than your best configuration.

**CPU checks before launch.** Copy the patterns of tests 3 and 12 in `tests/as_jepa/test_anchor_reader.py`.

1. With the entropy and reader terms removed, the loss and gradients equal `conditional_rate_loss` with the isotropic rate, bitwise.
2. A constant added to every code receives a gradient from the entropy term of at most 1e-5 of the gradient without the relative coordinates.
3. With `"complement"`, `Q^T` times the entropy term's gradient with respect to the codes is at most 1e-5 of its norm.
4. The reader loss reaches only `reader.*`, and the entropy term reaches no `reader.*` parameter.

### 5.2 Arms

| Arm | Loss | Question |
| --- | --- | --- |
| R0 | plain Gaussian rate only (`conditional_rate_only`, isotropic, no normalization) | matched control; reuse your no-entropy control of the current-entropy comparison if its window, initial weights and data match |
| R1 | R0 plus the complement reward, anchor = window's first frame | does reward on non-action content help or hurt |
| R2 | R0 plus the full relative reward (no projection), anchor = window's first frame | the stale-anchor reader without the projection |

Use the same initial weights, the fixed 12,518 TRAIN logs, and 3 hours on 4 GPUs each. Evaluate every full hour with your 64 DEV cases, strict 2 mm success, mean final distance, and the frozen 384-parameter current-action head, plus training and held-out `reader_mse`. Report your current best configuration at the same hours for context. Equal hours are not equal updates, so report updates too.

### 5.3 Decisions

Strict success in your runs moves by up to 6 of 64 between adjacent hours (for example 24, 30, 29, 30, 26 over 8 to 12 h), and at about 40% success the binomial spread over 64 cases is about 4 cases. So 8 cases is the smallest difference worth acting on.

- **R1 and R2 within 8 cases of R0 at 3 h:** compatibility is shown. Stop Track 2; Reach3D cannot test the method further.
- **R1 or R2 at least 8 cases above R0 at 3 h:** non-action content matters on Reach. Continue that arm to 12 h under your hourly protocol and report the scaling curve.
- **R1 or R2 more than 8 cases below R0, with held-out `reader_mse >= 0.05`:** the reward interferes with your readout. Report the training and held-out `reader_mse` and stop.

## 6. Part G

Unchanged: [handoff-anchor-reader.md](handoff-anchor-reader.md) Section 4, commands in Section 11.1. The tool is `docs/lingjing/round2/as_jepa_tools/gap_buckets.py` on this branch.

## 7. Predictions, to record before launch

| # | Prediction | Basis | Confidence |
| --- | --- | --- | --- |
| 1 | At least one of Q1 to Q3 passes the gate, and Q0 fails | F7 to F8; every failed decoder-free run had `rho = 0` | moderate: your arm with a noise-free prior input (block C) also failed, so `rho` may not be the only cause |
| 2 | Q5 keeps the agent and loses the block | synthetic toy; the reward is capped at log 8 nats | high |
| 3 | In passing arms, held-out `reader_mse` is within 0.1 of training | the reader sees fresh windows every step during training, unlike a frozen fit | moderate |
| 4 | The best passing arm is non-inferior to L0 at 3 epochs | F8 reached 87.33% with the same noise and prior | low to moderate |
| 5 | R1 and R2 are within 8 strict cases of R0 at 3 h | Section 5 | high |

A failed prediction is a result. Report it as such.

## 8. What to send back, in the issue

1. Yes or no for each track, the expected start, and any change to the budget.
2. The `reader_mse` values of your earlier Reach3D anchor-reader runs (Section 0).
3. After the gate, one table per arm and seed: spread, training and held-out `reader_mse`, noisy agent and block R², and pass or fail, with the decision of Section 4.5.
4. At 3 epochs: the table of Section 4.4 with the references, GPU-hours and seconds per update.
5. Track 2: your hourly table for R0, R1 and R2, with updates and GPU-hours.

Commit full reports where you usually keep them and link them from the issue. We read the issue and your branch.

## 9. What you may change without asking

- Machine and GPU allocation, launch order within a track, checkpoint cadence.
- Throughput settings that leave the numbers unchanged.
- In Track 2, the exact code layout of Section 5.1, as long as the four CPU checks pass.

Anything that changes an objective, a named quantity, an arm or a decision rule needs our agreement first; ask in the issue.
