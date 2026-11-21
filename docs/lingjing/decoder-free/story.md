# Decoder-free AS-JEPA: formulation, theory and narrative

Written 2026-10-31 on branch `lingjing/decoder-free-as-jepa`, an exploratory line next to the renderer-based AS-JEPA.

- [../as-jepa-story.md](../as-jepa-story.md) is the canonical statement of the renderer version. This document reuses its data model, notation and propositions, and says exactly what changes.
- [../discussions/2026-10-31-decoder-free-as-jepa.md](../discussions/2026-10-31-decoder-free-as-jepa.md) records how the idea was reached, with a synthetic toy.
- [spec.md](spec.md) is the experiment and implementation handoff.

Notation as in the story: `x_t` frames, `a_t` actions, `z_t` dynamics state, `c` appearance, `eps_t` transient detail, `s_t = mu(x_t) + sigma0 * xi_t` the noisy state, `x_A` the anchor, nats throughout.

## 1. Summary

The renderer-based objective, with an optimal renderer, depends on the encoder only through entropies of the state. Its pixel loss contributes exactly one quantity: the state's information beyond the anchor, `I_A = I(x_t; s_t | x_A)`. Because the state's noise is known, `I_A` can be computed in state space. Replacing the renderer by that computation gives a decoder-free objective. At `lam = 0` it equals the round-1 objective up to a constant, so Propositions 1 to 3 hold unchanged when `I_A` is computed exactly:

```
J_DF(beta, lam) = beta * Entry  +  beta * sum_t Surprise_t  +  lam * sum_t Forecast_t  -  sum_t I_A(s_t)
```

- `Entry` and `Surprise` are the state's code length, as before. `Surprise` is the JEPA latent prediction loss.
- `Forecast_t` measures how well a forecast of `s_t` predicts it. `I_A - Forecast_t` is a lower bound on the information that forecast carries about the state beyond the anchor.
- With the canonical one-step forecast, `Forecast_t = Surprise_t`, and the total weight on the prediction loss is `beta + lam`.

The price of removing the decoder is that `I_A` must be estimated in the state's own space. The Gaussian estimate used here can be gamed by codes that are not Gaussian (Section 3.4). Both JEPA terms are derived. The prediction loss is the state's code length, and also the decoder of the forecast's information. The anti-collapse term `I_A` is the state's information beyond what the anchor already gives; the anchor tells it what not to count. There is no decoder, EMA, stop-gradient, augmentation or schedule.

## 2. Formulation

### 2.1 Unchanged

The data-generating process, the three blocks and the target are those of story Sections 2.1 and 2.2:

- `z`: persists and drives the dynamics;
- `c`: persists and does not;
- `eps`: does not persist;
- the target is block identifiability of `z`: `s_t = h(z_t)` plus code noise, with `h` injective and independent of `c` and `eps_t`.

### 2.2 The principle, restated

A state should carry what the past tells about the future, beyond what is available for free, at the smallest cost.

- **"What the past tells about the future"** is sufficiency: the textbook definition of a state.
- **"Beyond what is available for free"** is the side information: the anchor, any frame of the episode.
- **"At the smallest cost"** is the price `beta` on the state's code length.

This is a past-future information bottleneck with side information. The round-1 statement ("describe each video with as few nats as possible when the receiver holds a frame") is the `lam = 0` member of the same family. The story's three principles map onto the decoder-free form as follows:

| Story principle | Renderer version | Decoder-free version |
| --- | --- | --- |
| Information has a price | the state's code length, priced at `beta` | unchanged |
| Side information belongs at the decoder | the renderer reads the anchor, so appearance costs nothing to draw | the reward counts only information beyond the anchor, so appearance earns nothing |
| Sample what you cannot predict | a stochastic renderer samples unpredictable detail instead of storing it | unpredictable detail earns nothing through the forecast term, so it is never worth its cost when `beta + kappa * lam >= 1` |

## 3. Objective

### 3.1 From the renderer objective

Story Proposition 1 gives, for every per-frame encoder, with an optimal prior and renderer:

```
J*_beta(q) = C + beta * Entry + sum_t [ beta * h(s_t | s_<t, a_<t) - h(s_t | x_A) + (1 - beta) * h(s_t | x_t) ],    C = sum_t H(x_t | x_A)
```

`h(s_t | x_t)` is the code noise and is known. So the only quantity the renderer supplies is `h(s_t | x_A)`, through its likelihood. Writing `I_A(s_t) = h(s_t | x_A) - h(s_t | x_t)` and `Surprise_t = h(s_t | s_<t, a_<t) - h(s_t | x_t)` (the expected KL to an optimal prior),

```
J*_beta(q) - C = beta * Entry + sum_t [ beta * Surprise_t - I_A(s_t) ]  =  J_DF(beta, 0)
```

This is an identity; no approximation is involved.

### 3.2 The forecast term and the `lam` family

Let `c_t` be a forecast of `s_t` made from the past: a one-step mean, a sample, or an open-loop rollout. The information it carries about the state beyond the anchor has a variational lower bound with a Gaussian decoder `r(s_t | c_t)`:

```
I(s_t; c_t | x_A) = h(s_t | x_A) - h(s_t | c_t, x_A) >= I_A(s_t) - [ CE_r(s_t | c_t) - h(s_t | x_t) ] = I_A(s_t) - Forecast_t
```

Mixing the frame's own information with weight `1 - lam` and the forecast's information with weight `lam` gives `J_DF(beta, lam)`. The four forecast types, all in nats:

| Forecast | Centre `c_t` | Decoder | `Forecast_t` | Effective weight `kappa` |
| --- | --- | --- | --- | --- |
| one-step, mean (canonical) | `m_t = prior(s_<t, a_<t)` | `N(m_t, sigma0^2)` | `Surprise_t` | 1 |
| one-step, sample | `m_t + sigma0 * xi` | `N(c_t, 2 sigma0^2)` | `||s_t - c_t||^2 / (4 sigma0^2)`; expectation `Surprise_t / 2` plus a constant | 1/2 |
| open-loop, mean | rollout of prior means from the encoded `s_0` | `N(c_t, sigma0^2)` | `||mu_t - c_t||^2 / (2 sigma0^2)` | 1, against a multi-step error |
| open-loop, sample | rollout of prior samples from `s_0` | `N(c_t, 2 sigma0^2)` | `||s_t - c_t||^2 / (4 sigma0^2)` | 1/2, plus the rollout's spread |

The rate always uses the one-step teacher-forced prior. The forecast type changes only the reward.

**Why noiseless forecasts are fine here.** In the renderer version a noiseless forecast mean was an unpriced channel: the renderer could read unbounded information from it. Here the reward is information about the target state, which the target's own noise `sigma0` caps. So nothing passes unpriced, and noise on the forecast only weakens the decoder.

### 3.3 Two effective numbers

The networks that read the state absorb any rescaling of it, so only the state in noise units, `u = mu / sigma0`, matters (story Section 4.4). In those units, with the canonical forecast:

```
J_DF = (beta * sigma0^2 / 2) * ||u_0||^2  +  (beta + lam) / 2 * sum_t ||u_t - m_t / sigma0||^2  -  sum_t I_A(u)
```

`I_A` depends only on `u`. So the decoder-free objective has two effective numbers:

- **the entry price** `pi = beta * sigma0^2`;
- **the prediction weight** `w = beta + kappa * lam`.

`sigma0` has no effect beyond `pi`. And `lam` is exactly the freedom to set the prediction weight separately from the entry price. Open-loop forecasts split the prediction weight into a one-step part (`beta`) and a multi-step part (`lam`).

### 3.4 Estimating `I_A` without a decoder

`s_t = mu(x_t) + sigma0 * xi`, so `h(s_t | x_A)` is the entropy of a Gaussian mixture with known component covariance. Three estimators matter. All are computed on the state's change relative to the anchor's state, `d_t = mu_t - mu_A`, where `mu_A` is the anchor encoded by the same encoder. Anything constant within an episode cancels in `d_t`.

- **Pooled Gaussian (canonical).** `I_A ~ 0.5 * logdet(I + Sigma_A / sigma0^2)`, with `Sigma_A = 0.5 * E[d d^T]` pooled over the batch. The factor 0.5 assumes the anchor and the frame are independent draws from the episode, which a distant anchor approximates. For a Gaussian `mu`, this is the information of the Gaussian channel `I(mu; mu + sigma0 xi)`, which MCR² calls the coding rate. It is the closest relative of the LogDet estimators the JEPA literature already uses.
- **Per-window Gaussian.** The same formula per window, from its `T` changes relative to its own anchor (a `T × T` determinant, by Sylvester's identity), then averaged over windows.
- **Collision (pairwise).** In population, for any distribution, `I_A >= -log E exp(-||d||^2 / (4 sigma0^2)) - (D/2) log(e/2)`, because Shannon entropy is at least the order-2 Rényi entropy. For Gaussian states the right side equals the log-det minus the constant.

**Which way each estimate errs, and why it matters.** The pooled log-det is an *upper* bound on `I_A`, for two reasons:

- a Gaussian has the most entropy of any distribution with that covariance;
- the log-det of an average covariance is at least the average log-det (Jensen's inequality across anchors).

The encoder can exploit both. In a constructed example, rotating each episode's change by an episode-level key, such as its colour, leaves the true `I_A` at 4.6 nats per frame and raises the pooled estimate to 40. Codes that curl a low-dimensional motion into many dimensions are over-credited in the same way.

- **What the per-window estimate fixes.** It removes the Jensen gap across anchors, so an episode-level key cannot raise it; a contract test checks this. It is still Gaussian within a window and small-sample (rank at most `T`), so it is conservative relative to the pooled estimate and under-credits components that do not change within a window.
- **Why the collision bound is only a diagnostic.** It holds in population, but on minibatches it is biased upward and dominated by the nearest pair. In one check it gave 149 nats against an exact 74 at an isotropic spread of `3 sigma0`. It is therefore logged as a rotation-invariant trend diagnostic, not trained on.

This is the main theoretical price of removing the decoder. Any estimate of information that assumes nothing about the distribution saturates near the log of the sample size (McAllester and Stratos 2020). The renderer avoided that by modelling pixels, and its likelihood bound erred in the safe direction: a lower bound on information. A decoder-free estimate avoids saturation only by assuming something about the state's distribution, and the encoder can exploit that assumption.

The experiments watch it with three diagnostics:
- the gap between the pooled and per-window estimates;
- the participation ratio of `Sigma_A` against the number of factors that linear probes find;
- the probes themselves.

### 3.5 Code

```python
mu_all = encoder(torch.cat([window_frames, anchor[:, None]], 1))    # every frame alone; the anchor is last
mu, mu_A = mu_all[:, :-1], mu_all[:, -1]
s = mu + sigma0 * torch.randn_like(mu)
m = prior(s[:, :-1], a[:, :-1])                                      # one-step means of s_1..s_T, identity carry
entry = 0.5 * mu[:, 0].pow(2).sum((-2, -1))                          # nats (plus a constant)
surprise = (mu[:, 1:] - m).pow(2).sum((-2, -1)) / (2 * sigma0 ** 2)  # nats per step: the JEPA loss
d = (mu[:, 1:] - mu_A[:, None]).flatten(0, 1).flatten(1)             # change relative to the anchor's state
info = 0.5 * torch.logdet(torch.eye(D) + d.T @ d / (2 * len(d)) / sigma0 ** 2)
loss = beta * entry.mean() + (beta + lam) * surprise.sum(1).mean() - T * info
```

## 4. Method

| Component | Decoder-free AS-JEPA | Change from the renderer version |
| --- | --- | --- |
| Encoder | per-frame ViT plus `QueryPool`, `mu`, noise `sigma0` | none |
| Prior | causal, anchor-blind, action-conditioned, identity carry, fixed variance tied to `sigma0`; reads noisy states | none |
| Rate | `Entry` plus one-step `Surprise`, priced at `beta` | none |
| Anchor | a frame of the same episode, at least `gap` away, encoded by the same encoder | used only inside `I_A`, as the reference for change; no renderer reads it |
| Reward | `I_A` and, with weight `lam`, the forecast term | replaces the renderer's flow loss |
| Renderer | none in training. A decoder may be trained afterwards on the frozen state, for visualization and pixel-based evaluation | removed |
| Initial state scale | the pool head is rescaled once so the state's spread is about `sigma0` (`calibrate_state_std`) | now part of the empirical design (Section 5.6) |

Design invariants:

1. The encoder reads one frame.
2. The prior never reads frames or the anchor.
3. Actions reach only the prior.
4. The anchor enters only `I_A`.
5. Window frame 0 starts the chain and is never rewarded: `I_A` sums over frames 1 to `T`.
6. No EMA, stop-gradient, augmentation, masking or schedule.
7. Nothing normalizes `mu`.
8. `sigma0`, `beta` and `lam` are constant within a run.

## 5. Theory

### 5.1 Assumptions

- **A1 to A3 and A5 are unchanged.**
- **A4 becomes A4-DF:** the prior family contains the true conditional density, and `I_A` is computed exactly.
- **A4-G (Gaussian estimators):** the pooled log-det equals `I_A` when the state's change relative to the anchor is Gaussian with the same covariance in every episode. The per-window log-det equals it when each window's changes are Gaussian. Otherwise both over-estimate `I_A` (Section 3.4), and the statements below that use them hold only for codes that do not exploit the gap.

### 5.2 Proposition 1-DF: equivalence at `lam = 0`

Under A4-DF, `J_DF(beta, 0) = J*_beta - C` for every per-frame encoder, with `C` independent of the encoder.

*Proof.* Section 3.1. ∎

Consequence: every statement the story proves about `J*_beta` holds verbatim for `J_DF(beta, 0)`. That covers Proposition 2 (appearance), Proposition 3 (transients at `beta >= 1`), the A5 value condition, the worked example and the collapse corollary.

### 5.3 Proposition 2-DF: appearance

- **Exact `I_A`, any `lam`.** The proof of story Proposition 2 goes through. The anchor determines `c` (A1), so `I_A` is unchanged by appearance. The anchor-blind prior must pay for any appearance information a later state adds.
- **Pooled Gaussian estimator: the proposition fails in general.** Appearance that enters the state as a per-episode additive constant cancels exactly in `mu_t - mu_A`, and a uniform appearance-dependent rescaling gains nothing. But appearance used as a key that rotates or reshapes the dynamics code inflates the pooled estimate without adding information (the counterexample of Section 3.4). Such a code carries appearance and is rewarded for it.
- **Per-window Gaussian estimator.** An episode-level key cannot raise it, because each window's log-det is unchanged by a rotation that is constant within the episode. So the counterexample is closed. Keys that vary within a window are not excluded by any argument we have.
- **Without the anchor reference** (batch covariance), appearance earns `0.5 * log(1 + v_c / sigma0^2)` every frame and is kept. The toy confirms this.

### 5.4 Proposition 3-DF: transients

Take a transient component: independent of everything and fresh in each frame. Two regimes give the same form of answer.

- **Exact regime** (optimal prior and decoders, exact `I_A`). A transient carrying `i` nats in `s_t` adds `i` to `I_A`, `i` to the surprise, and `i` to the forecast term for every forecast type, because no forecast can predict it. The objective changes by `(beta + lam - 1) * i`, so the transient is excluded if and only if `beta + lam >= 1`. At equality, the window's first state, which pays `Entry` and is never rewarded, makes exclusion strict, as in the story.
- **Implemented regime** (fixed-variance prior and decoders, Gaussian estimator). Take one Gaussian direction of variance `v`. It earns `0.5 * log(1 + v / sigma0^2)` and costs `(beta + kappa * lam) * v / (2 sigma0^2)`, with `kappa = 1/2` for sampled forecasts (their decoder variance is `2 sigma0^2`) and 1 otherwise. So it is excluded if and only if `w = beta + kappa * lam >= 1`, because `log(1 + x) < x`.

The threshold survives the estimator's over-credit. The log-det is subadditive, and `0.5 * logdet(I + A) <= 0.5 * tr(A)`. So a transient's contribution to the estimate is at most what the surprise term charges at weight 1. A transient with any distribution or embedding is therefore excluded at `w >= 1`, even under the Gaussian estimators.

Below the threshold, the kept amount shrinks continuously. The best signal-to-noise ratio per direction is `1/w - 1`, so a linear probe on the noisy state reads at most about `R^2 = 1 - w`: about 0.7 at `w = 0.3` and 0.2 at `w = 0.8`. Every term is in nats, so the threshold can be tested at known prices; the renderer version could not do that.

### 5.5 Dynamics and fixed properties

- **Dynamics.** A component that varies given the anchor (variance `v`) and is predictable (one-step residual `r`) earns `0.5 * log(1 + v / sigma0^2) - w * r / (2 sigma0^2)` per frame, against its entry cost once per window. This is the A5 condition entirely in nats.
- **Fixed property the dynamics read** (block shape, a push gain `K`). It has no variance given the anchor, so it earns only by lowering the residual of the components it governs. Storing it costs `beta * H(K)` at entry. Leaving it for the prior to infer after a contact costs `(beta + kappa * lam) * I_reveal` in surprise and forecast terms.
  - At `lam = 0` an exact code never stores it (`I_reveal <= H(K)`). A fixed-variance code may store it anyway, because its prior cannot widen its uncertainty before a contact. The toy keeps `K` at `lam = 0`, and the static-component bullet of story Section 5.6 describes this fixed-variance behaviour.
  - At `lam > 0` an exact code stores it whenever contacts reveal enough of it: `beta * H(K) < (beta + kappa * lam) * I_reveal`.
  - Open-loop forecasts strengthen this: a rollout from `s_0` cannot infer `K` from observations, so it pays at every contact, not only the first.

### 5.6 Collapse

**At the optimum.** The collapsed state has `I_A = 0` and pays nothing. A state carrying predictable change beyond the anchor scores better when that change pays for itself: the story's collapse corollary, unchanged.

**Locally, collapse can be a minimum, depending on what the actions reveal.** At small amplitude a KL-optimal prior cannot read the state through the noise of its noisy inputs. It still reads the actions, though, and nothing prices them. So its residual is the part of the component that the actions do not determine, `Var(z_t | a_<t)`. The net value per frame at amplitude `a` is about

```
a^2 * (v - w * Var(z_t | a_<t)) / (2 sigma0^2)
```

So a barrier exists exactly when `w > v / Var(z_t | a_<t)`.

- **Components the actions determine** (the PushT agent, whose actions are target positions) have `Var(z_t | a_<t) ~ 0` and face no barrier at any `w`.
- **Components the actions do not reveal** (the block) have `Var(z_t | a_<t) ~ v_tot >= v`. They face a barrier whenever `w > v / v_tot`, which can be below 1.

This is the same kind of barrier as in the story's worked example.

**What helps.**
- **The identity carry.** At initialization the prior copies the noisy previous state. The noise part of its error is constant, so the encoder sees only the frame-to-frame change as residual, and no first-order barrier. That holds until the prior learns to shrink its predictions.
- **No reader has to learn first.** The renderer version escaped at a rate proportional to the square of the state's scale, because the renderer had to learn to read the state (round-2 spec, Appendix A.6). Here the reward is the state's own spread, so the escape rate does not vanish at small scale.
- **The initialization rule.** Start the state's spread at about `sigma0`, as revision 2 does on TwoRoom.

**Toy evidence** (scratch `.tmp/latent-toy/initscale.py`, 3,000 steps). The trap selects components that are predictable from the start:

| Setting | Tiny start (0.001 `sigma0`) | Start at `sigma0` |
| --- | --- | --- |
| prediction weight 2 (`beta = 1`, `lam = 1`) | only the agent kept, on both seeds; the block and its push gain `K` lost | the block and `K` partly kept on seed 0; `K` lost on seed 1 |
| prediction weight 1.3 (`beta = 0.3`, `lam = 1`) | the agent and `K` kept; the block only partly (one of its two coordinates) | same, at every start tried (0.001, 1 and 3 `sigma0`), both seeds |

Two more toy runs, at 6,000 steps with two seeds each (probe R² from the noisy state):

| Objective at `beta = 0.3` | agent | block | `K` | appearance | transient |
| --- | --- | --- | --- | --- | --- |
| pooled, `lam = 0` (`w = 0.3`) | 1.00 / 1.00 | 1.00 / 1.00 | 0.99 / 0.99 | 0.00 / 0.00 | 0.70 / 0.70 |
| per-window, `lam = 0` | 1.00 / 1.00 | 1.00 / 1.00 | 0.99 / 0.99 | 0.00 / 0.00 | 0.00 / 0.00 |
| per-window, `lam = 1` (`w = 1.3`) | 1.00 / 1.00 | 0.19 / 0.00 | 0.61 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |

What they show:

- **Below the threshold, the transient behaves as predicted.** At `w = 0.3` the pooled objective keeps it at the predicted `R^2 ~ 1 - w = 0.7`, and keeps the block fully, where `w = 1.3` kept it only partly. So purity (`w >= 1`) and completeness for components the actions do not reveal pull in opposite directions along `w`.
- **The per-window estimator under-credits slow components.** At `w = 1.3` it loses most of the block and `K`. At `w = 0.3` its rank cap (at most `T` directions per window) also squeezes out the transient.

The block cannot become predictable until `K` is stored, and `K` earns nothing until the block is encoded. Together with the barrier above, that is the decoder-free version's main optimization risk: components the actions do not reveal can stay off. The experiments therefore map block retention across the prediction weight `w` and test the initial scale.

### 5.7 What is weaker than the renderer version

- **Sufficiency is certified only in the state's own space.** The state is guaranteed to keep every frame-visible quantity that varies relative to the anchor and pays for itself, not to be sufficient for pixels.
- **Information carries no salience weighting.** A predictable change earns full credit regardless of its pixel area. That is good for small objects, but it also credits small predictable distractors.
- **The information reward is an estimate in the state's own space.** It can be gamed by non-Gaussian or episode-dependent codes (Section 3.4); the renderer's likelihood could not be.
- **The Gaussian estimator is rotation-invariant.** The state is identifiable only up to rotation. The component-wise claim (story Section 5.7) needs learned variances.
- **The equivalence with the renderer objective is exact only at `lam = 0` with exact `I_A`.** With the Gaussian estimator, the thresholds above are re-derived per component. The general block-identifiability proofs would need restating for the estimator.

## 6. How the explored techniques behave in the decoder-free form

| Technique | In the renderer version | In the decoder-free form | Theory prediction |
| --- | --- | --- | --- |
| `lam` family | probability that the renderer reads a forecast | weight of the forecast-information reward; with one-step forecasts it is extra weight on the prediction loss (`w = beta + kappa * lam`) | transients excluded iff `beta + kappa * lam >= 1`; an exact code stores fixed properties only when `lam > 0` and contacts reveal enough; larger `w` raises the barrier for components the actions do not reveal (Section 5.6) |
| Noisy versus noiseless **forecast** | noise was required: a noiseless mean was an unpriced channel to the renderer | noise is not needed (the target's noise caps the information); with fixed-variance decoders it halves the forecast weight (`kappa = 1/2`) | (`lam = 1`, sample) behaves like (`lam = 0.5`, mean); canonical is the mean |
| Noisy versus noiseless **prior input** | the prior reads posterior samples | unchanged canonical: samples, so the rate is the code length of what a receiver holds. Reading means (LeWM-like) lets forecasts use precision the rate does not charge | means: more information used than paid, a higher rate at the same `beta`, possibly better short-horizon prediction |
| Noisy versus noiseless **state** | the collaborator's noise-free variants were held without gains | the noise is not optional: it defines `I_A` (with `sigma0 -> 0`, `I_A` diverges). Its value matters only through `pi = beta * sigma0^2` | runs with equal `pi` and `w` are equivalent |
| One-step versus open-loop | what the renderer reads: teacher-forced or free-running | which forecast's information is rewarded: one step ahead of real states, or many steps from `s_0`. Open loop adds a multi-step prediction loss (latent overshooting) to the reward, not to the rate | open loop: less drift over long horizons, the object the planner uses, fixed properties stored more strongly; but compounding errors penalize hard dynamics more |
| AR(1) posterior noise (`rho`) | part of TF | not used; it would change both the surprise and `I_A`'s noise model | out of scope for now |
| Anchor gap | how much of the current state the anchor shows | how far the reference for change is | gap 0 rewards only change since the window start, so slow components are under-rewarded; gap 8 canonical |
| Initial scale | needed so the renderer can read the state | needed so the prior can read the state from the past before learning to predict the mean (Section 5.6) | default initialization can trap components the actions do not reveal at high `w` |
| Estimator of `I_A` | (the renderer's likelihood, a lower bound) | pooled or per-window Gaussian log-det; both can over-credit (Section 3.4) | the pooled-minus-per-window gap stays small unless the code exploits episode-level keys |

## 7. Compatibility with the renderer formulation

| Element | Status |
| --- | --- |
| Data model, blocks, target (story Sections 2.1, 2.2) | identical |
| Assumptions A1 to A3, A5 | identical; A4 becomes A4-DF |
| Encoder, prior, rate, noise, identity carry, window, anchor rule | identical |
| Objective at `lam = 0` | identical up to a constant, when `I_A` is exact (Proposition 1-DF) |
| Propositions 1, 2 and 3; A5; collapse corollary | hold verbatim at `lam = 0` with exact `I_A`; Proposition 3 extends to `beta + kappa * lam >= 1` |
| `lam` family | new meaning (a weight). The exact-regime threshold `beta + lam >= 1` matches the renderer version's `beta >= 1 - lam`. But the renderer's condition dropout (`p = 0.1`) moved its threshold to `(1 - p)(1 - lam)`, and sampled forecasts change `kappa` here, so the thresholds coincide only for mean forecasts without dropout. At `lam > 0`, the decoder-free reward is information about the noisy state, which by data processing is at most the information about the frame that the renderer rewarded |
| Narrative "the JEPA loss is derived" | strengthened: the anti-collapse term is derived too |
| Narrative "we do not reconstruct pixels, we price them" | replaced: there is no reconstruction |
| Image-editing interpretation (story Section 6.3) | moves to the optional post-hoc decoder |
| Calibration of `beta` (story Section 5.8) | solved for the price and the transient threshold, because every term is in nats. The reward itself is an estimate that can over-credit (Section 3.4) |
| Static dynamics factors (story Section 5.6) | that bullet describes the fixed-variance code. An exact code stores them only at `lam > 0` (Section 5.5) |
| Generation, swap tests, pixel-level evaluation | need the post-hoc decoder |
| Component-wise identifiability (story Section 5.7) | still needs learned variances; the log-det adds a rotation symmetry |

So the decoder-free form is compatible in the strong sense: same model, same assumptions, and, for `lam = 0` with an exact information term, the same objective. The differences are the estimator of one term, the meaning of `lam`, and what is lost with the decoder.

## 8. Narrative

1. **JEPAs say what to predict, not what to keep.** Their anti-collapse devices (EMA, VICReg, SIGReg) decide content by accident and favour whatever is easy to spread and predict. In the toy, a SIGReg-like regularizer filled its budget with appearance and noise and lost the factor the dynamics needed.
2. **A state should carry what the past tells about the future, beyond what is free, at the smallest cost.**
3. **In representation space, with a noisy state, that information has two halves.** The JEPA prediction loss bounds one; the anti-collapse term is the other. A price on the state's code length gives minimality. Every term is in nats, so the weights are not tuned.
4. **The anchor is free side information.** Measuring change relative to it removes appearance.
5. **Consequences, block by block:** appearance excluded at every price (with an exact or per-window estimate); unpredictable detail excluded iff `beta + kappa * lam >= 1`; dynamics kept when they pay; fixed properties the dynamics read kept at `lam > 0` when contacts reveal them.
6. **Reconstruction, contrastive learning and spread regularizers estimate the same information term.** They do it generatively, discriminatively (saturating at `log K`) and by spread (unconditioned and unpriced). Ours estimates it in state space, conditioned and priced.

One-line pitch: JEPA's regularizer is half of the predictive information; condition it on what the model gets for free, and price what it keeps.

Anticipated objections:

- **"It is MSE plus a log-det, a VICReg variant."** The terms are familiar, but here they are derived. Their weights are fixed by units; the log-det is conditioned on side information; a price is added. Each choice makes a prediction that VICReg does not (Section 6). Shwartz-Ziv, Balestriero, Kawaguchi, Rudner and LeCun (NeurIPS 2023) already read VICReg as entropy maximization with LogDet and pairwise estimators; we supply the noise, the conditioning and the price.
- **"Batch statistics."** Like VICReg and SIGReg. `I_A` needs many same-episode pairs per batch; each window supplies `T` of them.
- **"Gaussian assumption."** The collision bound needs none, at the cost of noisier gradients.
- **"Can't the encoder game the log-det?"** Yes. The pooled log-det over-credits non-Gaussian or episode-dependent codes. The per-window estimator closes the episode-level case, and the experiments watch the gap between the two. A renderer's likelihood cannot be gamed this way; that is the honest price of removing the decoder.
- **"Why not just SIGReg?"** SIGReg counts appearance and noise as information, and nothing prices the state. The toy shows both.

## 9. Relation to prior work

| Line of work | Relation |
| --- | --- |
| I-JEPA, V-JEPA, BYOL | avoid collapse by EMA or stop-gradient; nothing taken |
| LeJEPA, LeWM (SIGReg); VICReg, PLDM | spread regularizers across the batch; replaced by `I_A` relative to the anchor, plus a price |
| DINO-WM | predicts frozen features; not used (one stage, no frozen target) |
| Shwartz-Ziv et al. 2023 | entropy estimators (LogDet, Kolchinsky-Tracey pairwise) as anti-collapse terms; the closest precedent |
| MEC (Liu et al. 2022), MCR² (Yu et al. 2020) | log-det coding length at a precision; ours uses the code noise as the precision and conditions within episodes |
| Wang and Isola 2020 | the collision bound is a within-episode uniformity |
| Bialek, Nemenman and Tishby 2001; Creutzig et al. 2009; DCA (Clark et al. 2019); CPIC (Meng et al. 2022); CEB and PI-SAC | predictive information with compression; our principle's ancestry. We add actions, a planning prior, nonlinear per-frame encoders and side information |
| Conditional contrastive learning (Tsai et al. 2021) | removes a nuisance by fixing it when sampling pairs; the anchor does this inside an entropy estimator |
| Sobal et al. 2022; slow feature analysis | prediction plus spread prefers slow features; the anchor discounts what is constant within an episode |
| RSSM, VRNN, beta-VAE, DBSE | KL chain plus a pixel decoder; the KL chain is kept and the decoder replaced |

We did not find a JEPA or latent world model that conditions its anti-collapse entropy on side information or prices the state; the search was not exhaustive.

## 10. Risks

1. **The information reward can be gamed.** The Gaussian estimate over-credits non-Gaussian or episode-dependent codes (Section 3.4).
2. **The current prior gates what the state keeps.** Components the actions do not reveal face a barrier when `w > v / Var(z | a)`, so the state can settle without them (Section 5.6).
3. **No salience.** Small predictable distractors earn full credit.
4. **Estimation details:** batch size, and conditioning on the anchor's state rather than its frame.
5. **The price is mandatory.** Without it, predictable components inflate without bound.
6. **No generative side in training.** Videos and pixel metrics need a post-hoc decoder.
7. **Weaker certificates** (Section 5.7).
8. **Novelty can be misread** as a VICReg variant unless the predictions are tested.

## 11. Open questions

- A proof of Proposition 2-DF for arbitrary appearance entanglement under the Gaussian estimator.
- Learned variances in the decoder-free form, which make `kappa = 1` for sampled forecasts and allow component-wise identifiability.
- Whether open-loop forecasts should be weighted per horizon (a planning-horizon prior).
- Natural video, where the anchor does not show everything the episode will reveal.
