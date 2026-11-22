# Discussion log: AS-JEPA without a decoder, and how the objective itself prevents collapse (2026-10-31)

The question. JEPA proponents accept JEPA objectives but reject pixel-level reconstruction. Can AS-JEPA's renderer be replaced by a principled term that keeps the method a pure JEPA, prevents collapse without EMA, stop-gradient, augmentation, multistage training or other heuristics, and keeps the theory (sufficiency for the future, minimality by a price, appearance to the anchor, transient detail excluded, static dynamics factors kept)?

Short answer: yes, and without changing the theory. With an optimal renderer, the round-1 objective depends on the encoder only through entropies of the state (story Proposition 1). The renderer's only job is to estimate one of them, the state's entropy given the anchor. Because the state's noise is known, that entropy can be computed in state space. The result is a decoder-free objective that equals the round-1 objective up to a constant at `lam = 0`, extends to the `lam` family, and has both JEPA terms derived: the latent prediction loss, and an anti-collapse term that measures the state's information beyond the anchor in nats.

## 1. Collapse is missing information

A JEPA prediction loss alone is minimized by a constant state, because a constant is perfectly predictable. Any collapse-free method therefore contains something that rewards information. The families differ in how they measure it.

| Family | How information is rewarded | Status for us |
| --- | --- | --- |
| Training dynamics (EMA teacher, stop-gradient, asymmetric predictor; BYOL, I-JEPA, V-JEPA) | not by the objective: collapse is avoided by how training moves, and the losses do not in general correspond to a well-defined objective | excluded by our constraints |
| Contrastive (InfoNCE, CPC) | discriminate the true target from negatives; a lower bound on mutual information | capped at the log of the number of negatives; easy features suppress hard ones; the JEPA position paper argues against it |
| Distribution regularizers (VICReg, Barlow Twins, SIGReg, MCR², MEC) | spread the embedding (variance, decorrelation, Gaussianity, coding rate) | entropy proxies; they count any spread, including appearance and noise (Section 6) |
| Generative (decoder, our renderer) | likelihood of the observation given the state (a Barber-Agakov bound) | measures information about pixels; needs a decoder; the JEPA objection |

The JEPA position paper (LeCun 2022, Section 4.5) states the requirement in the same terms: maximize the information content of the representations, make the target predictable from the context, minimize the information in the predictor's latent variable. Shwartz-Ziv, Balestriero, Kawaguchi, Rudner and LeCun (NeurIPS 2023, "An Information-Theoretic Perspective on VICReg") read VICReg's variance and covariance terms as entropy maximization and replace them with established entropy estimators: a LogDet estimator, and the pairwise-distance lower bound of Kolchinsky and Tracey (2017). The pairwise lower bound gave the best results. So "anti-collapse = an entropy estimate of the representation" is already accepted in the JEPA community. What is missing is a reason to measure entropy at a particular resolution, a reason to exclude appearance, and a reason to stop: the noise scale, the conditioning and the price.

## 2. The renderer only estimates an entropy of the state

Story Proposition 1, with an optimal prior and renderer:

```
J*_beta(q) = C + beta * Entry + sum_t [ beta * h(s_t | s_<t, a_<t) - h(s_t | x_0) + (1 - beta) * h(s_t | x_t) ],   C = sum_t H(x_t | x_0)
```

Every encoder-dependent term is an entropy of the state `s_t`. `C` is a constant of the data. The renderer's negative log-likelihood, at its optimum, is `H(x_t | x_0) - I(x_t; s_t | x_0)`, and `I(x_t; s_t | x_0) = h(s_t | x_0) - h(s_t | x_t)`. So the pixel loss contributes exactly one encoder-dependent quantity: the state's entropy given the anchor, `h(s_t | x_0)`. The other entropy, `h(s_t | x_t)`, is the code's own noise and is known: `(d/2) log(2 pi e sigma0^2)` for the fixed-variance code.

Define the state's information beyond the anchor,

```
I_A(s_t) := I(x_t; s_t | x_A) = h(s_t | x_A) - h(s_t | x_t)          (nats)
```

Then, exactly,

```
J*_beta(q) - C = beta * Entry + sum_t [ beta * Surprise_t - I_A(s_t) ],    Surprise_t = E KL(q(s_t|x_t) || p*(s_t | s_<t, a_<t))
```

The renderer is one estimator of `I_A`. Any other estimator of `I_A` gives the same objective. That is the whole move.

## 3. The decoder-free objective and the `lam` family

At `lam = 0` the decoder-free objective is the expression above. For `lam > 0` the round-2 renderer read a forecast; in state space the forecast's information about the next state has a variational lower bound whose decoder is the prior itself:

```
I(s_t; m_t | x_A) = h(s_t | x_A) - h(s_t | m_t, x_A) >= I_A(s_t) - Surprise_t
```

(take the decoder `N(s_t; m_t, sigma0^2 I)`; its cross-entropy is `Surprise_t + h(s_t | x_t)`). Mixing the two rewards with weights `1 - lam` and `lam`:

```
J_lat(beta, lam) = beta * Entry + sum_t [ (beta + lam) * Surprise_t - I_A(s_t) ]
```

- **`lam` becomes the weight on the JEPA prediction loss.** The total weight is `w = beta + lam`.
- **At `lam = 1` the reward is the predictive information beyond the anchor.** `I_A - Surprise` is a lower bound on `I(s_t; s_<t, a_<t | x_A)`, the information the past carries about the next state beyond what the anchor shows. That is the sufficiency principle, measured in representation space, with the anchor as side information and `beta` as the price per nat of state: a past-future information bottleneck without a decoder.
- **JEPA's two terms are the two halves of that information.** `I(s_t; past | A) = h(s_t | A) - h(s_t | past, A)`. The prediction loss bounds the second half; the anti-collapse term is the first. Nothing is added by hand.
- **Everything is in nats.** Unlike the flow loss, `I_A` and the rate share a unit, so `beta` is a calibrated price and `beta = 1` is the code length.
- **Noiseless forecast means are fine here.** In the renderer world a noiseless mean was an unpriced channel. Here the reward is information about the target state, which is capped by the target's own noise, so nothing can pass unpriced.

## 4. Computing `I_A` without a decoder

`s_t = mu(x_t) + sigma0 * xi`, so `h(s_t | x_A)` is the entropy of a Gaussian mixture whose components (one per frame consistent with the anchor) have known covariance `sigma0^2 I`. Two standard estimators:

- **Gaussian closed form (coding rate).** If the state's variation given the anchor is Gaussian with covariance `Sigma_A`, then `I_A = 0.5 * logdet(I + Sigma_A / sigma0^2)`. This is the rate-distortion function of a Gaussian source at distortion `sigma0^2`, the "coding rate" of MCR² (Yu et al. 2020) and MEC (Liu et al. 2022), and the LogDet estimator family. `sigma0` sets its resolution, so it brings no new knob.
- **Distribution-free lower bound (collision entropy).** For any code distribution, `I_A >= -log E_{x, x' | A} exp(-||mu(x) - mu(x')||^2 / (4 sigma0^2)) - (d/2) log(e/2)`, because Shannon entropy is at least the Rényi entropy of order 2. For a Gaussian code the right side evaluates to the log-det above minus that constant. This is the within-episode version of the "uniformity" potential of contrastive learning, and a relative of the Kolchinsky-Tracey pairwise bound used by Shwartz-Ziv et al.

The anchor enters only through the conditioning. A practical estimate of `Sigma_A` uses the anchor's own state as the reference: `Sigma_A ~ 0.5 * E[(mu_t - mu_A)(mu_t - mu_A)^T]`, the covariance of the state's change relative to the anchor's state (frames of the same episode at least the anchor gap apart). Anything constant within an episode cancels in the difference.

```python
mu = encoder(frames)                       # window frames 0..T, each encoded alone
mu_A = encoder(anchor)                     # a frame of the same episode, at least `gap` away; same encoder
s = mu + sigma0 * torch.randn_like(mu)
m = prior(s[:, :-1], a[:, :-1])            # one-step means of s_1..s_T (identity carry)
entry = 0.5 * mu[:, 0].pow(2).sum(-1)                              # nats
surprise = (mu[:, 1:] - m).pow(2).sum(-1) / (2 * sigma0 ** 2)      # nats per step: the JEPA prediction loss
d = (mu[:, 1:] - mu_A[:, None]).flatten(0, 1).flatten(1) / 2 ** 0.5
info = 0.5 * torch.logdet(torch.eye(D) + d.T @ d / len(d) / sigma0 ** 2)   # nats per frame beyond the anchor
loss = beta * entry.mean() + (beta + lam) * surprise.sum(1).mean() - T * info
```

No decoder, no negatives, no EMA, no stop-gradient, no augmentation, one stage, one searched knob (`beta`).

## 5. What the objective keeps

Take one Gaussian component of the state with variance `v` given the anchor and one-step prediction residual `r`, both in state units. Its net value per frame is

```
0.5 * log(1 + v / sigma0^2)  -  w * r / (2 sigma0^2),        w = beta + lam
```

minus `beta` times its entry cost, paid once per window.

- **Transient detail** (fresh each frame, so `r = v`): the net value is negative for every `v > 0` exactly when `w >= 1`, because `log(1 + x) < x`. Excluded if and only if `beta + lam >= 1`: the same threshold as the renderer theory.
- **Appearance** (constant within an episode, so `v = 0`): no reward, positive entry cost. Excluded at every `beta > 0`. Measuring spread across the batch instead of relative to the anchor would reward appearance every frame, which is the slow-feature failure of prediction-plus-spread objectives (Sobal et al. 2022).
- **Dynamics** (varies given the anchor and predictable, `r << v`): kept when the window's reward exceeds its entry cost, the A5 condition now entirely in nats.
- **Static dynamics factor** (block shape, a mass class; `v = 0`): it earns only by lowering the residual of the components it governs, at weight `w`, against entry cost `beta`. At `lam = 1` storing it up front beats inferring it after a contact whenever contacts reveal it (`beta * H(K) < (1 + beta) * I_reveal`); in an exact code at `lam = 0` it does not. Same conclusion as the renderer theory.
- **Collapse (corrected later the same day).** An earlier version of this bullet said collapse is not a local minimum. That assumed the prior predicts a tiny state as well as a large one, which it cannot: a KL-optimal prior reads noise-dominated inputs at small amplitude, predicts the mean, and leaves the state's whole variance as residual. The net value is then about `a^2 (v - w v_tot) / (2 sigma0^2)` for amplitude `a`, so collapse is a local minimum whenever `w >= 1`. This is the same barrier as the story's worked example. What helps: the identity carry (at initialization it copies the noisy previous state, so the encoder sees only the frame-to-frame change as residual); the reward reads the state's spread directly, so no reader has to learn first; and starting the state's spread at about `sigma0`. In the toy, a tiny start at `w = 2` kept only the agent (block and push gain lost, both seeds), while `w = 1.3` kept the agent and the push gain from every start tried, and the block partly ([../decoder-free/story.md](../decoder-free/story.md), Section 5.6). There is still no decoder that could bypass the state, so the renderer version's posterior collapse does not arise.

## 6. Toy evidence

A synthetic check (scratch script `.tmp/latent-toy/toy.py`). Each episode contains five kinds of information:

- an agent `p` (2-D), moved by actions;
- a block `q` (2-D), moved by `K` times a push action;
- a static factor `K` in {0.5, 1.5}, which the block's dynamics read and the frame shows;
- an appearance vector `c` (3-D), constant per episode;
- transient noise `eps` (3-D), fresh in every frame.

Frames are a fixed random rotation of all five. The model has a linear encoder to an 8-dimensional state and an MLP prior with the identity carry, with `sigma0 = 0.1`. The anchor lies at least 8 frames before the 8-frame window. Training uses 6,000 Adam steps, on two seeds. Each entry is the `R^2` of a linear probe from the noisy state, so it measures information at the code's resolution. Values are seed 0 / seed 1.

| Objective | `beta` | `lam` | agent | block | static `K` | appearance | transient |
| --- | --- | --- | --- | --- | --- | --- | --- |
| decoder-free, anchor-relative `I_A` | 1 | 1 | 1.00 / 1.00 | 0.58 / 1.00 | 0.87 / 0.99 | 0.00 / 0.00 | 0.00 / 0.00 |
| decoder-free, anchor-relative `I_A` | 1 | 0 | 1.00 / 1.00 | 0.63 / 1.00 | 0.85 / 0.99 | 0.00 / 0.00 | 0.00 / 0.00 |
| decoder-free, anchor-relative `I_A` | 0.1 | 1 | 1.00 / 1.00 | 0.64 / 1.00 | 0.89 / 1.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| decoder-free, anchor-relative `I_A` | 0.1 | 0 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 0.00 / 0.00 | **0.90 / 0.90** |
| same, spread measured across the batch (no anchor) | 1 | 1 | 1.00 / 0.94 | 0.51 / 0.55 | 1.00 / 1.00 | **1.00 / 1.00** | 0.00 / 0.00 |
| LeWM-like: prediction loss + moment matching to `N(0, I)` | | | 0.98 / 0.94 | 0.59 / 0.79 | **0.03 / 0.06** | **0.93 / 0.81** | **0.66 / 0.65** |
| contrastive, prior as critic, 8 same-episode candidates | 1 | 1 | 0.99 / 0.99 | **0.00 / 0.00** | **0.00 / 0.00** | 0.00 / 0.00 | 0.00 / 0.00 |
| contrastive, prior as critic, 8 same-episode candidates | 0.1 | 1 | 0.99 / 0.98 | 0.07 / 0.66 | 0.08 / 0.14 | 0.25 / 0.09 | 0.00 / 0.00 |

What it shows:

- **The decoder-free objective behaves as the theory predicts.** It keeps the agent and the static factor, and keeps the block fully on seed 1. It drops appearance at every setting. It drops the transient exactly when `beta + lam >= 1`, and keeps it when `beta + lam = 0.1`.
- **The anchor is what removes appearance.** Measuring spread across the batch instead of relative to the anchor keeps appearance fully.
- **The LeWM-like objective fills its fixed spread budget with whatever is available.** It uses the prediction loss plus moment matching to `N(0, I)`, a stand-in for SIGReg, not LeWM's exact regularizer. It keeps appearance and the transient, and drops the static factor that the dynamics need.
- **The contrastive critic saturates.** With 8 candidates it can reward at most `log 8`, about 2.1 nats per frame. Once the agent identifies the frame, nothing else is rewarded, so the block and the static factor are dropped.
- **One caveat.** On seed 0, one block coordinate stays off in every decoder-free setting (`R^2` 0.58 to 0.64), while seed 1 keeps it fully. This is a local optimum of the optimization, not a property of the objective, but it is worth watching on PushT.

## 7. Against the JEPA criteria

| JEPA criterion | Decoder-free AS-JEPA |
| --- | --- |
| No pixel reconstruction | none in the objective; a decoder may be trained afterwards on the frozen state, for visualization only, as LeWM and V-JEPA 2 do |
| Prediction in representation space | the latent prediction loss, derived as the state's code length and as the decoder of the forecast's information |
| Maximize the information content of the representation | `I_A`, the state's information beyond the anchor, in nats, by an entropy estimator his group already uses |
| Minimize the information in the latent variable | the price `beta` on the state's code length |
| No collapse heuristics | collapse is excluded by the objective (Section 5); no EMA, stop-gradient, augmentation or schedule |

Relative to LeWM (prediction loss plus 0.09 times SIGReg): SIGReg is replaced by `I_A` computed relative to the anchor, plus the entry price, and the weights between terms are fixed by their common unit instead of tuned. Relative to Shwartz-Ziv et al. 2023: the noise that makes the entropy estimate meaningful is part of the code instead of being moved to the data, the entropy is conditioned on side information, and there is a price.

## 8. Costs and open risks

- **Self-measured information.** `I_A` measures information in the state's own space. The guarantee becomes "the state keeps every frame-visible quantity that varies relative to the anchor and is predictable enough to pay for itself", not "the state is sufficient for the pixels". A renderer certifies the stronger, pixel-level statement; that is its real remaining value (for example for natural video with complex appearance).
- **The estimator assumes something.** Any estimate of information from a batch that assumes nothing about the distribution is capped near the log of the sample size (McAllester and Stratos 2020). The log-det avoids the cap by assuming a Gaussian state; the renderer avoided it by modelling pixels. A Gaussian assumption on a 64-dimensional state is much weaker than a pixel model, but it over-credits clustered or discrete features, bounded by a few nats each. The collision bound needs no assumption, at the price of noisier gradients dominated by near pairs.
- **The scale needs a price.** At `beta = 0` with a perfect prior, `I_A` grows without bound as the state is scaled up. `beta > 0` (or a nonzero residual) fixes the scale; this is the price's job.
- **No pixel generation from the objective.** Generation and the swap tests need the post-hoc decoder.
- **Unproven at scale.** The toy is linear and synthetic; the encoder sometimes settles in a local optimum (in an earlier run one block coordinate stayed off on one seed).

## 9. Options and recommendation

| Option | Pixel reconstruction | Collapse prevented by | Appearance | Transients | Static factors | Main risk |
| --- | --- | --- | --- | --- | --- | --- |
| A. Keep the renderer, reframe it as an estimator | yes | the likelihood | excluded | `beta + lam >= 1`, `beta` uncalibrated | kept at `lam = 1` | JEPA proponents reject it; 8 times LeWM's compute; posterior collapse |
| B. Contrastive critic (round-2 Block I) | no | discrimination | excluded with same-episode negatives | excluded at `lam = 1` | weakly | saturation at log K drops hard features; the JEPA position paper argues against contrastive methods |
| C. **Information beyond the anchor in state space (recommended)** | no | `I_A` in nats | excluded | `beta + lam >= 1`, `beta` calibrated | kept at `lam = 1` | Gaussian estimator; self-measured information; unproven at scale |
| D. Predict frozen features (DINO-WM style) | no pixels, but reconstruction of a fixed target | the fixed target | not excluded | not excluded | depends on the frozen encoder | two-stage; imports the frozen encoder's content |
| E. LeWM plus a price and an anchor-relative SIGReg | no | SIGReg | excluded | not guaranteed | not guaranteed | a heuristic combination without the derivation |

Recommendation: make C the canonical AS-JEPA objective, and present the renderer as the generative estimator of the same quantity, kept where pixel-level sufficiency matters. The story's principle is unchanged; what changes is that the anti-collapse term is now derived and computed in representation space.

What would change this: if C clearly underperforms the renderer version on PushT planning or rollouts, or fails to keep the block, the renderer stays canonical and C becomes the JEPA-compatible variant. The cheapest check is one PushT run of C against L0 and L2 at matched budget, with the dot and colour domains for purity. Without a renderer, it costs a fraction of an L2 run.

## 10. What would change in the story (proposal, not applied)

- **Summary and Section 3.** The objective is `J_lat(beta, lam)`. Proposition 1 is stated first in its decoder-free form, and the renderer moves to an "estimators of `I_A`" subsection (generative, Gaussian, collision).
- **Section 3.3, "how the JEPA loss emerges".** Both JEPA terms emerge: the prediction loss as the state's code length, and the anti-collapse term as the information beyond the anchor. The claim "SIGReg is replaced by the code length plus a renderer" becomes "SIGReg is replaced by the information beyond the anchor plus a price".
- **Section 4.** The anchor is encoded by the same encoder and used only in `I_A`; the renderer becomes an optional decoder trained afterwards. Design invariant 2 becomes "the anchor reaches only the information term".
- **Section 5.** Propositions 2 and 3 carry over (the objective is the same at `lam = 0`), and the `lam` versions follow from Section 5 above.
- **Section 6.** The pitch becomes: "AS-JEPA is a JEPA whose regularizer is derived: both of its terms are halves of one information quantity, the anchor tells the regularizer what not to count, and a price decides what to keep."
- **An earlier analysis (2026-10-29, not in the repository).** Its claim that only a likelihood-type term can certify sufficiency was too strong. A known-noise state makes an entropy term in state space sufficient, at the cost of a modelling assumption on the state distribution.

Sources: LeCun 2022 (position paper, Section 4.5); Shwartz-Ziv, Balestriero, Kawaguchi, Rudner and LeCun, NeurIPS 2023 (arXiv 2303.00633); Kolchinsky and Tracey, Entropy 2017 (arXiv 1706.02419); Yu et al. 2020 (MCR²); Liu et al. 2022 (MEC); Meng, Luo and Bouchard 2022 (CPIC, arXiv 2203.02051); Creutzig, Globerson and Tishby 2009 (past-future information bottleneck); McAllester and Stratos 2020; Sobal et al. 2022; Balestriero and LeCun 2025 (LeJEPA, arXiv 2511.08544); LeWM (arXiv 2603.19312). Toy script: `.tmp/latent-toy/toy.py` (scratch).

## 11. Follow-up: where it sits in prior work

| Line of work | What it does about collapse or content | What the decoder-free AS-JEPA takes or changes |
| --- | --- | --- |
| I-JEPA, V-JEPA, BYOL | EMA teacher or stop-gradient; collapse avoided by training dynamics | nothing taken: the objective itself has no trivial minimizer |
| LeJEPA, LeWM (SIGReg); PLDM, VICReg | match or spread the embedding distribution across the batch | the spread term becomes the information beyond the anchor, with a weight fixed by units, plus a price |
| DINO-WM | predict frozen pretrained features | no frozen target, one stage |
| Shwartz-Ziv, Balestriero, Kawaguchi, Rudner, LeCun 2023 | VICReg read as entropy maximization; LogDet and Kolchinsky-Tracey pairwise estimators as plug-ins | the same estimators, with the noise in the code (so the entropy is finite and in nats), conditioned on the anchor, and priced |
| MEC (Liu et al. 2022), MCR² (Yu et al. 2020) | log-det coding length at a chosen precision `eps`; MCR² expands the whole set and compresses classes | `eps` is the code noise `sigma0`, not a free choice; the conditioning runs the other way from MCR²: within-episode variation is rewarded, between-episode variation is not |
| Wang and Isola 2020 (uniformity) | pairwise potential `log E exp(-t ||f(x) - f(y)||^2)` | the collision lower bound is a within-episode uniformity with `t = 1 / (4 sigma0^2)` |
| Predictive information: Bialek, Nemenman and Tishby 2001; past-future IB (Creutzig et al. 2009); DCA (Clark et al. 2019); CPIC (Meng et al. 2022); CEB and PI-SAC (Fischer 2020; Lee et al. 2020) | maximize what the past says about the future, with compression; DCA uses Gaussian log-dets for linear projections | the same principle with actions, a nonlinear per-frame encoder, a learned prior used for planning, and the anchor as side information |
| Conditional contrastive learning (Tsai et al. 2021) | remove an unwanted variable by fixing it when sampling pairs | the same idea in an entropy estimator: the anchor fixes the episode, so appearance cancels |
| Slow feature analysis; Sobal et al. 2022 | prediction plus spread prefers slow features | the anchor conditioning discounts exactly what is constant within an episode |
| RSSM, VRNN, beta-VAE, DBSE | KL chain plus a pixel decoder | the KL chain is kept as the rate; the decoder is replaced by a state-space estimate of the one quantity it supplied |

A search did not turn up a JEPA or latent world model that conditions its anti-collapse entropy on side information or prices the state; the pieces exist separately. The search was not exhaustive.

## 12. Follow-up: the narrative

1. JEPAs predict in representation space, but their objectives do not say what the representation should keep. Collapse prevention (EMA, SIGReg, VICReg) decides content by accident, and favours whatever is easy to spread and predict: appearance, slow features, even noise when the spread budget must be filled.
2. Principle: a state should carry what the past tells about the future, beyond what is available for free, at the smallest cost. That is a predictive-information bottleneck with side information.
3. Written in representation space with a noisy state, it has two halves. The JEPA prediction loss is one; an information (anti-collapse) term is the other. A price on the state's code length gives minimality. Nothing is added by hand, and every term is in nats.
4. The anchor is the side information: any frame of the episode. Measuring the state's information relative to it removes appearance, because appearance is identical in both.
5. Consequences, provable block by block: appearance excluded at every price; unpredictable detail excluded when the prediction weight is at least one; dynamics kept when they pay for themselves; fixed properties the dynamics read kept at `lam = 1`; collapse not even a local minimum.
6. Reconstruction, contrastive learning and spread regularizers are three estimators of the same information term: generative (non-saturating, needs a decoder), discriminative (saturates at `log K`), and Gaussian or pairwise (in state space). The decoder-free form uses the third.

One-line pitch: JEPA's regularizer is half of the predictive information; condition it on what the model gets for free, and price what it keeps.

## 13. Follow-up: downsides, in order of importance

1. **Prediction gates information, so the current prior's weakness biases the state.** With weight `w = beta + lam >= 1`, a component earns only what the prior can predict. Hard dynamics (contacts) are under-rewarded while the prior is weak, which can lock in a state without them. This is a plausible reading of the seed-0 block coordinate in the toy.
2. **No salience: information is counted in the state's own currency.** A small but predictable visual change earns full credit regardless of its pixel area. That fixes the small-object problem of pixel losses (relevant to the TwoRoom collapse). But it also gives full credit to small predictable distractors (indicators, shadows, progress bars) that a pixel loss would down-weight. Pixel-level sufficiency is no longer certified.
3. **The estimator assumes something.** The Gaussian log-det over-credits discrete or clustered features by up to a few nats each. The pairwise bound is assumption-free but noisy. Both are batch statistics: they need many same-episode pairs per batch, and for states larger than the batch the log-det sees at most as many directions as samples.
4. **The conditioning is approximate.** It uses the anchor's state, linearly, and assumes the anchor and the frame are independent draws from the episode. Components that change little over the gap are under-rewarded (the A5 issue again).
5. **The price is mandatory.** Without it, predictable components inflate without bound, so near-zero-price controls lose meaning.
6. **The generative side is lost from training.** Rollout videos, swap tests, pose-based action identification and the image-editing interpretation need a decoder trained afterwards. Planning is unaffected.
7. **Theory moves to the estimator.** The exact equality with the renderer objective holds only at `lam = 0` with exact entropies. With the Gaussian estimator, the key thresholds (Section 5) are re-derived, but the general block-identifiability proofs would need restating. The log-det is rotation-invariant, so the state is identifiable only up to rotation without learned variances.
8. **The novelty is easy to misread.** On paper it looks like a prediction loss plus a log-det (MEC or VICReg family). The case rests on the derivation, the anchor conditioning, the price, and empirical purity and planning results against LeWM.

## 14. Corrections later the same day

An independent review of the theory and the spec, against the collaborator's code, changed three statements above. The current versions are in [../decoder-free/story.md](../decoder-free/story.md).

1. **The Gaussian estimator can be gamed.** The pooled log-det is an upper bound on `I_A`. It assumes Gaussian states, and pooling across anchors adds a Jensen gap. In a constructed example, an encoder that rotates each episode's change by an episode-level key (its colour) raises the estimate from 4.6 to 40 nats per frame without adding information. So Proposition 2 does not hold for the pooled Gaussian estimator in general. Two consequences:
   - **A per-window estimate closes the episode-level case.** It applies the same formula to each window's changes relative to its own anchor; a contract test checks it. In the toy, though, it under-credited the slowly moving block.
   - **The minibatch pairwise (collision) estimate is not a usable bound.** It is biased upward and dominated by the nearest pair, so it is only logged as a diagnostic.

   This is the honest price of removing the decoder. The renderer's likelihood was a lower bound and erred in the safe direction; a decoder-free estimate avoids saturating near the log of the sample size only by assuming something about the state's distribution, and the encoder can exploit that assumption.
2. **The collapse barrier depends on what the actions reveal.** The prior reads unpriced actions, so at small amplitude its residual is `Var(z | a)`, and a barrier exists exactly when `w > v / Var(z | a)`. Action-determined components, such as the agent, face no barrier. Components the actions do not reveal, such as the block, can face one even below `w = 1`.
3. **Fixed properties are stored at `lam = 0` in a fixed-variance code.** "Stored iff `lam > 0`" holds only for an exact code, and only when contacts reveal enough of the property.

Toy runs added the same day show two things.

- **The transient follows the threshold.** At `w = 0.3` (pooled, `lam = 0`) the transient is kept at the predicted `R^2 ~ 1 - w = 0.7`, and the block fully.
- **Purity and completeness trade off along `w`.** At `w = 1.3` the block was kept only partly. The plan therefore maps that trade-off across the prediction weight.
