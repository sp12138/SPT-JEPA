# Anchored-State JEPA: formulation, method, theory and narrative

Written 2026-10-26. This is the canonical statement of AS-JEPA: the problem, the formulation, the objective and how the JEPA loss emerges from it, the architecture, the theory, and the narrative for the JEPA community. The design notes ([anchored-state-jepa-design.md](anchored-state-jepa-design.md)) keep the decision history and component details; where the two differ, this document is current. Section 5 replaces the theory sketch in design-notes Section 5 (Section 5.8 says what changed and why). The build plan is [as-jepa-spt-implementation-plan.md](as-jepa-spt-implementation-plan.md). Nothing has been implemented or trained.

**Notation.** `x_t` is frame `t`, `a_t` the action taken after it, `z_t` the true dynamics state, `c` the episode's appearance, `eps_t` the transient detail of frame `t`, and `s_t` the learned noisy state, with encoder mean `mu(x_t)` (the design notes call the learned state `z`). A training window holds frames `x_0 .. x_T` and actions `a_0 .. a_{T-1}`; `x_0` is the anchor. Pixels are 8-bit, so `H` is discrete entropy; learned states are continuous, so `h` is differential entropy; `I` is mutual information. All quantities are in nats.

## Summary

We want a world model whose learned state holds what the dynamics need and leaves out persistent appearance and unpredictable detail, so that it supports latent planning, generation and probing from single frames. We obtain such a model from one principle and add no JEPA ingredient by hand: describe each video with as few nats as possible, when the receiver already holds the first frame and can sample. The resulting objective has three parts:

- the cost of the first state;
- the cost of each later state under a learned predictor;
- the cost of the frames under a renderer that sees the state and the first frame.

With Gaussian codes, the second part is exactly a JEPA latent prediction loss with an attached target. With an optimal predictor and renderer, the whole objective reduces to the entry cost minus how predictable each state is from its past states and the actions, plus how predictable it is from the first frame. In short, the objective rewards information that the past predicts and the first frame does not, which is what we mean by a high-level state. We prove that appearance is excluded at every price and transient detail at every price of at least one, and we state the condition under which every dynamics component is kept.

The three claims, in order of strength:

1. **Emergence (exact).** The JEPA latent prediction loss is the code length of the state under the predictor (Section 3.3).
2. **Identity (exact, Proposition 1).** With optimal predictor and renderer, the objective is `entry cost - sum_t [ I(s_t; s_<t, a_<t) - I(s_t; x_0) ]`.
3. **Block identifiability (Propositions 2 and 3, and a conditional claim in Section 5.6).** Appearance is excluded for every price `beta > 0`, transient detail for `beta >= 1`, and each dynamics component is kept when it pays for its entry within the window.

## 1. Problem

A world model for planning and generation needs a state that holds what the dynamics need: positions, velocities, contact geometry. It should leave out what the dynamics do not need: colours, textures, background, and detail that changes unpredictably from frame to frame. Three existing objectives do not settle what the state keeps.

- **Prediction alone keeps appearance.** A state trained to be predictable from its past drops only what the past cannot predict. Persistent appearance is the easiest thing to predict, so it stays.
- **JEPA objectives have no term that decides content.** A JEPA objective is a latent prediction loss plus something that prevents collapse (EMA or stop-gradient, augmentations, masking, or a regularizer such as SIGReg). Every representation that is predictable and non-degenerate satisfies it, and which information survives is decided by those heuristics.
- **Deterministic reconstruction keeps everything.** A pixel decoder trained with squared error pushes the state to store every detail, including detail that cannot be predicted, to reduce blur.

The question this work answers is what a representation should keep, and whether an objective can say so without heuristics.

## 2. Formulation

### 2.1 Data-generating process

Each episode draws an appearance `c` and an initial state `z_0`, independently. The state evolves under actions, possibly depending on the last `L` states, and every frame is rendered from the current state, the appearance and fresh transient detail:

```
z_{t+1} = f(z_{t-L+1:t}, a_t, eta_t)        f does not read c
x_t     = g(z_t, c, eps_t)                  eps_t i.i.d. over time
```

Graph: `c -> x_t` for every `t`; `z_t -> x_t`; `(z_{t-L+1:t}, a_t) -> z_{t+1}`; `eps_t -> x_t`; no edge from `c` or `eps` into `z`; actions depend on the states only.

This defines three blocks by what they do over time:

| Block | Persists over time | Affects the dynamics | PushT examples |
| --- | --- | --- | --- |
| dynamics state `z` | yes | yes | agent position, block position and angle; also block shape and scale, which are constant within an episode but change contacts (their transition is the identity) |
| appearance `c` | yes | no | agent, block and background colours; goal-overlay pose |
| transient `eps` | no | no | sensor noise; the random dot in our test variant |

Velocity is a property of the state sequence, not of a single frame. Allowing the transition to read the last `L` states keeps `z_t` the configuration visible in frame `t`, so a single frame determines it (Section 5.2, A1).

### 2.2 What we want

We want the learned state to depend on the frame only through `z_t` and to determine `z_t`: `s_t = h(z_t)` plus code noise, with `h` injective and independent of `c` and `eps_t`. This is block identifiability of the dynamics block. Proposition 1 gives the same target in information terms: the state should carry information that is predictable from the earlier states and the actions, and not predictable from the first frame.

## 3. Principle and objective

### 3.1 Coding with side information

A sender transmits a video to a receiver who already holds the first frame of the window and can draw random numbers. The sender turns each frame into a noisy state, transmits the states, and the receiver renders each later frame from its state and the first frame. Three principles decide what is worth sending:

1. **Information has a price** (minimum description length). Every nat in the state is paid for. A predictive code pays for a nat once, when it enters; afterwards a predictor that has seen the earlier states can predict it for free.
2. **Side information belongs at the decoder** (Wyner-Ziv). The receiver already has the first frame, so whatever it shows, in particular the appearance, costs nothing to render and is not worth sending.
3. **Sample what you cannot predict.** Storing a detail that no past predicts costs its information in rate and saves at most the same amount in rendering. A stochastic renderer can sample it instead.

Each principle has one architectural channel: the priced state, the anchor, and the renderer's noise. In the data's terms, each block of Section 2.1 gets exactly one cheap channel: `z` through the state, `c` through the anchor, `eps` through the noise.

### 3.2 The objective

Generative model: `s_0 ~ N(0, I)`; `s_t ~ p_phi(s_t | s_<t, a_<t)` for `t >= 1`, computed by a causal predictor that never reads frames; `x_t ~ p_psi(x_t | s_t, x_0)` for `t >= 1`, a flow-matching renderer. The anchor `x_0` is never rendered. Inference: a per-frame encoder `q_theta(s_t | x_t)`, applied to every frame of the window including the anchor. The objective is the expected code length, with `beta` the price per nat of state:

```
J_beta = sum_{t=1}^{T} E[ -log p_psi(x_t | s_t, x_0) ]
       + beta * ( E KL(q(s_0|x_0) || N(0, I))  +  sum_{t=1}^{T} E KL(q(s_t|x_t) || p_phi(s_t | s_<t, a_<t)) )
```

At `beta = 1` this is the negative evidence lower bound of the generative model, and by the bits-back argument (Hinton and van Camp 1993) the KL terms are the code length of the noisy states. In the simplest practical form both variances are fixed, `q(s_t | x_t) = N(mu(x_t), sigma_0^2 I)` and `p_phi = N(m_t, s_0^2 I)`:

```python
mu = encoder(x)                                  # [B, T+1, K, d]; each frame encoded alone
s = mu + sigma0 * torch.randn_like(mu)           # the code; its noise sets the resolution
m = s[:, :-1] + prior(s[:, :-1], a[:, :-1])      # predicted means of s_1..s_T, from states and actions only
entry = 0.5 * mu[:, 0].pow(2).sum((-2, -1))                       # first state against N(0, I)
surprise = (mu[:, 1:] - m).pow(2).sum((-2, -1)) / (2 * s0 ** 2)   # later states against the predictor
fm = flow_loss(renderer, y=x[:, 1:], state=s[:, 1:], anchor=x[:, 0])   # frames given state and anchor
loss = fm + beta * (entry + surprise.sum(1)).mean()               # nothing else
```

No stop-gradient, target network, EMA, augmentation or masking appears anywhere.

### 3.3 How the JEPA loss emerges

The cheapest way to send a sequence of states is predictive coding: the first state against a fixed prior, each later state against a predictor that has seen the earlier states and the actions. With fixed variances, the cost of state `t` is

```
KL(q(s_t|x_t) || p_phi(s_t | s_<t, a_<t)) = ||mu(x_t) - m(s_<t, a_<t)||^2 / (2 s_0^2) + const
```

That is the JEPA latent prediction loss. A predictor maps earlier embeddings and actions to the next frame's embedding, and is scored by squared error against the same encoder's embedding of that frame, with the target attached, as in LeWM. Nobody chose this loss; it is what the state costs under the predictor. With learned variances it becomes a Gaussian negative log-likelihood, a JEPA loss with calibrated uncertainty.

The entry term contains what JEPA adds as an anti-collapse regularizer, and more. By the ELBO decomposition (Hoffman and Johnson 2016),

```
E_x KL(q(s|x) || N(0, I)) = I(x; s) + KL(q(s) || N(0, I))
```

The second half matches the aggregate embedding distribution to an isotropic Gaussian, the role SIGReg plays in LeJEPA and LeWM. The first half is the price of information, which SIGReg leaves out. The cost of the frames is the flow loss of a renderer that sees the first frame. That term is what makes information worth paying for, and it is what prevents collapse.

So AS-JEPA is LeWM's objective with SIGReg replaced by the full code length plus a renderer: the same encoder, predictor and attached-target squared error, with noise on the embedding, an entry price, and an anchored stochastic renderer added. Two smaller differences: LeWM's predictor reads clean embeddings while ours reads noisy states, because the noise is part of the code; and LeWM regularizes every embedding while we price only the first state, because later states are priced by the predictor.

### 3.4 Why collapse does not happen

A collapsed encoder (states independent of the frames) pays the smallest entry cost, but its renderer must then generate every frame from the anchor alone, so the flow loss pays for all motion. Rescaling the states cannot help either: the noise `sigma_0` is fixed, so shrinking `mu` lowers the prediction error only by transmitting less information, and the flow loss charges for the loss. Section 5.6 states exactly when collapse is not optimal: when the windows contain action-explained change relative to the anchor that is worth more than its entry cost.

## 4. Method

### 4.1 Encoder

A ViT reads one frame. `K` learned query tokens cross-attend to its patch tokens (Perceiver-style pooling), and a linear head gives `mu`; the state is `K` tokens of `d` dimensions (`K = 4`, `d = 16` to start), and `s = mu + sigma_0 * xi`.

- **Per frame, with no access to history or anchor.** The state must be a function of its own frame: goal images encode without history, the predictor owns all prediction, and the encoder cannot express "pose relative to the anchor", so it cannot move position into the renderer's side information. In the theory, the encoder's output is the object the identifiability claim is about.
- **Shared across time, including the anchor.** The anchor is encoded too, and its state is the first state of the chain. Together with the anchor never being rendered, this makes transient information strictly unprofitable (Proposition 3).
- **Learned queries.** A fixed slot identity across examples without masking, and a state that is small by construction.
- **Noise of fixed size.** The noise is what makes the code length a real price. A noiseless continuous vector carries unbounded information; with noise `sigma_0`, each dimension carries about `log(range / sigma_0)` nats.
- **No normalization of `mu`.** Batch or layer normalization applied to `mu` itself would pin each dimension's spread, so the entry price could never switch a dimension off.

### 4.2 Prior

A causal transformer over past noisy states and actions outputs the next state's distribution. The mean is residual, `m_t = s_{t-1} + delta_t`, with the `delta` head zero-initialized so the prior starts as the identity carry; a variance head is optional (Section 4.6). Attention is causal over steps, and the predictor may attend to the last `L` states, which is where velocity lives.

- **Trained only by the code length.** The prior emits a distribution over the next state and never feeds the renderer; during training the renderer reads the encoded state. This separation is what makes bits chargeable. In SPT, by contrast, the causal transformer outputs a context that conditions the renderer directly, so it carries whatever helps the renderer, appearance included.
- **Anchor-blind.** If the prior saw the anchor, everything visible in the first frame would become free to predict. Appearance would then cost nothing in the state (no exclusion), and dynamics-relevant static factors such as block shape would never need to enter the state.
- **Actions enter here only.** Actions are interventions on the state mechanism. A renderer that saw actions could render motion without the state.
- **Residual mean.** Unchanging dimensions cost nothing to carry after entry.
- **One step.** The prior predicts `s_t` from `s_<t`; multi-step targets (SPT's random offset, latent overshooting) do not serve the argument and are removed.

### 4.3 Renderer

A flow-matching DiT renders frame `t` from noise, conditioned on `s_t` and the anchor, with a fixed target (pixels or a frozen codec). On a static camera the anchor is concatenated with the noisy target along channels, so every renderer position sees the corresponding anchor pixel. The state tokens enter through cross-attention, after a linear projection with no per-token normalization. Dropping the state with small probability during training enables classifier-free guidance on the state at inference, and the guidance scale needed is a direct readout of how much the renderer relies on the state. For video with camera motion, the anchor's tokens are cross-attended (ReferenceNet style) or placed in context (FLUX Kontext, OmniGen).

- **History-blind and action-blind.** The renderer sees only the current state and the anchor, so it cannot do the dynamics itself.
- **Stochastic.** Given the current state and the anchor, the only randomness left in a frame is its within-frame residual: transient detail and whatever appearance the anchor does not pin down. Alternative futures live in the prior. One observed outcome per context is enough to learn a conditional distribution, as one label per input is enough for a classifier.
- **Fixed target.** The learned state is the condition, not the target, so the collapse reported for learned targets under a diffusion loss (REPA-E) does not apply.
- **Noise weighting.** The state's gradient comes from noise levels at which the noisy target has not yet revealed the answer, so weighting toward high noise makes the state responsible for coarse structure. SPT's schedule is already high-noise weighted.

### 4.4 Training

One pass trains encoder, prior and renderer together on `J_beta`. Windows hold `T + 1 = 8` frames at frameskip 5, one frame per chunk; the anchor is the window's first frame and is never rendered. `beta` ramps from 0 over the first few percent of training, so the renderer learns to use the state before the price applies. Entry and surprise have separate weights in practice (`beta_entry`, `beta_surprise`, starting equal). There is no augmentation, EMA, stop-gradient, target network or masking.

### 4.5 Uses of the trained model

- **Planning.** Encode the history and the goal image with `mu`, roll out the prior's means under candidate actions, and score the distance to `mu(goal)`. The renderer and the anchor are not needed.
- **Generation.** Sample the next state from the prior and render it with the first observed frame as the anchor; append the sampled state or re-encode the generated frame.
- **Representation.** `mu(x_t)` is the high-level state, appearance stays in the anchor, and the prior's hidden state is the belief over the history.

### 4.6 Practical codes

All codes optimize the same objective, distortion plus `beta` times code length; they differ in the code.

| Code | Posterior | Rate | Use |
| --- | --- | --- | --- |
| Gaussian, fixed variances | `N(mu, sigma_0^2)` | `||mu_t - m_t||^2 / (2 s_0^2)`; entry `||mu_0||^2 / 2` | first choice: three lines, no learned variances |
| Gaussian, learned variances | `N(mu, sigma^2)` | closed-form diagonal KL | exact bound, calibrated uncertainty; needed for the component-wise claim (Section 5.7) |
| FSQ, discrete | quantized `mu` | categorical cross-entropy, exact bits | if `beta` is painful: the code size is a hard budget |
| noisy continuous plus latent flow prior | `N(mu, sigma_0^2)` | latent flow-matching loss, an ELBO-type bound | multimodal futures, action-free video |

Masking is not one of these codes. Dropping input tokens at random hides nothing from a network that sees them at other times, gives no entry cost and trains no predictor; as an information restriction it is at most an erasure channel on the state, an alternative noise rather than a price.

### 4.7 What each component does

| Component | Principle | What breaks without it |
| --- | --- | --- |
| per-frame encoder, shared across time | the state is a function of its frame | an encoder that sees history makes the prior trivial; an encoder that sees the anchor can encode pose relative to it |
| noise on the state | a finite price per nat | an unbounded code; the price means nothing |
| entry term against `N(0, I)` | information enters at a cost | appearance is predictable after the first state and costs nothing |
| anchor to the renderer only | side information at the decoder | appearance must enter the state; with the anchor in the prior, appearance is free in the state too |
| anchor-blind, action-conditioned prior | predictive coding of the state | an unpriced history summary carries appearance (plain SPT) |
| stochastic renderer | sample what you cannot predict | a squared-error renderer pushes transient detail into the state |
| history-blind, action-blind renderer | the state is the only route across time | the renderer does the dynamics and bypasses the state |

## 5. Theory

### 5.1 Setting

The model is the one in Section 3.2, with the encoder allowed to be any per-frame stochastic map `q(s | x)` with finite entry cost: shared across time, applied independently to each frame of the window including the anchor, and reading nothing else. The fixed-variance Gaussian code is one parametrization of this class. We study `J*_beta(q)`, the objective after the prior and the renderer have been optimized for a given encoder.

### 5.2 Assumptions

- **(A1) Injective rendering.** `x_t = g(z_t, c, eps_t)` with `g` injective. In particular, any frame determines the appearance `c`.
- **(A2) Block structure.** `c` is independent of `(z_0, eta, eps)`; `eps_t` is i.i.d. over time and independent of everything else; `f` does not read `c`.
- **(A3) Actions see only states.** `a_t` depends on `(z_<=t, a_<t)` and independent noise, not on `c` or `eps`. A scripted policy that reads the simulator state satisfies this.
- **(A4) Expressive prior and renderer.** The prior family contains the true conditional density of `s_t` given `(s_<t, a_<t)` under the joint law of the data and the encoder, and the renderer family contains the true conditional law of `x_t` given `(s_t, x_0)`.

A4 is where practice departs from theory: a Gaussian prior and a flow renderer approximate these optima.

### 5.3 Proposition 1: the objective in terms of the encoder alone

For every encoder `q`,

```
J*_beta(q) = C + beta * E KL(q(s_0|x_0) || N(0, I))
               + sum_{t=1}^{T} [ beta * h(s_t | s_<t, a_<t) - h(s_t | x_0) + (1 - beta) * h(s_t | x_t) ]
```

where `C = sum_t H(x_t | x_0)` does not depend on `q`. At `beta = 1`, with `Entry(q) = E KL(q(s_0|x_0) || N(0, I))`,

```
J*_1(q) = C + Entry(q) - sum_{t=1}^{T} [ I(s_t; s_<t, a_<t) - I(s_t; x_0) ]
```

For the fixed-variance code, `h(s_t | x_t) = (d/2) log(2 pi e sigma_0^2)` and `Entry(q) = (1/2) E||mu(x_0)||^2 + const`.

*Proof.* Fix the encoder. Because it reads only its own frame and adds independent noise, `s_t` is independent of all other variables given `x_t`, and `s_t` has a density.

- Renderer: by Gibbs' inequality `E[-log p(x_t | s_t, x_0)] >= H(x_t | s_t, x_0)`, with equality for the true conditional (A4). Write `H(x_t | s_t, x_0) = H(x_t | x_0) - I(x_t; s_t | x_0)`, and `I(x_t; s_t | x_0) = h(s_t | x_0) - h(s_t | x_t)` by the conditional independence.
- Prior: `E KL(q(s_t|x_t) || p(s_t | s_<t, a_<t)) = E[log q(s_t|x_t)] - E[log p(s_t | s_<t, a_<t)]`. The first term is `-h(s_t | x_t)` by the conditional independence. The second is at least `h(s_t | s_<t, a_<t)`, with equality for the true conditional density (A4).
- The entry term is not optimized.

Adding the renderer terms, `beta` times the prior terms and `beta` times the entry term gives the first identity. At `beta = 1` the `h(s_t | x_t)` terms cancel, and `h(s_t | A) = h(s_t) - I(s_t; A)` gives the second. For the fixed-variance code the noise entropy is constant and `E KL(N(mu, sigma_0^2 I) || N(0, I)) = (1/2) E||mu||^2 + const`. ∎

**Reading.** The objective rewards information in each state that the earlier states and actions share, and penalizes information in it that the anchor shares. By the chain rule,

```
I(s_t; s_<t, a_<t) - I(s_t; x_0) = I(s_t; s_<t, a_<t | x_0) - I(s_t; x_0 | s_<t, a_<t)
```

The first term is what the history and actions tell about the state beyond the anchor; the second is what the anchor tells about it beyond the history. Appearance is shared with the anchor; transient detail is shared with nothing; the dynamics state is shared with the history and the actions. The next three subsections make this precise, block by block.

### 5.4 Proposition 2: appearance is excluded at every price

Under A1 to A4, for every `beta > 0` and every encoder `q`, there is an encoder `q_bar` whose states are independent of `c` such that

```
J*_beta(q) >= J*_beta(q_bar) + beta * sum_{t=1}^{T} I(s_t; c | s_<t, a_<t)
```

*Proof.* For a fixed appearance value `c`, let `q_c(s | z, eps) = q(s | g(z, c, eps))` be the encoder restricted to that appearance. Let `F_beta(nu)` be the right-hand side of Proposition 1 for the population of `(z, eps, a)` encoded by `nu`, with prior and renderer allowed to depend on `c`. By A2 and A3 the joint law of `(z, eps, a)` does not depend on `c`, so `F_beta` is one functional for all `c`. By A1 the anchor determines `c`, so `h(s_t | x_0) = h(s_t | x_0, c)` and `h(s_t | x_t) = h(s_t | x_t, c)`, and the entry term is an average over `c`. The only term that does not split over `c` is the prior term, where `h(s_t | s_<t, a_<t) = h(s_t | s_<t, a_<t, c) + I(s_t; c | s_<t, a_<t)`. This gives the exact decomposition

```
J*_beta(q) = E_c F_beta(q_c) + beta * sum_{t=1}^{T} I(s_t; c | s_<t, a_<t)
```

Let `nu*` minimize `F_beta` (or approach its infimum), and let `q_bar` apply `nu*` to every frame, ignoring `c`. Its states and actions have the same joint law for every `c`, so its information term is zero, the `c`-blind prior is already optimal, and `J*_beta(q_bar) = F_beta(nu*) <= E_c F_beta(q_c)`. ∎

**Consequences.** The minimum is attained by encoders that carry no appearance information, and every minimizer adds no appearance information after its first state: `I(s_t; c | s_<t, a_<t) = 0` for all `t >= 1`. With Gaussian noise on every state, earlier states never pin an appearance code down exactly, so in non-degenerate cases a later state adds appearance information whenever the earlier ones carry some, and the inequality is strict. A minimizer's states and actions therefore have the same joint law for every appearance, and probes on the states cannot recover it. Appearance-dependent reparametrizations that leave this law unchanged (for example, rotating an isotropic state differently for each appearance) remain possible, but they carry no appearance information.

**Why it works.** The anchor gives the renderer appearance for free (A1), and the prior, being anchor-blind, must pay for any appearance information the states add. Both design choices are needed: with the anchor also in the prior, the information term vanishes and nothing excludes appearance; without the anchor in the renderer, appearance becomes useful to the state.

### 5.5 Proposition 3: transient detail is excluded at every price of at least one

Under A1 to A4, for any encoder `q`, let `q'` encode a fresh independent copy of the transient instead of the actual one, `q'(s | x_t) = E_{eps' ~ P_eps} q(s | g(z_t, c, eps'))`. Then

```
J*_beta(q) - J*_beta(q') = beta * I(s_0; eps_0 | z_0, c) + (beta - 1) * sum_{t=1}^{T} I(s_t; eps_t | z_t, c)
```

*Proof.* Because `eps_t` enters only frame `t` and is independent of everything else (A2, A3), the states of `q'` have the same joint law with `(z, c, a)` as the states of `q`. Hence the prior terms `h(s_t | s_<t, a_<t)` are unchanged, and so are the anchor terms: for `t >= 1`, `s_t` is independent of `eps_0` given `(z_0, c)`, so `h(s_t | x_0) = h(s_t | z_0, c)` for both encoders. The aggregate law of `s_0` is unchanged, so in `Entry = I(s_0; x_0) + KL(q(s_0) || N(0, I))` only the first term changes, by `I(s_0; x_0) - I(s_0; z_0, c) = I(s_0; eps_0 | z_0, c)`. Finally `h(s_t | x_t) - h(s'_t | x_t) = h(s_t | z_t, c, eps_t) - h(s_t | z_t, c) = -I(s_t; eps_t | z_t, c)`. Collecting the coefficients from Proposition 1 gives the identity. ∎

**Consequences.** For `beta >= 1`, removing transient information never increases the objective, and strictly decreases it whenever the first state carries transient information; for `beta > 1`, strictly whenever any state does. Windows start at random times within episodes, so the anchor's state distribution covers the states at other times, and a minimizer at `beta = 1` already carries no transient information on the data's support. When transient information is the same at every frame, exclusion holds down to `beta > T / (T + 1)`.

**Why it works.** A transient detail in frame `t >= 1` costs its information in surprise and saves the same amount in rendering, a wash at `beta = 1`. The anchor breaks the tie: the shared encoder would also put the anchor's transient into the first state, where it pays the entry price and is never used, because the anchor is never rendered. The proof needs `q'` to be a per-frame stochastic encoder, which is generally not of the fixed-variance Gaussian form; the proposition describes the optimum over all per-frame codes, of which the Gaussian code is one parametrization.

### 5.6 Dynamics: what is kept, and when collapse is not optimal

After Propositions 2 and 3, a minimizer's state depends on the frame only through `z_t` (for `beta >= 1`). For such codes, `I(s_t; x_0) = I(s_t; z_0)`, and at `beta = 1`

```
J*_1(q) = C + Entry(q) - W_T(q),        W_T(q) = sum_{t=1}^{T} [ I(s_t; s_<t, a_<t) - I(s_t; z_0) ]
```

`W_T` is the window value of the code: what the history and actions tell about the states beyond what the anchor tells. Which components of `z` a minimizer keeps is decided by comparing each component's contribution to `W_T` with its contribution to the entry cost.

- **Changing components** (agent position). The value accumulates frame by frame, roughly the uncertainty about the position given the anchor minus its uncertainty given the earlier states and the actions, and grows with the window length. The entry cost is paid once.
- **Static, dynamics-relevant components** (block shape and scale). Their direct value is about zero (slightly negative): the anchor shows them exactly and earlier states repeat them noisily. They pay the entry cost and are worth it only through prediction, `sum_t I(position_t; shape | earlier states, actions)`, the improvement in predicting positions through contacts. They are kept when contacts within the window carry more information than their entry cost.
- **Slowly changing components** (block pose when the block is rarely pushed). When the block does not move within a window, the anchor already shows its pose and the state gains nothing from it. Short windows, or a policy that rarely pushes the block, can leave block pose to the anchor; the renderer then copies the block, and predicted pushes fail.
- **Anchor distance.** The value of every changing component grows with how little the anchor reveals about the current state. The window's first frame is the most distant frame inside the window. An anchor drawn from a distant time in the same episode would raise the value of slow components at no cost to appearance exclusion, since any frame of the episode shows the appearance. This is an open design choice (Section 9).

**Condition (A5): every component pays for itself within the window.** For every non-constant function `phi` of the state, encoding `phi(z_t)` more precisely increases `W_T - Entry` at the optimum.

**Claim (block identifiability).** Under A1 to A5 at `beta = 1`, a minimizer's state depends on the frame only through `z_t` (Propositions 2 and 3) and is informative about every component of `z_t`; in the noiseless limit, `s_t = h(z_t)` with `h` injective and independent of `c` and `eps_t`. Status: Propositions 1 to 3 are proved. The inclusion half is a claim: A5 is an assumption about the objective's landscape that we have not reduced to primitive conditions on the dynamics, and with finite noise "injective" is a statement about the encoder mean `mu`, not the noisy state.

**Corollary (when collapse is not optimal).** The collapsed code, whose states are independent of the frames, has `W_T = 0` and the smallest entry cost. It is not a minimizer exactly when some code has `W_T` larger than its extra entry cost, that is, when the windows contain action-explained change relative to the anchor worth more than the cost of describing the first state. Very short windows, static scenes, or anchors adjacent to the rendered frames can make collapse optimal.

**Away from `beta = 1`.** By Proposition 1, `J*_beta - J*_1 = (beta - 1) * [Entry + sum_t I(s_t; x_t | s_<t, a_<t)]`: raising `beta` adds `(beta - 1)` times the total rate, the entry cost plus each state's information that the past does not predict. Below `beta = 1` transient detail can be worth storing; by `beta = 1` it has left (Proposition 3); beyond that, dynamics components leave in order of value per nat, contact-only factors before frequently changing ones. Appearance is excluded at every `beta > 0`.

### 5.7 Component-wise identifiability inside the state (bonus)

Add two conditions from the temporal identifiability literature:

- **(B1) Conditionally independent innovations.** The components of `eta_t` are mutually independent given the past states and the action.
- **(B2) Sufficient variability.** The transition's conditional log-density varies enough with the past states and the action, in the linear-independence sense of TDRL and LEAP. Actions supply extra variability, the role played by the auxiliary variable in iVAE and by interventions in CITRIS and BISCUIT.

Then `h` reduces to a permutation composed with per-dimension maps (proof idea from TDRL: equate the latent log-densities under `h`, differentiate with respect to current components and the past to remove the Jacobian term, use independence to remove cross terms, and use variability to force the Jacobian to be a generalized permutation). The architectural counterpart is a diagonal prior whose means and variances depend on the past and the actions, so it needs the learned-variance code. On PushT this is measured by the mean correlation coefficient (MCC) against simulator factors. It is a bonus; the method does not depend on it.

### 5.8 Scope, and what changed from the earlier sketch

Not covered by the theory: parametric priors and renderers instead of optimal ones (A4), including SPT's flow loss, whose noise-time weighting is not an evidence bound, so `beta` is not calibrated in nats in practice (a renderer trained with a monotone, ELBO-type weighting would calibrate it, by Kingma and Gao 2023); optimization, including posterior collapse as a local optimum; non-injective rendering such as occlusion; camera motion, where the anchor does not show everything the episode will reveal; predictable but control-irrelevant dynamics, which belong to `z` by definition and need an action-based criterion to remove; multimodal futures under a Gaussian prior.

The design notes (Section 5.3) sketched a three-step proof: matching, coupling through the state, rate minimality. Writing it carefully changed four things:

1. **The matching step does not hold as stated.** It claimed the objective's minimum is the data's conditional entropy given the anchor, reached when the model matches the data. Because the state chain is anchor-blind by design, the model must pay for the first state and can at best match the conditional given the appearance. Proposition 1 replaces the step with an exact identity.
2. **Appearance exclusion holds at every price**, with an exact gap (Proposition 2).
3. **Transient exclusion is strict at `beta = 1`**, because the shared encoder also encodes the never-rendered anchor. No minimum-rate tie-break is needed (Proposition 3).
4. **The coupling assumption becomes a per-window value condition** (A5). It makes window length, contact frequency and anchor distance part of the theory, and it corrects the design notes' remark that a static dynamics-relevant factor's value "accrues over the whole episode": the entry cost is paid per training window, so the value must accrue within the window.

Also corrected: the design notes said the entry term rules out appearance-dependent symmetries of the state; it does not rule out all of them (Section 5.4), but those it allows carry no appearance information.

### 5.9 Literature map

| Ingredient | Source | Where it enters |
| --- | --- | --- |
| KL as the code length of a noisy code (bits-back) | Hinton and van Camp 1993; Frey and Hinton 1997 | Section 3.2 |
| ELBO as rate-distortion | Alemi et al. 2018 | the objective; `beta` as the price |
| aggregate-posterior decomposition of the KL | Hoffman and Johnson 2016 | entry term versus SIGReg (Section 3.3) |
| side information at the decoder | Wyner and Ziv 1976 | the anchor route; Proposition 2 |
| shared versus changing blocks; minimal change | von Kugelgen et al. 2021; Lyu et al. 2022; Yao et al. 2024; Kong et al. 2022 | block identifiability (Sections 5.4 to 5.6) |
| temporal latent processes with time-delayed, independent innovations | iVAE (Khemakhem et al. 2020); TDRL and LEAP (Yao et al. 2022) | lag-`L` transitions; Section 5.7 |
| actions as interventions | CITRIS, BISCUIT (Lippe et al.); mechanism sparsity (Lachapelle et al.) | actions in the prior only |
| weighted diffusion losses as ELBOs | Kingma and Gao 2023 | calibrating `beta` (Section 5.8) |
| level selection by masking in hierarchical latents | Kong et al. 2023 | chunk length as the slow/fast knob (Section 9) |

## 6. Narrative for the JEPA community

### 6.1 The main line

JEPA's goal is to predict in representation space and not spend capacity on unpredictable pixel detail. Its objective, prediction plus anti-collapse, says what to predict but not what to keep, and the heuristics around it decide.

Our message: **the JEPA objective is not designed here, it is derived.** Price the state's information, give the renderer the first frame for free, and let it sample what nobody can predict. The latent prediction loss then appears as the cost of each state under the predictor (Section 3.3). What the price buys is stated exactly (Proposition 1): the best state is the one that is most predictable from its own past and the actions, and least predictable from the first frame. Appearance is predictable from the first frame, so it leaves at any price; transient detail is predictable from nothing, so it leaves at a price of at least one; the dynamics state is predictable from the past with actions, so it stays.

One-line pitch: **we do not reconstruct pixels, we price them.**

### 6.2 Why the derived objective needs no heuristics

JEPA avoids reconstruction because a pixel loss forces the latent to encode everything and is dominated by high-entropy detail. Both complaints are about reconstruction that is unpriced and unexcused.

- **Two alibis.** Appearance is excused by the anchor; unpredictable detail is excused by the noise. Whatever the renderer cannot excuse must come from the state, and every nat of state costs.
- **Fixed points.** A JEPA objective is satisfied by every representation that is predictable and non-degenerate. Ours singles out one kind of optimum, the minimal predictive state (Section 5), and collapse is the degenerate end of the same continuum: the distortion term measures it, so EMA, stop-gradient and augmentation are not needed.
- **Capacity goes to the renderer, not the representation.** The state is tens of dimensions, and the renderer is discarded at planning time. JEPA's wasted-capacity objection is about the encoder's output, which here cannot hold pixels. The cost is training compute.
- **Pixel detail and gradients.** In flow matching, fine detail lives at low noise levels, where the noisy target already contains it; the state's gradient comes from high noise levels, which is coarse structure.
- **The field is moving this way.** V-JEPA 2 trains a diffusion decoder for visualization, VANE adds a light decoder, and FlowWM shows that deterministic latent predictors blur multimodal futures. All add back a stochastic or auxiliary renderer; we give it a principled role.

### 6.3 Interpretation: rendering as image editing

The renderer edits the first frame according to the state, `x_t = edit(x_0, instruction = s_t)`, so world prediction becomes predicting the next edit instruction. This is the intuitive picture of the side-information argument: an instruction is low-bandwidth and carries no appearance. The swap test literally applies episode B's instructions to episode A's image. The picture is an interpretation of the renderer's role, not the argument, and the method borrows interfaces from editing models (reference-image concatenation, instruction cross-attention, guidance on the instruction), not pretrained weights. It holds only with a fixed reference frame: editing from the current frame with the action as the instruction (ImageWAM, Orca, ChronoEdit) predicts in pixel space and lets the editor do the dynamics.

### 6.4 Anticipated objections

- *"VAEs with a KL give weak, blurry latents."* The state is small and never asked to carry pixels, and blur is impossible because the renderer samples. Dropping low-value nats at large `beta` is the intended behaviour, tuned by action following rather than reconstruction quality.
- *"Posterior collapse with a powerful decoder."* Real. An anchor-conditioned renderer can produce plausible frames without the state. Section 5.6 states when collapse is not optimal; in practice, ramp `beta`, keep high-noise weighting, and use windows long enough for motion relative to the anchor to outweigh the entry cost. The discrete code is a fallback.
- *"Isn't this Dreamer?"* RSSM is the template for a KL to a learned transition prior. It differs in three ways that carry the argument: a deterministic decoder with no side information (so its state holds appearance and detail), a deterministic recurrent path the decoder reads, and no account of what the state keeps.
- *"Why not keep SIGReg and add a renderer?"* Then nothing charges for information, and appearance stays in the state. Our experiments test exactly this: LeWM plus the anchored renderer, with SIGReg kept, is a rung of the comparison ladder.
- *"Why not a latent target?"* A learned latent target trained through a diffusion loss collapses (REPA-E), and a frozen semantic target (VANE) imports that encoder's biases. Here the target is fixed data and the state is the condition.
- *"One more knob."* `beta` replaces LeWM's SIGReg weight, which is also a single knob; ours has a meaning (the price per nat) and a prediction for how the state changes as it moves.

## 7. Relation to prior work

- **RSSM (PlaNet, Dreamer):** KL to a learned transition prior and latent planning; deterministic decoders, no anchor, so the state holds appearance.
- **SVG-LP:** a learned prior, but the decoder sees the previous frame, so the latent becomes a change code.
- **DBSE:** a static code from a fixed first frame, dynamic codes from a sequential prior; no actions, planning or stochastic renderer.
- **LeWM and LeJEPA:** an end-to-end JEPA with an attached target and SIGReg. AS-JEPA replaces SIGReg by the full code length plus an anchored stochastic renderer (Section 3.3).
- **LRC-JEPA:** keeps context out of the predictor but reconstructs the same frame deterministically, so it needs SIGReg plus a VICReg-style term. The closest JEPA baseline.
- **SODA:** a bottleneck encoder on one view and a diffusion decoder generating another; strong partly disentangled latents, no dynamics.
- **AnimateAnyone, MagicAnimate, InstructPix2Pix, FLUX Kontext:** reference-image conditioning mechanics and the reference-plus-control decomposition.
- **ImageWAM, Orca, ChronoEdit, EditWorld:** editing models as world or action models, editing from the current frame; the contrast in Section 6.3.
- **D-JEPA, VANE, Flow-JEPA, FlowWM:** diffusion losses with JEPAs, light decoders, stochastic latent dynamics; the latent-flow-prior variant follows Flow-JEPA and FlowWM.
- **Genie:** discrete latent actions with a decoder that sees past frames.
- **TDRL, LEAP, iVAE, CITRIS, BISCUIT, von Kugelgen et al., Kong et al.:** the identifiability lineage (Section 5.9).
- **SPT:** the per-frame encoder, causal transformer and flow renderer modules, and the end-to-end flow-matching recipe.

The pieces exist. Not found together: a flow renderer with anchor side information, a code-length chain with no anti-collapse heuristics, latent planning, and an identifiability analysis that matches the architecture. The value has to come from experiments.

## 8. Predictions the experiments must test

1. **Appearance.** With the anchor, colour, background and goal-overlay information in `mu` falls to chance once `beta > 0` has had time to act. Without the anchor it stays. LeWM keeps it, and so does LeWM plus the anchored renderer with SIGReg kept, because nothing prices it (Proposition 2).
2. **Transients.** A random dot redrawn every frame is excluded with the flow renderer, and enters the state or blurs with a squared-error renderer (Proposition 3).
3. **Dynamics.** Agent position and block pose are decodable from `mu`; block shape and scale are decodable when contacts are frequent within windows; decodability of slow components rises with window length and anchor distance (Section 5.6).
4. **Price ordering.** As `beta` increases, transient information leaves first (by the calibrated `beta = 1`), then contact-only dynamics, before frequently changing components.
5. **Collapse.** Collapse appears when windows are too short or motion too small relative to the entry cost; when the model works, the measured entry cost sits near the state's information budget (roughly 40 nats on PushT), and well above it signals leakage.
6. **Generation.** Rendered prior rollouts follow the actions, including through contact; the swap test shows the anchor's appearance with the other episode's motion.
7. **Planning.** Parity with LeWM on standard PushT, and an advantage when goal images have different colours from the observations.

The empirical plan will specify datasets, metrics, baselines and ablations for each prediction.

## 9. Open questions

- **Anchor distance:** the window's first frame, or a frame from a distant time in the same episode (Section 5.6)?
- **Window length** relative to motion and contact frequency (A5).
- **Entry versus surprise weights:** separate knobs, starting equal.
- **One state token versus several.**
- **Chunk length:** one frame now; multi-frame chunks would carry velocity and set the slow/fast level of the state.
- **An action-relevance term** to remove predictable but control-irrelevant dynamics (multi-step inverse dynamics, Delta-JEPA).
- **Camera motion:** anchor refresh, and content revealed after the anchor.
- **Calibrated `beta`** via an ELBO-weighted renderer loss.
- **The latent flow prior** for action-free, multimodal video.
