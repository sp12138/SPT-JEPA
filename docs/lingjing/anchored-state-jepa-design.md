# Anchored State JEPA: design notes

Updated 2026-10-25. Working name: AS-JEPA. Status: design only; nothing has been implemented or trained. The SPT-specific build plan is [as-jepa-spt-implementation-plan.md](as-jepa-spt-implementation-plan.md). This supersedes the residual-SPT sketch in [residual-jepa-spt-proposal.md](superseded/residual-jepa-spt-proposal.md) for the question of separating dynamics from appearance. The fixed-slot idea from those notes survives here as the learned state queries.

## 1. Goal and principles

The goal is a video world model, trained end to end, whose learned state holds the high-level information (what the dynamics need) and leaves out persistent appearance and unpredictable detail. The state must support latent planning, generation and probing from a single frame.

Three principles, each with one architectural consequence:

1. **Information has a price** (minimum description length, rate-distortion). Every bit in the state is charged once, when it enters; bits the predictor can predict are free afterwards. Consequence: a code-length term from a fixed initial prior through a learned transition prior (a KL chain).
2. **Side information belongs at the decoder** (Wyner-Ziv). Information the renderer can obtain elsewhere need not be in the state. Consequence: the first chunk (the anchor) feeds the renderer, and only the renderer.
3. **Sample what you cannot predict** (bits-back). Storing an unpredictable detail costs its entropy in rate and saves at most that much distortion, so a stochastic renderer makes sampling cheaper than storing. Consequence: a flow-matching renderer.

In causal-representation terms, the data has three latent blocks (dynamics state `z`, episode appearance `c`, transient nuisance `eps`), and the architecture gives each block exactly one cheap channel (state, anchor, noise). Section 5 makes this precise.

Decisions carried over from discussion: no visual augmentation, no EMA, no stop-gradient, no target network of any kind, one-step prediction only, no masking as the information restriction.

## 2. Model

Generative story for one episode of chunks `x_0 .. x_T` with optional actions `a_t`:

- `z_0 ~ N(0, I)`.
- `z_t ~ p_phi(z_t | z_<t, a_<t)`: a diagonal Gaussian computed by a causal transformer, with mean `z_{t-1} + delta`.
- `x_t ~ p_psi(x_t | z_t, x_0)` for `t >= 1`: a flow-matching renderer. The anchor `x_0` is never rendered.

Inference: `q_theta(z_t | x_t)` is a diagonal Gaussian from a per-chunk encoder. The encoder sees only its own chunk.

Objective (negative evidence lower bound; `beta = 1` is the exact bound only when the renderer term is an exact negative log-likelihood. SPT's flow loss is not, so with SPT `beta` is a rate-distortion weight to sweep; see [the implementation plan](as-jepa-spt-implementation-plan.md), Section 2):

```
L = sum_{t>=1} FlowLoss(x_t | z_t, x_0)
  + beta * [ KL(q(z_0|x_0) || N(0,I)) + sum_{t>=1} KL(q(z_t|x_t) || p(z_t | z_<t, a_<t)) ]
```

```python
mu, sig = Encoder(x)                      # [B, T+1, Kh, dh]; each chunk encoded alone
z = mu + sig * torch.randn_like(mu)       # posterior samples
m, s = Prior(z[:, :-1], a[:, :-1])        # causal; predicts z[:, 1:] as z_t + change

kl = (kl_normal(mu[:, :1], sig[:, :1], 0.0, 1.0).sum()          # first state: pay for every bit
      + kl_normal(mu[:, 1:], sig[:, 1:], m, s).sum()) / (B * T)  # later: pay only for surprise

y = x[:, 1:]                              # every chunk after the anchor; pixels or frozen codec
tn, eps = sample_noise_time(y), torch.randn_like(y)
v = DiT((1 - tn) * y + tn * eps, tn, state=z[:, 1:], anchor=x[:, 0])
fm = (v - (eps - y)).pow(2).mean()

loss = fm + beta * kl                     # nothing else
```

The simplest practical form fixes both variances (`sig = sigma_0`, `s = s_0`). The KL chain then reduces to three lines: Gaussian noise of fixed size on the state, a squared prediction error between the prior's mean and the next state, and a squared-norm penalty on the first state. See Section 6.

PushT starting point. Superseded for SPT by [as-jepa-spt-implementation-plan.md](as-jepa-spt-implementation-plan.md), which uses 4-frame RGB4 chunks at 256 px and SPT's ViT-E, causal blocks and pixel DiT; the table below is the original LeWM-style sketch:

| Module | Start | Note |
| --- | --- | --- |
| Encoder | ViT-tiny at 224 px, patch 14 (as LeWM), plus 4 learned state queries of 16 dims | state = 64 dims per chunk |
| Prior | LeWM's depth-6 predictor, residual mean, per-dimension log-variance head (or fixed variance) | block-causal over state tokens and actions |
| Renderer | DiT-S on 64 x 64 pixels; anchor channel-concatenated with the noisy target; state tokens through cross-attention | static camera, so the anchor is spatially aligned with the target |
| Data | windows of 8 frames, frameskip 5 | chunk = one frame |
| beta | ramp from 0 over the first few percent of training, then sweep | the main knob |

Three uses of the trained model:

- **Planning**: encode history and goal image with `mu`, roll out the prior's means under candidate actions, score `||z_hat_T - mu(goal)||^2`. No renderer or anchor is needed. Keeping LeWM's `encode` and `rollout` signatures (`stable_worldmodel/wm/lewm/lewm.py`) lets the repo's planners and `GoalMSE` run unchanged.
- **Generation**: sample the next state from the prior and render it with the first observed chunk as anchor; append the new state or re-encode the generated chunk.
- **Representation**: `mu(x_t)` is the high-level state; appearance stays in the anchor; the prior's hidden state is the belief over the history.

## 3. Components and their justification

### 3.1 Per-chunk encoder with learned state queries

What: a ViT over one chunk; `K_h` learned queries cross-attend to the patch tokens (Perceiver-style pooling); linear heads give `mu` and `log sigma`. The state is `K_h` tokens of `d_h` dims.

Why per-chunk, with no access to history or anchor: the state must be a function of the frame alone. This gives three things at once. Goal images encode without a history. The state is Markov, so the prior owns all prediction. And the encoder cannot express "pose relative to the anchor", so it cannot offload position into the renderer's side information. In the theory, `z_hat_t = E(x_t)` is the object the identifiability claim is about.

Why learned queries: a fixed slot identity across examples (the "fixed J" of the residual-SPT notes) without masking, and a state that is small by construction.

Why Gaussian noise on the state: the noise is what turns the code length into a real price. A noiseless continuous vector carries unbounded information and the price per bit would be meaningless. With fixed noise `sigma_0`, each dimension carries about `log(range / sigma_0)` nats, so the noise sets the resolution of the code.

Without it: an encoder that sees history makes the KL chain degenerate (the encoder does the prediction and the prior becomes trivial), and the state stops being plannable from a single image.

### 3.2 Anchor route

What: the anchor `x_0` enters the renderer only. It never reaches the encoder or the prior.

How it enters: on PushT the camera is static, so the anchor is spatially aligned with the target and channel concatenation with the noisy target (InstructPix2Pix style) is the cheapest and strongest route; each pixel of the renderer sees the corresponding anchor pixel. For video with camera motion, concatenate the anchor's tokens into the DiT sequence with their own positional ids (in-context conditioning as in FLUX Kontext and OmniGen) or cross-attend to them (ReferenceNet style). The state tokens enter through cross-attention or adaLN in either case.

The image-editing analogy is exact and worth using in the story: rendering is `x_t = edit(x_0, instruction = z_t)`. The anchor is the reference image, the state is the edit instruction. An edit instruction is low-bandwidth and never carries appearance; that is why the state ends up high-level. The closest precedent is the reference-plus-control family in video generation (AnimateAnyone, MagicAnimate): a reference frame supplies appearance through spatial attention, a low-bandwidth pose signal supplies motion, and training pairs the reference with targets from other times of the same clip. Our state is that pose signal, learned instead of given.

Why to the renderer: appearance (colors, texture, background, goal overlay) becomes free for rendering, so the rate removes it from the state.

Why not to the prior: dynamics-relevant persistent factors (block shape and size) would also become free and would never enter the state. The prior could then not roll out from the state alone, and planning would need the anchor. With a blind prior, shape enters the state at a one-time cost because it makes later contacts predictable, and that gain accrues over the whole episode.

Why the first chunk and not the previous chunk: a recent anchor lets the renderer extrapolate motion and ignore the state; a fixed first chunk gives appearance plus information about pose that decays with time. DBSE found a fixed first-frame anchor works and random anchors leak dynamics into the static code.

Two tools from the editing literature. Dropping the state with small probability during training enables classifier-free guidance on the state at inference; the guidance scale needed is a direct diagnostic of how much the renderer relies on the state. Noising the anchor (Diffusion Forcing style) reduces over-reliance on it; not needed when the anchor is meant to be copied.

Principle: side information at the decoder. The state carries only what the anchor does not explain. In the theory this is the channel for block `c`.

Without it: appearance is predictable, so it is paid once and kept forever; the state is clean of transients but not of appearance.

### 3.3 Transition prior

What: a causal transformer over past state samples and actions; per-dimension residual mean and log-variance for `z_t`. Block-causal attention: the state tokens of chunk `t` attend to each other and to all earlier chunks' tokens and actions.

What is different from SPT: in SPT the causal transformer outputs an unconstrained context vector that conditions the DiT, and the flow loss is its only training signal, so it carries whatever helps the DiT, including appearance. Here the transformer outputs a distribution over the next state, never touches the renderer, and is trained only by the code-length term. The renderer reads the encoded (posterior) state during training and the prior's sample during generation; the code-length term keeps the two close.

Why a code-length term and not just "as little information as possible": the KL to the predictive prior is `E_q[log q(z_t | x_t) - log p(z_t | past)]`, the expected description length of the state under the predictor. With Gaussians it is the prediction error in units of the prior's uncertainty plus variance terms. Two consequences. It is the JEPA latent prediction loss with learned per-dimension variance. And it prices surprise, not size: a static dimension is paid once at entry and free afterwards, a changing dimension is paid every step. A sparsity penalty on active dimensions is a cruder code: it charges a color dimension and a pose dimension the same and cannot reward predictability. The KL is the sparsity constraint in the right coordinates, sparse in surprise rather than in activation.

Why diagonal and conditioned on the past: this is the TDRL/LEAP model class (conditionally independent innovations), under which temporal latents are identifiable up to permutation and per-dimension maps. A mixed representation pays KL because mixing independent innovations creates dependence the diagonal cannot express.

Why a residual mean: constant dimensions cost zero, so static dynamics-relevant factors ride along for free after entry (the SlowVAE intuition).

Why actions enter here only: actions are interventions on the state mechanism (CITRIS, BISCUIT). If the renderer saw actions it could render motion without the state.

Why one step only: the prior predicts `z_t` from `z_<t`; the renderer renders `x_t` from `z_t`. Multi-step targets (SPT's random offset, PlaNet's latent overshooting) were removed because they do not serve the separation argument; Dreamer v2/v3 also do without overshooting.

### 3.4 Flow-matching renderer

What: a DiT over the noisy target chunk (pixels or frozen codec), conditioned on the state and the anchor, regressing the velocity, as in SPT.

What randomness it models: not alternative futures. Given the state of the current chunk and the anchor, the only randomness left in `x_t` is the within-frame residual (transient detail and whatever appearance the anchor does not pin down). Alternative futures live in the prior over states. So the renderer's noise is matched to something observed at every frame, and the "we only see one future" worry does not apply to it.

Why one observed outcome per context is enough: a conditional flow model is trained like a classifier, one label per input. The regression target `eps - y` averages over all training frames with similar conditions, and the learned field transports the noise distribution to the conditional distribution. The noise drawn at training time is the regression input (where on the path), not a guess about another outcome.

Why stochastic: bits-back. The renderer samples the residual instead of the state storing it, and there is no blur pressure on the state.

Why the target is fixed (pixels or a frozen codec): the learned state is the condition, not the target, so the collapse REPA-E reports for a learned target under a diffusion loss does not apply.

Why noise-time weighting matters: the gradient reaching the state comes from noise levels at which the noisy target has not already revealed the answer, so high-noise weighting makes the state responsible for coarse structure (Mittal et al. 2023). SPT's weighting (logit-normal with mean 0.8 plus a resolution shift, median noise level about 0.8) is already biased toward high noise; keep it.

Without it (an MSE renderer): the state is pushed to store transient detail to reduce blur, which is the MAE failure mode.

### 3.5 The objective

What: one evidence bound. Rate is the code-length chain; distortion is the flow loss.

Why the chain is both prediction loss and rate: `KL(q || p)` is the prediction error in units of the prior's variance plus variance terms. It is invariant to rescaling the latent space jointly, and the `N(0, I)` term at `t = 0` pins the scale. So the latent cannot shrink to cheat the way an attached MSE target can, and the trivial minimum (the encoder ignores the frame) is blocked because the renderer then cannot reproduce the temporal coupling in the data.

Why no stop-gradient, EMA or augmentation: Section 4.

`beta` is the price per nat. `beta = 1` is the exact bound; the sweep finds the knee of the rate-distortion curve. Ramp it from 0 so the renderer learns to use the state before the price is applied.

## 4. The story for the JEPA community

JEPA's goal is to predict in representation space and not spend representation capacity on unpredictable pixel detail. Its stated reason for avoiding reconstruction is that a pixel loss forces the latent to encode everything and is dominated by high-entropy detail.

Both complaints are about reconstruction that is unpriced and unexcused. Give the renderer alibis and charge for bits, and the reconstruction loss becomes the one thing JEPA is missing: a quantitative account of what the representation should keep.

- **Two alibis.** Appearance is excused by the anchor; unpredictable detail is excused by the noise. Whatever the renderer cannot excuse must come from the state, and every bit of state costs. The state therefore ends up with exactly the predictable, dynamics-relevant information, which is the JEPA target, reached by accounting rather than by heuristics.
- **The state is an edit instruction.** Rendering is editing the anchor according to the state. Edit instructions are low-bandwidth and carry no appearance. This is the intuitive version of the Wyner-Ziv argument.
- **Fixed points.** A JEPA objective (prediction plus anti-collapse) is satisfied by every representation that is predictable and non-degenerate. It has no term that says "you dropped something that mattered" and none that says "you kept something useless". Which information survives is decided by masking, augmentation and the regularizer. Here the optimum is the minimum-description-length predictive state (Section 5). Collapse is the degenerate end of the same continuum; the distortion term measures it, so EMA, stop-gradient and augmentation are not needed.
- **The latent predictor is still there and still does all the prediction.** The code-length chain is a Gaussian-NLL JEPA loss with learned variance, plus a fixed-prior entry term. LeJEPA's SIGReg is that entry term without the price: by the ELBO decomposition `E_x KL(q(z|x) || p(z)) = I(x; z) + KL(q(z) || p(z))`, SIGReg keeps the second term and drops the first. We add the first term and the renderer.
- **Capacity goes to the renderer, not the representation.** The state is tens of dimensions and the renderer is discarded at planning time. JEPA's "wasted capacity" objection is about the encoder's output; here the output cannot hold pixels. The cost is training compute, as in SPT.
- **Pixel detail and gradients.** In flow matching, fine detail lives at low noise levels, where the noisy target already contains it; the state's gradient comes from high-noise levels, which is coarse structure. With a frozen codec target (as in SPT) texture is already gone.
- **The field is moving this way.** V-JEPA 2 trains a diffusion decoder for visualization; VANE adds a light decoder and reports state of the art among unified encoders; FlowWM shows deterministic latent predictors blur multimodal futures. All three add back a stochastic or auxiliary renderer.

One-line pitch: we do not reconstruct pixels, we price them.

Anticipated objections:

- *"VAEs with a KL give weak, blurry latents."* The latent is small and never asked to carry pixels; blur is impossible because the renderer samples. The information loss of beta-VAEs at large beta is the intended behavior here (dropping low-value bits), tuned by action following rather than by reconstruction quality.
- *"Posterior collapse with a powerful decoder."* Real. An anchor-conditioned renderer can produce plausible frames without the state. Mitigations: beta warm-up, high-noise weighting, and the hard-budget discrete variant (Section 6). The renderer cannot reproduce temporal coupling without the state, so collapse is a local optimum, not the global one.
- *"Why not a latent target?"* A learned latent target trained through a diffusion loss collapses (REPA-E). A frozen semantic target (VANE) works but imports that encoder's biases. Here the target is fixed data and the latent is the condition.

## 5. Theory: three blocks, three channels

### 5.1 Data-generating process

Per episode: appearance `c ~ p(c)`; initial state `z_0 ~ p(z_0)` independent of `c`; transition `z_{t+1} = f(z_t, a_t, eta_t)` with exogenous noise `eta_t` and `f` not depending on `c`; actions `a_t` from any policy (we condition on them); rendering `x_t = g(z_t, c, eps_t)` with `eps_t` i.i.d. over time and independent of everything else.

Graph: `c -> x_t` for every `t`; `z_t -> x_t`; `(z_t, a_t) -> z_{t+1}`; `eps_t -> x_t`; no edge from `c` into `z`.

Channels in the architecture: the state is priced and is the only path to the prior and across time; the anchor is free and reaches only the renderer; the noise is free, reaches only the renderer, and is not predictable.

### 5.2 Assumptions for block identifiability of `z`

- **(A1) Injective smooth mixing.** `g` is injective in `(z, c, eps)` jointly. Standard in iVAE, TDRL, von Kugelgen et al. 2021 and Kong et al. 2022.
- **(A2) Block structure.** `c` is independent of `(z_0, eta, eps)`; `eps_t` is i.i.d. over time; `f` does not read `c`. This defines the partition: `c` is what persists and never affects motion, `eps` is what never persists, `z` is the rest.
- **(A3) Coupling.** Every non-constant function of `z_t` is dependent on the other frames given the anchor: `I(phi(z_t); x_{!=t} | x_0, a) > 0`. Equivalently, `z_t` is the minimal sufficient statistic of `x_t` for the rest of the sequence given `x_0`. This is the analogue of sufficient variability: the dynamics must actually use every state direction. A direction that is independent across time is `eps` by definition.
- **(A4) Capacity and optimization.** Encoder, prior and renderer are expressive enough and a global optimum is reached, at `beta = 1` with the minimum-rate tie-break or at any `beta` slightly above 1.

**Claim (informal).** Under (A1) to (A4), the learned state satisfies `z_hat_t = h(z_t)` for an injective `h` that does not depend on `c` or `eps_t`. This is block identifiability of the high-level partition.

### 5.3 Proof idea

1. **Matching.** By Gibbs' inequality the objective is bounded below by the data's conditional entropy `H(x_{1:T} | x_0, a)`, with equality only if the model's conditional distribution of the frames matches the data's and the bound is tight (`q` equals the model posterior).
2. **Coupling must pass through the state.** In the model, `x_t` is independent of the other frames given `(z_hat_t, x_0)`, because the renderer draws each frame separately. Tightness transfers this conditional independence to the data joint with `z_hat_t ~ q(. | x_t)`. By (A3) nothing that couples `x_t` to the other frames may be lost, so `(z_hat_t, x_0)` determines `z_t`. Since the encoder reads only `x_t`, this means `z_hat_t = h(z_t, c, eps_t)` with `h` injective in `z_t` for each fixed `(c, eps_t)`. This is the shared-block argument of multi-view identifiability: the views are `x_t` and the rest of the sequence, the shared block is `z_t`.
3. **Rate minimality strips the rest.** A dependence on `eps_t` costs rate every step and buys at most the same amount of distortion (bits-back), so its net value is at most zero. A dependence on `c` costs entry rate (`z_0` is independent of `c`, so any `c`-dependence raises `E||z_hat_0||^2` or the surprise terms) and buys nothing, because the renderer reads `c` from `x_0` and `f` ignores `c`; its net value is negative. A dependence on `z_t` has positive net value by (A3): one bit of innovation explains structure in other frames as well as the current one. So the minimum-rate optimum, or any `beta` slightly above 1, keeps exactly `h(z_t)`. This is the minimal-change argument (Kong et al. 2022) in information-theoretic form, and the rate-distortion reading of the ELBO (Alemi et al. 2018) explains why `beta = 1` alone leaves `eps` undetermined.

Residual ambiguity: an invertible reparameterization of `z`. Symmetries of the dynamics (a translation offset when the dynamics are translation-invariant) would also be ambiguities if the offset could depend on `c`; the entry term rules that out when `z_0` is independent of `c`.

### 5.4 Component-wise identifiability inside the state (bonus)

Add:

- **(B1) Conditionally independent innovations.** The components of `eta_t` are mutually independent given `(z_t, a_t)`.
- **(B2) Sufficient variability.** The transition's conditional log-density varies enough with the past state and the action, in the linear-independence sense of TDRL/LEAP. Actions provide extra variability, the role played by the auxiliary variable in iVAE and by interventions in CITRIS and BISCUIT.

Then `h` reduces to a permutation composed with per-dimension maps. Proof idea from TDRL: equate the latent log-densities under `h`, differentiate with respect to current components and the past to remove the Jacobian term, use independence to zero the cross terms, and use variability to force the Jacobian of `h` to be a generalized permutation. The architectural counterpart is the diagonal conditional prior with per-dimension variances and actions entering the prior. On PushT this is measured by MCC against the simulator's factors; it is a bonus, not something the method depends on.

### 5.5 Literature map

| Ingredient | Source | Architectural counterpart |
| --- | --- | --- |
| Invertible mixing, exponential-family or nonparametric conditional prior | iVAE (Khemakhem et al. 2020), TDRL (Yao et al. 2022), LEAP (Yao et al. 2022) | diagonal conditional prior with learned variances |
| Shared block across views is identifiable | von Kugelgen et al. 2021; Lyu et al. 2022; Yao et al. 2024 (partial observability) | step 2: the state is the block shared between a frame and the rest of the sequence |
| Invariant block versus minimally changing block | Kong et al. 2022 (partial disentanglement) | step 3: appearance is excluded because it is free elsewhere and costs rate |
| Actions and interventions as the source of variability | CITRIS, BISCUIT (Lippe et al.), mechanism sparsity (Lachapelle et al.) | actions enter the prior only |
| ELBO as rate-distortion; indifference at `beta = 1` | Alemi et al. 2018 | minimum-rate tie-break or `beta` slightly above 1 |
| Side information at the decoder | Wyner and Ziv 1976 | anchor route |
| Level selection by masking in hierarchical latents | Kong et al. 2023 (MAE) | chunk length selects the level of dynamics (slow versus fast) |

### 5.6 What the theory does not cover

Finite capacity and optimization (posterior collapse), non-injective rendering, non-static context such as camera motion, and `beta` well above 1 (a rate-distortion point drops low-value dynamics bits). Predictable but control-irrelevant dynamics (a background clock) belong to `z` under this definition; removing them needs an action-based criterion (multi-step inverse dynamics, Delta-JEPA), which is a separate step.

## 6. Practical forms of the rate term, and why masking is not one

Three codes, one objective (distortion plus `beta` times code length):

| Code | Posterior | Prior and rate | Entry term | Pros | Cons |
| --- | --- | --- | --- | --- | --- |
| Gaussian, fixed variances (first choice) | `N(mu, sigma_0^2)`, `sigma_0` fixed | Gaussian with fixed variance `s_0`; rate = `||mu_t - m_t||^2 / (2 s_0^2)` | `||mu_0||^2 / 2` | three lines: noise, MSE, weight decay on the first state; no learned variances | unimodal prior; `beta` to sweep |
| Gaussian, learned variances | `N(mu, sigma^2)` | diagonal Gaussian; closed-form KL | KL to `N(0, I)` | exact bound; calibrated uncertainty for planning | more to tune; posterior collapse |
| Noisy continuous plus flow prior | `N(mu, sigma_0^2)` | conditional flow-matching model in latent space; rate = latent flow loss, an ELBO-type bound (Kingma and Gao 2023) | `||mu_0||^2 / 2` | multimodal futures; same loss form as the renderer | a bound, not exact; two flow models |
| FSQ discrete | FSQ codes (levels times tokens) | categorical autoregressive; rate = cross-entropy, exact bits | hard budget, no `beta` | exact description length; no collapse onto the prior; Genie and Cosmos style | discrete state; sampling-based planners only (iCEM is fine) |

Recommendation: fixed-variance Gaussian first on PushT, where actions make futures near-deterministic. FSQ second if `beta` is painful; its rate is exact and the budget replaces the entry term. The flow prior for action-free video, where futures are multimodal.

Masking. SPT drops 90% of each history state's patches before its per-state encoder blocks (ViT-E), so about 102 of 1024 tokens per RGB16 state reach the encoder and the causal ViT-C. This saves compute and acts as a mild regularizer, and SPT fine-tunes unmasked afterwards. It does not hide information (each retained token can encode anything in its chunk, including appearance), it gives no entry cost, and it trains no latent predictor. As an information restriction it is at most an erasure channel on the state (the Information Dropout view), an alternative noise to Gaussian, not a simpler one. In this design the small state already provides SPT's compute saving in the causal model (4 tokens per chunk instead of about 102). Because SPT's masking acts inside the encoder, reusing ViT-E as the AS-JEPA encoder requires masking off: the state must be a function of the whole chunk, with matching inputs at training and inference.

## 7. Relation to SPT

Kept: the three modules (SPT's per-state ViT-E as the encoder, its state-causal blocks as the prior, its pixel DiT as the renderer); one end-to-end training pass; a flow-matching renderer on pixels or a frozen codec; state reuse across denoising steps; streaming generation.

Changed:

1. The DiT conditions on the current state sample and the anchor, not on the causal transformer's history summary.
2. The causal transformer becomes a prior over states with its own code-length loss; it no longer feeds the DiT.
3. SPT's 90% random token masking is replaced by a small state with a rate.
4. SPT's random future offset `k` is removed; the prior is one-step and the renderer renders the current chunk.

Why: the renderer can no longer bypass the state, the state is per-chunk and plannable, and the history lives in the prior's cache. Cost: two loss terms instead of one, but they are a single evidence bound.

## 8. Risks and guards

| Failure | How it shows | Guard |
| --- | --- | --- |
| Posterior collapse: renderer copies the anchor and ignores the state | code length per dimension near zero early; samples from pure noise do not change when the state is zeroed; large guidance scale on the state needed | ramp `beta` from 0; high-noise weighting; FSQ variant |
| Appearance stays in the state | color probes on `mu` above chance; entry cost far above the state budget; colors follow the state in the swap test | raise `beta`; check the anchor route is actually used |
| A contact-only factor is squeezed out at large `beta` | action-following error worse in contact than in free motion | choose `beta` by contact action-following, not by the flow loss |
| Unimodal prior averages distinct futures | latent rollouts drift without actions although samples look fine | flow prior (Section 6) |
| Multi-step rollouts drift although one-step error is low | compounding error in latent rollouts | Dreamer v2/v3 style training without overshooting is the default; revisit only if drift appears |
| Content not static (camera motion) | generation forgets regions revealed after the anchor | start with PushT; anchor refresh is open |
| Background motion fills the state | probes find distractor state in `mu` | multi-step inverse dynamics or Delta-JEPA action decoding, if control is the goal |

Calibration: PushT's state at 64 x 64 precision is roughly 50 to 60 bits (about 40 nats). Randomized colors plus goal overlay would add a similar amount. A measured entry cost near the state budget with color probes at chance is the success picture; an entry cost near twice that is the leak picture. These are order-of-magnitude guides; the code length only upper-bounds the information.

## 9. First experiment (PushT)

Use PushT's variation space (`docs/envs/pusht.md`) to randomize two kinds of factors:

- render-only: agent, block and background colors, and the goal-overlay pose;
- dynamics-relevant: block shape and scale.

Collect with `WeakPolicy`; ground-truth factors come free. To test the noise route, overlay a small random dot on every frame, large enough that blurring it is costly; it is transient, so it should be sampled rather than stored.

Train the model across a `beta` sweep, plus two ablations: no anchor route (colors should enter the state) and a deterministic MSE renderer (the dot should enter the state or blur). Compare against LeWM and, if code is available, LRC-JEPA. Check:

- the entry cost against the roughly 40-nat budget;
- probes and MCC from `mu`: pose, shape and scale in; colors, goal pose and dot out;
- a swap test: anchor from episode A with states from episode B; generated frames should show A's colors and B's poses;
- planning with goal images in different colors (the proposed model should barely notice; LeWM likely degrades);
- action following in contact versus free motion.

Refutation: colors stay decodable at every `beta` that preserves action following, or the dot enters the state as readily with the flow renderer as with the MSE renderer.

## 10. Prior work

- **RSSM (PlaNet, Dreamer)**: the template for a KL to a learned transition prior and planning in latent space; deterministic decoders and no anchor, so the state holds appearance.
- **SVG-LP**: a learned prior, but the decoder sees the previous frame, so the latent becomes a change code.
- **DBSE**: static code from a fixed first frame, dynamic codes from a sequential prior, subtraction bias; no actions, planning or flow renderer.
- **LRC-JEPA**: keeps context out of the predictor but reconstructs the same frame deterministically, so it needs SIGReg plus a VICReg-style term. Closest JEPA baseline.
- **SODA**: a bottleneck encoder on one view and a diffusion decoder generating another view; a tight bottleneck plus a stochastic decoder learns strong, partly disentangled latents. No dynamics.
- **AnimateAnyone, MagicAnimate, InstructPix2Pix, FLUX Kontext**: reference-image conditioning mechanics (spatial attention to a reference, channel concatenation, in-context tokens) and the reference-plus-control decomposition.
- **D-JEPA**: adds a diffusion loss to a JEPA for generation; reconstructs the same image's masked tokens rather than a rendered frame with side information.
- **Flow-JEPA, FlowWM**: stochastic latent dynamics with flow matching; the model for the flow-prior variant.
- **Genie**: VQ latent actions with a decoder that sees past frames; the discrete analogue of the state and anchor.
- **TDRL, LEAP, CITRIS, BISCUIT, iVAE**: the identifiability lineage for the factorized conditional prior and action-as-intervention.
- **von Kugelgen et al. 2021; Lyu et al. 2022; Yao et al. 2024; Kong et al. 2022**: block identifiability of shared versus changing latents; Section 5 is of this kind, with time and side information replacing augmentations and domains.
- **Alemi et al. 2018**: the ELBO as a rate-distortion objective.
- **REPA-E**: why the renderer's target must stay fixed.
- **SPT**: the three modules and the end-to-end flow-matching recipe.

The pieces exist. Not found: SPT's flow renderer with anchor side information, a code-length chain with no collapse tricks, latent planning, and an identifiability evaluation in one model. Its value has to come from experiments.

## 11. Decisions and open questions

Decisions: no augmentation; no EMA or stop-gradient; anchor is the first chunk and is never rendered; anchor enters by channel concatenation on PushT; encoder is per-chunk; prior is anchor-blind and one-step; renderer is history-blind and action-blind; masking is not the information restriction; fixed-variance Gaussian code first. On SPT (2026-10-25): no target network, and SPT's evaluation-only weight EMA is also off; the state reaches the renderer without per-token LayerNorm; first chunks are 4-frame RGB4 states; `beta` is swept, not fixed at 1.

Open: anchor refresh for videos with camera motion; an action-relevance term for control; one state token versus several; chunk length for the slow/fast hierarchy; whether the entry term and the surprise terms should have different weights.
