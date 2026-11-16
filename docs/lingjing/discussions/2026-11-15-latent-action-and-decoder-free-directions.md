# Latent action, anchor gap and decoder-free directions

Written 2026-11-15. Starting point: the objection that any pixel reconstruction
loss forces the model to encode unforeseen future information.
This note records the evidence read from the collaborator's branch
`origin/as-jepa/lingjing-fullspec-20261102`, the literature search, and the
proposed split into three tracks. Nothing here is implemented or launched.

## 1. Evidence from our own experiments

### 1.1 The Meta-World "future frame" result

Source: `experiments/metaworld_kernel_20261106/THEORY.md` and commit `b266a09b`.

The model that scored well is an actor `q(a_t | o_t, o_{t+k})`. It reads the
current image and a later image of the same expert episode (a hindsight goal)
and is trained on the recorded action `a_t`. The future image enters an action
predictor, not the latent state predictor. The collaborator withdrew it because
it supervises the next action to execute, which broke the rule "no action
targets", not because the predictor saw the future.

| Quantity | Value |
| --- | --- |
| Strict single-action actor with own world-model feedback, 50 tasks, 1,000 cases x 2 learners | 86.80% |
| Same actor, no world-model feedback (five samples on the unchanged image) | 71.05% |
| Six manipulation tasks, with proper future goal image | 85.83% |
| Six manipulation tasks, goal slot filled with the current image (no future information) | 71.67% |
| World model plus CEM on latent distance, 50 tasks (LeWM, SMWM, AC-MTM) | 13.4% to 14.2% |
| LeWM plus inverse-dynamics auxiliary, 50 tasks | 9.17% |

How inference worked: the benchmark supplies a goal image, which takes the place
of the future frame. The camera refreshes every five native steps; between
refreshes the actor reads images predicted by its own forward model.

Reading: most of the score is goal-conditioned behaviour cloning on expert data
(71.67% without any future image). The future image adds about 14 points and the
forward model's intermediate predictions add about 15 points. It is evidence
that an amortised, action-space way of choosing actions beats CEM on a terminal
latent distance. It is not evidence that the representation or the predictor
improved.

### 1.2 Inverse dynamics as a representation loss

- `metaworld_input_inverse_20261107`: 32 models, inverse reconstruction of the
  executed action from the latent pair as an auxiliary, 5,000 updates. One
  manipulation success in 768 episodes. The auxiliary's raw encoder gradient was
  about 400 to 1,400 times the main loss's at the tested weights.
- `metaworld_causal_inverse_20261109`: 0/12 on the six hard tasks in every arm.
- A five-step response window improves action identifiability (inverse MSE 0.523
  to 0.353, zero-action baseline 0.455) without any control gain.
- 32.94% of action components sit exactly at a box boundary in the Meta-World
  corpus; in RoboCasa about 45.8% of large z commands move the arm under 2 mm.
  The commanded action and the realised effect differ.

### 1.3 Our decoder-free variant

Source: `experiments/lingjing_fullspec_20261102/CLOSEOUT_20261103.md`,
`experiments/branch_trials_20261102/RESULTS.md`.

62 PushT recipes of the coding-rate decoder-free objective, none qualified.
Window estimator at `lam = 0`: 1.91 / 1.65 nats of estimated information, noisy
agent probe R^2 0.011 / 0.006, planning 4% / 2%. The information reward was
earned by content unrelated to the task state. The reward is also sensitive to
numerical precision (5.756 nats under BF16 against 0.0238 under FP32 on the same
windows). Native LeWM (SIGReg) reaches 90.89% / 87.33% on PushT at 10 epochs.

### 1.4 Anchor placement

Round-2 TwoRoom cells (original or calibrated scale, near or past anchor) scored
3% to 8%. Anchor placement did not move planning. The collaborator's default on
TwoRoom, Cube and RoboCasa is the initial frame, so the anchor-to-target gap
equals the elapsed time in the episode. PushT uses a frame at least 8 kept
frames away.

## 2. Theory points

**Which entropy may be raised.** A latent prediction loss lowers
`h(s_t | past, actions)`. Something else must raise an entropy of `s_t`. That
objection applies to any term that raises `h(s_t | past)`, because that
rewards content the past cannot predict. Rendering `x_t` from `s_t = enc(x_t)`
does so. So does a volume-controlled innovation match (below). A term is safe
when it raises entropy conditional only on what the state should go beyond
(the anchor or the episode), or when it rewards predictable information directly
(`I(forecast; s_t)`).

**Innovation matching is not an anti-collapse term.** Let
`e_t = g(s_t; history, a)` be matched to `N(0, I)`.
- If `g` is unconstrained, a state with a tiny spread satisfies the match,
  because `g` can amplify any residual variation or the code noise. With
  `s_t = mu + sigma0 xi` and `mu` constant, `e_t = xi` matches exactly.
- If `g` is invertible in `s_t` with bounded volume change, the match gives
  `h(s_t | history, a) >= h(e) - d log L`. That forces unpredictable content of
  full dimension into the state, which is the same objection again.
- The maximum-likelihood form (`0.5 |e|^2 - log det`) is the state's code length.
  The encoder lowers it by contracting. It is our Surprise term generalised, a
  rate, not a reward.
- In the current AS-JEPA, `e_t = (s_t - f(history, a)) / sigma0` and
  `Surprise = 0.5 |e_t|^2`. The innovation already exists in affine form.
- TDRL, LEAP and iVAE obtain identifiability from independent noise conditions,
  but they need the mixing to be inverted, and reconstruction is what enforces
  that. The decoder-free identifiability results are contrastive (TCL, PCL,
  CPC-style density ratios).

**Where contrastive learning sits.** It is implemented on pairs of instances and
it constrains a distribution-level quantity. In `InfoNCE = alignment +
log-sum-exp over negatives`, the second term is a kernel estimate of the
marginal entropy (Wang and Isola 2020). Garrido et al. 2022 show sample-contrastive
and dimension-contrastive criteria are algebraically related. Equivalently it is
grounding to the instance index. A cleaner split of anti-collapse instruments:
(1) grounding to an external per-sample target `Y` (pixels, frozen features,
actions, reward, flow, an EMA teacher), where `Y` decides what is kept;
(2) holding up an entropy of the state itself (VICReg, Barlow Twins, whitening,
MCR2, SIGReg, contrastive uniformity), where the paired prediction term decides
what is kept; (3) asymmetric optimisation (stop-gradient, EMA), excluded here.

## 3. Literature

Collected by two search agents on 2026-11-15. They report checking the arXiv ids
against abstract pages; entries marked [U] were not opened. I have not re-read
the 2026 papers myself. Recheck before citing in a paper.

Latent action with latent prediction:
- Garrido et al. 2026, "Learning Latent Action World Models in the Wild",
  2601.05230. Inverse model on frozen V-JEPA 2 features `(s_t, s_{t+1})`,
  continuous 128-d `z` with sparsity, noise/KL or VQ. Inference: a small
  controller maps real actions and past features to `z`, then CEM. They list
  end-to-end training as future work. Closest prior work.
- DynaMo 2409.12192. Encoder from scratch with latent inverse and forward
  models, low-dimensional `z`, but stop-gradient or EMA; `z` is discarded.
- LAOM (Nikulin et al.) 2502.00379. Inverse model sees `(o_t, o_{t+k})`, `k` up
  to 10; a few percent of action labels gives a large gain; without supervision
  a tight bottleneck encodes distractors.
- Sensorimotor World Models 2606.20104, AC-MTM 2608.17542, Delta-JEPA 2606.31232,
  PLDM 2502.14819: end-to-end JEPA with inverse dynamics toward the real action
  only.
- "Why Latent Actions Fail" 2605.20223 (reconstruction makes `z` absorb exogenous
  future information); "What Matters for Latent Actions" 2608.19613 (41 design
  choices; regularisation strength matters more than type; about 32 dimensions).
- LAPO 2312.10812, Genie 2402.15391, LAPA 2410.11758, AdaWorld 2503.18938 [U]:
  VQ or beta-VAE latent action with pixel targets; inference by a latent policy
  plus a small action decoder, or user-chosen codes.
- Dreamer RSSM and SVG-style learned priors [U]: a latent inferred from the
  future with a KL to a prior, on top of real actions.

Decoder-free anti-collapse:
- LeJEPA/SIGReg 2511.08544; LeWorldModel 2603.19312 (its own limits: loses on
  Two-Room, misses end-effector angle on Cube).
- VISReg 2606.02572 (SIGReg gradient vanishes near collapse; scale and shape
  coupled). UR-JEPA 2606.01443 (isotropic target against the manifold
  hypothesis). Pandya, NeurIPS 2026 (SIGReg as a score is scale-sensitive).
- MotionJEPA 2609.23881: predicts embeddings of temporal difference images;
  standard JEPA training favours slow features.
- MC-JEPA 2307.12698: optical flow plus VICReg in one encoder; the flow loss
  likely includes photometric warping [U].
- ATM 2401.00025, Track2Act 2405.01527, Im2Flow2Act 2407.15208: tracks from
  pretrained trackers [U]. FlowVLA 2508.18269: generative.
- VJEPA 2601.14354: variational JEPA with a flow predictive distribution; its
  collapse guarantee is unverified.
- Innovations Autoencoder 2106.12382: matches an i.i.d. innovation sequence but
  keeps a decoder. Closest precedent to innovation matching.
- CPC 1807.03748; C-SWM 1911.12247; Dreaming 2007.14535; TPC 2106.07156.
- REPA-E 2504.10483 and SimFlow 2512.04084: end-to-end latent likelihood
  contracts the latent unless the scale is pinned.
- Inverse dynamics as grounding: AC-State 2207.08229, Efroni et al. 2110.08847,
  ACRO 2211.00164; failures in Levine et al. 2403.11940 and Rakelly et al.
  2106.07278.
- Taxonomy: Garrido et al. 2206.02574; Wang and Isola 2005.10242; Balestriero
  and LeCun 2205.11508; Shwartz-Ziv et al. 2303.00633.

No paper was found that trains an encoder with next-latent prediction plus
innovation matching alone, and none that learns a future-inferred latent as a
residual on labelled actions in an end-to-end JEPA without EMA.

## 4. Proposed tracks

**Track A, renderer line: anchor gap.** The predictor never sees the anchor, so
the gap does not concern it. Do not condition the renderer on the gap by
default: with an initial-frame anchor the gap is the episode's time index, which
in scripted expert data predicts the arm pose and gives the renderer a route
around the state. First step is free: bucket render loss, rate and probe R^2 by
gap on existing checkpoints. Run a fixed-gap arm and a gap-conditioned arm only
if the buckets show a dependence.

**Track B, innovation variable (latent action).** One inference network
`q(e_t | history, a_t, s_{t+h})`, priced by a KL to a learned prior
`p(e_t | history, a_t)` in the same nats as the state, consumed by the
predictor. With action labels `e` is the residual the action does not
determine; with labels masked it also carries the action. Inference: sample or
take the mean of the prior; for action-free data, a controller from real actions
to `e` (Garrido et al.) or a latent policy with a small decoder. First step is
offline on frozen checkpoints: fit `q` and the prior, report forecast-error
reduction per nat, whether `e` predicts contact events, and the scene-cut test.

**Track C, decoder-free.** Candidates in order: (C1) predictive InfoNCE with the
Gaussian prior as critic and negatives from the same episode; (C2) a marginal
regulariser with scale and shape separated (VISReg-style) as the baseline to
beat alongside SIGReg; (C3) motion targets as an auxiliary, in the
difference-image-embedding form or classical flow statistics predicted from the
forecast. First step: the existing toy (`.tmp/latent-toy`), then PushT against
native LeWM at 90.89% / 87.33%.

Open across all tracks: every objective tried so far favours the arm (which the
actions determine) over the object. None of the three tracks addresses that by
construction; each should report object-pose probes and hard-task success.

## 5. Update 2026-11-16: EBReg and the score-based form of Track C

The user pointed to Kappiyath et al., "Self-Supervised Collapse Prevention with
Score-Based Entropy Maximization" (EBReg), arXiv 2611.11379, with LeCun and Bar
as co-authors. Read from the arXiv HTML through an automated summary; the last
2% of the page was not read.

- Objective: `L_align + gamma * E[ sg(score(h~))^T h~ ]`, with `h~ = h + sigma eps`.
  It maximises the entropy of the noise-smoothed marginal of the embeddings.
- The score comes from a small MLP trained by denoising score matching on
  detached embeddings: predict `eps` from `h + sigma eps`; `score = -eps_hat / sigma`.
  Default 16 score updates per encoder step.
- Scale is fixed by normalisation; under that constraint the entropy maximiser
  is the isotropic Gaussian, the same target as SIGReg.
- Experiments: ImageNet only, with augmented views. ViT-L linear probe 75.57%
  against SIGReg 75.32%. No video or world-model experiment. The theory covers
  gradient directions under an exact score, not full training dynamics.

Mapping to AS-JEPA. Our noisy state `s_t = mu(x_t) + sigma0 xi` is already the
smoothed embedding, and the decoder-free objective needs
`I_A = h(s_t | anchor) - const`. So the renderer can be replaced by a small
denoiser of the state conditioned on the anchor:

```
score model:  min_phi  | eps_phi(sg(mu_t) + sigma0 xi, anchor) - xi |^2
encoder:      beta * Entry + w * Surprise + E[ sg(-eps_phi(s_t, anchor) / sigma0)^T s_t ]
```

This is the working form of the user's "lightweight network whose output is the
noise": the noise is the injected code noise, the conditioning is the anchor,
and the encoder is trained to make that noise hard to guess. The form objected
to in Section 2 (innovation given the past, matched to a fixed Gaussian) stays
rejected. Differences from EBReg: conditional on the anchor, no augmentation,
alignment replaced by the action-conditioned prediction loss, scale set by the
priced rate instead of normalisation. All of these are untested.

Open question the first experiment must settle: the coding-rate decoder-free
runs failed either because the Gaussian estimator was gamed or because the
objective itself does not favour the task state. Run the score-based estimator
on the same PushT recipe and read the agent probe.

Track C candidates after this update: score-based conditional entropy first;
InfoNCE with same-episode negatives as the control that estimates the same
quantity with a `log K` cap; native LeWM (SIGReg) as the baseline. The
scale/shape-separated regulariser is dropped as a direction. Track B does not
prevent collapse and is parked until a state objective is settled. Track A:
checkpoints live on the collaborator's EFS, not on this machine, so the
gap-bucket diagnostic is a handoff.

## 6. Update 2026-11-16 (later): direction of the game, toy check, contrastive as a special case

### 6.1 Who minimises what

The user's reading of the denoiser is right: it reads the noisy state and the
anchor, and it recovers the injected noise perfectly exactly when the anchor
already determines the clean state. That point is the collapsed one, and the
encoder is trained to move away from it. There are two losses and two optimisers.

- Denoiser parameters minimise `|xi_hat(s_t, anchor) - xi|^2` on detached states.
  This keeps it an honest measurement of what the anchor explains.
- Encoder parameters receive no gradient from that loss. They descend
  `sg(score) . mu_t` with `score = -xi_hat / sigma0`, which is the gradient of
  `-h(s_t | anchor)`. In a Gaussian case the update is
  `mu_t += lr * (s_t - m_A) / (a^2 + sigma0^2)`: each state moves away from the
  denoiser's guess, and the push weakens as the spread `a` grows.

The denoiser must read the noisy state, not the anchor alone. An anchor-only
predictor trained by MSE, with the encoder maximising that MSE, gives a reward
quadratic in the scale; it cannot balance a quadratic rate. The score of the
noisy state gives the entropy, which grows like the log of the spread.

What the denoiser reads of the anchor (corrected 2026-11-16; the first version
of this paragraph gave a wrong reason). The encoder is shared, so any function
of appearance that leaks into `s_t` also sits in the anchor's own state, and a
denoiser reading the anchor's detached state can already predict it. The
anchor's state is therefore enough to deny reward to content that is constant
within an episode. Reading the anchor image as well helps only with codes that
mix appearance with the changing state (for example a position whose sign
depends on colour), which the anchor's state need not reveal. The principle says
the reader holds the anchor frame, so the canonical reader gets the anchor's
detached state plus its own small network on the anchor image; `reader_anchor =
"state"` is the cheap comparison.

Stop-gradient. The score is detached in the encoder's term. This is not the
JEPA target trick: prediction targets stay attached. It is how a two-player
objective is written, each player taking the gradient of its own loss; at the
denoiser's optimum it equals the gradient of the conditional entropy (envelope
argument). The decoder-free README's "no stop-gradient" sentence needs this
qualification when the denoiser estimator is used.

Approved 2026-11-16: one objective, three estimators (K same-episode frames,
Gaussian log-det, learned denoiser) behind one switch. Reference class
`AnchorReaderLeWM(DecoderFreeLeWM)` with `estimator in {"denoiser", "nce"}` is
being written in the worktree `.tmp/wt-decoder-free` under
`docs/lingjing/decoder-free/`. Training fits the collaborator's single-optimizer
trainer: one backward pass with gradient routing (the denoising loss on detached
states, the entropy term with a detached score) and a separate parameter group
for the reader through `training_optimizer_config`.

Two-reader statement of the objective: the state should be cheap to describe for
a reader who holds the past states and actions (the predictor, which the encoder
helps) and expensive for a reader who holds only the anchor (the denoiser, which
the encoder works against). `J = beta Entry + w sum_t CE_predictor(s_t) - sum_t CE_anchor(s_t)`.
This is a difference of two cross-entropies (McAllester and Stratos 2020, the
difference-of-entropies estimator): it has no `log K` cap and it is not a bound.
The reward side is an upper bound on the entropy when the denoiser is imperfect,
so a lagging denoiser over-credits. That is the main risk.

### 6.2 Toy check

Script `.tmp/latent-toy/reader.py` (world, linear encoder, prior and probes of
`toy.py`); SLURM job 3071246, partition `cpuonly`, 16 CPUs, 6 min, exit 0;
results in `.tmp/latent-toy/reader_results.json`. 18 runs, 6,000 steps each,
4 denoiser steps per encoder step unless noted. Probe R^2 from the noisy state.
`w = beta + lam` is the total weight on the prediction loss.

| Run | agent | block | K | appearance | transient |
| --- | --- | --- | --- | --- | --- |
| denoiser with anchor, w = 0.3, seeds 0 / 1 | 1.00 / 1.00 | 1.00 / 1.00 | 0.99 / 0.99 | 0.00 / 0.00 | 0.69 / 0.69 |
| same, start at 0.001 sigma0, seeds 0 / 1 | 1.00 / 1.00 | 1.00 / 1.00 | 0.99 / 0.99 | 0.00 / 0.00 | 0.69 / 0.69 |
| denoiser with anchor, beta 1 lam 0 | 1.00 / 1.00 | 0.72 / 1.00 | 0.87 / 0.99 | 0.00 / 0.00 | 0.00 / 0.00 |
| denoiser with anchor, beta 0.3 lam 1 | 1.00 / 1.00 | 0.61 / 1.00 | 0.87 / 0.99 | 0.00 / 0.00 | 0.00 / 0.00 |
| same, start at 0.001 sigma0 | 1.00 / 1.00 | 1.00 / 1.00 | 0.99 / 0.99 | 0.00 / 0.00 | 0.00 / 0.00 |
| denoiser with anchor, beta 1 lam 1 | 1.00 / 1.00 | 0.57 / 1.00 | 0.88 / 0.99 | 0.00 / 0.00 | 0.00 / 0.00 |
| denoiser without anchor, beta 1 lam 1 | 1.00 / 1.00 | 0.58 / 0.55 | 0.89 / 0.88 | 1.00 / 1.00 | 0.00 / 0.00 |
| denoiser without anchor, beta 0.3 lam 0, seed 0 | 1.00 | 1.00 | 0.98 | 1.00 | 0.00 |
| earlier: same-episode InfoNCE, beta 1 lam 1 | 0.99 / 0.99 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| earlier: pooled Gaussian estimator, beta 0.3 lam 0 | 1.00 / 1.00 | 1.00 / 1.00 | 0.99 / 0.99 | 0.00 / 0.00 | 0.70 / 0.70 |

- From a start at 0.001 sigma0 the spread reached 1.5 to 3.2 sigma0 within 50
  steps in all 5 such runs. The encoder is not being collapsed.
- Appearance R^2 is at most 0.002 in all 15 runs where the denoiser reads the
  anchor and at least 0.996 in all 3 where it does not. The anchor conditioning
  is what excludes appearance; the unconditional form is EBReg without
  normalisation.
- The transient is dropped in all 8 runs with `w >= 1` and kept at 0.69 at
  `w = 0.3`, the predicted `1 - w`.
- Block retention at `w >= 1` is partial on seed 0 (0.57 to 0.72) and complete
  on seed 1, as with the earlier estimators. Two seeds only.
- Removing the control variate, or using 1 denoiser step, changed nothing here.

Limits: the toy has a linear encoder and Gaussian factors, where the Gaussian
estimator is exact. It shows that the alternating training runs in the intended
direction and reproduces the closed-form estimator. It cannot show an advantage
over the coding-rate version, and it does not explain the PushT failure.

### 6.3 Contrastive as a special case

With the fixed-noise code, the density of the noisy state given the anchor is
exactly a mixture of Gaussians centred on the encoder means of frames the anchor
allows. Averaging over `K` such frames gives InfoNCE with the code's own
likelihood as critic: no learned critic, temperature `sigma0^2`, nats comparable
with the rate, and with a forecast as the query the positive term is the
prediction loss. Negatives must therefore be frames of the same episode.

- Precedent: CRTR (Ziarko et al., NeurIPS 2025, 2508.13113) uses negatives from
  the same trajectory and reads the loss as a bound on `I(X; X+ | context)`.
  The negative rule is not ours to claim.
- Evidence: in the toy the same-episode InfoNCE keeps the agent and drops the
  block (R^2 0.00 on both seeds at beta = 1). With 8 candidates the reward caps
  at 2.1 nats, which the agent alone supplies; after that the block has no
  gradient.
- The principled removal of the cap is telescoping over intermediate
  distributions (Rhodes et al. 2020, 2006.12204), whose continuous limit is
  score matching (Choi et al. 2022, 2111.11010). Noise levels are the natural
  intermediate distributions here, so the scalable contrast is the denoiser.

Proposal: one objective, three estimators of the same reward (K frames, one
Gaussian, learned denoiser), behind one switch in one class.

### 6.4 Sources and overlap (search summaries, 2026-11-16; not re-read in full)

- Taken from: EBReg 2611.11379 (training loop); AR-DAE, Lim et al. ICML 2020,
  2006.05164 (entropy gradient through a denoiser for implicit distributions);
  MINDE, Franzese et al. ICLR 2024, 2310.09031, and the I-MMSE relation
  (information from denoising error; a denoiser trained over a range of noise
  levels can report `I_A` in nats as `integral of E|xi - xi_hat|^2 d sigma / sigma`
  from `sigma0` upward; the Gaussian case checks to `0.5 log(1 + a^2 / sigma0^2)`).
- Neighbours: JEDI 2605.13013 (encoder trained through a latent denoising loss
  for the next state, with stop-gradient on the target and latent clamping: the
  encoder helps the denoiser, the opposite sign); SplitJEPA 2610.12349
  (invariant and variant blocks without reconstruction, matched pairs plus
  variance/covariance regularisation); DSReg 2610.09457 (post-hoc rotation of a
  LeJEPA checkpoint); CRTR 2508.13113.
- Not found in about 12 searches: an entropy conditional on a reference frame as
  the anti-collapse term of a latent world model, paired with a priced
  prediction loss. EBReg is a few days old; a video version from its authors is
  plausible.

Track A: the diagnostic tool (`round2/as_jepa_tools/gap_buckets.py` with tests)
was restarted after an interrupted first attempt; first targets are the PushT
and Cube temporal-forecast checkpoints.

### 6.5 Corrections and state at handoff (2026-11-16, evening)

- **Anchor gap, corrected.** Section 4 said that with an initial-frame anchor the
  gap is the episode's time index. That was wrong. On Cube, TwoRoom and RoboCasa
  the default anchor is the window's own first frame (`render[0]` in
  `as_jepa_tools/benchmark_data.py`; 8-frame windows of 17-frame episodes in the
  Reach3D loader), so the gap is the frame's index in the window, 1 to 7. Only
  PushT has large random gaps (a frame at least 8 kept frames away, either
  side). The time-index shortcut argument therefore does not apply as stated;
  the handoff's reason for not conditioning the renderer on the gap is that a
  first-frame gap equals the forecast horizon the renderer can already infer.
- **Collaborator's latest experiments** (their head `3131efaa`): RoboCasa
  Reach3D with `AnalyticResidualSPT`, frozen action heads on 17,920 / 1,536
  current-action labels, 64 DEV cases, hourly evaluation; the encoder-heavy
  shared-mean reference reaches 81.83% / 42.23 mm at 1 h and 91.35% / 17.78 mm
  at 6 h, with strict 2 mm success still rare. They are now also testing EMA
  targets of their own encoder. Both trainers clip the global gradient norm at 1.
- **Handoff.** `docs/lingjing/decoder-free/handoff-anchor-reader.md` on branch
  `lingjing/decoder-free-as-jepa`: Part G (gap diagnostic on PushT, Cube and
  the current Reach3D checkpoints), Part P (denoiser objective on PushT:
  w = 1.3, 0.3, 1; reader without its own view of the anchor image; contrastive
  control), Part R (the same objective in the Reach3D protocol, gated on Part P).
