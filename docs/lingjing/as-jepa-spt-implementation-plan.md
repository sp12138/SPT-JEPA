# AS-JEPA implementation plan: SPT, LeWM and editing pieces

**Superseded for execution (2026-10-26)** by [as-jepa-experiment-spec.md](as-jepa-experiment-spec.md), the handoff spec the collaborator follows. Kept for its rationale and the appendix on what SPT's code actually does.

Written 2026-10-25 after reading the bundled SPT implementation (`runtime/vforge/models/rep_world/`, `runtime/vforge/tasks/rep_world/`, `jobs/`) and LeWM in `~/code/sparse_jepa` (`stable_worldmodel/wm/lewm/`, `scripts/train/lewm.py`). Third revision, same day: it follows the consolidated story in [anchored-state-jepa-design.md](anchored-state-jepa-design.md) Section 4 (the JEPA objective derived from code length; editing as an interpretation), checked point by point against the design session's final answer (Happier session `cmvmnyra9gqc1qq5775z6t4j9`, seq 144). Nothing here has been implemented or run.

The rule for this plan: **the design decides, the code adapts.** Section 4 traces every design point to where it is implemented.

## 1. Where each piece comes from

The model is the one in the design doc: a per-frame encoder with a noisy state, a predictive prior trained only by the code length, and a flow renderer that sees the state and the first frame. The image-editing story adds no module and no pretrained weights. What varies by stage is which code base supplies each module.

| Module | Stages 1-2 (PushT suite) | Stage 3 (SPT scale) | Role in the story |
| --- | --- | --- | --- |
| Encoder `q(z_t \| x_t)` | LeWM's ViT-tiny (224 px, patch 14), 4 learned query tokens pooling its patch tokens, `LayerNorm -> Linear` head to `mu` (4 x 16 = 64 dims) | SPT's ViT-E (per-state), same query pooling | the JEPA encoder; its noisy output is the code |
| Prior `p(z_t \| z_<t, a_<t)` | LeWM's `Predictor` (depth 6, AdaLN-zero action conditioning, causal) on the flattened 64-dim state; output is `delta`, mean `z_{t-1} + delta` | SPT's `CausalViTBlock` stack with `state_causal_sparsity`, over `K` state tokens plus an action token per step | the JEPA predictor, which is the predictive coder |
| Action encoder | LeWM's `Embedder` (5 actions per step at frameskip 5) | a linear action token | actions enter the prior only |
| Renderer `p(x_t \| z_t, x_0)` | SPT's `WorldDiT` at DiT-S size (width 384, depth 12, heads 6), one 64 px frame packed as a 16x16 grid of 4x4 patches (48 channels per token), plain output layer | SPT's pixel `WorldDiT` with `PixelRefiner` | the "editor": first frame by channel concatenation, state by cross-attention, guidance via condition dropout |
| Flow recipe | SPT's timestep schedule (logit-normal plus shift), velocity target `y - eps`, `condition_dropout 0.1` | same | stochastic rendering (sample what you cannot predict) |
| Objective | `fm + ramp * (beta_surprise * surprise + beta_entry * entry)` | same | the code length; no SIGReg, stop-gradient, EMA or augmentation |
| Harness | stable-worldmodel: LeWM's `.lance` loader and normalization, Lightning trainer, planners, `GoalMSE` | SPT's Native trainer, FSDP, FVD protocol | |
| Editing models | interfaces only (reference image by channel concatenation, instruction by cross-attention, guidance on the instruction) | optional later variant: initialize the renderer from a pretrained editor for natural video | interpretation, not mechanism |

Why stages 1-2 use LeWM's harness and modules rather than SPT's: the design doc's PushT table already specifies LeWM's encoder and predictor, and keeping them makes the comparison with LeWM a literal diff (Section 2). The planners, environments and data pipeline the JEPA community evaluates with live there. SPT supplies the renderer, which is the piece LeWM lacks. Running it outside SPT's trainer needs one call, `initialize_ops(OpsConfig(backend="sdpa"), device=..., parallel_state=None)`, exactly as SPT's own CPU tests do (`runtime/tests/models/rep_world/test_cross_attention_order.py`); the uninitialized parallel state defaults to no sequence parallelism. No hash-pinned SPT file is edited.

Why not a pretrained editing model: the derivation says the instruction space is whatever the code length and the renderer make it. A pretrained editor brings its own notion of an instruction, trained on text, which would weaken the claim that the principle alone produces the state, and its cost and resolution do not fit PushT. It remains an option for natural video in stage 3, reported as a separate variant.

## 2. The diff from LeWM, in code

LeWM's training step (`scripts/train/lewm.py`, `lejepa_forward`) is `pred_loss = (pred_emb - tgt_emb).pow(2).mean()` with an attached target, plus `lambda * SIGReg(emb)`. AS-JEPA changes it as follows:

```python
out = model.encode(batch)                       # ViT-tiny + query head -> mu: (B, T+1, 64); no BatchNorm on mu
mu = out["mu"]
z = mu + sigma0 * torch.randn_like(mu)          # NEW: noise sets the code's resolution
m = z[:, :-1] + model.predict(z[:, :-1], out["act_emb"][:, :-1])   # LeWM predictor, residual mean
surprise = (mu[:, 1:] - m).pow(2).sum(-1) / (2 * s0**2)   # LeWM's pred_loss, in nats per step
entry = 0.5 * mu[:, 0].pow(2).sum(-1)                      # replaces SIGReg (Section 4.1 of the design)

y = pack64(batch["pixels"][:, 1:])              # every frame after the anchor, 64 px
anchor = pack64(batch["pixels"][:, :1])
fm = model.render_loss(y, state=z[:, 1:], anchor=anchor)   # SPT WorldDiT, SPT schedule and sign

loss = fm + ramp(step) * (beta_s * surprise.sum(1) + beta_e * entry).mean() / NORM
```

Two LeWM details must change, for the same reason as SPT's LayerNorm projector below: LeWM's `projector` and `pred_proj` are MLPs with `BatchNorm1d`. Batch normalization on `mu` would force every dimension to unit spread, so the entry price could never switch a dimension off. `mu` and `delta` therefore come from plain linear heads (zero-initialized for `delta`, so the prior starts as the identity carry). `NORM` is the number of rendered pixel elements per window, which puts both `beta`s on a per-element scale.

## 3. Module details

### 3.1 Renderer (both stages)

`AnchoredWorldDiT` subclasses SPT's `WorldDiT` with `condition_mode cross_attention`, `cross_attention_scope global`, `cross_attention_3d_rope false` (state tokens have no spatial position), `patch_size 1`, and in stages 1-2 `channels 48`, `state_planes 1`, `grid 16`, `pixel_depth 0`. Two changes at construction, none to pinned files:

- **Anchor.** `x_embedder` is wrapped so it returns `base(noisy) + anchor_embed(x_0)`, with `anchor_embed` a zero-initialized `PatchEmbed3D` of the same shape: algebraically, channel concatenation with the new weights starting at zero. The anchor is set on the wrapper inside `forward` under `try/finally`; activation checkpointing recomputes only blocks, never the embedding. In stage 3 the `PixelRefiner`'s per-pixel embedding gets the same wrapper.
- **State projection.** `semantic_raw_projector` (`LayerNorm(condition_width) -> Linear`) would normalize each 16-dim state token, removing its mean and scale. It is replaced by `nn.Sequential(nn.Identity(), nn.Linear(d, width))` with `condition_width = d`; `WorldDiT._condition` only requires element `[1]` to have a weight.

The renderer receives no history and no actions. The 4 state tokens are the cross-attention context. For camera motion (stage 3), the anchor's token embeddings are appended to that context instead (ReferenceNet style), which `WorldDiT` supports unmodified; an in-context variant would need a `WorldDiT` change.

### 3.2 State codes (both stages)

All codes implement `sample(posterior)`, `entry_rate(posterior)` and `surprise_rate(posterior, prior)`, in nats.

- **FixedGaussian (first).** `z = mu + sigma_0 * eps`; entry `||mu_0||^2 / 2`; surprise `||mu_t - m_t||^2 / (2 s_0^2)`; constants dropped.
- **LearnedGaussian.** Closed-form diagonal KL to `N(0, I)` and to `N(m_t, s_t^2)`, with `s_t` from the prior. Needed for the component-wise (MCC) claim, since TDRL and LEAP need variances that depend on the past.
- **FSQ (second, if `beta` is painful).** Quantized `mu`, a categorical prior head, rate = cross-entropy.

The latent flow prior (design Section 6, for action-free multimodal video) is not built yet; the interface leaves room for it.

### 3.3 Inference (both stages)

- **Encode**: `mu` of a frame, goal images included.
- **Rollout**: prior means, or samples, under candidate actions, with LeWM's `encode`/`rollout` signatures so the stable-worldmodel planners and `GoalMSE` run unchanged. The rollout history equals the training context.
- **Generate**: sample the next state from the prior and render it with the first observed frame as the anchor, at a chosen guidance scale on the state. Two continuation modes, reported separately: append the sampled state, or re-encode the generated frame.

### 3.4 Stage 3 on SPT's Native stack

A new Native family under `runtime/vforge/models/as_jepa/` and `runtime/vforge/tasks/as_jepa/` with an experiment YAML, following `runtime/AGENTS.md` (no new Trainer, checkpoint manager or attention backend) and its own launchers under `jobs/as_jepa/` with their own hash manifest, so the RGB16 worker's integrity check is untouched. Encoder: `StateViTEncoder` with `context_mask_ratio 0` plus query pooling. Prior: `CausalViTBlock` stack, block-causal over `[K state tokens, action token]` per step, RoPE time = step, row = token index; cached rollout via `forward_cached`. Renderer: Section 3.1 with the pixel refiner. Evaluation: SPT's FVD protocol against the RGB16 baseline, and, for action-free video, the per-step surprise as a violation-of-expectation score (IntPhys-style).

## 4. Design points and where they are implemented

| Design point | Where | Note |
| --- | --- | --- |
| JEPA loss emerges as the surprise code length; entry term in place of SIGReg | 2, 3.2 | the diff from LeWM is literal |
| Anchor = first frame, renderer only, never rendered | 2, 3.1 | encoder and prior never receive `x_0` |
| Static camera: channel concatenation (InstructPix2Pix) | 3.1 | zero-initialized anchor embedding |
| State through cross-attention or adaLN | 3.1 | cross-attention, no spatial RoPE |
| Camera motion: in-context or ReferenceNet-style anchor | 3.1, 3.4 | cross-attention route works unmodified; in-context deferred |
| Editing analogy `x_t = edit(x_0, z_t)` | 1, design 4.3 | interfaces only, no pretrained editor |
| State dropout for CFG; guidance scale as reliance diagnostic | 3.1, 6 | SPT's `condition_dropout 0.1` |
| Prior emits a distribution, never touches the renderer, trained only by the code length | 2 | renderer reads the posterior sample in training |
| Residual mean; static dimensions free after entry | 2 | zero-initialized `delta` |
| Fixed-variance three-line form with separate prices for entry and surprise | 2, 3.2 | `beta_e`, `beta_s`, start equal |
| Learned-variance, FSQ and latent-flow codes | 3.2 | flow prior later |
| Actions enter the prior only | 1, 2 | |
| Renderer noise = within-frame residual; one observed outcome suffices | 5.1 | holds exactly with one-frame chunks |
| One-step prediction, no offsets or overshooting | 2 | |
| Masking is not the information restriction | 5.4 | off; one optional comparison |
| Per-frame encoder | 1, 7 (test) | |
| Identifiability step 2: frames rendered independently given state and anchor | 3.1 | one frame per renderer call |
| A2 data conditions: appearance independent of initial state; `eps` i.i.d. | 6 | data collection |
| Lag-`L` transitions (design 5.1) | 2, 3.3 | prior reads past states; planning starts from the last frames |
| `beta = 1` meaning | 5.2 | sweep with SPT's loss; ELBO weighting as variant |
| Risks and guards (design Section 8) | 7 | each has a logged signal |
| Three uses: planning, generation, representation | 3.3 | |
| First experiment and the LeWM-to-AS-JEPA ladder (design Section 9) | 6, 8 | |

## 5. Decisions made, with reasons

1. **One-frame chunks at frameskip 5.** The renderer's noise then models only within-frame residual, the identifiability step that renders frames independently holds exactly, goal images encode directly, and the setup matches LeWM. SPT's `WorldDiT` renders one 64 px frame with its plain output layer; the 4-frame refiner is only needed in stage 3.
2. **`beta` is a rate-distortion weight with SPT's loss.** SPT's flow loss is a velocity MSE under a non-monotone logit-normal time weighting, so by Kingma and Gao (2023) it is not an ELBO term, and it is measured per pixel element. Log rates in nats and sweep. If we want `beta = 1` to mean the exact bound, as the theory states it, a variant trains the renderer with a monotone (ELBO) weighting, which Kingma and Gao show is an ELBO of noise-augmented data.
3. **Separate entry and surprise weights**, starting equal; the leak and collapse diagnostics decide whether they should differ.
4. **Masking off, with one optional comparison** that replaces the Gaussian state noise with SPT-style input-patch dropping, so the disagreement about masking is measured rather than argued.
5. **Velocity lives in the prior's context.** One frame does not show velocity ("Frozen in a Frame", arXiv 2610.04585, finds single-frame JEPA states miss it). The state is the visible configuration and the prior reads several past states (design 5.1). A velocity probe on the prior's hidden state, versus on `z_t`, checks this.
6. **No edits to pinned SPT files; EMA off; AdamW** (LeWM's optimizer; SPT's EMA is evaluation-only).

## 6. Data

The environments, `WeakPolicy` and collectors are in `~/code/sparse_jepa` (`stable_worldmodel/envs/pusht`, `scripts/data/collect_weak_pusht.py`, variation space in `docs/envs/pusht.md`; TwoRoom, DMC Reacher and OGBench Cube are alongside).

1. **PushT with variations** (stage 1): `WeakPolicy`, 224 px as LeWM. Per episode, sample render-only factors (agent, block and background colours, goal-overlay pose) independently of the initial state, and dynamics-relevant factors (block shape and scale). Record factors and per-frame simulator state for probes and MCC. A second variant overlays a random dot redrawn independently every frame. The renderer's target is the same frames downsampled to 64 px.
2. **Standard LeWM datasets** (stage 2): PushT, TwoRoom, Reacher and Cube as LeWM uses them, for comparability with published numbers.
3. Windows of 8 frames at frameskip 5, split by episode. Datasets stay outside Git, at a path the user chooses.

## 7. Diagnostics tied to the design's risks

| Signal | What it detects | Response |
| --- | --- | --- |
| entry rate in nats against the rough 40-nat PushT budget; colour probes on `mu` | appearance leaking into the state | raise `beta_e`; check the anchor branch weights moved off zero |
| per-dimension rate averaged over data | effective state size; early collapse when all are near zero | longer ramp; FSQ code |
| renders with the state zeroed or swapped; guidance scale needed to follow the state | renderer ignoring the state | as above |
| swap test: anchor from episode A, states from episode B | correct routing: A's colours, B's poses | |
| dot probe on `mu`, flow versus MSE renderer | transient detail stored instead of sampled | |
| action following in contact versus free motion | a contact-only factor dropped at large `beta` | choose `beta` on contact action following, not flow loss |
| prior-mean rollouts; drift over steps | unimodal-prior averaging; compounding error | flow prior; revisit training only if drift appears |
| velocity probes on `z_t` and on the prior's hidden state | where velocity lives | if the prior lacks it, two-frame chunks |
| probes and MCC against simulator factors | block identifiability; component-wise bonus | |

## 8. Order of work and success checks

1. **Data and harness.** Frames match simulator renders; a known action sequence moves the agent as recorded between the frames the prior links. `WorldDiT` imports and runs in the stable-worldmodel environment after `initialize_ops`.
2. **CPU contract tests**, small configs: encoder output for frame `t` unchanged when other frames change; prior output for step `t` unchanged when later frames or actions change, with no anchor input; renderer output unchanged by history and actions, and equal to plain `WorldDiT` at the anchor branch's zero initialization; gradients reach encoder, prior and renderer; each code's rates equal hand-computed values.
3. **One-GPU smoke run**: finite loss, rates moving, renders change when the state is zeroed.
4. **Stage 1, visual results first**: rendered prior rollouts under recorded actions next to ground truth; different action sequences from one start; the swap figure; entry rate and probes; the LeWM-to-AS-JEPA ladder (design Section 9) across a `beta` sweep. Success: colour probes near chance for rung 3 but not rungs 1, 2 and 4, rollouts that follow actions in contact, and the dot excluded with the flow renderer but not the MSE one.
5. **Stage 2, planning** under LeWM's protocol on PushT, TwoRoom, Reacher and Cube, plus colour-shifted goals on the variation dataset. Success: on par with LeWM on the standard tasks, and better with colour-shifted goals.
6. **Stage 3, SPT scale** (Section 3.4): natural video, FVD against the SPT baseline, surprise-based physics tests.

## 9. Decisions needed from the user

- **Compute placement.** This host is a login node. Collection and training need an allocated machine; dev3 and dev8 belong to the SPT maintainers.
- **Where the stage 1-2 code lives.** Recommended: `stable_worldmodel/wm/as_jepa/` and `scripts/train/as_jepa.py` in `~/code/sparse_jepa`, next to LeWM, with this repository's `runtime` installed as a package for `WorldDiT`. That environment must meet `runtime`'s requirements (Python 3.12 or newer, torch 2.9.1 or newer). Docs stay on this branch.
- **Whether to include the masking comparison** in the first round.

## Appendix: what SPT's code actually does

The design notes described SPT from its manuscript. The code differs in ways that shaped this plan.

| Topic | Actual SPT code | Consequence |
| --- | --- | --- |
| Encoder | `SplitViTConditioner`: ViT-E (3 blocks, each state alone) then ViT-C (18 blocks, state-causal) | ViT-E and ViT-C's blocks are stage 3's encoder and prior |
| Chunk | RGB16 state = 16 frames at 256 px (4 planes x 16x16 patches = 1024 tokens); `PixelRefiner` hard-codes 4 frames per plane | stage 1 uses `WorldDiT` without the refiner for one 64 px frame |
| Masking | 90% of patches dropped before ViT-E blocks (`mask_stage: before_e`), about 102 of 1024 kept | reusing ViT-E as the encoder requires masking off |
| DiT conditioning | cross-attention to the latest history state's tokens with spatially aligned 3D RoPE | the recent-anchor bypass AS-JEPA removes |
| Condition projection | `LayerNorm -> Linear` per token | replaced (Section 3.1) |
| Noise schedule | logit-normal mean 0.8, std 0.8, automatic shift; median noise level about 0.8; target `y - eps` | already high-noise biased; reused |
| Condition dropout | 0.1, learned null token | reused for guidance on the state |
| EMA | evaluation-only weight average, decay 0.99 | off |
| Actions | none; SolarWM data are action-free with camera motion | PushT suite from stable-worldmodel |
| Optimizer | Muon on every parameter with two or more dimensions | AdamW |
| Integrity | `jobs/worker.py` refuses to start if hash-pinned runtime files change | no pinned edits |
