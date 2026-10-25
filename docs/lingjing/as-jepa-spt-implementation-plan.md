# AS-JEPA on the SPT code base: implementation plan

Written 2026-10-25 after reading the bundled SPT implementation (`runtime/vforge/models/rep_world/`, `runtime/vforge/tasks/rep_world/`, `jobs/`). Revised the same day against the final answer of the design session (Happier session `cmvmnyra9gqc1qq5775z6t4j9`, seq 144: anchor conditioning, the prior and its code-length term, the flow noise, masking, block identifiability). The design is in [anchored-state-jepa-design.md](anchored-state-jepa-design.md); this file says how to build it in this repository. Nothing here has been implemented or run.

The rule for this plan: **the design decides, the code adapts.** Where SPT's code makes a design choice inconvenient, the plan finds a route that keeps the design, and says so when no such route exists. Section 2 checks every design point against the plan.

## 1. What SPT actually is

The design notes described SPT from the manuscript. The code differs in ways that matter.

| Topic | Design notes assumed | Actual SPT code | Consequence for AS-JEPA |
| --- | --- | --- | --- |
| Encoder | one ViT, then masked tokens go to a causal transformer | `SplitViTConditioner` = **ViT-E** (`StateViTEncoder`, 3 blocks, each state encoded alone) then **ViT-C** (`TokenCausalViT`, 18 blocks, state-causal mask) | ViT-E is the per-chunk encoder AS-JEPA needs. ViT-C's pieces (`CausalViTBlock`, `state_causal_sparsity`, cached prefill) are the prior's skeleton. |
| Chunk | one frame on PushT | RGB16 state = 16 frames at 256x256 (4 planes x 16x16 patches = 1024 tokens); the `PixelRefiner` hard-codes 4 frames per plane at 256 px | The RGB16 renderer cannot render one frame. `WorldDiT` without the refiner can (Section 3.3). |
| Masking | 90% of history tokens dropped *after* the ViT; "26 retained of 256" | `mask_stage: before_e`: 90% of patches dropped *before* the ViT-E blocks, so the encoder trains on about 102 of 1024 tokens per state and sees all of them at inference | Reusing ViT-E as the encoder requires masking off; Section 4.4. |
| DiT conditioning | "conditioning projection" of a history summary | cross-attention (blocks 0 and 4) to ViT-C tokens of the latest origin state, with 3D RoPE aligning each condition token to the same spatial position in the target | SPT's renderer has a spatially aligned view of the most recent chunk: the recent-anchor bypass the design forbids. AS-JEPA removes it by construction. |
| Condition projection | not discussed | `semantic_raw_projector = LayerNorm(condition_width) -> Linear` | Per-token LayerNorm on a 16-dim state token removes its mean and scale. Replaced in AS-JEPA (Section 3.3). |
| Target | pixels or frozen codec | direct pixels, losslessly packed into patch tokens | fixed target, as the design requires |
| Future offset | random `k` | target is state `i+1` or `i+2` (`distant_target_prob 0.5`) | removed: one-step prior, renderer renders the current chunk |
| Noise schedule | "bias high if the state is under-used" | `noisy = (1-t) y + t eps`, `t = shift(sigmoid(0.8 + 0.8 n))`; median `t` about 0.8 | already high-noise biased; keep. Velocity target is `y - eps` (design pseudocode used `eps - y`). |
| Condition dropout | proposed as new, for CFG on the state | `condition_dropout 0.1` swaps the condition for a learned null token | reuse; it is the training half of the guidance-scale diagnostic |
| EMA | "no EMA" | `ema.decay 0.99` is an evaluation-only weight average, not a target network | keep off to honour the decision |
| Actions | `a_t` into the prior | none anywhere; SolarWM data are action-free, with camera motion | PushT with actions must be brought in (Section 5) |
| Optimizer | unspecified | `MuonWithAuxAdam` puts every `ndim >= 2` parameter in Muon | AdamW for the first runs |
| Code integrity | edit SPT modules | `jobs/worker.py` refuses to start if a hash-pinned runtime file changes; `tests/test_portability.py` checks the same | do not edit `rep_world`; AS-JEPA is a new Native family importing its components |

## 2. Design points and where the plan implements them

This table walks through the final design answer and the design doc. "Plan" names the section below; "Change" marks where the code forces a deviation.

| Design point | Plan | Change |
| --- | --- | --- |
| **Anchor = first chunk, renderer only, never rendered; prior and encoder never see it** | 3.3, 3.5 (renderer is the only module taking `x_0`; loss renders `t >= 1`) | none |
| Static camera: anchor channel-concatenated with the noisy target (InstructPix2Pix) | 3.3: zero-initialized anchor embedding added to the DiT patch embedding, algebraically channel concatenation | none |
| State tokens enter through cross-attention or adaLN | 3.3: cross-attention, global scope, no spatial RoPE (state tokens have no position) | none |
| Camera motion: anchor tokens in context (FLUX Kontext, OmniGen) or cross-attended (ReferenceNet) | 3.3: in-context is blocked by `WorldDiT` (position-aware cross-attention and target-only constraints); cross-attending to `[state tokens; anchor tokens]` works unmodified | in-context variant needs a `WorldDiT` change; deferred until SolarWM |
| Editing analogy `x_t = edit(x_0, z_t)` | the renderer's interface is literally `render(noisy, t, state=z_t, anchor=x_0)` | none |
| State dropout enables CFG on the state; the guidance scale needed is the reliance diagnostic | 3.3 (`condition_dropout 0.1`), 6 (diagnostic sweeps guidance scale) | none |
| Anchor noising: not wanted | off, not implemented | none |
| **Prior emits a distribution over the next state, never touches the renderer, trained only by the code-length term** | 3.2, 3.5 (prior output enters only the rate term during training; renderer reads the posterior sample) | none |
| Code length prices surprise, not size; residual mean so static dimensions cost nothing after entry | 3.2: mean `z_t + delta`, zero-initialized `delta` head | none |
| **Fixed-variance three-line form: fixed noise on the state; squared prediction error; squared norm on the first state. Noise sets resolution, two weights set the price of surprise and of entry** | 3.4, 3.5: `sigma_0` fixed, separate `beta_surprise` and `beta_entry` | my first plan collapsed the two weights into one; restored |
| Learned-variance Gaussian (exact KL), FSQ with categorical prior, noisy encoder plus latent flow prior (design Section 6) | 3.4: one `StateCode` interface with these as alternative implementations; fixed-variance first, FSQ second | none |
| Diagonal prior whose means *and variances* depend on the past and actions (TDRL/LEAP condition for component-wise identifiability) | 3.4: the learned-variance code; required for the MCC experiment | fixed variances forfeit the component-wise bonus, not the block claim |
| Actions enter the prior only | 3.2 (action token per step in the prior); renderer and encoder take no actions | none |
| **Renderer noise is the within-frame residual, not alternative futures; one observed outcome per context suffices** | 3.3, 4.1: one-frame chunks keep this argument exact | my first plan used 4-frame chunks, which put intra-chunk motion inside the renderer's noise; reverted |
| **One-step prediction; multi-step offsets removed** | 3.5: no target offsets, no overshooting | none |
| **Masking is not the information restriction** (user leaned toward it; not settled) | 4.4: off in the main model; one optional erasure-channel comparison | none |
| Encoder per chunk, sees only its own chunk | 3.1, CPU test in 7 | none |
| Block identifiability step 2: frames are rendered independently given state and anchor | 3.3: each frame is one renderer call | holds only with one-frame chunks; another reason for 4.1 |
| A2: appearance independent of the initial state; `eps` i.i.d. over time | 5: data collection samples colours independently of initial poses; dot redrawn every frame | none |
| `beta = 1` is the exact bound | 4.2: not with SPT's flow loss; sweep `beta` | yes, documented |
| Risks: posterior collapse (ramp `beta`, high noise, FSQ); appearance leak; contact factor squeezed at large `beta`; unimodal prior; rollout drift | 6: each risk has a logged signal and a response | none |
| Three uses: planning via `encode`/`rollout`, generation (sample from prior, render with anchor; append state or re-encode), representation | 3.6, 6 | none |
| First experiment: render-only vs dynamics-relevant factors, dot, `beta` sweep, ablations no-anchor and MSE renderer, baselines LeWM and LRC-JEPA, probes, MCC, swap test, colour-shifted goals, contact vs free motion | 5, 6, 8 | none |

## 3. Code layout and modules

New Native family, following `runtime/AGENTS.md` (models in `models/`, assembly in `tasks/`, model differences in the experiment YAML; no new Trainer, checkpoint manager or attention backend):

```text
runtime/vforge/models/as_jepa/
  config.py      # StateConfig, PriorConfig, RendererConfig, CodeConfig
  pixels.py      # pack/unpack one frame <-> [B, 3*p*p, 1, 16, 16] patch tokens
  encoder.py     # StateEncoder: rep_world StateViTEncoder (latent input, mask 0) + query pooling
  codes.py       # StateCode: FixedGaussian, LearnedGaussian, FSQ (rate and sampling)
  prior.py       # StatePrior: CausalViTBlock stack over [state tokens, action token] per step
  renderer.py    # AnchoredWorldDiT: rep_world WorldDiT + anchor embedding + linear state projector
  model.py       # ASJepa root: loss, encode, rollout, render
runtime/vforge/tasks/as_jepa/
  data.py  train.py  shared.py  eval.py
runtime/vforge/conf/experiment/as_jepa_pusht.yaml
runtime/tests/models/as_jepa/  runtime/tests/tasks/as_jepa/
jobs/as_jepa/    # prepare/worker launchers with their own hash manifest
```

### 3.1 Encoder

One frame at 64x64 is packed losslessly into a 16x16 grid of 4x4 patches, 48 channels per token (`pixels.py`, a reshape like `PixelRGB4Codec`). `StateViTEncoder` in latent-input mode applies `Linear(48, width)` per patch, its blocks, and a final norm, with each frame encoded alone. `K` learned queries cross-attend to its 256 output tokens and a linear head gives `mu` (and `log sigma` for the learned-variance code). Masking is off (`context_mask_ratio 0`).

### 3.2 Prior

One sequence block per time step: `K` state tokens (`Linear(d -> W)` of the posterior sample) plus one action token (`Linear(5 * 2 -> W)` of the 5 actions between consecutive frames at frameskip 5). Block-causal attention via `state_causal_sparsity(length, K + 1)` (FA4) or the boolean block mask (SDPA), as `TokenCausalViT` does. RoPE positions: time = step, row = token index, col = 0. Heads on the `K` state positions of block `t` give `delta` (zero-initialized) and, for the learned code, `log s`; the mean for step `t + 1` is `z_t + delta`. Inference uses `CausalViTBlock.forward_cached`.

The prior sees posterior samples `z_<t` and actions, nothing else: no anchor, no pixels, no renderer features. In training its output goes only to the rate term.

### 3.3 Renderer

`AnchoredWorldDiT` subclasses `WorldDiT` with `channels 48`, `state_planes 1`, `grid 16`, `patch_size 1`, `pixel_depth 0` (the plain `FinalLayer` output, which accepts any channel count), `condition_mode cross_attention`, `cross_attention_scope global`, `cross_attention_3d_rope false`, `condition_dropout 0.1`. Two changes at construction, none to pinned files:

- **Anchor.** `x_embedder` is wrapped so that it returns `base(noisy) + anchor_embed(x_0)`, with `anchor_embed` a zero-initialized `PatchEmbed3D` of the same shape. That equals channel concatenation with the new half of the weights at zero. The anchor tensor is set on the wrapper inside `forward` under `try/finally`; activation checkpointing recomputes only the blocks, never the embedding.
- **State projection.** `semantic_raw_projector` is replaced by `nn.Sequential(nn.Identity(), nn.Linear(d, width))` with `condition_width = d`, removing the per-token LayerNorm. `WorldDiT._condition` only requires element `[1]` to have a weight.

The renderer receives no history and no actions. For a camera-motion dataset, the anchor's own token embeddings would be appended to the cross-attention context instead (ReferenceNet style), which `WorldDiT` supports unmodified; the in-context variant needs a `WorldDiT` change and is deferred.

### 3.4 State codes

All codes implement the same interface: `sample(posterior) -> z`, `entry_rate(posterior) -> nats`, `surprise_rate(posterior, prior) -> nats`.

- **FixedGaussian (first).** `z = mu + sigma_0 * eps`; entry `||mu_0||^2 / 2`; surprise `||mu_t - m_t||^2 / (2 s_0^2)`; constants dropped. Three lines, as the design says.
- **LearnedGaussian.** Closed-form diagonal KL to `N(0, I)` and to `N(m_t, s_t^2)`, with `s_t` from the prior. Needed for the component-wise (MCC) claim, since TDRL/LEAP need variances that depend on the past.
- **FSQ (second, if `beta` is painful).** Finite scalar quantization of `mu`; categorical prior head; rate = cross-entropy in nats; the code size is the hard budget.

The latent flow prior (for action-free multimodal video) is not built now; the interface leaves room for it.

### 3.5 Loss

```python
post = encoder(frames)                          # [B, T+1, K, d]; each frame alone
z = code.sample(post)
prior = prior_net(z[:, :-1], actions[:, :-1])   # mean (and scale) for steps 1..T
entry = code.entry_rate(post[:, 0])             # nats per window
surprise = code.surprise_rate(post[:, 1:], prior).sum(1)

y = pack(frames[:, 1:])                         # every frame after the anchor
t = draw_timesteps(...)                         # SPT schedule, imported
noisy = (1 - t) * y + t * eps
v = renderer(noisy, t, state=z[:, 1:], anchor=pack(frames[:, 0]))
fm = (y - eps - v).square().mean()              # SPT sign convention

loss = fm + (ramp(step) / NORM) * (beta_entry * entry + beta_surprise * surprise).mean()
```

`NORM` is the number of pixel elements per window, so the two `beta`s are on a fixed per-element scale. `ramp` goes linearly from 0 to 1 over the first 5% of steps. No stop-gradient, no target network, no EMA, no augmentation.

### 3.6 Inference

- **Encode**: `mu` of a frame (goal images included).
- **Rollout**: prior means (or samples) under candidate actions, cached.
- **Generate**: sample the next state from the prior and render it with the first observed frame as the anchor, at a chosen guidance scale on the state. Two continuation modes, reported separately: append the sampled state, or re-encode the generated frame.
- **Planning adapter**: `encode` and `rollout` with LeWM's signatures (`stable_worldmodel/wm/lewm/lewm.py` in `~/code/sparse_jepa`), so its planners and `GoalMSE` run unchanged.

## 4. Decisions made, with reasons

1. **One-frame chunks at 64 px with frameskip 5, as in the design.** The renderer's noise then models only within-frame residual, the block-identifiability step that renders frames independently given state and anchor holds exactly, goal images encode directly, and the setup matches LeWM for comparison. My first plan chose 4-frame RGB4 chunks to reuse the pixel refiner; that broke three of these properties for a code convenience that turns out to be unnecessary. Multi-frame chunks stay available as the design's open "slow/fast level" knob.
2. **`beta` is a rate-distortion weight, not 1.** SPT's flow loss is a velocity MSE under a non-monotone logit-normal time weighting, so by Kingma and Gao (2023) it is not an ELBO term and is measured per pixel element. Log the rates in nats and sweep. The block-identifiability statement is about the idealized model, and its "beta = 1 or slightly above" condition has no numeric counterpart here.
3. **Separate entry and surprise weights** (design: "the two weights set the price of surprise and of entry"; open question in design Section 11). Start equal; the leak and collapse diagnostics decide whether to separate them.
4. **Masking off, with one comparison.** The design argues masking is at most an erasure channel on the state, not a price; the user's counter-point was that it is empirically proven in SPT. One run replaces the Gaussian state noise with SPT-style token dropping on the encoder's input (the entry and surprise terms then reduce to a plain prediction error with no resolution) so the comparison is measured, not argued.
5. **Anchor wrapper, not edits to pinned files**, so the RGB16 worker's integrity check keeps passing.
6. **EMA off; AdamW.** EMA is evaluation-only in SPT and the design removes EMA; Muon puts small learned tokens through orthogonalization.

## 5. Data: PushT with actions

The environment, `WeakPolicy` and collectors are in `~/code/sparse_jepa` (`stable_worldmodel/envs/pusht`, `scripts/data/collect_weak_pusht.py`, variation space in `docs/envs/pusht.md`).

1. Collect at `resolution=64` with `WeakPolicy`. Per episode, sample render-only factors (agent, block and background colours, goal-overlay pose) independently of the initial state, and dynamics-relevant factors (block shape and scale). Record factors and per-frame simulator state for probes and MCC. A second variant overlays a random dot redrawn independently every frame.
2. Export each episode as frames plus `actions.npy` and `factors.json`, with a JSONL manifest. Split by episode before windowing.
3. `PushTDataset` returns `frames [T+1, 3, 64, 64]` at frameskip 5 and `actions [T+1, 5, 2]` (the actions between consecutive kept frames), with a resume contract mirroring `rep_world/data.py`. Window `T + 1 = 8`.

Datasets live outside Git, at a path the user chooses (not `/efs/prescottsun/SPT`).

## 6. Diagnostics tied to the design's risks

| Signal (logged or evaluated) | What it detects | Response |
| --- | --- | --- |
| entry rate (nats) vs the rough 40-nat PushT budget; colour probes on `mu` | appearance leaking into the state (near twice the budget, probes above chance) | raise `beta_entry`; check the anchor branch weights moved off zero |
| per-dimension rate averaged over data | effective state size; collapse when all are near zero early | longer ramp; FSQ code |
| renders with the state zeroed or swapped; guidance scale needed for state-following samples | renderer ignoring the state (posterior collapse) | as above |
| swap test: anchor from episode A, states from episode B | correct routing: A's colours, B's poses | |
| dot probe on `mu`, flow renderer vs MSE renderer | transient detail stored instead of sampled | |
| action following in contact vs free motion | a contact-only factor dropped at large `beta` | choose `beta` on contact action following, not on flow loss |
| prior-mean rollouts without actions; latent drift over steps | unimodal-prior averaging; compounding error | flow prior; revisit training only if drift appears |
| probes and MCC against simulator factors | block identifiability (pose, shape, scale in; colours, goal, dot out); component-wise bonus | |

## 7. Order of work and checks

Follow the feedback ladder in `runtime/AGENTS.md`.

1. **Data export and loader.** Frames match simulator renders; a known action sequence moves the agent in the recorded direction between the frames the prior links.
2. **CPU contract tests**, small configs:
   - encoder output for frame `t` unchanged when other frames change;
   - prior output for step `t` unchanged when frames or actions after `t` change, and it has no anchor or pixel input;
   - renderer output unchanged by history and actions; with the anchor branch at its zero initialization, output equals plain `WorldDiT`;
   - gradients reach encoder, prior and renderer, with no detached path;
   - each `StateCode`'s rates equal hand-computed values.
3. **One-GPU smoke run** on a small shard: finite loss, rates moving, renders change when the state is zeroed.
4. **Main run, `beta` sweep, ablations** (no anchor; deterministic MSE renderer, which regresses `y` from state and anchor with a fixed input), the masking comparison (4.4), and the learned-variance code for MCC.
5. **Evaluation** (Section 6), then planning through the adapter, against LeWM and, if code is available, LRC-JEPA.

## 8. Decisions needed from the user

- **Compute placement.** This host is a login node. Collection and training need an allocated machine; dev3 and dev8 belong to the SPT maintainers.
- **Whether to run the masking comparison** (4.4) in the first round, given the earlier disagreement about masking.
