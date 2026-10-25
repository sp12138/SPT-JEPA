# AS-JEPA on the SPT code base: implementation plan

Written 2026-10-25 after reading the bundled SPT implementation (`runtime/vforge/models/rep_world/`, `runtime/vforge/tasks/rep_world/`, `jobs/`). It replaces the "PushT starting point" table in [anchored-state-jepa-design.md](anchored-state-jepa-design.md) Section 2, which assumed a LeWM-style stack (ViT-tiny at 224 px, LeWM predictor, DiT-S at 64 px). The design itself (objective, routing, theory) is unchanged except where noted in Section 2 below. Nothing here has been implemented or run.

## 1. What SPT actually is (and where the design notes were wrong about it)

The design notes and the October collaborator handoff described SPT from the manuscript. The code differs in ways that matter for the plan.

| Topic | Design notes assumed | Actual SPT code | Consequence for AS-JEPA |
| --- | --- | --- | --- |
| Encoder | one ViT, then masked tokens go to a causal transformer | `SplitViTConditioner` = **ViT-E** (`StateViTEncoder`, 3 blocks, each state encoded alone) then **ViT-C** (`TokenCausalViT`, 18 blocks, state-causal mask) | ViT-E is already the per-chunk encoder AS-JEPA needs. ViT-C's block-causal machinery (`CausalViTBlock`, `state_causal_sparsity`, cached prefill) is the prior's skeleton. |
| Chunk | "chunk" of unspecified length | RGB16: one state = 16 RGB frames at 256x256 = 4 temporal planes x 16x16 patches = 1024 tokens; clip = 10 states = 160 frames. RGB4 (`pixel/rgb4`): 4 frames, 1 plane, 256 tokens. | A chunk is several frames. The state encodes a short motion segment, not an instantaneous pose. |
| Masking | 90% of history tokens dropped *after* the ViT; "26 retained of 256" | `mask_stage: before_e`: 90% of patches dropped *before* ViT-E blocks, so the encoder itself is trained on about 10% of each state's patches (about 102 of 1024 in RGB16) and sees all patches at inference | If ViT-E becomes the AS-JEPA encoder, masking must be off: the state must be a deterministic function of the whole chunk, and train/test inputs must match. |
| DiT conditioning | "conditioning projection" of a history summary | Cross-attention (blocks 0 and 4) to ViT-C output tokens of the latest origin state, with 3D RoPE aligning each condition token with the same spatial position in the target | SPT's DiT already has a spatially aligned view of the most recent chunk. This is the "recent anchor" bypass the design warns about; AS-JEPA removes it by design. |
| Target | pixels or frozen codec | Direct pixels, losslessly packed (`PixelRGB4Codec`): 16x16 patch x 4 frames = 3072 channels per token; two-level `WorldDiT` (8 blocks, width 768) + `PixelRefiner` (per-pixel PiT, 2 blocks) | Target is fixed data, as the design requires. No codec to freeze. |
| Future offset | random `k` | `distant_target_prob 0.5`, `future_target_choices 2`: target is state `i+1` or `i+2` | Removed in AS-JEPA (one-step prior, renderer renders the current chunk). |
| Noise schedule | "default to SPT's weighting; bias high if the state is under-used" | `noisy = (1-t) y + t eps`, `t = shift(sigmoid(0.8 + 0.8 n))`, automatic shift about 1.74 for a 16x16 grid. Median `t` is about 0.8: already strongly biased toward high noise | The guard "bias toward high noise" is already the default. Velocity sign is `y - eps` (design pseudocode used `eps - y`); use SPT's convention. |
| Condition dropout | proposed as a new tool for CFG on the state | `condition_dropout 0.1` already replaces the condition with a learned null token | Reuse as is: with the anchor still present, it trains the anchor-only renderer needed for the guidance-scale reliance diagnostic. |
| EMA | "no EMA" | `ema.decay 0.99` is a **weight average for evaluation only** (demos and FVD load EMA weights); it is not a target network | Not in conflict with the theory. Keep it off in AS-JEPA runs to honour the decision and report online weights; an EMA-evaluated variant can be reported separately later. |
| Actions | `a_t` into the prior | No actions anywhere in `rep_world`; data are action-free SolarWM videos with camera motion | PushT data with actions must be brought in (Section 4). SolarWM violates the static-content assumption (design 5.6) and is a later, action-free scale test. |
| Optimizer | unspecified | `MuonWithAuxAdam`: every parameter with `ndim >= 2` goes to Muon, including learned tokens | Small new parameters (state queries) would be orthogonalized by Muon. Use AdamW for the first PushT runs (LeWM's choice) and keep Muon as a later scaling option. |
| Code integrity | edit SPT modules | `jobs/worker.py` refuses to start if any file in `provenance/upstream_runtime_hashes.json` changed; `tests/test_portability.py` checks the same hashes | Do not edit `rep_world` files. AS-JEPA goes in a new Native family that imports `rep_world` components (Section 3). |

## 2. Changes to the design that follow from the code

1. **Chunk = one RGB4 state (4 frames at 256x256), not one frame at 64 px.** RGB4 reuses the validated pixel path with no renderer changes: `PixelRefiner` and `PixelRGB4Codec` hard-code 4 frames per plane and a 16x16 grid of 16x16 patches. A one-frame chunk needs an `rgb1` geometry and a generalized refiner (about 40 lines plus tests). On PushT at 10 Hz, 4 frames span 4 control steps, close to LeWM's frameskip 5. Planning then needs a goal *chunk*; a resting goal image repeated 4 times is the natural choice. Move to `rgb1` only if goal-image planning or probing shows the multi-frame chunk is in the way.
2. **The anchor enters by channel concatenation implemented as an added, zero-initialized embedding**, at two places: the coarse patch embedding of `WorldDiT` and the per-pixel embedding of `PixelRefiner`. The in-context alternative (anchor as prefix frames, as `flow.mode: zprev` does) is blocked in this code: `PixelRefiner` and position-aware cross-attention both require exactly one target state with no prefix. Zero-initialized addition `x_embed(y_noisy) + anchor_embed(x_0)` is algebraically channel concatenation with the new weights initialized at zero.
3. **The state reaches the renderer without SPT's per-token LayerNorm.** `WorldDiT.semantic_raw_projector` is `LayerNorm(condition_width) -> Linear`. With 16-dim state tokens, LayerNorm removes each token's mean and scale (2 of 16 degrees of freedom) and makes the renderer insensitive to the magnitude that the fixed noise `sigma_0` is calibrated against. Feeding an already-projected input does not avoid this, because the projector would still normalize each token. AS-JEPA therefore sets `condition_width = d_h` and replaces the projector at construction with `nn.Sequential(nn.Identity(), nn.Linear(d_h, width))`; `WorldDiT._condition` only requires a sequential whose element `[1]` has a weight.
4. **`beta = 1` is not an exact evidence bound with SPT's renderer loss.** SPT's flow loss is a velocity MSE under a logit-normal, shifted time distribution. Kingma and Gao (2023) show a weighted diffusion loss is an ELBO only for weightings monotone in time; this one is not, and its scale is per pixel-element, not nats per chunk. So `beta` has no natural value of 1: treat the objective as rate-distortion (Alemi et al. 2018) with distortion = SPT's flow loss, log the KL in nats per chunk, and sweep `beta` on a log grid. The block-identifiability argument (design Section 5) is unaffected as a statement about the idealized model, but its "beta = 1 or slightly above" condition does not transfer to a numeric `beta` here.
5. **Render a random subset of chunks per step if compute requires it.** The KL chain is cheap and computed on all chunks; the renderer runs on `B x T` targets. Rendering `R < T` uniformly chosen chunks per window and scaling by `T / R` keeps the estimator unbiased.

## 3. Code layout

New Native family, following `runtime/AGENTS.md` (models in `models/`, assembly and step in `tasks/`, differences only in the experiment YAML; no new Trainer, checkpoint manager, or attention backend):

```text
runtime/vforge/models/as_jepa/
  config.py      # StateConfig (K_h, d_h, sigma_0, s_0), PriorConfig, AnchorConfig
  encoder.py     # StateEncoder: rep_world StateViTEncoder (mask ratio 0) + query pooling + mu head
  prior.py       # StatePrior: CausalViTBlock stack over [state tokens, action token] per chunk
  renderer.py    # AnchoredWorldDiT: rep_world WorldDiT + zero-init anchor embeddings
  model.py       # ASJepa root: loss, encode, rollout, render
runtime/vforge/tasks/as_jepa/
  data.py        # PushTChunkDataset: frames + per-chunk action blocks from a JSONL manifest
  train.py       # build_train_components (reuses rep_world loader pattern, EMA off)
  shared.py      # parse_config, model_spec
  eval.py        # rollout + render, action-following metrics
runtime/vforge/conf/experiment/as_jepa_pusht.yaml
runtime/tests/models/as_jepa/  runtime/tests/tasks/as_jepa/
jobs/as_jepa/    # prepare/worker launchers modeled on jobs/prepare_run.py and jobs/worker.py,
                 # with their own hash manifest; the RGB16 worker stays untouched
```

### 3.1 Encoder (`encoder.py`)

```python
class StateEncoder(nn.Module):
    def __init__(self, vit_cfg, geometry, state):         # vit_cfg.context_mask_ratio == 0
        self.vit = StateViTEncoder(vit_cfg, geometry)        # imported from rep_world
        self.queries = nn.Parameter(torch.randn(state.k, vit_cfg.width) * 0.02)
        self.pool = nn.MultiheadAttention(vit_cfg.width, vit_cfg.heads, batch_first=True)
        self.norm = nn.LayerNorm(vit_cfg.width)
        self.mu = nn.Linear(vit_cfg.width, state.d)          # fixed-variance code: no sigma head

    def forward(self, rgb):                                   # [B, 3, S*4, 256, 256]
        tokens = self.vit(rgb)                                # [B, S, 256, W], each chunk alone
        b, s, n, w = tokens.shape
        q = self.queries.expand(b * s, -1, -1)
        pooled, _ = self.pool(q, tokens.reshape(b * s, n, w), tokens.reshape(b * s, n, w))
        return self.mu(self.norm(pooled)).reshape(b, s, -1, self.mu.out_features)  # [B, S, K, d]
```

`StateViTEncoder` raises nothing for mask ratio 0 and returns `norm(hidden)` per state, so it is reused without modification.

### 3.2 Prior (`prior.py`)

One sequence block per chunk: `K` state tokens (`Linear(d -> W)` of the posterior sample) plus one action token (`Linear(4 * 2 -> W)` of the chunk's 4 two-dimensional actions). Use `state_causal_sparsity(length, K + 1)` under FA4 or the boolean block mask under SDPA, exactly as `TokenCausalViT` does. RoPE positions: time = chunk index, row = token index within the block, col = 0. Output heads read the `K` state positions of block `t` and produce `delta`; the mean for chunk `t + 1` is `z_t + delta` (zero-initialize the head so the prior starts as the identity carry). Inference uses `CausalViTBlock.forward_cached` for incremental rollout.

Action alignment must be explicit and tested: chunk `t` holds frames `4t .. 4t+3`; the action token in block `t` is the four actions applied from frame `4t+3` to frame `4t+7`, i.e. those that produce chunk `t + 1`.

### 3.3 Renderer (`renderer.py`)

`AnchoredWorldDiT` subclasses `WorldDiT` and wraps two embeddings with anchor-aware versions:

```python
class _AnchoredEmbed(nn.Module):        # wraps net.x_embedder and net.pixel_refiner.embed
    def __init__(self, base, anchor_branch):
        self.base, self.anchor = base, anchor_branch      # anchor_branch zero-initialized
        self.value = None                                 # set only inside AnchoredWorldDiT.forward
    def forward(self, x):
        if self.value is None:
            raise RuntimeError("anchor missing")
        return self.base(x) + self.anchor(self.value)
```

`AnchoredWorldDiT.forward(noisy, t, *, state, anchor)` reshapes the anchor once into each embedding's input layout, sets both `value` fields in a `try/finally`, calls `WorldDiT.forward(..., condition=state_tokens)`, and clears them. This is safe with the existing activation checkpointing because only `blocks` and refiner blocks are recomputed, never the embeddings. The subclass also swaps `semantic_raw_projector` as described in Section 2, item 3. Settings: `condition_mode: cross_attention`, `cross_attention_scope: global`, `cross_attention_3d_rope: false` (state tokens have no spatial position), no `target_offsets`.

The alternative is an `anchor=None` keyword added to `WorldDiT.forward` and `PixelRefiner.forward`. It is cleaner code but edits hash-pinned files and breaks the RGB16 worker's integrity check, so it is not the default.

### 3.4 Loss (`model.py`)

```python
mu = encoder(rgb)                                         # [B, T+1, K, d]
z = mu + sigma0 * torch.randn_like(mu)
m = prior(z[:, :-1], actions[:, :-1])                     # mean for z[:, 1:]
rate0 = 0.5 * mu[:, 0].square().sum((-2, -1))             # entry: KL to N(0, I) up to constants
rate = (mu[:, 1:] - m).square().sum((-2, -1)) / (2 * s0**2)   # surprise per chunk, nats + const
kl = (rate0 + rate.sum(1)).mean()                         # nats per window

y = target_chunks(rgb)[:, 1:]                             # packed RGB4, chunks 1..T
t = model.draw_timesteps(...)                             # SPT schedule, unchanged
noisy = (1 - t) * y + t * eps
v = renderer(noisy, t, state=z[:, 1:], anchor=packed(rgb)[:, 0])
fm = (y - eps - v).square().mean()                        # SPT sign convention

loss = fm + beta(step) * kl / NORM                        # NORM: elements per window, fixed
```

`NORM` (the number of pixel elements per window) puts `beta` on a per-element scale so its sweep range does not depend on window length. Log `kl` in nats per chunk, `rate0` separately, and the per-dimension KL averaged over the batch (effective state size). `beta(step)` ramps linearly from 0 over the first 5% of steps.

## 4. Data: PushT with actions

The PushT environment, `WeakPolicy`, and collectors exist in `~/code/sparse_jepa` (`stable_worldmodel/envs/pusht`, `scripts/data/collect_weak_pusht.py`, variation space in `docs/envs/pusht.md`). They write `.lance` shards at 224 px.

1. Collect with `resolution=256`, `WeakPolicy`, per-episode variation of agent, block and background colours, goal-overlay pose (render-only factors) and block shape and scale (dynamics-relevant). Record the sampled factors and per-frame simulator state for probes. Add the transient random-dot overlay as a separate dataset variant.
2. Export each episode to an MP4 (lossless or high-quality) plus an `actions.npy` and a `factors.json`; write a JSONL manifest in the style `VideoClipDataset` reads (`rgb_path`, plus `actions_path`, `start_frame`). Split by episode before windowing.
3. `PushTChunkDataset` returns `rgb [3, (T+1)*4, 256, 256]` and `actions [T+1, 4, 2]` with a fixed resume contract, mirroring `rep_world/data.py`.

Start with `T + 1 = 8` chunks (32 frames, 3.2 s).

Datasets and exports live outside Git, under a path the user chooses (not `/efs/prescottsun/SPT`, which belongs to the SPT maintainers' runs).

## 5. Starting configuration (proposed, not tuned)

| Module | Setting | Reason |
| --- | --- | --- |
| Geometry | `pixel/rgb4`, 256x256, 4 frames per chunk | validated pixel path, no refiner changes |
| ViT-E | width 384, depth 6, heads 6, patch 16, mask 0 | PushT is simple; LeWM uses ViT-tiny |
| State | `K = 4` tokens x `d = 16`, `sigma_0 = 0.1`, `s_0 = 0.1` initially | design default (64 dims) |
| Prior | 6 `CausalViTBlock`s, width 384 | LeWM predictor depth |
| Renderer | `WorldDiT` width 384, depth 8, heads 6, cross-attention every 2, `pixel_depth 2`, `pixel_width 16`, `condition_dropout 0.1` | SPT structure at smaller width |
| Optimizer | AdamW, lr 1e-4 warmup 100, weight decay 0, grad clip 1.0, bf16 | standard; Muon later |
| EMA | off | design decision; SPT's EMA is evaluation-only |
| `beta` | ramp over 5%, then sweep on a log grid | the main knob |

Profile one step before fixing batch size and steps; the RGB16 baseline's 1.65 s per step on 8 H200s is for a 423M-parameter model at 4x the tokens and is not a guide.

## 6. Order of work and checks

Follow the feedback ladder in `runtime/AGENTS.md`; do not go to GPUs until the CPU contracts pass.

1. **Data export and loader.** Check: decoded frames match the simulator renders; action alignment test (a known action sequence moves the agent in the recorded direction between the chunks the prior links).
2. **Modules with CPU contract tests**, each with a small config:
   - encoder output for chunk `t` is unchanged when other chunks change (per-chunk property);
   - prior output for block `t` is unchanged when chunks `> t` or actions `> t` change (causality);
   - renderer output is unchanged by history and actions (it receives neither); zero-initialized anchor branch gives exactly `WorldDiT` output at initialization;
   - gradients reach encoder, prior and renderer from the loss, with no detached path;
   - `kl` equals a hand-computed value on a fixed tensor.
3. **Tier-1 run on one GPU**: a few hundred steps on a small PushT shard; loss finite, `kl` per chunk moving, samples from pure noise change when the state is zeroed.
4. **First real run**, then the `beta` sweep and the two ablations from design Section 9 (no anchor; deterministic MSE renderer).
5. **Evaluation** (`tasks/as_jepa/eval.py`): latent rollout from a prefix with the prior's means under recorded actions, rendered with the first chunk as anchor; action following from matched simulator states; probes and swap test from design Section 9. Planning comes after action following works, through an adapter exposing LeWM's `encode`/`rollout` signatures so the `stable_worldmodel` planners and `GoalMSE` can run unchanged.

## 7. Decisions needed before coding

- **Compute placement.** This host is a login node. Data collection and every training step need an allocated machine; dev3 and dev8 are allocated to the SPT maintainers' runs and must not be used without their agreement.
- **Chunk length.** RGB4 (recommended, no renderer changes) or a new `rgb1` geometry (matches the design's one-frame state and single-image planning; about 40 lines plus tests).
- **Anchor wiring.** Wrapper subclass (recommended, leaves pinned files untouched) or a keyword added to the pinned `WorldDiT`/`PixelRefiner` files with refreshed hashes.
