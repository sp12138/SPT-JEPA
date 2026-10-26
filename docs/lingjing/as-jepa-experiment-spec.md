# AS-JEPA on PushT: experiment and implementation spec (handoff)

Written 2026-10-26 for the collaborator who will implement and run the first round of AS-JEPA experiments. It is self-contained: follow it as written. Where a choice is left to you, Section 13 lists it; any other change needs our agreement first. The idea and its theory are in [as-jepa-story.md](as-jepa-story.md); you do not need them to execute this spec, but each component below names the design reason it serves, so that implementation shortcuts do not silently break the method. Decisions behind this spec are logged in [discussions/2026-10-25-spt-code-review-of-plan.md](discussions/2026-10-25-spt-code-review-of-plan.md).

## 0. Overview

**What AS-JEPA is, operationally.** A per-frame encoder maps each frame to a small noisy state. A causal predictor (the prior) reads past states and actions and predicts the next state's mean. A flow-matching renderer draws each frame from its own state and one anchor frame. Training minimizes one objective: the renderer's flow loss plus `beta` times a code length, which is the first state's cost against `N(0, I)` plus each later state's squared prediction error under the prior. There is no augmentation, EMA, stop-gradient, target network, SIGReg or masking in AS-JEPA.

**What this round must show, on PushT only.**

- Main criteria: generation (rendered rollouts follow the actions) and planning (LeWM's PushT protocol).
- For understanding, not gating: linear probes of the state, appearance and transient purity, rate diagnostics.

**Two branches, run in parallel, as two ladders that meet at AS-JEPA.**

- **Branch L (JEPA side)**, in LeWM's code base (stable-worldmodel):
  - L0: LeWM as published.
  - L1: LeWM plus the anchored renderer, with SIGReg kept and nothing priced.
  - L2: AS-JEPA built from LeWM's encoder and predictor plus SPT's renderer.
- **Branch S (generator side)**, in SPT's code base (vforge Native):
  - S0: SPT with actions, on PushT.
  - S1: S0 plus the anchor.
  - S2: AS-JEPA built from SPT's modules.

Both branches use the same PushT data, the same 64-pixel rendering target, the same renderer architecture, and one shared evaluation package. Rungs inside a branch are independent runs and should run in parallel once implemented.

**Two anchor arms for every AS-JEPA model (L2, S2), both main arms.**

- `first`: the anchor is the window's first frame.
- `far`: the anchor is a frame from a distant time in the same episode, and the window's first frame still starts the state chain but is never rendered.

## 1. Repositories, branches, environments

### 1.1 Repositories and branches

- **SPT-JEPA** (this repository). Create three branches from `lingjing/anchored-state-jepa`:
  - `as-jepa/shared`: shared model components under `runtime/vforge/models/as_jepa/`, and data plus evaluation tools under `as_jepa_tools/`;
  - `as-jepa/lewm`: from `as-jepa/shared`; adds the branch-L package `as_jepa_lewm/`;
  - `as-jepa/spt`: from `as-jepa/shared`; adds `runtime/vforge/tasks/as_jepa/` and `jobs/as_jepa/`.

  Merge `as-jepa/shared` into both implementation branches whenever it changes.
- **stable-worldmodel** (public, `https://github.com/galilai-group/stable-worldmodel`). Pin the commit you start from and record its hash in every report. It provides the PushT environment and its variation space, `WeakPolicy`, the LeWM reference training script `scripts/train/lewm.py` with its config `scripts/train/config/lewm.yaml`, the LeWM model `stable_worldmodel/wm/lewm/`, `SIGReg`, the planners, and the PushT planning evaluation `scripts/plan/eval_wm.py` with `scripts/plan/config/pusht.yaml`. Do not modify it; install it as a package.

### 1.2 Hard rule for SPT code

Do not edit any file listed in `provenance/upstream_runtime_hashes.json`. `jobs/worker.py` and `tests/test_portability.py` check those hashes, and the RGB16 baseline must keep working. Everything in this spec is new files. New files under `jobs/` must not contain the machine-specific path strings that `tests/test_portability.py` rejects (for example `/threed-code/` and `/efs/stormyxie`). Two existing extension points make that possible:

- `vforge.tasks.registry.register(name, ModelTasks(...))`, the public hook "for out-of-tree plugins";
- subclassing the `rep_world` modules.

### 1.3 Environments

- **Env S:** SPT's validated environment (Python 3.12, torch 2.9.1+cu130, FlashAttention-4, `pip install -e ./runtime`), as in the repository README.
- **Env L:** a clone of env S plus `pip install -e "<stable-worldmodel at the pinned commit>[train,data,format]"` and the PushT packages `pygame pymunk shapely opencv-python-headless` (the PushT environment imports them at module load). Install with a constraints file pinning torch and torchvision to env S's versions, then confirm torch is still 2.9.1+cu130. Check that `import stable_worldmodel`, `swm.World("swm/PushT-v1")` and `from vforge.models.rep_world.world import WorldDiT` all work. Inside any process that runs SPT modules outside SPT's trainer, call once at startup:

  ```python
  import torch
  from vforge.ops import OpsConfig, initialize_ops
  initialize_ops(OpsConfig(backend="sdpa", allow_fallback=False), device=torch.device("cuda", local_rank), parallel_state=None)
  ```

  SPT's own CPU tests do the same (`runtime/tests/models/rep_world/test_cross_attention_order.py`). With no parallel state, sequence parallelism is off.

### 1.4 SPT smoke test (no baseline reproduction)

Run the repository's CPU tests (`PYTHONPATH=runtime python -B -m unittest discover -s tests`) and the `rep_world` model tests under `runtime/tests/models/rep_world/`. Then run a short RGB16 training smoke (at most 100 updates) on one 8-GPU node following the README's procedure, record step time and peak memory, and stop. Do not run the 50k-step baseline, FVD-2048, or SolarWM experiments in this round.

## 2. Data

### 2.1 Datasets

| Name | Content | Source | Used for |
| --- | --- | --- | --- |
| `D-std` | standard PushT, fixed appearance | LeWM's public PushT dataset (`pusht_expert_train`, Hugging Face `galilai-group/lewm-pusht`), as used by `scripts/train/lewm.py` | all rungs; the benchmark; planning |
| `D-var` | PushT with appearance randomized per episode | collected (Section 2.2) | appearance purity, generation under varied appearance |
| `D-dot` | `D-std` frames with a random dot redrawn every frame | derived in the loader (Section 2.3) | transient purity |

Use PushT only. Do not vary block shape or scale, and do not use other environments, in this round.

### 2.2 Collecting `D-var`

Use stable-worldmodel's collection API as `scripts/data/collect_pusht_fov.py` does:

```python
world = swm.World("swm/PushT-v1", num_envs=10, max_episode_steps=<same as D-std episodes; report it>,
                  image_shape=(224, 224), render_mode="rgb_array")
world.set_policy(WeakPolicy(dist_constraint=100))
world.collect(path, episodes=<equal to the number of D-std episodes>, seed=20261026,
              options={"variation": ("agent.color", "block.color", "background.color",
                                     "goal.color", "goal.position", "goal.angle",
                                     "agent.start_position", "block.start_position", "block.angle")})
```

Check the names against `swm fovs PushT-v1` at your pinned commit. Store each episode's sampled variation values (colours, goal pose) alongside the data; probes need them. Block shape, block scale, agent shape and agent scale stay at their defaults.

`D-std` was collected with a different policy, so `D-var` differs in dynamics distribution as well as appearance. That is acceptable: `D-var` is used to train and evaluate separate models, not mixed with `D-std`.

### 2.3 `D-dot`

Applied in the data loader to `D-std` frames at 224 pixels, before any resizing:

- one filled disc of radius 9 pixels at 224 (about 2.5 at 64);
- centre uniform over the image;
- colour uniform in RGB;
- redrawn independently for every frame from `torch.Generator().manual_seed(hash((episode_idx, step_idx, 7)) & 0x7FFFFFFF)`.

The same frame therefore always shows the same dot, at both resolutions. Record the dot's centre and colour for probes.

### 2.4 Splits

- Split `D-std` and `D-var` by episode with seed 0: 95% train, 5% test. All our rungs (including L0 retrained) train on the train episodes; generation, probes and galleries use test episodes only.
- Planning follows LeWM's protocol unchanged, which samples start and goal states from `pusht_expert_train` itself (Section 4.8). Report that as is.

### 2.5 Frames, actions, windows

- **Frameskip 5.** Consecutive kept frames are 5 environment steps apart. The action for step `t` is the 5 raw 2-D actions taken after frame `t`, concatenated to a 10-D vector. It "leaves frame `t`" and produces frame `t+1`, as in LeWM. Normalize actions with stable-worldmodel's column normalizer, as `scripts/train/lewm.py` does.
- **Encoder input (branch L):** 224 pixels, LeWM's preprocessing (`get_img_preprocessor`: to image, ImageNet statistics, resize 224).
- **Renderer target and anchor (both branches), and encoder input (branch S):** 64 pixels, computed from the raw `uint8` 224-pixel frame by exactly these functions (shared, in `as_jepa_tools/data/frames.py`). The area filter is rounded back to `uint8`, so both branches see identical values:

  ```python
  def to64(frames_uint8):            # [N, 3, 224, 224] uint8 -> [N, 3, 64, 64] uint8
      x = torch.nn.functional.interpolate(frames_uint8.float(), size=(64, 64), mode="area")
      return x.round().clamp_(0, 255).to(torch.uint8)

  def unit(frames64_uint8):          # uint8 -> float in [-1, 1]
      return frames64_uint8.float() / 127.5 - 1.0
  ```

  Everywhere below, "64-pixel frame" means `unit(to64(raw))`.

- **Packing for SPT modules** (shared, `runtime/vforge/models/as_jepa/pixels.py`). A 64-pixel frame is a 16x16 grid of 4x4 patches, 48 channels per token, one temporal plane:

  ```python
  def pack64(frames):                # [N, 3, 64, 64] -> [N, 48, 1, 16, 16]
      n = frames.shape[0]
      x = frames.reshape(n, 3, 16, 4, 16, 4)          # c, gy, py, gx, px
      return x.permute(0, 3, 5, 1, 2, 4).reshape(n, 48, 1, 16, 16)

  def unpack64(packed):              # [N, 48, 1, 16, 16] -> [N, 3, 64, 64]
      n = packed.shape[0]
      x = packed.reshape(n, 4, 4, 3, 16, 16).permute(0, 3, 4, 1, 5, 2)
      return x.reshape(n, 3, 64, 64)
  ```

- **Windows:**
  - L1, L2 and S2: 8 kept frames (`T = 7` rendered frames).
  - Window ablations: 4 and 16 frames.
  - L0: LeWM's own window (history 3 plus 1).
  - S0 and S1: SPT's own structure, 10 kept frames.
- **Anchor arms (L2, S2, and their ablations):**
  - `first`: anchor = window frame 0.
  - `far`: anchor = a frame of the same episode, at frameskip-aligned steps, at least 8 kept frames (40 environment steps) away from both ends of the window, before or after. Choose uniformly among the eligible frames with `torch.Generator().manual_seed(hash((episode_idx, window_start, epoch)) & 0x7FFFFFFF)`. If none is eligible, use the farthest frame of the episode, and count how often that happens.
  - In both arms, window frame 0 is encoded as the first state of the chain and is never a rendering target.
- **Branch S export.** Write each dataset once as memory-mapped arrays: the `uint8` output of `to64` as `[N_rows, 64, 64, 3]`, actions, the 7-D simulator state, and an index table (episode id, step index, row). Both branches must produce bit-identical 64-pixel targets. Test this on 1000 random rows.

### 2.6 Dataset card

For each dataset, report:

- episode count and length distribution;
- number of valid windows per window length;
- fraction of 8-frame windows in which the block moves by more than 10 pixels (in the 512-pixel workspace);
- for `far`, the fallback rate.

The PushT state is `[agent_x, agent_y, block_x, block_y, block_angle, agent_vx, agent_vy]` in a 512-pixel workspace.

## 3. Shared model components (`runtime/vforge/models/as_jepa/`, branch `as-jepa/shared`)

Pure model code, importable from both environments. No training-loop code here.

### 3.1 Flow utilities (`flow.py`)

The renderer is trained and sampled exactly as SPT does.

```python
import torch
from vforge.models.rep_world.model import shift_timestep

SHIFT = 2.667 * (16 ** 2 / 600) ** 0.5            # SPT's automatic shift for a 16x16 token grid, about 1.742

def draw_timesteps(n, device, uniform=None):       # SPT: logit-normal(0.8, 0.8), then shifted; t = 1 is pure noise
    u = uniform if uniform is not None else torch.rand(n, 1, device=device).clamp_(1e-5, 1 - 1e-5)
    t = torch.sigmoid(0.8 + 0.8 * torch.special.ndtri(u.double())).float()
    return shift_timestep(t, SHIFT)

def flow_loss(renderer, y, state, anchor):        # y, anchor: [N, 48, 1, 16, 16]; state: [N, K, d]
    t = draw_timesteps(y.shape[0], y.device)
    eps = torch.randn_like(y)
    tv = t.view(-1, 1, 1, 1, 1)
    with renderer.anchored(anchor):
        v = renderer(((1 - tv) * y + tv * eps), t, condition=state)
    return (y - eps - v).square().mean()          # SPT's sign: the target velocity is y - eps

@torch.no_grad()
def sample(renderer, state, anchor, steps=50, cfg_scale=1.0, generator=None):
    """SPT's Euler sampler (RepWorldInference._denoise), with optional guidance on the state.
    Call with the renderer in eval mode: in train mode WorldDiT drops the state for 10% of samples."""
    n = state.shape[0]
    z = torch.randn((n, 48, 1, 16, 16), device=state.device, generator=generator)
    sched = shift_timestep(torch.linspace(1, 0, steps + 1, device=state.device), SHIFT)
    drop = torch.ones(n, dtype=torch.bool, device=state.device)
    with renderer.anchored(anchor):
        for t_now, t_next in zip(sched[:-1], sched[1:]):
            t = t_now.expand(n, 1)
            v = renderer(z, t, condition=state)
            if cfg_scale > 1 and 0.2 < float(t_now) <= 1.0:      # SPT's guidance interval (0.2, 1.0]
                vu = renderer(z, t, condition=state, condition_drop=drop)
                v = vu + cfg_scale * (v - vu)
            z = z - (t_next - t_now) * v
    return z
```

Check that `draw_timesteps` matches `RepWorld.draw_timesteps` for identical uniforms (Section 9.1).

### 3.2 Anchored renderer (`renderer.py`)

SPT's `WorldDiT` with two changes and no edits to its file. Design reason: the anchor is side information that reaches the renderer only; the state must reach the renderer without per-token normalization, which would remove each token's mean and scale and stop the code length from working. The state tokens also get learned slot embeddings after the linear projection: cross-attention without positions treats its context as an unordered set, which would discard which token is which.

```python
import contextlib
import torch
from torch import nn
from vforge.models.rep_world.layers import PatchEmbed3D
from vforge.models.rep_world.world import WorldDiT

class SlotLinear(nn.Linear):
    """Linear projection of state tokens plus a learned embedding per token slot. No normalization."""
    def __init__(self, in_features, out_features, num_slots):
        super().__init__(in_features, out_features)
        self.slot = nn.Parameter(torch.zeros(num_slots, out_features))
    def forward(self, x):                          # x: [N, num_slots, in_features]
        return super().forward(x) + self.slot[: x.shape[1]]

class AnchoredWorldDiT(WorldDiT):
    """WorldDiT whose patch embedding also reads an anchor frame: channel concatenation
    implemented as a zero-initialized added embedding. state_dim=None keeps SPT's
    condition projector (used by S1); an integer replaces it by SlotLinear."""

    def __init__(self, config, state_dim=None, num_slots=None, use_anchor=True):
        super().__init__(config)
        c = config
        self.use_anchor = use_anchor
        if use_anchor:
            self.anchor_embed = PatchEmbed3D((c.state_planes, c.grid, c.grid), (1, c.patch_size, c.patch_size), c.channels, c.width)
            self.x_embedder.register_forward_hook(self._add_anchor)
        if state_dim is not None:
            if c.condition_width != state_dim:
                raise ValueError("condition_width must equal the state token width")
            self.semantic_raw_projector = nn.Sequential(nn.Identity(), SlotLinear(state_dim, c.width, num_slots))
        self._anchor = None
        self._init_added()

    def _init_added(self):
        if getattr(self, "use_anchor", False) and hasattr(self, "anchor_embed"):
            nn.init.zeros_(self.anchor_embed.proj.weight)
            nn.init.zeros_(self.anchor_embed.proj.bias)
        proj = self.semantic_raw_projector[1]
        if isinstance(proj, SlotLinear):
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)
            nn.init.normal_(proj.slot, std=0.02)

    def initialize_weights(self):                  # WorldDiT.__init__ calls this before the added modules exist
        super().initialize_weights()
        if hasattr(self, "use_anchor"):
            self._init_added()

    def _add_anchor(self, module, inputs, output):
        if self._anchor is None:
            raise RuntimeError("AnchoredWorldDiT used outside renderer.anchored(...)")
        return output + self.anchor_embed(self._anchor)

    @contextlib.contextmanager
    def anchored(self, anchor):                    # anchor: [N, 48, 1, 16, 16], one per rendered target
        self._anchor = anchor if self.use_anchor else None
        try:
            yield self
        finally:
            self._anchor = None
```

Notes:

- Activation checkpointing in `WorldDiT` recomputes only blocks, never `x_embedder`, so the hook is safe.
- The `noanchor` ablations use `use_anchor=False`: `anchor_embed` is not created, the hook is not registered, and `anchored(...)` passes nothing. Everything else, including the projector, is unchanged.
- `num_slots` is the number of state tokens: 4 for L2 and S2, 12 for L1.
- WorldDiT zero-initializes its output layer, so a freshly built renderer outputs exactly zero and passes no gradient upstream on its first step. That is intended for training; Section 9 says how tests must account for it.

**Renderer configuration**, identical for L1, L2 and S2 (S1 differs only where noted in Section 5.3):

```python
WorldConfig(channels=48, state_planes=1, grid=16, width=384, depth=12, heads=6, patch_size=1,
            mlp_ratio=4.0, qk_norm=True, adaln_mode="adaln_lora", adaln_lora_dim=256,
            condition_width=16, condition_dropout=0.1, cross_attention_every=2,
            cross_attention_position="before_self", cross_attention_scope="global",
            cross_attention_3d_rope=False, condition_mode="cross_attention", pixel_depth=0)
```

That is DiT-S size, with the plain output layer (no pixel refiner) and cross-attention in blocks 0, 2, 4, 6, 8 and 10. `condition_dropout=0.1` replaces the state by a learned null token for 10% of samples, which enables guidance on the state. The state tokens have no spatial position, so cross-attention uses no spatial RoPE.

### 3.3 Code lengths (`codes.py`)

The fixed-variance Gaussian code: `sigma_0 = s_0 = 0.1`, in fp32.

```python
import math

LOG_TERM = lambda sigma0: 0.5 * sigma0 ** 2 - 0.5 - math.log(sigma0)   # per-dimension constant of KL(N(mu, sigma0^2) || N(0, 1))

def entry_nats(mu0, sigma0):                     # mu0: [B, K, d] -> full KL to N(0, I) per window, [B]
    mu0 = mu0.float()
    return 0.5 * mu0.pow(2).sum((-2, -1)) + mu0[0].numel() * LOG_TERM(sigma0)

def surprise_nats(mu, m, s0):                    # mu, m: [B, T, K, d] -> [B, T]; exact KL when sigma0 == s0
    return (mu.float() - m.float()).pow(2).sum((-2, -1)) / (2 * s0 ** 2)
```

The constant in `entry_nats` does not affect gradients. Keep it, so that the logged entry cost is the true KL in nats.

### 3.4 State prior for branch S (`prior.py`)

Used only by S2; branch L uses LeWM's predictor (Section 4.4). One sequence block per step: `K` state tokens plus one action token.

```python
from vforge.models.rep_world.config import ConditionerConfig
from vforge.models.rep_world.conditioner import CausalViTBlock
from vforge.models.rep_world.sparsity import state_causal_sparsity, use_segmented_attention

class StatePrior(nn.Module):
    def __init__(self, K=4, d=16, action_dim=10, width=384, depth=6, heads=6):
        super().__init__()
        cfg = ConditionerConfig(width=width, heads=heads, mlp_ratio=4.0)
        self.K = K
        self.state_in = nn.Linear(d, width)
        self.action_in = nn.Linear(action_dim, width)
        self.slot = nn.Parameter(torch.zeros(K + 1, width))          # K state slots plus the action slot
        self.blocks = nn.ModuleList(CausalViTBlock(cfg) for _ in range(depth))
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.delta = nn.Linear(width, d)                             # zero-initialized: the prior starts as the identity carry

    def forward(self, s, a):                     # s: [B, T, K, d] past states; a: [B, T, action_dim] actions leaving them
        B, T, K, d = s.shape
        tokens = torch.cat([self.state_in(s), self.action_in(a)[:, :, None]], dim=2) + self.slot   # [B, T, K+1, W]
        h = tokens.reshape(B, T * (K + 1), -1)
        L = T * (K + 1)
        step = torch.arange(L, device=s.device) // (K + 1)
        mask = state_causal_sparsity(L, K + 1) if use_segmented_attention() else step[:, None] >= step[None, :]
        positions = (step.float(), (torch.arange(L, device=s.device) % (K + 1)).float(), torch.zeros(L, device=s.device))
        for block in self.blocks:
            h = block(h, mask=mask, positions=positions)
        out = self.norm(h).reshape(B, T, K + 1, -1)[:, :, :K]        # state-slot outputs of each step
        return s + self.delta(out)               # means of s_1 .. s_T
```

Initialization: `trunc_normal_(std=0.02)` for all linear weights, zeros for biases, `slot` normal with std 0.02, and the residual scaling of `vforge.models.rep_world.split_conditioner._initialize` for the blocks. Then zero `delta` (weight and bias) last.

### 3.5 Query pooling (`pooling.py`)

```python
class QueryPool(nn.Module):                      # K learned queries cross-attend to one frame's tokens
    def __init__(self, width, heads, K=4, d=16):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(K, width) * 0.02)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, d)          # plain linear head: no normalization after it
    def forward(self, tokens):                   # [N, L, width] -> mu: [N, K, d]
        q = self.queries.expand(tokens.shape[0], -1, -1)
        pooled, _ = self.attn(q, tokens, tokens, need_weights=False)
        return self.head(self.norm(pooled))
    def initialize(self):                        # must run after meta-device materialization (branch S)
        nn.init.normal_(self.queries, std=0.02)
        self.attn._reset_parameters()
        nn.init.ones_(self.norm.weight); nn.init.zeros_(self.norm.bias)
        nn.init.trunc_normal_(self.head.weight, std=0.02); nn.init.zeros_(self.head.bias)
```

`tokens` is 3-D: flatten batch and time before pooling, `pool(tokens.flatten(0, 1)).view(B, W, K, d)`. Design reason: a small state of fixed slot identity. Nothing may normalize `mu` itself (batch or layer norm after `head`); that would pin each dimension's spread and stop the entry price from switching dimensions off.

## 4. Branch L: the LeWM ladder (`as_jepa_lewm/`, branch `as-jepa/lewm`)

Mirror the structure of stable-worldmodel's `scripts/train/lewm.py` (Lightning via `stable_pretraining`, `save_pretrained` checkpoints with a model config carrying `_target_`). Models must be loadable by `swm.wm.utils.load_pretrained` and plannable by `scripts/plan/eval_wm.py`, which needs `encode(info)` and `rollout(info, action_sequence)` with LeWM's signatures.

### 4.1 L0: LeWM as published

Copy `scripts/train/lewm.py` and its config into `as_jepa_lewm/train_lewm.py` and change only the dataset split (upstream splits windows at random; Section 2.4 needs an episode split) and the output paths. At the time of writing, upstream uses:

- encoder: ViT-tiny, 224 pixels, patch 14, `embed_dim 192`;
- predictor: depth 6, 16 heads, `mlp_dim 2048`, dropout 0.1;
- windows: `history_size 3`, `num_preds 1`, frameskip 5;
- loss: `SIGReg` weight 0.09 (17 knots, 1024 projections);
- optimization: AdamW lr `5e-5`, weight decay `1e-3`, 1% warmup then cosine, batch 128, 100 epochs, bf16, gradient clip 1.0.

Train three seeds on `D-std` and one seed each on `D-var` and `D-dot`.

### 4.2 Data module for L1 and L2

A window dataset over the pinned stable-worldmodel dataset, using its row access (`get_col_data`, `get_row_data`, as `scripts/plan/eval_wm.py` does) and an index table built once from `episode_idx` and `step_idx`. Each sample returns:

- `pixels`: `[W, 3, 224, 224]` after LeWM's preprocessing (encoder input);
- `pixels64`: `[W, 3, 64, 64]`, `unit(to64(raw))` on the raw `uint8` frames, computed before any normalization;
- `anchor64`: `[3, 64, 64]` (Section 2.5);
- `action`: `[W, 10]`, normalized;
- `state`: `[W, 7]`;
- `episode_idx`, `start_step`.

`W` is the window length. Sampling is uniform over valid windows of the train episodes.

### 4.3 L1: LeWM plus the anchored renderer, SIGReg kept

Same encoder, projector, action encoder and predictor as L0, except the predictor's `num_frames = 7`, so it reads up to 7 past embeddings over the 8-frame window. Loss per window:

```python
out = model.encode(batch)                              # emb: [B, 8, 192], LeWM's encoder + BatchNorm projector
emb, act = out["emb"], out["act_emb"]
pred = model.predict(emb[:, :-1], act[:, :-1])         # [B, 7, 192]
pred_loss = (pred - emb[:, 1:]).pow(2).mean()          # LeWM's loss, attached target
sigreg = sigreg_fn(emb.transpose(0, 1))                # LeWM's SIGReg, weight 0.09
state = emb[:, 1:].reshape(B * 7, 12, 16)              # the embedding read as 12 tokens of 16
y = pack64(batch["pixels64"][:, 1:].flatten(0, 1))
anchor = pack64(batch["pixels64"][:, :1]).repeat_interleave(7, dim=0)   # anchor = window frame 0
fm = flow_loss(renderer, y, state, anchor)
loss = pred_loss + 0.09 * sigreg + fm
```

The renderer reads the encoded current embedding, not a prediction, as L2 does. The renderer is `AnchoredWorldDiT(<renderer configuration>, state_dim=16, num_slots=12)`. Run the `first` anchor only. Train three seeds on `D-std` and one each on `D-var` and `D-dot`.

### 4.4 L2: AS-JEPA in LeWM's modules

Components:

- **Encoder:** LeWM's ViT-tiny (224, patch 14, `embed_dim 192`, random init), all 257 output tokens. Then `QueryPool(width=192, heads=3, K=4, d=16)`, giving `mu: [B, W, 4, 16]`. No LeWM projector.
- **Action encoder:** LeWM's `Embedder(input_dim=10, smoothed_dim=10, emb_dim=64, mlp_scale=4)`. The width must be 64, not LeWM's 192: LeWM's `Transformer` projects the action embedding with `cond_proj = Linear(input_dim, hidden_dim)` whenever the state width (`input_dim`, here 64) differs from `hidden_dim`, so the action embedding must have the state's width.
- **Prior:** LeWM's `Predictor(num_frames=7, depth=6, heads=16, mlp_dim=2048, input_dim=64, hidden_dim=192, output_dim=64, dim_head=64, dropout=0.1, emb_dropout=0.0)` on the flattened 64-D state, with a zero-initialized `delta_head = nn.Linear(64, 64)`. No `pred_proj`.
- **Renderer:** `AnchoredWorldDiT(<renderer configuration>, state_dim=16, num_slots=4)`.

Loss per window (`W = T + 1` frames):

```python
mu = encoder(batch["pixels"])                          # [B, W, 4, 16]; each frame encoded alone
s = mu + 0.1 * torch.randn_like(mu)                    # the code: fixed noise sigma_0 = 0.1
flat = s.flatten(2)                                    # [B, W, 64]
m = (flat[:, :-1] + delta_head(predictor(flat[:, :-1], act_emb[:, :-1]))).view_as(mu[:, 1:])
entry = entry_nats(mu[:, 0], 0.1)                      # [B]
surprise = surprise_nats(mu[:, 1:], m, 0.1)            # [B, T]
y = pack64(batch["pixels64"][:, 1:].flatten(0, 1))     # frames 1..T; frame 0 is never rendered
anchor = pack64(batch["anchor64"]).repeat_interleave(T, dim=0)
fm = flow_loss(renderer, y, s[:, 1:].flatten(0, 1), anchor)
rate = beta * (entry + surprise.sum(1))                # beta_entry = beta_surprise = beta in this round
loss = fm + ramp(step) * rate.mean() / (T * 12288)     # 12288 = 64 * 64 * 3 elements per rendered frame
```

- `ramp(step)` rises linearly from 0 to 1 over the first 5% of optimizer steps and stays at 1.
- Dividing by `T * 12288` puts `beta` on a per-pixel-element scale, so the same `beta` means the same price for every window length.
- The renderer reads the posterior sample `s_t` (noisy), never `mu` and never the prior's output.
- The prior reads noisy past states and actions only.

### 4.5 Optimization for L1 and L2

AdamW with two parameter groups:

- encoder, pooling, action encoder and prior: lr `5e-5`, weight decay `1e-3`, betas `(0.9, 0.999)` (LeWM's);
- renderer: lr `1e-4`, weight decay 0, betas `(0.9, 0.95)` (SPT's).

Schedule: exactly L0's scheduler configuration, through the same code path, for both groups. Upstream configures `LinearWarmupCosineAnnealingLR` with warmup and maximum given in optimizer steps but `interval: epoch`; depending on the installed `stable-pretraining`, that can advance once per epoch. Log the learning rate at every step in the smoke runs (Section 9.3) for L0, L1 and L2. If the curve is degenerate (zero for the first epoch, or still warming up at the end), stop and ask us before Wave 1; we will then switch all branch-L rungs, L0 included, to a per-step schedule. Global batch 128 windows, bf16 autocast (rate terms in fp32), gradient clip 1.0.

Budget: the full budget is 100 epochs over the train windows of the run's window length. The screening budget is 40% of that.

Checkpoints: every 10% and at the end, in `save_pretrained` format with the full model config.

### 4.6 Inference interface for L1 and L2 (`as_jepa_lewm/model.py`)

- `encode(info)`: `info["emb"] = mu.flatten(2)` with shape `[B, H, 64]` for L2, or LeWM's embedding for L1; and `info["act_emb"]` when actions are present. Same keys and shapes as LeWM.
- `rollout(info, action_sequence, history_size=None)`: LeWM's rollout loop, with each step `next = last + delta_head(predictor(context)[:, -1])` for L2. `history_size` defaults to `self.predictor.num_frames`, as in LeWM (the planner calls `rollout` without it, and the predictor's positional table is only `num_frames` long). It keeps the last `history_size` states as context and returns `info["predicted_emb"]` of shape `[B, S, H + T, 64]`. For planning, roll out prior means (no noise). For generation, Section 7.2 specifies means and samples.
- `render(states, anchor64, *, steps=50, cfg_scale=1.0, seed)`: `flow.sample` in eval mode, then `unpack64`.

### 4.7 L ablations (all on `D-std`, at `beta*` from Section 10)

| Id | Change from L2 | Arms |
| --- | --- | --- |
| `L2-noanchor` | renderer without the anchor branch | none |
| `L2-mse` | deterministic renderer: call it with zero input and `t = 1` and train `(v - y)^2`, i.e. a direct regressor of `y` | `first` |
| `L2-w4` | 4-frame windows (`T = 3`), predictor `num_frames = 3` | `first`, `far` |
| `L2-w16` | 16-frame windows (`T = 15`), predictor `num_frames = 15` | `first`, `far` |

### 4.8 Planning for branch L

Run `scripts/plan/eval_wm.py` with `scripts/plan/config/pusht.yaml` unchanged except `policy` (the checkpoint) and the history length, set with the Hydra override `+plan_config.history_len=3` for every model (the key is commented out upstream). Current upstream settings:

- CEM solver: 300 samples, 30 iterations, top 30;
- horizon 5, receding horizon 5, action block 5;
- `GoalMSE` with sum reduction;
- 50 episodes, goal offset 25 steps, budget 50 steps.

Use three evaluation seeds (42, 43, 44) for each model seed, and report success rate as mean and standard deviation over all nine. This is P0.

If an AS-JEPA model passes the generation gate but its P0 success is more than 10 points below L0, run these variants in order and report each:

- **P1:** the goal cost summed over every predicted step instead of only the last (a custom objective following `GoalMSE`).
- **P2:** the iCEM solver at the same sample budget.
- **P3:** horizon 10, receding horizon 5.
- **P4:** the cost averaged over 8 prior-sample rollouts instead of the mean rollout.

## 5. Branch S: the SPT ladder (`runtime/vforge/tasks/as_jepa/`, `jobs/as_jepa/`, branch `as-jepa/spt`)

Native family `as_jepa_spt`, following `runtime/AGENTS.md`: model code in `runtime/vforge/models/as_jepa/`, task code in `runtime/vforge/tasks/as_jepa/`, launchers and full configs in `jobs/as_jepa/`. Register the family at the top of `jobs/as_jepa/train_entry.py`, before training starts:

```python
from vforge.tasks.registry import ModelTasks, register
register("as_jepa_spt", ModelTasks(train="vforge.tasks.as_jepa.train:build_train_components",
                                   train_preflight="vforge.tasks.as_jepa.shared:validate_train_request"))
from vforge.tasks.train import run_train  # then: run_train(OmegaConf.load(sys.argv[1]))
```

Mirror `runtime/vforge/tasks/rep_world/train.py` for `build_train_components`. Keep its pieces: `build_resumable_loader`, `ModelSpec` with meta initialization and `initialize_weights`, the `StepOutput` contract, and a dataset `resume_contract`. Change the dataset, the step function and the optimizer: AdamW with `use_muon` false, and no EMA. Full configs in `jobs/as_jepa/configs/` follow the layout of `jobs/configs/pretrain_50k.yaml`, with `model.name: as_jepa_spt` and the settings below under `model.extras`.

### 5.1 Geometry and data

```python
from vforge.models.rep_world.config import CodecGeometry
PUSHT64 = CodecGeometry(family="pixel64", variant="rgb1p4", rgb_frames=1, channels=48, planes=1, grid=16, spatially_ordered=True)
```

Construct it directly; do not add it to the pinned `CODECS` table. The dataset reads the Section 2.5 export and returns:

- `latents`: `pack64` frames arranged `[48, W, 16, 16]`, the packed layout `[P*C, S, H, W]` that `rep_world` uses;
- `actions`: `[W, 10]`;
- `anchor64`;
- `state`;
- `sample_id`.

### 5.2 S0: SPT with actions

`RepWorld` with SPT's split encoder in latent-input mode, actions added to each state's encoded tokens:

```python
class ActionSplitViT(SplitViTConditioner):
    def __init__(self, config, geometry, action_dim=10):
        super().__init__(config, geometry)
        self.action_proj = nn.Linear(action_dim, config.width)
        self._actions = None
        self._zero()
    def _zero(self):
        nn.init.zeros_(self.action_proj.weight); nn.init.zeros_(self.action_proj.bias)
    def initialize_weights(self):
        super().initialize_weights()
        if hasattr(self, "action_proj"):
            self._zero()
    def encode_states(self, source, *, return_indices=False):
        out = super().encode_states(source, return_indices=return_indices)
        encoded, idx = out if return_indices else (out, None)
        if self._actions is None:
            raise RuntimeError("actions not set")
        a = self.action_proj(self._actions[:, : encoded.shape[1]].to(encoded.dtype))[:, :, None, :]
        encoded = encoded + a                    # action a_t joins state t, so origin t's context sees a_<=t
        return (encoded, idx) if return_indices else encoded

class ActionRepWorld(RepWorld):
    def __init__(self, geometry, world, conditioner, flow, anchored=False):
        super().__init__(geometry, world, conditioner, flow)
        self.causal_semantic_encoder = ActionSplitViT(conditioner, geometry)
        if anchored:
            self.net = AnchoredWorldDiT(world, state_dim=None)     # S1
    def forward(self, latents=None, actions=None, anchor=None, **kwargs):
        enc = self.causal_semantic_encoder
        enc._actions = actions
        try:
            if isinstance(self.net, AnchoredWorldDiT) and kwargs.get("operation", "loss") == "loss":
                count = self.flow.target_count(latents.shape[2])
                with self.net.anchored(pack64(anchor).repeat_interleave(count, dim=0)):
                    return super().forward(latents=latents, **kwargs)
            return super().forward(latents=latents, **kwargs)
        finally:
            enc._actions = None
```

Configuration:

- **Conditioner:** `ConditionerConfig(input="latent", width=384, depth=9, encoder_depth=3, heads=6, mlp_ratio=4.0, feature_layers=(8,), pretrained_grid=16, context_mask_ratio=0.9, architecture="split_ec", mask_stage="before_e", temporal_patch=2)`. This is SPT's design at small width: ViT-E with 3 blocks per state, then a state-causal ViT-C with 6 blocks, and 90% of patches dropped before ViT-E during training.
- **World:** the Section 3.2 renderer configuration, with `condition_width=384` and `cross_attention_3d_rope=True` (SPT's spatially aligned cross-attention). For S0 use a plain `WorldDiT`.
- **Flow:** `FlowConfig(mode="no_zprev", mean=0.8, std=0.8, shift=-1.0, base_shift=2.667, distant_target_prob=0.0, target_layout="single_state", target_origin_reserve=1, future_target_choices=2)`. Every origin `i` predicts state `i+1`.

  This deviates from SPT's pretraining, which also targets `i+2` half the time. The skipped intermediate action would be invisible to the context, so we use SPT's next-state target throughout.
- **Window:** 10 kept frames.

Train three seeds on `D-std`.

### 5.3 S1: S0 plus the anchor

`ActionRepWorld(..., anchored=True)`, everything else as S0. The anchor is window frame 0, the same for all targets of a window; the targets are frames 1 to 9, so the anchor is never a target. Train three seeds on `D-std`.

### 5.4 S2: AS-JEPA in SPT's modules (`ASJepaSPT` in `runtime/vforge/models/as_jepa/model.py`)

Components:

- **Encoder:** `StateViTEncoder(ConditionerConfig(input="latent", width=384, depth=7, encoder_depth=6, heads=6, mlp_ratio=4.0, feature_layers=(6,), architecture="split_ec", context_mask_ratio=0.0), PUSHT64)`, then `QueryPool(width=384, heads=6, K=4, d=16)`. Each frame is encoded alone, with no masking.
- **Prior:** `StatePrior(K=4, d=16, action_dim=10, width=384, depth=6, heads=6)` (Section 3.4).
- **Renderer:** `AnchoredWorldDiT(<renderer configuration>, state_dim=16, num_slots=4)`, identical to L2's.
- **Loss:** exactly L2's (Section 4.4), with `mu = pool(encoder(latents).flatten(0, 1)).view(B, W, 4, 16)`, `m = prior(s[:, :-1], actions[:, :-1])`, the same `ramp`, `beta`, `sigma_0`, `s_0` and normalization.
- **Initialization:** the Native trainer builds the model on the meta device and then calls only `initialize_weights`, so `ASJepaSPT.initialize_weights` must initialize every parameter: `encoder.initialize_weights()`, `pool.initialize()`, the prior as in Section 3.4, and `renderer.initialize_weights()`.
- **Window:** 8 kept frames. Both anchor arms.

### 5.5 Optimization for all S rungs

Native trainer:

- AdamW (`use_muon: false`), lr `1e-4`, weight decay 0, betas `(0.9, 0.95)`, eps `1e-8`;
- `constant_with_warmup` with 100 warmup steps;
- bf16 autocast, rate terms in fp32, gradient clip 1.0;
- EMA disabled;
- global batch 128 windows, on any number of GPUs.

Use `attention.backend: sdpa`, or `flash_attention_4` if it runs these shapes; the semantics must match. Budget: `max_steps` chosen so each S run sees the same number of windows as the corresponding L run (100 epochs, or 40% for screens). Checkpoint every 10% in full DCP.

### 5.6 Inference for branch S

Write `rollout_s01(model, context_latents, actions, anchor, steps=50, cfg_scale=1.0)`, mirroring `RepWorldInference._semantic`. At each step, set the conditioner's actions to those of the states being encoded, generate the next frame with SPT's sampler, append it, and re-encode it. The history window is 9 states, the context length S0 and S1 were trained with.

For S1, wrap every velocity call of the sampler in `model.net.anchored(pack64(anchor))`; `ActionRepWorld.forward` sets the anchor only for the training loss.

For S2, the rollout is L2's: encode the context frames, roll out the prior, render with the anchor.

Planning for S2 is optional in this round. If you run it, write an adapter exposing `encode` and `rollout`. The adapter must undo the evaluation's ImageNet normalization, rescale to `uint8`, and apply `unit(to64(...))`.

### 5.7 S ablations (on `D-std`, at S2's `beta*`)

`S2-noanchor` (no anchor branch), and `S2-w16` with both anchors.

## 6. Design invariants: what must not change

Each line is a property the method depends on.

1. The AS-JEPA encoder reads one frame, nothing else: no history, no anchor, no actions. (The state is a function of its frame; goal images encode without history.)
2. The anchor reaches only the renderer. The prior never reads frames or renderer features. (Otherwise appearance becomes free in the state and is never excluded.)
3. Actions reach only the prior, plus S0 and S1's conditioner by design. The AS-JEPA renderer never sees actions or history. (Otherwise the renderer does the dynamics.)
4. The renderer reads the posterior sample `s_t`, the encoded current frame, never the prior's prediction. In training, the prior's output enters only the code length.
5. Window frame 0 is encoded and starts the chain, and is never rendered.
6. No augmentation, EMA (target or evaluation), stop-gradient, target network, SIGReg, VICReg or masking in L2 and S2. L1 keeps LeWM's SIGReg by design; S0 and S1 keep SPT's masking by design.
7. Nothing normalizes `mu`, `s` or the state tokens before the renderer's linear projection.
8. Noise is fixed: `sigma_0 = s_0 = 0.1`. One-step prediction only; no multi-step targets or overshooting.
9. The renderer's target is fixed data (packed 64-pixel frames), never a learned representation.
10. `beta` is ramped from 0 over the first 5% of steps.

## 7. Shared evaluation package (`as_jepa_tools/eval/`)

Every number in the reports comes from this package, applied identically to every model through a small adapter. Each model provides:

- `encode(frames)`: states, for probes;
- `rollout(context, actions)`: predicted states, or frames for S0 and S1;
- `render(states, anchor)`: frames.

### 7.1 Pose estimator

A ResNet-18 (torchvision, random initialization) on 64-pixel frames regresses `[agent_x, agent_y, block_x, block_y]` (divided by 512) and `[sin, cos]` of the block angle.

- Training: train episodes, 30 epochs, AdamW lr `1e-3`, cosine schedule, batch 512, with mild Gaussian blur and pixel noise so it reads imperfect generated frames. This augmentation belongs to the evaluator only, not to any model.
- Train one estimator on `D-std` and one on `D-var`. The `D-std` estimator also scores `D-dot`.
- Report its error on real test frames as the floor of every pose metric.

Errors are measured in workspace pixels for positions and degrees for the angle, wrapped to `[-180, 180]`.

### 7.2 Generation protocol (main criterion)

Fixed test windows: 256 windows of 16 kept frames from test episodes, chosen once with seed 0 and stratified half and half by whether the block moves more than 10 pixels.

- Context: frames 0 to 2 and all 15 actions. Predict frames 3 to 15 (13 steps).
- Anchor at test time: frame 0, for both anchor arms.
- Rollout context: AS-JEPA priors and the L1 predictor keep the last `num_frames` states (7 for 8-frame windows, 3 and 15 for the window ablations); S0 and S1 keep the last 9 states, the context length they were trained with (`target_count(10) = 9`). Run every model in eval mode.

Variants:

- **mean:** AS-JEPA context states are `mu` of the context frames; roll out prior means.
- **sample:** four samples per window; context states `mu + 0.1 * noise`; roll out prior samples `m + 0.1 * noise`.

Render with 50 Euler steps and guidance 1.0, and report both variants. S0 and S1 use their own rollout (Section 5.6).

Metrics at every horizon (report horizons 1, 3, 7 and 13):

- agent position error, block position error and block angle error, from the pose estimator against true simulator states;
- LPIPS (AlexNet), SSIM and PSNR against the true frame;
- FVD as a secondary metric, using SPT's bundled I3D detector on clips upsampled to 224 pixels.

Reference: copy the last context frame (frame 2) forward. Report every error relative to this copy baseline as well as absolute.

### 7.3 Counterfactual actions (D-std)

Setup:

- 64 start points: a test episode and step `k`, with context frames `k-10`, `k-5`, `k`.
- At step `k`, restore the simulator to the recorded state, using the environment's `_set_state` as the planning evaluation does.
- Execute 4 action sequences of 8 steps (40 environment steps): the recorded one, and 3 recorded sequences taken from other test windows.

The simulator renders the true futures. The model generates one rollout per sequence from the same context and anchor.

Report:

- pose errors as in Section 7.2;
- action identification accuracy: assign each generated rollout to the true future with the smallest pose error, and count the fraction assigned to their own sequence (chance 0.25).

### 7.4 Planning

Branch L: Section 4.8. Save the planning evaluation's videos, five successes and five failures per model, in the gallery (Section 7.7).

### 7.5 Probes (understanding, not gating)

Ridge regression (closed form; `alpha` chosen by 5-fold cross-validation on train features from 50,000 frames), reporting `R^2` on test episodes.

Features:

- L2 and S2: `mu`, flattened to 64-D;
- L0 and L1: LeWM's embedding;
- S0 and S1: `model.causal_semantic_encoder.vit_e(...)` output tokens, mean-pooled (called directly, so no action embedding is added).

Targets:

- agent position, block position, and block angle as sin and cos;
- agent velocity;
- on `D-var`: the RGB of agent, block, background and goal, and the goal pose;
- on `D-dot`: the dot's centre and colour.

### 7.6 Rate diagnostics (AS-JEPA only)

Log during training:

- `entry_nats` per window;
- `surprise_nats` per step;
- `fm`;
- `rate_share = ramp * beta * rate / (T * 12288) / fm`;
- per-dimension rate (the mean over a held-out batch of each dimension's `0.5 * mu_0^2` and summed surprise), and the number of dimensions above 0.1 nats per window;
- gradient norms of encoder, prior and renderer.

### 7.7 Galleries for human inspection

For every evaluated checkpoint, under `<run>/eval/<checkpoint>/gallery/`:

- `fixed/<case>/`: 32 fixed cases, the first 32 of the Section 7.2 windows, identical for all models. Each holds `gt.mp4`, `pred_mean.mp4`, `pred_sample_{0..3}.mp4`, `copy.mp4`, a `grid.png` (rows: truth, mean, samples; columns: horizons 1 to 13), and `meta.json` (episode, start step, actions, per-horizon errors, model, checkpoint, seed).
- `best/` and `worst/`: the 8 windows with the lowest and highest mean pose error over horizons, same format.
- `counterfactual/<case>/`: 8 start points, each with the 4 true futures, the 4 generated rollouts, a grid, and metadata.
- `swap/<pair>/`: anchor models only, 8 pairs. Render episode B's predicted states with episode A's anchor; save A's anchor image, B's true video, the swapped video, and metadata with B's pose errors. On `D-var`, also record whether the rendered colours at B's object positions are closer to A's or B's recorded colours.
- `planning/`: success and failure videos.

Mark each fixed case `pass` or `fail` in `fixed/review.csv`. A pass means the motion is plausible and the objects are intact, with no duplicated, vanished or deformed blocks.

### 7.8 Reports

Per wave, write `reports/wave<k>.md` with:

- tables for all metrics, as mean ± standard deviation over seeds;
- the dataset cards;
- links to galleries;
- configs, commit hashes (SPT-JEPA and stable-worldmodel), and the GPU-hours used.

Also write `reports/wave<k>.json` with one record per run: `run_id`, `branch`, `rung`, `dataset`, `arm`, `beta`, `window`, `seed`, `checkpoint`, and every metric name exactly as in this section.

## 8. Run matrix

Run IDs follow `<rung>-<dataset>-<arm>-b<beta>-w<window>-s<seed>`. Priorities: P1 must run, P2 should run, P3 if time permits.

### Wave 1 (start as soon as each rung passes its tests; all parallel)

| Runs | Dataset | Seeds | Budget | Priority |
| --- | --- | --- | --- | --- |
| L0 | D-std | 0, 1, 2 | full | P1 |
| L1 | D-std | 0, 1, 2 | full | P1 |
| L2 `beta` screen: `beta` in {1, 4, 16, 64, 256, 1024} x {`first`, `far`} | D-std | 0 | 40% | P1 |
| S0 | D-std | 0, 1, 2 | full | P1 |
| S1 | D-std | 0, 1, 2 | full | P1 |
| S2 `beta` screen: 4 values (Section 10) x {`first`, `far`} | D-std | 0 | 40% | P1 |

### Wave 2 (after `beta*` is chosen; all parallel)

| Runs | Dataset | Seeds | Budget | Priority |
| --- | --- | --- | --- | --- |
| L2 at `beta*`, `first` and `far` | D-std | 0, 1, 2 | full | P1 |
| S2 at `beta*`, `first` and `far` | D-std | 0, 1, 2 | full | P1 |
| L2-noanchor, L2-mse, L2-w4 (both arms), L2-w16 (both arms) | D-std | 0 | full | P2 |
| S2-noanchor, S2-w16 (both arms) | D-std | 0 | full | P2 |
| L0, L1, L2 (both arms) | D-var | 0 | full | P2 |
| S2 (both arms) | D-var | 0 | full | P3 |
| L0, L2 (both arms), L2-mse | D-dot | 0 | full | P2 |

Before Wave 1, measure throughput in the 200-step smoke runs (Section 9.3), report the projected GPU-hours per run, and wait for our acknowledgement if the total exceeds what was agreed.

## 9. Tests and checks before any long run

Zero initializations make several checks trivial or impossible at initialization: WorldDiT's output layer (`final_layer.linear` and `final_layer.adaLN_modulation[-1]`), the prior's `delta` (S2) and `delta_head` (L2), and S0's `action_proj`. In every test below that probes gradients, causality, or the effect of an input, first re-initialize those layers with `normal_(std=0.02)`; the equivalence tests run on the true initialization.

### 9.1 Shared components (CPU, small configs, `sdpa_math` backend)

- `pack64` then `unpack64` is the identity; `to64` matches between the branch-L loader and the branch-S export on 1000 rows.
- `draw_timesteps` equals `RepWorld.draw_timesteps` on identical uniforms.
- `AnchoredWorldDiT`:
  - With the output layer perturbed and `anchor_embed` at zero, its output equals a plain `WorldDiT` with the same weights, for any anchor.
  - With the output layer perturbed, one optimizer step on a nonzero loss makes `anchor_embed`'s weights nonzero.
  - Outside `anchored(...)` it raises.
  - Scaling the state tokens by 2 changes the output (no normalization before the projector).
  - Permuting the state tokens changes the output (slot identity).
- `entry_nats` and `surprise_nats` equal `torch.distributions.kl_divergence` between the corresponding Normals.
- `StatePrior`:
  - Its output for step `t` is unchanged when states or actions at steps later than `t` change.
  - At initialization its output equals its input (identity carry).

### 9.2 Per rung (CPU, small configs)

- **Encoder locality:** changing frame `j` leaves `mu` of every other frame unchanged (L2, S2).
- **Prior causality:** changing frames or actions after step `t` leaves the predicted means for steps up to `t+1` unchanged.
- **Anchor isolation:** the gradient of `mu` and of the prior's means with respect to the anchor is exactly zero.
- **Renderer isolation:** the gradient of the rendered output for frame `t` with respect to actions and to states other than `s_t` is exactly zero.
- **Gradient reach:** every loss term produces nonzero gradients in encoder, prior and renderer. In L2 and S2, the rate term reaches the encoder and the prior; the flow loss reaches the encoder through `s_t`.
- **Action alignment:** the action vector paired with frame `t` is the 5 raw actions recorded after frame `t`, checked against dataset indices.
- **`far` sampler:** the anchor comes from the same episode and respects the minimum distance; it is deterministic given the seed.
- **S0:** with `action_proj` at zero, `ActionSplitViT` equals `SplitViTConditioner`; changing `a_t` changes contexts at origins `>= t` only.
- **Branch L checkpoints:** a saved L2 checkpoint loads with `swm.wm.utils.load_pretrained` and runs one planning step of `eval_wm.py`.
- **Branch S:** a 3-step Native training on CPU with a tiny config, and a checkpoint save-and-resume round trip.

### 9.3 GPU smoke (200 steps per rung, one GPU)

Check:

- the loss is finite;
- `entry_nats`, `surprise_nats` and `fm` all move;
- for AS-JEPA, renders from pure noise change when the state is replaced by zeros;
- throughput and peak memory are recorded.

## 10. Choosing `beta*`

- **L2:** from the Wave 1 screen, per anchor arm. Compute the main generation score: the mean of agent and block position error at horizon 7, relative to the copy baseline (mean rollout, `D-std` fixed windows). Let `best` be the lowest score in the grid. `beta*` is the largest `beta` whose score is within 10% of `best`, because among near-equal generators we prefer the more strongly priced state the method is about. If planning (P0) separates the candidates by more than 10 points, prefer the better planner.
- **Grid adjustment:** if no `beta` changes `rate_share` or the per-dimension rates meaningfully, extend the grid upward by factors of 4. If every `beta` collapses (renders ignore the state, `entry_nats` near its floor), extend it downward.
- **S2:** screen `{beta_L*/16, beta_L*/4, beta_L*, 4 * beta_L*}`, where `beta_L*` is L2's choice for the same arm, and select by the same rule. If S2's Wave 1 must start before L2's screen finishes, screen `{4, 16, 64, 256}`.

## 11. Gates: when PushT counts as working

Move beyond PushT, to shape and scale variation and to other environments, only with our agreement. Agreement requires at least one AS-JEPA configuration (L2 or S2, either arm, at `beta*`, three seeds) to meet the generation gate:

- **Rollouts:** agent position error at horizon 7 at most half the copy baseline's, and block position error in moving-block windows below the copy baseline's (mean rollout, `D-std` test windows, mean over seeds). Half of the copy baseline means the model captures at least half of the motion; the block condition makes sure contact is handled, not only free agent motion.
- **Counterfactuals:** action identification accuracy at least 0.75 (chance 0.25).
- **Inspection:** at least 24 of the 32 fixed cases marked `pass`.

Planning is the second main criterion. Report P0 for every branch-L model, and P1 to P4 when the generation gate passes but P0 trails L0 by more than 10 points (Section 4.8). Probes, purity, rates and FVD are reported for understanding and do not gate.

## 12. Failure responses

| Symptom | Likely cause | Response |
| --- | --- | --- |
| Renders ignore the state: zeroing it changes little; `entry_nats` near its floor; all dimensions inactive | collapse | Lengthen the ramp to 20% of steps; lower `beta`; check the projector replacement and anchor isolation tests. Compare the `far` arm, which has a weaker pull toward collapse. If both arms collapse at every `beta`, stop and report. |
| Good at horizon 1, fast drift after a few steps | prior errors compound in open loop | Report mean and sample rollouts and the per-horizon curve; check the prior's context length at rollout. Do not add multi-step training without our agreement. |
| Block copied from the anchor and never moves in rollouts; block-pose probe low | block pose left out of the state | Compare against the `far` arm and the 16-frame ablation; report the block-motion fraction from the dataset card. |
| Appearance decodable from `mu` on `D-var` at every `beta` | anchor not used, or a leak | Check that `anchor_embed` weights left zero and rerun the isolation tests. Report; do not add augmentation. |
| NaN or loss spikes | numerics | Check that rate terms are fp32; halve the learning rate; keep gradient clipping. |
| S0 or S1 renders noise after training | SPT at this scale needs more updates | Compare with L1 at the same number of windows seen. Extend S budgets by 50% (Section 13); report. |
| Good generation, poor planning | the planning cost reads the state poorly | Run P1 to P4 (Section 4.8) and report each. |

## 13. Latitude: what you may change without asking

- Learning rates by a factor of 0.5 or 2, applied to the same rung across all its runs.
- Per-GPU batch and number of GPUs, keeping the global batch of 128 windows (use accumulation if needed).
- Extending the `beta` grid (Section 10).
- Training budget up to 1.5 times for all rungs of a branch equally, if losses are still falling at the end. Report it.
- Evaluation sample counts upward.
- The attention backend, if the semantics are identical.
- Implementation mechanics not fixed here: Lightning wiring, file splitting, logging tools.

Everything else, especially anything in Section 6, needs our agreement. When unsure, stop and ask.

## 14. Order of work

1. **Environments and SPT smoke** (Section 1). In parallel:
   - data (Section 2);
   - shared components and their tests (Sections 3 and 9.1);
   - the evaluation package with the pose estimators, run on ground-truth data to produce the copy baselines and estimator floors (Section 7).
2. **Implement the rungs** (Sections 4 and 5), then the per-rung tests (9.2) and GPU smokes (9.3). Report throughput and projected GPU-hours.
3. **Wave 1** (Section 8). Evaluate the screens, choose `beta*` (Section 10), and report.
4. **Wave 2** (Section 8). Full evaluation and galleries, then the report and the gate check (Section 11).

Send us the Wave 1 report before launching Wave 2 when `beta*` selection needed the grid extension or any failure response from Section 12 was used.
