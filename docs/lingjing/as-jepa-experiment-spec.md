# AS-JEPA on PushT: experiment and implementation spec (handoff)

Written 2026-10-26 for the collaborator who will implement and run the first round of AS-JEPA experiments, and revised the same day after review (decisions logged in [discussions/2026-10-25-spt-code-review-of-plan.md](discussions/2026-10-25-spt-code-review-of-plan.md), last section). It is self-contained: follow it as written. Where a choice is left to you, Section 14 lists it; any other change needs our agreement first. The idea and its theory are in [as-jepa-story.md](as-jepa-story.md). You do not need them to execute this spec, but each component below names the design reason it serves, so that implementation shortcuts do not silently break the method.

## 0. Overview

**What AS-JEPA is, operationally.** A per-frame encoder maps each frame to a small noisy state, `s_t = mu(x_t) + sigma0 * noise`. A causal predictor (the prior) reads past states and actions and predicts the next state's mean. A flow-matching renderer draws each frame from its own state and one anchor frame of the same episode. Training minimizes one objective at a constant price `beta`: the renderer's flow loss plus `beta` times the code length of the states, which is the first state's cost against `N(0, I)` plus each later state's squared prediction error under the prior, both in nats. There is no augmentation, EMA, stop-gradient, target network, SIGReg, masking or loss schedule in AS-JEPA.

**What this round must show, on standard PushT (`D-std`) only.**

- Main criteria: generation (rendered rollouts follow the actions) and planning (LeWM's PushT protocol).
- For understanding, not gating: probes of the state and rate diagnostics.
- Report the `D-std` results and wait for our agreement before using any other dataset or environment. Appendix A defines the next datasets.

**Main models and comparisons, in two code bases.**

- **Branch L**, in LeWM's code base (stable-worldmodel). The main model is **L2**: AS-JEPA built from LeWM's encoder and predictor plus SPT's renderer, tuned by a hyperparameter search. Comparisons: **L1**, the same architecture trained with LeWM's objective (prediction loss plus SIGReg) and the same renderer, with nothing priced; **L0**, LeWM as published.
- **Branch S**, in SPT's code base (vforge Native). The main model is **S2**: AS-JEPA built from SPT's modules. Comparisons: **S0**, SPT with actions; **S1**, S0 plus the anchor.

Both branches use the same data, the same 64-pixel rendering target, the same renderer architecture and one shared evaluation package. L2 and S2 come first. Runs that do not depend on each other run in parallel on whatever compute you have; Section 9 gives the dependencies and the priorities when compute is short.

## 1. Named quantities

Code uses these names. Derived quantities are computed from the data and the configs as defined here, never retyped as numbers. Values are for PushT.

### 1.1 Derived quantities

| Name | Definition | PushT value |
| --- | --- | --- |
| `C` | image channels | 3 |
| `ENC_PX` | branch-L encoder input side, read from LeWM's config | 224 |
| `RENDER_PX` | side of a rendered frame (a fixed choice) | 64 |
| `PATCH` | side of the pixel patch packed into one renderer token (a fixed choice) | 4 |
| `GRID` | `RENDER_PX // PATCH`, renderer tokens per side | 16 |
| `TOKEN_CH` | `C * PATCH ** 2`, channels per renderer token | 48 |
| `FRAME_ELEMS` | `C * RENDER_PX ** 2`, values in one rendered frame | 12288 |
| `FRAMESKIP` | environment steps between kept frames, read from LeWM's config | 5 |
| `RAW_ACTION_DIM` | the environment's action dimension | 2 |
| `ACTION_DIM` | `RAW_ACTION_DIM * FRAMESKIP` | 10 |
| `W`, `T` | window length in kept frames; `T = W - 1` rendered frames per window | 8 and 7 by default |
| `W_SPT` | SPT's own window, used by S0 and S1; their rollout context is `W_SPT - 1` | 10 |
| `STATE_DIM` | `K * d` | 64 |
| `SHIFT` | SPT's automatic flow-time shift for the renderer's grid (Section 4.1) | about 1.742 |
| `WORKSPACE` | side of the PushT workspace in simulator pixels, read from the environment | 512 |

### 1.2 Hyperparameters

| Name | Meaning | Values | Chosen in |
| --- | --- | --- | --- |
| `beta` | price of one nat of state, relative to the flow loss of one rendered value | {1, 4, 16, 64, 256, 1024} | Stage A (9.2) |
| `sigma0` | noise of the code, in units of the first state's prior `N(0, I)`; the prior's predictive spread is tied to it | {0.03, 0.1, 0.3} | Stage A |
| `g` (anchor gap) | kept frames between the anchor and the window; `g = 0` means the window's first frame (Section 3.3) | {0, 8} in Stage A; {0, 2, 4, 8} in Stage B | Stages A and B |
| `W` | window length in kept frames | 8 in Stage A; {4, 8, 16} in Stage B | Stage B |
| `K`, `d` | number of state tokens and their width | 4 and 16, fixed | not searched (below) |
| `LAMBDA_SIGREG` | L1 only: weight of SIGReg | {0.03, 0.09, 0.27} | L1's screen (9.6) |

Why only these. The code length has exactly two free numbers, `beta` and `sigma0`; Section 4.3 shows that its other constants are fixed by the mathematics or would duplicate these two. The anchor gap and the window length are the two knobs that, by the theory, decide which slowly changing components the state keeps (story Section 5.6); `sigma0` is the third such knob. The state size is not searched, because the price switches unneeded dimensions off; it only has to be large enough, which Section 11 checks. Everything else, including the architectures and optimizer settings, is inherited from LeWM or SPT and fixed.

### 1.3 Evaluation and decision constants

| Name | Value | Meaning |
| --- | --- | --- |
| `SCREEN_FRAC` | 0.4 | screening runs train for this fraction of the full budget |
| `SEL_TOL` | 0.10 | configurations within 10% of the best generation score count as tied (Section 11) |
| `PLAN_MARGIN` | 10 points | a difference in planning success rate that counts as a separation |
| `MOVE_PX` | 10 workspace pixels (about 2% of `WORKSPACE`) | a window moves the block if the block moves more than this |
| `ACTIVE_NATS` | 0.1 | a state dimension is active if it carries more than this many nats per window |
| `COLLAPSE_RATIO` | 0.9 | threshold of the collapse test (Section 13) |
| `FALLBACK_MAX` | 0.2 | largest acceptable share of training windows without an eligible anchor at gap `g` |
| `SAMPLER_STEPS`, `CFG_SCALE` | 50, 1.0 | Euler steps and guidance scale when rendering for evaluation |
| `EVAL_W`, `EVAL_CTX` | 16, 3 | generation test windows: 16 kept frames, the first 3 given as context |
| `HORIZONS` | 1, 3, 7 and `EVAL_W - EVAL_CTX` | horizons reported in tables |
| `GEN_RATIO_MAX` | 0.5 | gate: agent error at most this fraction of the copy baseline's |
| `ACTION_ID_MIN` | 0.75 | gate: counterfactual action identification (chance 0.25) |
| `GALLERY_PASS_MIN` | 24 of 32 | gate: fixed gallery cases marked pass |

## 2. Repositories, branches, environments

### 2.1 Repositories and branches

- **SPT-JEPA** (this repository). Create three branches from `lingjing/anchored-state-jepa`:
  - `as-jepa/shared`: shared model components under `runtime/vforge/models/as_jepa/`, and data plus evaluation tools under `as_jepa_tools/`;
  - `as-jepa/lewm`: from `as-jepa/shared`; adds the branch-L package `as_jepa_lewm/`;
  - `as-jepa/spt`: from `as-jepa/shared`; adds `runtime/vforge/tasks/as_jepa/` and `jobs/as_jepa/`.

  Merge `as-jepa/shared` into both implementation branches whenever it changes.
- **stable-worldmodel** (public, `https://github.com/galilai-group/stable-worldmodel`). Use the latest `main` when you start, pin that commit, and record its hash in every report. This spec was checked against `main` at `21446f1` (2026-10-02). It provides the PushT environment and its variation space, `WeakPolicy`, the LeWM reference training script `scripts/train/lewm.py` with its configs `scripts/train/config/lewm.yaml` and `scripts/train/config/data/pusht.yaml`, the LeWM model `stable_worldmodel/wm/lewm/`, `SIGReg` (`stable_worldmodel/wm/loss.py`), the planners, and the PushT planning evaluation `scripts/plan/eval_wm.py` with `scripts/plan/config/pusht.yaml`. Do not modify it; install it as a package. If your pinned commit differs from `21446f1` in any file this spec cites, check the cited behaviour before relying on it.

### 2.2 Hard rule for SPT code

Do not edit any file listed in `provenance/upstream_runtime_hashes.json`. `jobs/worker.py` and `tests/test_portability.py` check those hashes, and the RGB16 baseline must keep working. Everything in this spec is new files. New files under `jobs/` must not contain the machine-specific path strings that `tests/test_portability.py` rejects (for example `/threed-code/` and `/efs/stormyxie`). Two existing extension points make that possible:

- `vforge.tasks.registry.register(name, ModelTasks(...))`, the public hook "for out-of-tree plugins";
- subclassing the `rep_world` modules.

### 2.3 Environments

- **Env S:** SPT's validated environment (Python 3.12, torch 2.9.1+cu130, FlashAttention-4, `pip install -e ./runtime`), as in the repository README.
- **Env L:** a clone of env S plus `pip install -e "<stable-worldmodel at the pinned commit>[train,data,format]"` and the PushT packages `pygame pymunk shapely opencv-python-headless` (the PushT environment imports them at module load). Install with a constraints file pinning torch and torchvision to env S's versions (stable-worldmodel leaves both unpinned), then confirm torch is still 2.9.1+cu130. Check that `import stable_worldmodel`, `swm.World("swm/PushT-v1", num_envs=1, image_shape=(ENC_PX, ENC_PX))` and `from vforge.models.rep_world.world import WorldDiT` all work (`World` has no default for `num_envs` and requires `image_shape`). Inside any process that runs SPT modules outside SPT's trainer, call once at startup:

  ```python
  import torch
  from vforge.ops import OpsConfig, initialize_ops
  initialize_ops(OpsConfig(backend="sdpa", allow_fallback=False), device=torch.device("cuda", local_rank), parallel_state=None)
  ```

  SPT's own CPU tests do the same (`runtime/tests/models/rep_world/test_cross_attention_order.py`). With no parallel state, sequence parallelism is off.

### 2.4 SPT smoke test (no baseline reproduction)

Run the repository's CPU tests (`PYTHONPATH=runtime python -B -m unittest discover -s tests`) and the `rep_world` model tests under `runtime/tests/models/rep_world/`. Then run a short RGB16 training smoke (at most 100 updates) on one 8-GPU node following the README's procedure, record step time and peak memory, and stop. Do not run the 50k-step baseline, FVD-2048 or SolarWM experiments in this round.

## 3. Data

### 3.1 Dataset

| Name | Content | Source | Used for |
| --- | --- | --- | --- |
| `D-std` | standard PushT, fixed appearance | LeWM's public PushT dataset, as used by `scripts/train/lewm.py` | every run; the benchmark; planning |

Files. Nothing downloads them automatically: dataset names without a `/` are looked up only in `$STABLEWM_HOME/datasets/`. Record the checksum of each file in the dataset card.

- Training and our evaluation: `pusht_expert_train.lance` from Hugging Face `galilai-group/lewm-pusht` at revision `ea321e3`, the file `scripts/train/config/data/pusht.yaml` names. Its frames are stored as JPEG, so "raw frame" in this spec means the decoded `uint8` frame. Both branches read this same file.
- LeWM's planning evaluation: `pusht_expert_train.h5`, which `scripts/plan/config/pusht.yaml` names but `galilai-group/lewm-pusht` does not contain. Download `pusht_expert_train.h5.zst` from Hugging Face `quentinll/lewm-pusht` (13.1 GB) and decompress it into `$STABLEWM_HOME/datasets/`. This keeps the upstream planning protocol unchanged.

Expected contents (from an audit of revision `ea321e3`; confirm in the dataset card): 18,685 episodes, 2,336,736 rows, episode lengths 49 to 246 environment steps (median 123), that is 10 to 50 kept frames (median 25); 7-D state.

Use only `D-std` in this round. On `D-std` the appearance is fixed and the frames carry no transient detail, so this round tests generation, planning and which dynamics components the state keeps; it cannot test whether the state excludes appearance or transient detail. That needs the variants in Appendix A, after the `D-std` report.

### 3.2 Splits

- Split `D-std` by episode with seed 0: 95% train, 5% test. All our runs (including L0, retrained) train on the train episodes; generation, probes and galleries use test episodes only.
- Planning follows LeWM's protocol unchanged, which samples start and goal states from `pusht_expert_train` itself (Section 5.8). Report that as is.

### 3.3 Frames, actions, windows, anchors

- **Frameskip `FRAMESKIP`.** Consecutive kept frames are `FRAMESKIP` environment steps apart. The action for step `t` is the `FRAMESKIP` raw actions taken after frame `t`, concatenated into an `ACTION_DIM` vector. It leaves frame `t` and produces frame `t+1`, as in LeWM. The last row of each episode has a NaN action; set it to zero, as `scripts/train/lewm.py` does. Normalize actions with stable-worldmodel's z-score column normalizer, as `scripts/train/lewm.py` does.
- **Encoder input (branch L):** `ENC_PX` pixels, LeWM's preprocessing (`get_img_preprocessor`: to image, ImageNet statistics, resize to `ENC_PX`).
- **Renderer target and anchor (both branches), and encoder input (branch S):** `RENDER_PX` pixels, computed from the raw `uint8` frame by exactly these functions (shared, in `as_jepa_tools/data/frames.py`). The area filter is rounded back to `uint8`, so both branches see identical values:

  ```python
  def to_render(frames_uint8):           # [N, C, H, W] uint8 -> [N, C, RENDER_PX, RENDER_PX] uint8
      x = torch.nn.functional.interpolate(frames_uint8.float(), size=(RENDER_PX, RENDER_PX), mode="area")
      return x.round().clamp_(0, 255).to(torch.uint8)

  def unit(frames_uint8):                # uint8 -> float in [-1, 1]
      return frames_uint8.float() / 127.5 - 1.0
  ```

  Everywhere below, "rendered frame" means `unit(to_render(raw))`.

- **Packing for SPT modules** (shared, `runtime/vforge/models/as_jepa/pixels.py`). A rendered frame becomes a `GRID x GRID` grid of `PATCH x PATCH` patches, `TOKEN_CH` channels per token, one temporal plane:

  ```python
  def pack(frames, patch=PATCH):         # [N, C, H, W] -> [N, C * patch**2, 1, H // patch, W // patch]
      n, c, h, w = frames.shape
      x = frames.reshape(n, c, h // patch, patch, w // patch, patch)        # c, gy, py, gx, px
      return x.permute(0, 3, 5, 1, 2, 4).reshape(n, c * patch * patch, 1, h // patch, w // patch)

  def unpack(packed, patch=PATCH, c=C):  # inverse of pack
      n, _, _, gh, gw = packed.shape
      x = packed.reshape(n, patch, patch, c, gh, gw).permute(0, 3, 4, 1, 5, 2)
      return x.reshape(n, c, gh * patch, gw * patch)
  ```

- **Windows:** L2, S2 and L1 use `W` kept frames (8 unless Stage B selects otherwise); L0 uses LeWM's own window (history 3 plus 1); S0 and S1 use `W_SPT` kept frames.
- **Anchor with gap `g`** (L2, S2, L1 and their ablations):
  - `g = 0`: the anchor is window frame 0.
  - `g >= 1`: the anchor is drawn uniformly from the kept frames of the same episode (frameskip-aligned with the window) that lie at least `g` kept frames before window frame 0 or at least `g` kept frames after the window's last frame. Use `torch.Generator().manual_seed(hash((episode_idx, window_start, epoch)) & 0x7FFFFFFF)`. If no frame is eligible, use the farthest frame of the episode and count the window as a fallback. If the fallback share exceeds `FALLBACK_MAX` of training windows for some `g`, drop that `g` from the grid and report it.
  - For every `g`, window frame 0 is encoded as the first state of the chain and is never a rendering target.

  Episodes are short (median 25 kept frames), which bounds the useful gaps. With windows starting at every environment step, the expected share of training windows without an eligible anchor is:

  | `W` | `g = 2` | `g = 4` | `g = 8` | `g = 12` | `g = 16` |
  | --- | --- | --- | --- | --- | --- |
  | 4 | 0% | 0% | 3% | 17% | 40% |
  | 8 | 0% | 1% | 9% | 30% | 56% |
  | 16 | 1% | 7% | 30% | 61% | 82% |

  Hence the grid stops at `g = 8`, and the cell `W = 16, g = 8` fails `FALLBACK_MAX` and is skipped. Recompute this table for the dataset card.

  Design reason: the anchor supplies what the whole episode shares. The gap controls how much of the current configuration the anchor also shows. At `g = 0` it shows the configuration at the start of the window, so a component that does not move within the window (often the block) can be left to the anchor; larger gaps make the state carry it (story Section 5.6).
- **Branch S export.** Write the dataset once as memory-mapped arrays: the `uint8` output of `to_render` as `[N_rows, RENDER_PX, RENDER_PX, C]`, actions, the simulator state, and an index table (episode id, step index, row). Both branches must produce bit-identical rendered targets. Test this on 1000 random rows.

### 3.4 Dataset card

Report:

- episode count and length distribution;
- the number of valid windows per window length in the grid;
- the fraction of 8-frame windows in which the block moves by more than `MOVE_PX`;
- for each anchor gap in the grid, the fallback share.

The PushT simulator state is `[agent_x, agent_y, block_x, block_y, block_angle, agent_vx, agent_vy]` in a `WORKSPACE`-pixel workspace.

## 4. Shared model components (`runtime/vforge/models/as_jepa/`, branch `as-jepa/shared`)

Pure model code, importable from both environments. No training-loop code here.

### 4.1 Flow utilities (`flow.py`)

The renderer is trained and sampled exactly as SPT does. The flow settings come from SPT's `FlowConfig` defaults, and the time shift is computed by SPT's own rule for the renderer's grid, not typed in.

```python
import torch
from vforge.models.rep_world.model import FlowConfig, shift_timestep

FLOW = FlowConfig()                    # SPT's defaults: logit-normal mean and std, base shift, automatic shift

def timestep_shift(world_config, flow=FLOW):
    """SPT's automatic shift, exactly as RepWorld.__init__ computes it (vforge/models/rep_world/model.py)."""
    if flow.shift > 0:
        return flow.shift
    return flow.base_shift * ((world_config.grid // world_config.patch_size) ** 2 / 600) ** 0.5   # 600: SPT's reference token count

def draw_timesteps(n, device, shift, flow=FLOW, uniform=None):     # t = 1 is pure noise
    u = uniform if uniform is not None else torch.rand(n, 1, device=device).clamp_(1e-5, 1 - 1e-5)
    t = torch.sigmoid(flow.mean + flow.std * torch.special.ndtri(u.double())).float()
    return shift_timestep(t, shift)

def flow_loss(renderer, y, state, anchor, shift):   # y, anchor: [N, TOKEN_CH, 1, GRID, GRID]; state: [N, K, d]
    t = draw_timesteps(y.shape[0], y.device, shift)
    eps = torch.randn_like(y)
    tv = t.view(-1, 1, 1, 1, 1)
    with renderer.anchored(anchor):
        v = renderer((1 - tv) * y + tv * eps, t, condition=state)
    return (y - eps - v).square().mean()              # mean over all rendered values; SPT's sign: target velocity y - eps

@torch.no_grad()
def sample(renderer, state, anchor, shift, steps=SAMPLER_STEPS, cfg_scale=CFG_SCALE, generator=None):
    """SPT's Euler sampler (RepWorldInference._denoise), with optional guidance on the state.
    Call with the renderer in eval mode: in train mode WorldDiT drops the state for a fraction of samples."""
    n = state.shape[0]
    z = torch.randn((n, TOKEN_CH, 1, GRID, GRID), device=state.device, generator=generator)
    sched = shift_timestep(torch.linspace(1, 0, steps + 1, device=state.device), shift)
    drop = torch.ones(n, dtype=torch.bool, device=state.device)
    with renderer.anchored(anchor):
        for t_now, t_next in zip(sched[:-1], sched[1:]):
            t = t_now.expand(n, 1)
            v = renderer(z, t, condition=state)
            if cfg_scale > 1 and 0.2 < float(t_now) <= 1.0:        # SPT's guidance interval (0.2, 1.0]
                vu = renderer(z, t, condition=state, condition_drop=drop)
                v = vu + cfg_scale * (v - vu)
            z = z - (t_next - t_now) * v
    return z
```

`SHIFT = timestep_shift(renderer_config(d))` (Section 4.2). Section 10.1 checks `timestep_shift` and `draw_timesteps` against `RepWorld`.

### 4.2 Anchored renderer (`renderer.py`)

SPT's `WorldDiT` with two changes and no edits to its file. Design reason: the anchor is side information that reaches the renderer only, and the state must reach the renderer without per-token normalization, which would remove each token's mean and scale and stop the code length from working. The state tokens also get learned slot embeddings after the linear projection: cross-attention without positions treats its context as an unordered set, which would discard which token is which.

```python
import contextlib
import torch
from torch import nn
from vforge.models.rep_world.config import WorldConfig
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
    def anchored(self, anchor):                    # anchor: [N, TOKEN_CH, 1, GRID, GRID], one per rendered target
        self._anchor = anchor if self.use_anchor else None
        try:
            yield self
        finally:
            self._anchor = None

def renderer_config(state_width, condition_width=None, cross_attention_3d_rope=False):
    """The renderer for L1, L2 and S2 (S1 overrides the two keyword arguments, Section 6.5).
    DiT-S size, plain output layer (no pixel refiner), cross-attention in blocks 0, 2, 4, 6, 8, 10."""
    return WorldConfig(channels=TOKEN_CH, state_planes=1, grid=GRID, width=384, depth=12, heads=6, patch_size=1,
                       mlp_ratio=4.0, qk_norm=True, adaln_mode="adaln_lora", adaln_lora_dim=256,
                       condition_width=condition_width or state_width, condition_dropout=0.1,
                       cross_attention_every=2, cross_attention_position="before_self",
                       cross_attention_scope="global", cross_attention_3d_rope=cross_attention_3d_rope,
                       condition_mode="cross_attention", pixel_depth=0)
```

Notes:

- Activation checkpointing in `WorldDiT` recomputes only blocks, never `x_embedder`, so the hook is safe.
- The `noanchor` ablations use `use_anchor=False`: `anchor_embed` is not created, the hook is not registered, and `anchored(...)` passes nothing. Everything else, including the projector, is unchanged.
- `num_slots = K` for L1, L2 and S2.
- `condition_dropout=0.1` is SPT's value: it replaces the state by a learned null token for 10% of training samples, which enables guidance on the state. The state tokens have no spatial position, so cross-attention uses no spatial RoPE.
- WorldDiT zero-initializes its output layer, so a freshly built renderer outputs exactly zero and passes no gradient upstream on its first step. That is intended for training; Section 10 says how tests must account for it.

### 4.3 Code lengths (`codes.py`)

The Gaussian code with fixed noise `sigma0` per run (a hyperparameter, Section 1.2), with the prior's predictive spread tied to it. Computed in fp32.

```python
import math

def kl_const(sigma0):                    # per-dimension constant of KL(N(mu, sigma0^2) || N(0, 1))
    return 0.5 * sigma0 ** 2 - 0.5 - math.log(sigma0)

def entry_nats(mu0, sigma0):             # mu0: [B, K, d] -> KL(N(mu0, sigma0^2 I) || N(0, I)) per window, [B]
    mu0 = mu0.float()
    return 0.5 * mu0.pow(2).sum((-2, -1)) + mu0[0].numel() * kl_const(sigma0)

def surprise_nats(mu, m, sigma0):        # mu, m: [B, T, K, d] -> KL(N(mu, sigma0^2 I) || N(m, sigma0^2 I)), [B, T]
    return (mu.float() - m.float()).pow(2).sum((-2, -1)) / (2 * sigma0 ** 2)
```

The constant in `entry_nats` does not affect gradients. Keep it, so that the logged entry cost is the true KL in nats.

Why these are the only constants, and why there is one `beta`:

- The factors `1/2` and `kl_const` are the Gaussian KL itself, `KL(N(mu, s^2) || N(m, r^2)) = (s^2 + (mu - m)^2) / (2 r^2) - 1/2 + log(r / s)` per dimension. Changing them would make the logged rate stop being a code length.
- Tying the prior's spread to `sigma0` loses nothing: a different spread would only rescale the surprise term, which `beta` already does.
- Separate weights on entry and surprise would duplicate `sigma0`. The prior and the renderer read the state through linear layers, so they absorb any rescaling of it, and only the state measured in noise units, `u = mu / sigma0`, matters. In those units the objective is

  ```
  fm(u + noise) + beta * [ (sigma0^2 / 2) * ||u_0||^2 + (1/2) * sum_t ||u_t - m_t / sigma0||^2 ] + const
  ```

  so `sigma0^2` is exactly the price of entry relative to the price of surprise. One `beta` for both keeps the rate an exact code length, and `sigma0` sets the ratio.

### 4.4 State prior for branch S (`prior.py`)

Used only by S2; branch L uses LeWM's predictor (Section 5.2). This is SPT's native state-causal transformer applied to state tokens: one sequence block per step, `K` state tokens plus one action token, and a linear head that predicts the next state's mean directly, as JEPA predictors do.

```python
from torch import nn
from vforge.models.rep_world.config import ConditionerConfig
from vforge.models.rep_world.conditioner import CausalViTBlock
from vforge.models.rep_world.sparsity import state_causal_sparsity, use_segmented_attention

class StatePrior(nn.Module):
    def __init__(self, K, d, action_dim, width=384, depth=6, heads=6):
        super().__init__()
        cfg = ConditionerConfig(width=width, heads=heads, mlp_ratio=4.0)
        self.K = K
        self.state_in = nn.Linear(d, width)
        self.action_in = nn.Linear(action_dim, width)
        self.slot = nn.Parameter(torch.zeros(K + 1, width))          # K state slots plus the action slot
        self.blocks = nn.ModuleList(CausalViTBlock(cfg) for _ in range(depth))
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.head = nn.Linear(width, d)

    def forward(self, s, a):                     # s: [B, T, K, d] past states; a: [B, T, action_dim] actions leaving them
        B, T, K, d = s.shape
        tokens = torch.cat([self.state_in(s), self.action_in(a)[:, :, None]], dim=2) + self.slot   # [B, T, K+1, width]
        h = tokens.reshape(B, T * (K + 1), -1)
        L = T * (K + 1)
        step = torch.arange(L, device=s.device) // (K + 1)
        mask = state_causal_sparsity(L, K + 1) if use_segmented_attention() else step[:, None] >= step[None, :]
        positions = (step.float(), (torch.arange(L, device=s.device) % (K + 1)).float(), torch.zeros(L, device=s.device))
        for block in self.blocks:
            h = block(h, mask=mask, positions=positions)
        out = self.norm(h).reshape(B, T, K + 1, -1)[:, :, :K]        # state-slot outputs of each step
        return self.head(out)                    # means of s_1 .. s_T
```

Initialization: `trunc_normal_(std=0.02)` for all linear weights including `head`, zeros for biases, `slot` normal with std 0.02, and the residual scaling of `vforge.models.rep_world.split_conditioner._initialize` for the blocks. The head is not zero-initialized: there is no residual identity carry (Section 13 keeps one as a fallback).

### 4.5 Query pooling (`pooling.py`)

```python
class QueryPool(nn.Module):                      # K learned queries cross-attend to one frame's tokens
    def __init__(self, width, heads, K, d):
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

## 5. Branch L: LeWM's code base (`as_jepa_lewm/`, branch `as-jepa/lewm`)

Mirror the structure of stable-worldmodel's `scripts/train/lewm.py` (Lightning via `stable_pretraining`, `save_pretrained` checkpoints with a model config carrying `_target_`). Models must be loadable by `swm.wm.utils.load_pretrained` and plannable by `scripts/plan/eval_wm.py`, which needs `encode(info)` and `rollout(info, action_sequence)` with LeWM's signatures.

### 5.1 Data module for L2 and L1

A window dataset over the Lance table, with an index table built once from `get_col_data("episode_idx")` and `get_col_data("step_idx")` (the Lance reader's `get_row_data` and `column_names` leave the index columns out). A window may start at every environment step, as in upstream's dataset. Each sample returns:

- `pixels`: `[W, C, ENC_PX, ENC_PX]` after LeWM's preprocessing (encoder input);
- `pixels_render`: `[W, C, RENDER_PX, RENDER_PX]`, `unit(to_render(raw))` on the raw `uint8` frames, computed before any normalization;
- `anchor_render`: `[C, RENDER_PX, RENDER_PX]`, chosen by the anchor rule with the run's `g` (Section 3.3);
- `action`: `[W, ACTION_DIM]`, normalized;
- `state`: `[W, 7]`;
- `episode_idx`, `start_step`.

Sampling is uniform over valid windows of the train episodes.

### 5.2 L2: AS-JEPA in LeWM's modules (main model)

Components:

- **Encoder:** LeWM's ViT-tiny (`ENC_PX`, patch 14, `embed_dim 192`, random init), all output tokens. Then `QueryPool(width=192, heads=3, K=K, d=d)`, giving `mu: [B, W, K, d]`. No LeWM projector.
- **Action encoder:** LeWM's `Embedder(input_dim=ACTION_DIM, smoothed_dim=ACTION_DIM, emb_dim=STATE_DIM, mlp_scale=4)`. Its width must be `STATE_DIM`, not LeWM's 192: LeWM's `Transformer` projects the action embedding with `cond_proj = Linear(input_dim, hidden_dim)` whenever the state width (`input_dim`) differs from `hidden_dim`, so the action embedding must have the state's width.
- **Prior:** LeWM's `Predictor(num_frames=T, depth=6, heads=16, mlp_dim=2048, input_dim=STATE_DIM, hidden_dim=192, output_dim=STATE_DIM, dim_head=64, dropout=0.1, emb_dropout=0.0)` on the flattened state. Its output is used directly as the predicted mean of the next state; there is no residual head. No `pred_proj`: LeWM normalizes its prediction with a BatchNorm projector, which would put the predicted mean in different coordinates from `mu`, which is not normalized.
- **Renderer:** `AnchoredWorldDiT(renderer_config(d), state_dim=d, num_slots=K)`.

Loss per window (`W = T + 1` frames):

```python
mu = encoder(batch["pixels"])                          # [B, W, K, d]; each frame encoded alone
s = mu + cfg.sigma0 * torch.randn_like(mu)             # the code
act_emb = action_encoder(batch["action"])              # [B, W, STATE_DIM]
m = predictor(s.flatten(2)[:, :-1], act_emb[:, :-1]).view_as(mu[:, 1:])   # predicted means of s_1 .. s_T
entry = entry_nats(mu[:, 0], cfg.sigma0)               # [B]
surprise = surprise_nats(mu[:, 1:], m, cfg.sigma0)     # [B, T]
y = pack(batch["pixels_render"][:, 1:].flatten(0, 1))  # frames 1 .. T; frame 0 is never rendered
anchor = pack(batch["anchor_render"]).repeat_interleave(T, dim=0)
fm = flow_loss(renderer, y, s[:, 1:].flatten(0, 1), anchor, SHIFT)
rate = entry + surprise.sum(1)                         # nats per window, [B]
loss = fm + cfg.beta * rate.sum() / y.numel()          # rate per rendered value, priced at beta
```

- `y.numel()` equals `B * T * FRAME_ELEMS`, so `beta` prices the rate per rendered value and means the same price for every window length and batch size.
- `beta` is constant from the first step. There is no schedule.
- The renderer reads the posterior sample `s_t` (noisy), never `mu` and never the prior's output.
- The prior reads noisy past states and actions only.

### 5.3 L1: L2's architecture with LeWM's objective

The same encoder, pooling, action encoder, predictor and renderer as L2, built by the same code, with no noise on the state and no code length. L1 uses L2's selected window `W` and anchor gap `g`. Loss per window:

```python
mu = encoder(batch["pixels"])                          # [B, W, K, d]
flat = mu.flatten(2)                                   # [B, W, STATE_DIM]
act_emb = action_encoder(batch["action"])
pred = predictor(flat[:, :-1], act_emb[:, :-1])        # [B, T, STATE_DIM]
pred_loss = (pred - flat[:, 1:]).pow(2).mean()         # LeWM's prediction loss, attached target
sigreg = sigreg_fn(flat.transpose(0, 1))               # LeWM's SIGReg, called as LeWM calls it
y = pack(batch["pixels_render"][:, 1:].flatten(0, 1))
anchor = pack(batch["anchor_render"]).repeat_interleave(T, dim=0)
fm = flow_loss(renderer, y, mu[:, 1:].flatten(0, 1), anchor, SHIFT)
loss = pred_loss + cfg.lambda_sigreg * sigreg + fm
```

Design reason: L1 and L2 differ only in the objective (SIGReg on a noiseless, unpriced state versus the code length), so comparing them isolates the price. LeWM's own embedding passes through a BatchNorm projector; L1's does not, so that its state matches L2's. If L1 is unstable without it, add LeWM's BatchNorm projector after the pooling head in L1 only, and report the change.

### 5.4 L0: LeWM as published

Copy `scripts/train/lewm.py` and its config into `as_jepa_lewm/train_lewm.py` and change only the dataset split (upstream splits windows at random, 0.9 and 0.1 with seed 3072; Section 3.2 needs an episode split), the output paths, and `trainer.devices=1`. Upstream's `devices: auto` would make the batch of 128 a per-GPU batch, and `SIGReg` supports a single GPU only, so LeWM's published setting is one GPU. At commit `21446f1`, upstream uses:

- encoder: ViT-tiny, 224 pixels, patch 14, `embed_dim 192`;
- predictor: depth 6, 16 heads, `mlp_dim 2048`, dropout 0.1;
- windows: `history_size 3`, `num_preds 1`, frameskip 5;
- loss: `SIGReg` weight 0.09 (17 knots, 1024 projections);
- optimization: AdamW lr `5e-5`, weight decay `1e-3`, 1% warmup then cosine, batch 128, 100 epochs, bf16, gradient clip 1.0.

### 5.5 Optimization for L2 and L1

AdamW with two parameter groups:

- encoder, pooling, action encoder and prior: lr `5e-5`, weight decay `1e-3`, betas `(0.9, 0.999)` (LeWM's);
- renderer: lr `1e-4`, weight decay 0, betas `(0.9, 0.95)` (SPT's).

Learning-rate schedule: exactly L0's scheduler configuration, through the same code path, for both groups: `LinearWarmupCosineAnnealingLR`, 1% warmup then cosine over the total optimizer steps. This is the optimizer's schedule inherited from LeWM; nothing schedules the objective. Upstream's config says `interval: epoch`, but `stable-pretraining` (0.1.7 and 0.1.8) steps the scheduler after every optimizer step, so the schedule is per step. Log the learning rate at every step in the smoke runs (Section 10.3) for L0, L1 and L2; if the curve is not warmup then cosine over optimizer steps, stop and ask us before Stage A.

Batch and GPUs: global batch 128 windows, bf16 autocast (rate terms in fp32), gradient clip 1.0. L0 and L1 train on one GPU, because `SIGReg` supports a single GPU only. L2 has no `SIGReg` and may use several GPUs with the same global batch (per-GPU batch `128 / n_gpus`, or gradient accumulation).

Budget: the full budget is 100 epochs over the train windows of the run's window length, as in LeWM; a screening run uses `SCREEN_FRAC` of it. Because a window starts at every environment step, this is large: about 1.25 million optimizer steps at `W = 8`, 1.5 million at `W = 4` and 0.7 million at `W = 16` (each with twice as many frames per window). The throughput measurement in Section 10.3 turns these into GPU-hours.

Checkpoints: every 10% and at the end, in `save_pretrained` format with the full model config. `swm.wm.utils.load_pretrained` requires each checkpoint folder to hold exactly one `.pt` file plus its `config.json`.

### 5.6 Inference interface for L2 and L1 (`as_jepa_lewm/model.py`)

Subclass `stable_worldmodel.wm.lewm.LeWM` with `projector=None` and `pred_proj=None`; LeWM then uses identities for both, so its `predict(emb, act_emb)` returns the predictor's output directly, as L2 and L1 need. Add the pooling head and the renderer as submodules, and override only `encode`:

- `encode(info)`: run LeWM's encoder as LeWM does (`interpolate_pos_encoding=True`), pool all output tokens of `last_hidden_state` with `QueryPool` instead of taking the CLS token, and set `info["emb"] = mu.flatten(2)` with shape `[B, H, STATE_DIM]`, and `info["act_emb"]` when actions are present. Same keys as LeWM.
- `rollout(info, action_sequence, history_size=None)`: LeWM's `rollout` (`stable_worldmodel/wm/lewm/lewm.py`), inherited unchanged. Its contract, which the planner relies on: with `H` context frames, `action_sequence` holds strictly future candidates, and when `H > 1`, `info["action_history"]` of shape `[B, S, H - 1, ACTION_DIM]` holds the executed action blocks between the context frames (asserted); the current frame is paired with the first candidate. `history_size` defaults to `self.predictor.num_frames` (the predictor's positional table is only `num_frames` long). It returns `info["predicted_emb"]` of shape `[B, S, H + steps, STATE_DIM]`, whose first `H` entries are the encoded context frames. For planning, roll out prior means (no noise). For generation, Section 8.2 specifies means and samples.
- `render(states, anchor_render, *, steps=SAMPLER_STEPS, cfg_scale=CFG_SCALE, seed)`: `flow.sample` in eval mode, then `unpack`.

### 5.7 L ablation

| Id | Change from L2's final configuration | Seeds |
| --- | --- | --- |
| `L2-noanchor` | renderer without the anchor branch (`use_anchor=False`) | 0 |

The window length and the anchor gap are no longer ablations: Stage B searches them.

### 5.8 Planning for branch L

Run `scripts/plan/eval_wm.py` with `scripts/plan/config/pusht.yaml` unchanged except `policy` (the checkpoint) and the history length, set with the Hydra override `+plan_config.history_len=3` for every model (the key is commented out upstream; its default is 1). With it, the first plan of each episode sees one frame and later plans see three frames, 5 steps apart, plus `action_history`, so `rollout` is called with `H = 1` and with `H = 3`. Starts are drawn from `pusht_expert_train.h5` (Section 3.1), and the goal is the same episode 25 steps later. Upstream settings at `21446f1`:

- CEM solver: 300 samples, 30 iterations, top 30;
- horizon 5, receding horizon 5, action block 5;
- `GoalMSE` with sum reduction, scored at the last predicted step;
- 50 episodes, goal offset 25 steps, budget 50 steps.

This protocol is P0. Upstream's CEM now uses the population standard deviation, so success rates may differ slightly from the LeWM paper; compare only against our retrained L0. For final configurations use three evaluation seeds (42, 43, 44) for each model seed and report success rate as mean and standard deviation over all of them. For screening runs, Section 11 uses one evaluation seed (42).

Planning needs no anchor and no renderer, so the anchor gap does not enter it.

If an AS-JEPA model passes the generation gate but its P0 success is more than `PLAN_MARGIN` below L0's, run these variants in order and report each:

- **P1:** the goal cost summed over every predicted step instead of only the last: a new objective config following `GoalMSE`, selected by overriding the `objective` entry of the Hydra defaults. It must skip the first `H` entries of `predicted_emb`, which are the encoded context frames.
- **P2:** the iCEM solver at the same sample budget, through a new solver config (only `cem.yaml` and `adam.yaml` ship). iCEM clamps candidates to the action space's bounds, while the candidates are z-scored actions; before using it, confirm in a dry run that the clamp does not cut off normalized actions, and if it does, pass bounds that cover their range.
- **P3:** horizon 10, receding horizon 5.
- **P4:** the cost averaged over 8 prior-sample rollouts (`m + sigma0 * noise` at each step) instead of the mean rollout.

## 6. Branch S: SPT's code base (`runtime/vforge/tasks/as_jepa/`, `jobs/as_jepa/`, branch `as-jepa/spt`)

Native family `as_jepa_spt`, following `runtime/AGENTS.md`: model code in `runtime/vforge/models/as_jepa/`, task code in `runtime/vforge/tasks/as_jepa/`, launchers and full configs in `jobs/as_jepa/`. Register the family at the top of `jobs/as_jepa/train_entry.py`, before training starts:

```python
from vforge.tasks.registry import ModelTasks, register
register("as_jepa_spt", ModelTasks(train="vforge.tasks.as_jepa.train:build_train_components",
                                   train_preflight="vforge.tasks.as_jepa.shared:validate_train_request"))
from vforge.tasks.train import run_train  # then: run_train(OmegaConf.load(sys.argv[1]))
```

Mirror `runtime/vforge/tasks/rep_world/train.py` for `build_train_components`. Keep its pieces: `build_resumable_loader`, `ModelSpec` with meta initialization and `initialize_weights`, the `StepOutput` contract, and a dataset `resume_contract`. Change the dataset, the step function and the optimizer: AdamW with `use_muon` false, and no EMA. Full configs in `jobs/as_jepa/configs/` follow the layout of `jobs/configs/pretrain_50k.yaml`, with `model.name: as_jepa_spt` and the settings below under `model.extras`.

### 6.1 Geometry and data

```python
from vforge.models.rep_world.config import CodecGeometry
PUSHT_RENDER = CodecGeometry(family=f"pixel{RENDER_PX}", variant=f"rgb1p{PATCH}", rgb_frames=1,
                             channels=TOKEN_CH, planes=1, grid=GRID, spatially_ordered=True)
```

Construct it directly; do not add it to the pinned `CODECS` table. The family label must differ from SPT's `"pixel"`, so that SPT's RGB-codec code paths (which test `family == "pixel"`) are not triggered. The dataset reads the Section 3.3 export and returns:

- `latents`: `pack` frames arranged `[TOKEN_CH, W, GRID, GRID]`, the packed layout `[P*C, S, H, W]` that `rep_world` uses;
- `actions`: `[W, ACTION_DIM]`;
- `anchor_render`;
- `state`;
- `sample_id`.

### 6.2 S2: AS-JEPA in SPT's modules (main model; `ASJepaSPT` in `runtime/vforge/models/as_jepa/model.py`)

Components:

- **Encoder:** `StateViTEncoder(ConditionerConfig(input="latent", width=384, depth=7, encoder_depth=6, heads=6, mlp_ratio=4.0, feature_layers=(6,), architecture="split_ec", context_mask_ratio=0.0), PUSHT_RENDER)`, then `QueryPool(width=384, heads=6, K=K, d=d)`. Each frame is encoded alone, with no masking.
- **Prior:** `StatePrior(K, d, ACTION_DIM)` (Section 4.4).
- **Renderer:** `AnchoredWorldDiT(renderer_config(d), state_dim=d, num_slots=K)`, identical to L2's.
- **Loss:** exactly L2's (Section 5.2), with `mu = pool(encoder(latents).flatten(0, 1)).view(B, W, K, d)` and `m = prior(s[:, :-1], actions[:, :-1])`, the same `beta`, `sigma0` and normalization.
- **Initialization:** the Native trainer builds the model on the meta device and then calls only `initialize_weights`, so `ASJepaSPT.initialize_weights` must initialize every parameter: `encoder.initialize_weights()`, `pool.initialize()`, the prior as in Section 4.4, and `renderer.initialize_weights()`.
- **Window and anchor:** `W` kept frames and anchor gap `g`, from the S2 search (Section 9.5).

### 6.3 S0: SPT with actions (comparison)

`RepWorld` with SPT's split encoder in latent-input mode, actions added to each state's encoded tokens:

```python
class ActionSplitViT(SplitViTConditioner):
    def __init__(self, config, geometry, action_dim=ACTION_DIM):
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
                with self.net.anchored(pack(anchor).repeat_interleave(count, dim=0)):
                    return super().forward(latents=latents, **kwargs)
            return super().forward(latents=latents, **kwargs)
        finally:
            enc._actions = None
```

Configuration:

- **Conditioner:** `ConditionerConfig(input="latent", width=384, depth=9, encoder_depth=3, heads=6, mlp_ratio=4.0, feature_layers=(8,), pretrained_grid=GRID, context_mask_ratio=0.9, architecture="split_ec", mask_stage="before_e", temporal_patch=2)`. This is SPT's design at small width: ViT-E with 3 blocks per state, then a state-causal ViT-C with 6 blocks, and 90% of patches dropped before ViT-E during training.
- **World:** a plain `WorldDiT(renderer_config(d, condition_width=384, cross_attention_3d_rope=True))`, that is, the shared renderer configuration with SPT's conditioning width and spatially aligned cross-attention.
- **Flow:** `FlowConfig(mode="no_zprev", distant_target_prob=0.0, target_layout="single_state", target_origin_reserve=1, future_target_choices=2)` with SPT's defaults for the time distribution and shift. Every origin `i` predicts state `i+1`. This deviates from SPT's pretraining, which also targets `i+2` half the time: the skipped intermediate action would be invisible to the context, so we use SPT's next-state target throughout.
- **Window:** `W_SPT` kept frames.

### 6.4 S1: S0 plus the anchor (comparison)

`ActionRepWorld(..., anchored=True)` with `renderer_config(d, condition_width=384, cross_attention_3d_rope=True)`, everything else as S0. The anchor is window frame 0 (`g = 0`), the same for all targets of a window; the targets are frames 1 to `W_SPT - 1`, so the anchor is never a target.

### 6.5 Optimization for all S runs

Native trainer:

- AdamW (`use_muon: false`), lr `1e-4`, weight decay 0, betas `(0.9, 0.95)`, eps `1e-8`;
- `constant_with_warmup` with 100 warmup steps (SPT's learning-rate warmup; nothing schedules the objective);
- bf16 autocast, rate terms in fp32, gradient clip 1.0;
- EMA disabled;
- global batch 128 windows, on any number of GPUs.

Use `attention.backend: sdpa`, or `flash_attention_4` if it runs these shapes; the semantics must match. Budget: `max_steps` chosen so that each S run sees the same number of windows as the corresponding L run (100 epochs, or `SCREEN_FRAC` of that for screens). Checkpoint every 10% in full DCP.

### 6.6 Inference for branch S

Write `rollout_s01(model, context_latents, actions, anchor, steps=SAMPLER_STEPS, cfg_scale=CFG_SCALE)`, mirroring `RepWorldInference._semantic`. At each step, set the conditioner's actions to those of the states being encoded, generate the next frame with SPT's sampler, append it, and re-encode it. The history window is `W_SPT - 1` states, the context length S0 and S1 were trained with.

For S1, wrap every velocity call of the sampler in `model.net.anchored(pack(anchor))`; `ActionRepWorld.forward` sets the anchor only for the training loss.

For S2, the rollout is L2's: encode the context frames, roll out the prior, render with the anchor.

Planning for S2 is optional in this round. If you run it, write an adapter exposing `encode` and `rollout`. The adapter must undo the evaluation's ImageNet normalization, rescale to `uint8`, and apply `unit(to_render(...))`.

### 6.7 S ablation

`S2-noanchor`: S2's final configuration without the anchor branch, seed 0.

## 7. Design invariants: what must not change

Each line is a property the method depends on.

1. The AS-JEPA encoder reads one frame, nothing else: no history, no anchor, no actions. (The state is a function of its frame; goal images encode without history.)
2. The anchor reaches only the renderer. The prior never reads frames or renderer features. (Otherwise appearance becomes free in the state and is never excluded.)
3. Actions reach only the prior, plus S0 and S1's conditioner by design. The AS-JEPA renderer never sees actions or history. (Otherwise the renderer does the dynamics.)
4. The renderer reads the posterior sample `s_t`, the encoded current frame, never the prior's prediction. In training, the prior's output enters only the code length.
5. Window frame 0 is encoded and starts the chain, and is never rendered, for every anchor gap.
6. No augmentation, EMA (target or evaluation), stop-gradient, target network, SIGReg, VICReg or masking in L2 and S2. L1 keeps SIGReg by design; S0 and S1 keep SPT's masking by design.
7. Nothing normalizes `mu`, `s`, the prior's predicted mean, or the state tokens before the renderer's linear projection.
8. Within a run, the code noise `sigma0` is fixed, the prior's spread is tied to it, and the rate is the exact Gaussian KL of Section 4.3 with one `beta` for entry and surprise. One-step prediction only; no multi-step targets or overshooting.
9. The renderer's target is fixed data (packed rendered frames), never a learned representation.
10. `beta` is constant during training. The only exception is the documented collapse fallback (Section 13).

## 8. Shared evaluation package (`as_jepa_tools/eval/`)

Every number in the reports comes from this package, applied identically to every model through a small adapter. Each model provides:

- `encode(frames)`: states, for probes;
- `rollout(context, actions)`: predicted states, or frames for S0 and S1;
- `render(states, anchor)`: frames.

### 8.1 Pose estimator

A ResNet-18 (torchvision, random initialization) on rendered frames regresses `[agent_x, agent_y, block_x, block_y]` (divided by `WORKSPACE`) and `[sin, cos]` of the block angle.

- Training: train episodes of `D-std`, 30 epochs, AdamW lr `1e-3`, cosine schedule, batch 512, with mild Gaussian blur and pixel noise so that it reads imperfect generated frames. This augmentation belongs to the evaluator only, not to any model.
- Report its error on real test frames as the floor of every pose metric.

Errors are measured in workspace pixels for positions and in degrees for the angle, wrapped to `[-180, 180]`.

### 8.2 Generation protocol (main criterion)

Fixed test windows: 256 windows of `EVAL_W` kept frames from test episodes, chosen once with seed 0, stratified half and half by whether the block moves more than `MOVE_PX`, and starting at least `g_max` kept frames after the episode's start, where `g_max` is the largest anchor gap in the grid, so that every model has an eligible past anchor. With `g_max = 8`, about 57% of episodes are long enough, which leaves ample test windows. If fewer than 256 test windows satisfy this, use all that do and report the number.

- Context: frames 0 to `EVAL_CTX - 1` and all actions. Predict the remaining `EVAL_W - EVAL_CTX` frames. Through LeWM's `rollout` contract (Section 5.6), the `EVAL_CTX - 1` actions between context frames go in `action_history` and the rest are the candidates.
- Test-time anchor, by the model's anchor gap: for `g = 0`, context frame 0; for `g >= 1`, the frame `g` kept frames before context frame 0 (primary), and also context frame 0 (secondary: deployment at the start of an episode, where no distant past frame exists; reported separately, not used for selection or the gate). S1 uses context frame 0.
- Rollout context: AS-JEPA priors and L1's predictor keep the last `num_frames = T` states; S0 and S1 keep the last `W_SPT - 1` states. Run every model in eval mode.

Variants:

- **mean:** AS-JEPA context states are `mu` of the context frames; roll out prior means.
- **sample:** four samples per window; context states `mu + sigma0 * noise`; roll out prior samples `m + sigma0 * noise`.

Render with `SAMPLER_STEPS` Euler steps and guidance `CFG_SCALE`, and report both variants. S0 and S1 use their own rollout (Section 6.6).

Metrics at every horizon (tables report `HORIZONS`):

- agent position error, block position error and block angle error, from the pose estimator against true simulator states;
- LPIPS (AlexNet), SSIM and PSNR against the true frame;
- FVD as a secondary metric, using SPT's bundled I3D detector on clips upsampled to 224 pixels.

Reference: copy the last context frame forward. Report every error relative to this copy baseline as well as absolute.

### 8.3 Counterfactual actions

Setup:

- 64 start points: a test episode and a kept frame `k` with context frames `k - EVAL_CTX + 1` to `k`, where `k - EVAL_CTX + 1 >= g_max` so that every model has an eligible past anchor.
- At frame `k`, restore the simulator to the recorded state, using the environment's `_set_state` as the planning evaluation does.
- Execute 4 action sequences of 8 kept steps: the recorded one, and 3 recorded sequences taken from other test windows.

The simulator renders the true futures. The model generates one rollout per sequence from the same context, with the anchor chosen by the rule of Section 8.2.

Report:

- pose errors as in Section 8.2;
- action identification accuracy: assign each generated rollout to the true future with the smallest pose error, and count the fraction assigned to their own sequence (chance 0.25).

### 8.4 Planning

Branch L: Section 5.8. Save the planning evaluation's videos, five successes and five failures per final model, in the gallery (Section 8.7).

### 8.5 Probes (understanding, not gating)

Ridge regression (closed form; `alpha` chosen by 5-fold cross-validation on train features from 50,000 frames), reporting `R^2` on test episodes.

Features:

- L2, S2 and L1: `mu`, flattened to `STATE_DIM`;
- L0: LeWM's embedding;
- S0 and S1: `model.causal_semantic_encoder.vit_e(...)` output tokens, mean-pooled (called directly, so no action embedding is added).

Targets: agent position, block position, block angle as sin and cos, and agent velocity.

### 8.6 Rate diagnostics (AS-JEPA only)

Log during training:

- `entry_nats` per window and `surprise_nats` per step;
- `fm`;
- `rate_share = beta * rate.sum() / y.numel() / fm`;
- per-dimension rate (the mean over a held-out batch of each dimension's `0.5 * mu_0^2` and its summed surprise), and the number of active dimensions (above `ACTIVE_NATS` per window);
- gradient norms of encoder, prior and renderer.

### 8.7 Galleries for human inspection

For every final model and every evaluated checkpoint of it, under `<run>/eval/<checkpoint>/gallery/`:

- `fixed/<case>/`: 32 fixed cases, the first 32 of the Section 8.2 windows, identical for all models. Each holds `gt.mp4`, `pred_mean.mp4`, `pred_sample_{0..3}.mp4`, `copy.mp4`, a `grid.png` (rows: truth, mean, samples; columns: every predicted horizon), and `meta.json` (episode, start step, actions, per-horizon errors, model, checkpoint, seed, anchor index).
- `best/` and `worst/`: the 8 windows with the lowest and highest mean pose error over horizons, same format.
- `counterfactual/<case>/`: 8 start points, each with the 4 true futures, the 4 generated rollouts, a grid and metadata.
- `swap/<pair>/`: anchor models only, 8 pairs. Render episode B's predicted states with episode A's anchor; save A's anchor image, B's true video, the swapped video, and metadata with B's pose errors. On `D-std` the appearance is shared, so this checks that the render follows B's states rather than copying the configuration shown in A's anchor.
- `planning/`: success and failure videos.

Mark each fixed case `pass` or `fail` in `fixed/review.csv`. A pass means the motion is plausible and the objects are intact, with no duplicated, vanished or deformed blocks.

### 8.8 Reports

Write `reports/<stage>.md` after Stage B (selection) and after Stage C with the comparisons (the `D-std` report), with:

- tables for all metrics, as mean and standard deviation over seeds;
- for screens, the grid of generation scores, rates and collapse flags over `beta`, `sigma0`, `g` and `W`;
- the dataset card;
- links to galleries;
- configs, commit hashes (SPT-JEPA and stable-worldmodel) and the GPU-hours used.

Also write `reports/<stage>.json` with one record per run: `run_id`, `branch`, `model`, `dataset`, `beta`, `sigma0`, `gap`, `window`, `lambda_sigreg`, `seed`, `checkpoint`, `fallback`, and every metric name exactly as in this section.

## 9. Run plan

Run IDs follow `<model>-<dataset>-b<beta>-s0<sigma0>-g<gap>-w<window>-seed<seed>`, with fields that do not apply omitted. All runs use `D-std`.

### 9.1 Dependencies

- Stage A needs L2 implemented and tested (Section 10). Stage B needs Stage A's selection; Stage C needs Stage B's.
- L1 and `L2-noanchor` need Stage B's selection (L1 uses L2's `W` and `g`).
- L0, S0 and S1 depend on nothing in the search and can start as soon as they pass their tests.
- S2 depends on nothing in full mode; in reduced mode it needs L2's Stage B selection (Section 9.5).

### 9.2 Stage A: L2 screen

Full grid at `W = 8`, seed 0, `SCREEN_FRAC` budget: `beta` (6 values) x `sigma0` (3 values) x `g` in {0, 8}, 36 runs. If they cannot all run at once, launch `sigma0 = 0.1` first (12 runs), then `0.03`, then `0.3`. Then apply Section 11.

Grid extension: if the selected `beta` or `sigma0` lies at the edge of its grid, extend that axis by one step beyond the edge (a factor of 4 for `beta`, 3 for `sigma0`) at the selected values of the other hyperparameters, and select again.

### 9.3 Stage B: anchor gap and window length

At the selected `(beta*, sigma0*)`: `g` in {0, 2, 4, 8} x `W` in {4, 8, 16}, minus the two cells Stage A already ran and the cell that fails `FALLBACK_MAX` (`W = 16, g = 8`; Section 3.3), giving 9 runs at seed 0 and `SCREEN_FRAC` budget. Apply Section 11 to Stages A and B together and keep the top two configurations.

### 9.4 Stage C: confirmation

The top two configurations, seeds {0, 1, 2}, full budget: 6 runs. Of the two, the one Section 11 selects on three-seed means is L2's final configuration.

### 9.5 Branch S

- **S2, full mode** (enough compute for both branches at once): Stages A to C with the same grids and rules as L2, in parallel with branch L.
- **S2, reduced mode:** after L2's Stage B, run `beta` in {`beta*/4`, `beta*`, `4 * beta*`} at L2's `(sigma0*, g*, W*)`, seed 0, `SCREEN_FRAC` budget (3 runs); then the configuration Section 11 selects, seeds {0, 1, 2}, full budget (3 runs). The renderer, data and loss are shared with L2, so the rate-distortion balance should transfer approximately; the 3-point check covers the different encoder and prior.
- **S0** and **S1:** seeds {0, 1, 2}, full budget each.
- **`S2-noanchor`:** Section 6.7.

### 9.6 Comparisons and ablation in branch L

- **L0:** seeds {0, 1, 2}, full budget.
- **L1:** `LAMBDA_SIGREG` in {0.03, 0.09, 0.27} at L2's `(W*, g*)`, seed 0, `SCREEN_FRAC` budget; then the best (Section 11, steps 1 to 4, then the best score), seeds {0, 1, 2}, full budget. This gives L1 a tuning opportunity on its own knob; L2's search is larger, and the report must say so.
- **`L2-noanchor`:** Section 5.7.

### 9.7 Priorities when compute is short

You decide the allocation. Run in this order of priority:

1. L2: Stages A, B and C.
2. S2 in reduced mode, then its confirmation.
3. L0 and L1.
4. S0 and S1.
5. `L2-noanchor` and `S2-noanchor`.

If compute suffices for only one branch, run branch L (items 1, 3 and the L part of 5) and skip branch S. With ample compute, start everything without a dependency at once: Stage A, L0, S0, S1, and S2 in full mode.

Seeds: three seeds for every final configuration and comparison give a first estimate of seed-to-seed variation. If the difference between L2 (or S2) and a comparison is smaller than twice the larger of their seed standard deviations, add seeds 3 and 4 to both before drawing a conclusion.

Before Stage A, measure throughput in the smoke runs (Section 10.3), report the projected GPU-hours of the runs you intend to launch, and wait for our acknowledgement if the total exceeds what was agreed.

### 9.8 Reports and stopping point

- After Stage B: the selection report (Section 8.8).
- After Stage C and the comparisons: the `D-std` report, the gate check (Section 12), and stop. Do not start Appendix A or other environments without our agreement.

## 10. Tests and checks before any long run

Zero initializations make several checks trivial or impossible at initialization: WorldDiT's output layer (`final_layer.linear` and `final_layer.adaLN_modulation[-1]`), `anchor_embed`, and S0's `action_proj`. In every test below that probes gradients, causality, or the effect of an input, first re-initialize those layers with `normal_(std=0.02)`; the equivalence tests run on the true initialization.

### 10.1 Shared components (CPU, small configs, `sdpa_math` backend)

- `pack` then `unpack` is the identity; `to_render` matches between the branch-L loader and the branch-S export on 1000 rows.
- `timestep_shift(renderer_config(d))` equals `RepWorld.timestep_shift` built with the same world config and `FlowConfig()`, and `draw_timesteps` equals `RepWorld.draw_timesteps` on identical uniforms.
- `AnchoredWorldDiT`:
  - With the output layer perturbed and `anchor_embed` at zero, its output equals a plain `WorldDiT` with the same weights, for any anchor.
  - With the output layer perturbed, one optimizer step on a nonzero loss makes `anchor_embed`'s weights nonzero.
  - Outside `anchored(...)` it raises.
  - Scaling the state tokens by 2 changes the output (no normalization before the projector).
  - Permuting the state tokens changes the output (slot identity).
- `entry_nats` and `surprise_nats` equal `torch.distributions.kl_divergence` between the corresponding Normals, for every `sigma0` in the grid.
- `StatePrior`: its output for step `t` is unchanged when states or actions at steps later than `t` change.

### 10.2 Per model (CPU, small configs)

- **Encoder locality:** changing frame `j` leaves `mu` of every other frame unchanged (L2, S2, L1).
- **Prior causality:** changing frames or actions after step `t` leaves the predicted means for steps up to `t+1` unchanged.
- **Anchor isolation:** the gradient of `mu` and of the prior's means with respect to the anchor is exactly zero.
- **Renderer isolation:** the gradient of the rendered output for frame `t` with respect to actions and to states other than `s_t` is exactly zero.
- **Gradient reach:** every loss term produces nonzero gradients in encoder, prior and renderer. In L2 and S2, the rate term reaches the encoder and the prior; the flow loss reaches the encoder through `s_t`.
- **Same modules in L1 and L2:** they build modules with identical parameter names and shapes.
- **Action alignment:** the action vector paired with frame `t` is the `FRAMESKIP` raw actions recorded after frame `t`, checked against dataset indices.
- **Anchor sampler:** for every `g` in the grid, the anchor comes from the same episode and respects the gap; it is deterministic given the seed; `g = 0` returns window frame 0.
- **S0:** with `action_proj` at zero, `ActionSplitViT` equals `SplitViTConditioner`; changing `a_t` changes contexts at origins `>= t` only.
- **Branch L checkpoints and planning:** a saved L2 checkpoint loads with `swm.wm.utils.load_pretrained` and runs planning steps of `eval_wm.py` with `+plan_config.history_len=3`, which exercises `rollout` with `H = 1` and `H = 3`. A direct test checks that `rollout` with `H = 3` and `action_history` gives the same predictions as feeding the same frames and actions to the predictor by hand.
- **Branch S:** a 3-step Native training on CPU with a tiny config, and a checkpoint save-and-resume round trip.

### 10.3 GPU smoke (200 steps per model, one GPU)

Check:

- the loss is finite;
- `entry_nats`, `surprise_nats` and `fm` all move;
- for AS-JEPA, renders from pure noise change when the state is replaced by zeros;
- the learning-rate curve (Section 5.5);
- throughput and peak memory are recorded.

## 11. Selection rule

Applied to screening runs (one seed each) and to confirmed configurations (three-seed means):

1. Exclude collapsed runs (Section 13).
2. Generation score: the mean over agent and block of the position error at horizon 7 divided by the copy baseline's, mean rollout, `D-std` fixed test windows, primary test anchor. Lower is better.
3. Tied set: runs whose score is within `SEL_TOL` of the best score.
4. Run P0 planning with one evaluation seed on the tied set. If the best and worst planners in it differ by more than `PLAN_MARGIN`, keep only the runs within `PLAN_MARGIN` of the best planner.
5. Choose the run with the lowest rate per kept frame, `(entry + summed surprise) / W` on held-out windows: among equally good world models, the one whose state uses the fewest nats.

"The top two" applies steps 3 to 5 again after removing the first choice. L1 has no rate, so it uses steps 1 to 4 and then the best score.

State-size check: at L2's and S2's final configurations, count active dimensions. If all `STATE_DIM` dimensions are active, the state may be too small: rerun the final configuration with `d` doubled (seed 0) and report the result.

## 12. Gate: when `D-std` counts as working

Moving beyond `D-std` (Appendix A, other environments) needs our agreement after the `D-std` report. Agreement requires at least one AS-JEPA final configuration (L2 or S2, three seeds) to meet the generation gate:

- **Rollouts:** agent position error at horizon 7 at most `GEN_RATIO_MAX` times the copy baseline's, and block position error in moving-block windows below the copy baseline's (mean rollout, primary anchor, mean over seeds). Half of the copy baseline means the model captures at least half of the motion; the block condition makes sure contact is handled, not only free agent motion.
- **Counterfactuals:** action identification accuracy at least `ACTION_ID_MIN`.
- **Inspection:** at least `GALLERY_PASS_MIN` fixed cases marked `pass`.

Planning is the second main criterion. Report P0 for L0, L1 and L2's final configuration, and P1 to P4 when the generation gate passes but P0 trails L0 by more than `PLAN_MARGIN` (Section 5.8). Probes, rates and FVD are reported for understanding and do not gate.

## 13. Failure responses

**Collapse test**, run on every screening run at its end and on the smoke runs: render each held-out frame `t >= 1` from its own encoded state (`mu(x_t) + sigma0 * noise`) and the run's anchor, and measure the agent position error with the pose estimator, averaged over the frames whose agent lies more than `MOVE_PX` from its position in the anchor frame (elsewhere copying the anchor is already nearly exact, so the comparison says nothing). The run has collapsed if this mean error is at least `COLLAPSE_RATIO` times the mean error, over the same frames, of using the anchor frame itself as the prediction, or if fewer than 2 state dimensions are active (the agent's position alone needs two).

| Symptom | Likely cause | Response |
| --- | --- | --- |
| Collapse in some grid cells | the price is above the collapse barrier there, or training stalled before the renderer used the state | Rerun the cell with fallback 1: an identity-carry prior, `m_t = s_{t-1} + head(...)` with the head zero-initialized. If it still collapses, fallback 2: `beta` ramped linearly from 0 over the first 5% of steps (fixed; do not tune). Report which cells needed which fallback. A configuration selected with a fallback keeps it in all its later runs. |
| Collapse in every cell, with both fallbacks | the objective or an implementation error | Check the projector replacement and the anchor isolation tests. If they pass, stop and report. |
| Good at horizon 1, fast drift after a few steps | prior errors compound in open loop | Report mean and sample rollouts and the per-horizon curve; check the prior's context length at rollout. Do not add multi-step training without our agreement. |
| Block copied from the anchor and never moves in rollouts; block-pose probe low | block pose left out of the state | Compare across `g` and `W` in Stage B (larger gaps and longer windows should help) and across `sigma0` in Stage A (smaller noise should help); report the block-motion fraction from the dataset card. |
| NaN or loss spikes | numerics | Check that rate terms are fp32; halve the learning rate; keep gradient clipping. |
| S0 or S1 renders noise after training | SPT at this scale needs more updates | Compare with L1 at the same number of windows seen. Extend S budgets by 50% (Section 14); report. |
| L1 unstable | its state has no BatchNorm projector | Add LeWM's BatchNorm projector to L1 only (Section 5.3); report. |
| Good generation, poor planning | the planning cost reads the state poorly | Run P1 to P4 (Section 5.8) and report each. |

## 14. Latitude: what you may change without asking

- Compute allocation and the order of runs, within the dependencies and priorities of Section 9.
- Learning rates by a factor of 0.5 or 2, applied to the same model across all its runs.
- Per-GPU batch and number of GPUs, keeping the global batch of 128 windows (use accumulation if needed).
- Extending a grid by the rules of Section 9.2.
- Training budget up to 1.5 times for all runs of a branch equally, if losses are still falling at the end. Report it.
- Evaluation sample counts upward.
- The attention backend, if the semantics are identical.
- Implementation mechanics not fixed here: Lightning wiring, file splitting, logging tools.

Everything else, especially anything in Section 7, needs our agreement. When unsure, stop and ask.

## 15. Order of work

1. **Environments and SPT smoke** (Section 2). In parallel:
   - `D-std` data, export and dataset card (Section 3);
   - shared components and their tests (Sections 4 and 10.1);
   - the evaluation package with the pose estimator, run on ground-truth data to produce the copy baselines and estimator floors (Section 8).
2. **Implement L2**, then L1 and L0 (Section 5); in parallel, if branch S runs, S2 and then S0 and S1 (Section 6). Per-model tests (10.2) and GPU smokes (10.3). Report throughput and projected GPU-hours.
3. **Stage A**, plus everything without a dependency that compute allows (Section 9.7).
4. **Stage B**; selection report.
5. **Stage C**, L1, the ablations and the remaining S runs.
6. **`D-std` report and gate check**, then stop.

## Appendix A. Deferred datasets (after the `D-std` report, with our agreement)

These datasets test the two purity claims (appearance and transient detail are excluded from the state). Nothing here is collected or trained in this round.

- **`D-var`:** PushT with appearance randomized per episode, collected with stable-worldmodel's collection API as `scripts/data/collect_pusht_fov.py` does:

  ```python
  world = swm.World("swm/PushT-v1", num_envs=10, max_episode_steps=123,   # D-std's median length; report it
                    image_shape=(ENC_PX, ENC_PX), render_mode="rgb_array")
  world.set_policy(WeakPolicy(dist_constraint=100, seed=20261026))       # unseeded by default
  world.collect(path, episodes=<equal to the number of D-std episodes>, seed=20261026,
                options={"variation": ("agent.color", "block.color", "background.color",
                                       "goal.color", "goal.position", "goal.angle",
                                       "agent.start_position", "block.start_position", "block.angle")})
  ```

  Check the names against `swm fovs PushT-v1` at the pinned commit; passing `variation` replaces the default list. The default format is Lance with JPEG frames, the same as `D-std`. The sampled values are saved per step as `variation.<name>` columns; probes read them. Block shape and scale and agent shape and scale stay at their defaults. `D-var` trains and evaluates separate models; it is not mixed with `D-std`.
- **`D-dot`:** `D-std` frames with one filled disc redrawn independently every frame, applied in the loader before resizing: radius 4% of the frame side, centre uniform over the image, colour uniform in RGB, from `torch.Generator().manual_seed(hash((episode_idx, step_idx, 7)) & 0x7FFFFFFF)`, so a frame always shows the same dot at every resolution. Record the dot's centre and colour.
- **Additions to evaluation:** a pose estimator trained on `D-var` (the `D-std` estimator also scores `D-dot`); probes for the colours of agent, block, background and goal and for the goal pose on `D-var`, and for the dot's centre and colour on `D-dot`; on `D-var`, the swap gallery records whether rendered colours at B's object positions are closer to A's or B's recorded colours.
- **Runs, at the final configurations:** L0, L1 and L2 on `D-var`, then S2; L0, L2 and `L2-mse` (a deterministic renderer: called with zero input at `t = 1` and trained with `(v - y)^2`, a direct regressor of the frame) on `D-dot`.
