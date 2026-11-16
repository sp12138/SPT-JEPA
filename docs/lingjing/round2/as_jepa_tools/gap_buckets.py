"""Does the model's behaviour depend on how far the renderer's anchor is from the frame? Per-frame diagnostics in gap buckets.

The renderer draws frame x_t from the state s_t and an anchor frame of the same episode, and is not told how far
apart the two are. That distance (the gap, in kept frames) depends on the anchor rule: PushT draws the anchor at
least 8 kept frames from the window, on either side; the benchmark default is window frame 0, so the gap of a
frame is its index in the window; `GapBenchmarkWindows` draws a past frame. This tool measures, on held-out
windows of a trained checkpoint, which quantities move with the gap. It only evaluates the model: no weight
changes, every random draw is fixed by --seed, and the global random generators are left as they were.

One record per window frame. Frames 1..W-1 are the rendered frames, as in the model's loss; frame 0 is not
rendered, so its record carries only the entry code length and the model-free fields.
- gap: absolute distance between the anchor and the frame in kept frames, from the dataset's own fields:
  |rows[t] - anchor_row| over the row spacing of the window. `offset` keeps the sign (positive: the anchor lies
  in the frame's past). anchor_future and anchor_fallback are the 0/1 flags of a future anchor and of a window
  whose episode had no anchor that satisfies the rule.
- render_loss: the renderer's training loss of the frame (`flow_loss` before its mean over frames), given the
  frame's posterior sample s_t = mu(x_t) + sigma0 * noise (AR(1) noise when the model has rho > 0) and its
  anchor. It is the mean over a small grid of flow noise times, placed at quantiles of the training
  distribution of times, with one fixed noise draw per frame and time. The three conditions below reuse these
  times and draws, so their differences from render_loss are paired.
- render_loss_swapped, state_reliance: the same loss when the renderer reads the state of the same frame index
  in another window of the batch, and its excess over render_loss: what the renderer loses without the right
  state. The partner windows are a roll of the batch, chosen to avoid partners from the same episode;
  swap_same_episode flags the remaining ones.
- render_loss_swapped_time, state_reliance_time: the same with the state of another rendered frame of the same
  window, half a window away: the same episode at a different time.
- render_loss_zero_anchor, anchor_reliance: the same with the anchor zeroed, as `renderer_anchor_mode="zero"`
  does, and its excess over render_loss: what the renderer loses without the anchor.
- surprise_nats, entry_nats: the model's code lengths. `surprise_nats` per rendered frame, teacher-forced on the
  same posterior samples (the exact AR(1) form when rho > 0); `entry_nats` for window frame 0.
- anchor_pixel_mse: mean squared difference between the anchor and the frame at render resolution, in [-1, 1]
  pixel units, read from the dataset: a model-free measure of how stale the anchor is.
- probe_error_<label>, probe_base_<label> (only with --label): squared error of a ridge linear probe from the
  noisy state to a physical label, and of the constant prediction (the label's mean over the fitted frames).
  The probe is fit on every other episode (every other window when there is one episode), all gaps pooled, and
  the errors are reported for the remaining frames. Labels are read only here; they never reach the model.

The records are grouped into gap buckets: one per distinct gap when there are at most 12, otherwise 8 quantile
buckets. Per bucket and quantity: the number of frames, the mean and its standard error (windows are the
clusters: the frames of one window are not independent), and per label the probe R^2 = 1 - probe_error /
probe_base. Across frames: the Spearman correlation of every quantity with the gap, and the partial one after
the ranks of both have been residualised on the ranks of anchor_pixel_mse.

Two model families. The LeWM classes (ASJepaLeWM, RobustLeWM, ForecastScoredLeWM) are read as described above,
on PushT and benchmark windows. Models that expose their own `distortion` (the RoboCasa Reach3D family,
`AnalyticResidualSPT` and the variants `experiments/robocasa_stable_scaling/train.py` builds by patching its
methods) are read through that method, whatever it is (flow loss, future-image mean, profiled pixel likelihood,
point-state means): the tool calls `model.distortion` for one state and one rendered frame at a time, on the
two-frame slice (window frame 0, frame t), and seeds every generator identically before each call of a frame,
so the conditions of a frame are paired; a distortion that draws flow times and noise is averaged over --draws
seeds. The states are built as that family's loss builds them, and its rate terms follow its analytic rate:
surprise_nats is the code length of the encoder mean against the clean rollout. This family also reports
render_loss_forecast and forecast_excess = render_loss_forecast - render_loss: the loss when the renderer
reads the forecast condition of that loss (the clean rollout mean, with the coding noise that loss adds to it)
instead of the posterior sample. The RoboCasa anchor is window frame 0, so its gap is the frame index. When one
state renders two camera views (joint-view models) a record is the state and its value the mean over both views.

How to read it (two limits). With a first-frame anchor the gap IS the frame's index in the window, so a gap trend
cannot be told from a window-position trend (the prior has more context later in the window, a rollout has more
steps behind it); the other anchor rules separate the two, and the records keep `frame` for that. The LeWM
classes that render the prior's forecast for a share of frames (`forecast_fraction`, `lam`) are read on their
encoded branch only.
"""
import argparse
import contextlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from as_jepa_lewm.forecast_scored import ar1
from vforge.models.as_jepa.codes import entry_nats, surprise_nats
from vforge.models.as_jepa.constants import GRID, TOKEN_CH
from vforge.models.as_jepa.flow import draw_timesteps
from vforge.models.as_jepa.pixels import pack

TIME_QUANTILES = (.1, .3, .5, .7, .9)    # flow times of the grid, as quantiles of the training distribution of times
MAX_EXACT_GAPS = 12                      # up to this many distinct gaps: one bucket per gap
QUANTILE_BUCKETS = 8                     # above it: this many quantile buckets
RIDGE = 1e-3                             # ridge strength of the label probes, per fitted frame, on standardized states
COVARIATE = "anchor_pixel_mse"           # what the partial rank correlations are residualised on
DRAWS = 4                                # seeds per frame for a `distortion` that draws flow times and noise
ROBOCASA_CACHE = "/tmp/spt-time-compare-20261112"   # where `campaign.corpus` expects an episode's local files
# What identifies a record; every other column is a quantity. `view` is the camera of a multi-view window, -1 when
# the record covers all of them.
KEYS = ("window", "view", "episode", "start", "frame", "row", "anchor_row", "offset", "gap")


def frame_offsets(batch):
    """Signed distance from the anchor to every window frame, in kept frames: [B, W], positive when the anchor
    lies in the frame's past. It uses only the dataset's own fields: `rows`, the stored row of each window frame
    (consecutive window frames are one kept frame apart), and `anchor_row`. Every anchor rule in use is
    phase-aligned with its window, so the distance is an integer; anything else is an error."""
    rows = torch.as_tensor(batch["rows"]).long()
    anchor = torch.as_tensor(batch["anchor_row"]).long().reshape(-1, 1).to(rows.device)
    stride = rows[:, 1:2] - rows[:, :1]
    if bool((rows.diff(dim=1) != stride).any()) or bool(((rows - anchor) % stride).any()):
        raise ValueError("window rows are not evenly spaced, or the anchor is not phase-aligned with them")
    return (rows - anchor) // stride


def posterior_rho(model):
    """AR(1) correlation of the model's posterior noise across frames. `RobustLeWM` and `ForecastScoredLeWM`
    use their rho whenever it is set; `ResearchLeWM` stores one that only its 'correlated_noise' method uses."""
    rho = float(getattr(model, "rho", 0.))
    return rho if getattr(model, "method", "correlated_noise") == "correlated_noise" else 0.


def window_derangement(episode):
    """Partner window of every window of the batch: a roll, so no window is its own partner. Among the rolls,
    the one with the fewest partners from the same episode (the smallest shift on ties). None for one window."""
    episode = torch.as_tensor(episode)
    b = len(episode)
    if b < 2:
        return None
    shift = min(range(1, b), key=lambda k: int((episode == episode.roll(-k)).sum()))
    return (torch.arange(b, device=episode.device) + shift) % b


def time_derangement(frames, device=None):
    """Partner of every rendered frame within its window: a roll by half of them. None for a single frame."""
    return (torch.arange(frames, device=device) + frames // 2) % frames if frames > 1 else None


def frame_flow_loss(model, y, state, anchor, t, eps):
    """`flow_loss` (vforge.models.as_jepa.flow) with the time and noise draws passed in and without its mean over
    frames: [N]. Every frame has the same number of values, so the mean of these is the model's batch flow loss."""
    tv = t.view(-1, 1, 1, 1, 1)
    with model.renderer.anchored(anchor):
        v = model.renderer((1 - tv) * y + tv * eps, t, condition=state)
    return (y - eps - v).square().mean((1, 2, 3, 4)).float()


@torch.no_grad()
def window_quantities(model, batch, noise, uniform, eps):
    """Every model quantity of one batch as [B, W] tensors (NaN where it does not apply), plus the states.

    noise: [B, W, K, d] standard normal draws; they become the model's posterior noise (sigma0, AR(1) when rho > 0).
    uniform: [B * (W - 1), T] flow times of every rendered frame, as quantiles of the training distribution.
    eps: [B * (W - 1), T, TOKEN_CH, 1, GRID, GRID] flow noise.
    The steps are those of `ASJepaLeWM.loss`. With that loss's own draws (T = 1) the rendered frames'
    render_loss, surprise_nats and entry_nats average to its reported `fm`, `surprise` and `entry`.
    """
    if getattr(model, "training", False):
        raise ValueError("evaluate in eval mode: in train mode the renderer drops states at random")
    pixels, render, anchor = model.prepare(batch)
    info = model.encode(dict(pixels=pixels, action=batch["action"]))
    b, w = info["emb"].shape[:2]
    mu = info["emb"].reshape(b, w, model.state_slots, model.state_width)
    rho = posterior_rho(model)
    s = mu + model.sigma0 * ar1(noise.clone(), rho) if model.mode in {"L2", "S2"} else mu
    predicted = model.predict(s.flatten(2)[:, :-1], info["act_emb"][:, :-1]).reshape_as(mu[:, 1:])
    entry = entry_nats(mu[:, 0], model.sigma0)
    if rho:                                                          # exact KL of the AR(1) posterior's conditional
        target = mu[:, 1:] + rho * (s[:, :-1] - mu[:, :-1])
        surprise = surprise_nats(target, predicted, model.sigma0) + model.state_dim * .5 * (-rho ** 2 - math.log1p(-rho ** 2))
    else:
        surprise = surprise_nats(mu[:, 1:], predicted, model.sigma0)

    n, times = uniform.shape
    y = pack(render[:, 1:].flatten(0, 1))
    a = pack(anchor).repeat_interleave(w - 1, dim=0)
    t = draw_timesteps(n * times, y.device, model.training_shift, uniform=uniform.reshape(-1, 1))
    own = s[:, 1:]

    def losses(state, shown):                                        # [B, W - 1, K, d], [N, ...] -> [B, W - 1]: mean over times
        spread = [x.repeat_interleave(times, dim=0) for x in (y, state.flatten(0, 1), shown)]
        return frame_flow_loss(model, *spread, t, eps.flatten(0, 1)).view(b, w - 1, times).mean(2)

    missing = torch.full((b, w - 1), float("nan"), device=y.device)
    loss = losses(own, a)
    partner, shift = window_derangement(batch["episode"]), time_derangement(w - 1, y.device)
    swapped = losses(own[partner.to(y.device)], a) if partner is not None else missing
    swapped_time = losses(own[:, shift], a) if shift is not None else missing
    zero_anchor = losses(own, torch.zeros_like(a))
    same = missing if partner is None else (torch.as_tensor(batch["episode"]) == torch.as_tensor(batch["episode"])[partner]
                                            ).to(y.device)[:, None].expand(b, w - 1).float()

    def rendered(x):                                                 # [B, W - 1] -> [B, W], NaN at window frame 0
        return torch.cat([missing[:, :1], x.float()], 1).cpu()

    return dict(render_loss=rendered(loss), render_loss_swapped=rendered(swapped), state_reliance=rendered(swapped - loss),
                render_loss_swapped_time=rendered(swapped_time), state_reliance_time=rendered(swapped_time - loss),
                render_loss_zero_anchor=rendered(zero_anchor), anchor_reliance=rendered(zero_anchor - loss),
                swap_same_episode=rendered(same), surprise_nats=rendered(surprise),
                entry_nats=torch.cat([entry[:, None].float(), missing], 1).cpu(), state=s.flatten(2).float().cpu())


@contextlib.contextmanager
def reseeded(seed, device):
    """Seeds every generator a `distortion` may draw from (torch on all devices, numpy, random) and puts all of
    them back afterwards, so that calls under the same seed see the same draws and the caller's streams do not move."""
    kept = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=[torch.cuda.current_device() if device.index is None else device.index]
                               if device.type == "cuda" else []):
        torch.manual_seed(seed); np.random.seed(seed % 2 ** 32); random.seed(seed)
        try:
            yield
        finally:
            random.setstate(kept[0]); np.random.set_state(kept[1])


def standard_noise(shape, generator, laplace=False):
    """Unit-variance draws for the posterior noise from the tool's generator: Gaussian, or Laplace for a model
    with `coding_family == "laplace"` (the inverse-CDF form `torch.distributions.Laplace` samples with)."""
    if not laplace:
        return torch.randn(shape, device=generator.device, generator=generator)
    u = (2 * torch.rand(shape, device=generator.device, generator=generator) - 1).clamp(-1 + 1e-7, 1 - 1e-7)
    return -u.sign() * torch.log1p(-u.abs()) / math.sqrt(2)


@torch.no_grad()
def distortion_quantities(model, batch, noise, seeds, step=0):
    """`window_quantities` for a model that owns its `distortion` (AnalyticResidualSPT and its patched variants):
    [U, W] tensors over the model's U states (one per batch row, or per pair of rows for a joint-view model).

    noise: [U, W, K, d] unit-variance draws (`standard_noise`); they become the posterior noise as in the model's loss.
    seeds: [U, W - 1, R] integers. Every `distortion` call for state u and frame t runs under seed r, for all
      conditions alike; R > 1 is used only when the distortion turns out to depend on the seed.
    The steps are those of `AnalyticResidualSPT.loss`: `prepare`, `encode`, `posterior_means`, AR(1) noise, the
    clean rollout `mean_trajectory`, the forecast condition, and the analytic rate. The distortion is the model's
    own, called on the slice (window frame 0, frame t) of one state's rows with that frame's condition. When the
    distortion draws nothing (future-mean, profiled likelihood), render_loss and render_loss_forecast average to
    the loss's `fm` at forecast_fraction 0 and 1; surprise_nats and entry_nats average to its `surprise`, `entry`.
    """
    from experiments.robocasa_refine.analytic_model import innovation_error, laplace_shift_nats, root_marginal_error
    if getattr(model, "training", False):
        raise ValueError("evaluate in eval mode: in train mode the renderer drops states at random")
    pixels, render, anchor = model.prepare(batch)
    try:
        info = model.encode(dict(pixels=pixels, action=batch["action"]))
    finally:
        if getattr(model, "cached", None) is not None:              # the cached-feature models' loss clears it too
            model.cached = None
    flat = info["emb"]
    u, w = flat.shape[:2]
    mu = model.posterior_means(flat.reshape(u, w, model.state_slots, model.state_width), batch, step)
    rho, sigma0 = model.rho, model.sigma0
    stationary = getattr(model, "stationary_coding_channel", False)
    laplace = getattr(model, "coding_family", "gaussian") == "laplace"
    point = getattr(model, "state_mean_observation", False)         # the renderer reads point states, without noise
    noise = ar1(noise.clone(), rho)
    noisy = mu + sigma0 * noise
    means = model.mean_trajectory(flat[:, 0], info["act_emb"][:, :-1]).reshape_as(mu)
    if stationary:
        forecast = means[:, 1:] + sigma0 * noise[:, 1:]
    else:
        powers = mu.new_tensor([rho ** t for t in range(1, w)])[None, :, None, None]
        forecast = means[:, 1:] + powers * (noisy[:, :1] - mu[:, :1])
    if point or not getattr(model, "decoder_forecast_noise", True):
        forecast = means[:, 1:]
    own = mu[:, 1:] if point else noisy[:, 1:]

    if getattr(model, "forecast_rate_geometry", "markov") == "root-marginal":
        error, marginal = root_marginal_error(mu, means, rho)
    else:
        error, marginal = innovation_error(mu, means, rho), None
    entry = model.initial_rate(mu[:, 0])
    coding_sigma = sigma0 * math.sqrt(1 - rho ** 2) if stationary else sigma0
    if laplace:
        surprise = laplace_shift_nats(error, coding_sigma / math.sqrt(2)).sum((-2, -1))
    else:
        surprise = surprise_nats(error, torch.zeros_like(error), coding_sigma)
    if not stationary:
        surprise = surprise + model.state_dim * .5 * (-rho ** 2 - math.log1p(-rho ** 2))
        if marginal is not None:
            surprise = surprise + model.state_dim * marginal[None, :]

    rows = render.shape[0] // u                                      # batch rows per state: 2 for a joint-view model
    seeds = torch.as_tensor(seeds).tolist()

    def value(i, t, condition, shown, draws):                        # one state, rendered frame t (1-based), given seeds
        r = slice(i * rows, (i + 1) * rows)
        part = {k: v[r][:, [0, t]] if torch.is_tensor(v) and v.shape[:2] == render.shape[:2] else v for k, v in batch.items()}
        extra = dict(initial=mu[i:i + 1, 0]) if point else {}
        total = 0.
        for seed in draws:
            with reseeded(seed, render.device):
                total += float(model.distortion(part, render[r][:, [0, t]], shown[r], condition[None, None], **extra))
        return total / len(draws)

    first = seeds[0][0]
    random_distortion = len(first) > 1 and value(0, 1, own[0, 0], anchor, first[:1]) != value(0, 1, own[0, 0], anchor, first[1:2])
    episode = torch.as_tensor(batch["episode"])[::rows]
    partner, shift = window_derangement(episode), time_derangement(w - 1)
    blank = torch.zeros_like(anchor)
    names = ("render_loss", "render_loss_swapped", "render_loss_swapped_time", "render_loss_zero_anchor", "render_loss_forecast")
    out = {k: torch.full((u, w - 1), float("nan")) for k in names}
    for i in range(u):
        for t in range(1, w):
            draws = seeds[i][t - 1] if random_distortion else seeds[i][t - 1][:1]
            out["render_loss"][i, t - 1] = value(i, t, own[i, t - 1], anchor, draws)
            if partner is not None:
                out["render_loss_swapped"][i, t - 1] = value(i, t, own[int(partner[i]), t - 1], anchor, draws)
            if shift is not None:
                out["render_loss_swapped_time"][i, t - 1] = value(i, t, own[i, int(shift[t - 1])], anchor, draws)
            out["render_loss_zero_anchor"][i, t - 1] = value(i, t, own[i, t - 1], blank, draws)
            out["render_loss_forecast"][i, t - 1] = value(i, t, forecast[i, t - 1], anchor, draws)
    if hasattr(model, "_profiled_pixel_metrics"):                    # left by the profiled likelihood; its loss clears it too
        model._profiled_pixel_metrics = None
    missing = torch.full((u, w - 1), float("nan"))
    same = missing if partner is None else (episode == episode[partner]).cpu()[:, None].expand(u, w - 1).float()
    out.update(state_reliance=out["render_loss_swapped"] - out["render_loss"],
               state_reliance_time=out["render_loss_swapped_time"] - out["render_loss"],
               anchor_reliance=out["render_loss_zero_anchor"] - out["render_loss"],
               forecast_excess=out["render_loss_forecast"] - out["render_loss"], swap_same_episode=same,
               surprise_nats=surprise.float().cpu())
    out = {k: torch.cat([missing[:, :1], v], 1) for k, v in out.items()}             # NaN at window frame 0
    return dict(out, entry_nats=torch.cat([entry[:, None].float().cpu(), missing], 1),
                state=(mu if point else noisy).flatten(2).float().cpu())


@torch.no_grad()
def collect(model, data, indices, seed=0, quantiles=TIME_QUANTILES, batch_size=8, device="cpu", fp32=True, draws=DRAWS):
    """Records of the windows `indices` of `data`: a dict of flat arrays with one entry per state and window frame,
    and the noisy states [records, D] the label probes read. One generator seeded with `seed` supplies the
    posterior noise, the flow noise and the seeds of the `distortion` calls; the global random state is left as it was.
    A model with its own `distortion` is read through it (`distortion_quantities`), any other as a LeWM flow model."""
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    grid = torch.tensor(quantiles, dtype=torch.float32, device=device)[None]
    columns, states = {}, []
    for lo in range(0, len(indices), batch_size):
        picked = [int(i) for i in indices[lo:lo + batch_size]]
        raw = default_collate([data[i] for i in picked])            # a DataLoader would draw a seed from the global generator
        raw = {k: v.to(device) if torch.is_tensor(v) else v for k, v in raw.items()}
        batch = data.model_batch(raw) if hasattr(data, "model_batch") else raw      # multi-view windows: views become rows
        b, w = raw["rows"].shape
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda" and not fp32):
            if hasattr(model, "distortion"):
                units = batch["render"].shape[0] // (2 if getattr(model, "joint_views", False) else 1)
                noise = standard_noise((units, w, model.state_slots, model.state_width), generator,
                                       getattr(model, "coding_family", "gaussian") == "laplace")
                seeds = torch.randint(2 ** 31 - 1, (units, w - 1, draws), device=device, generator=generator)
                out = distortion_quantities(model, batch, noise, seeds)
            else:
                noise = torch.randn((b, w, model.state_slots, model.state_width), device=device, generator=generator)
                eps = torch.randn((b * (w - 1), len(quantiles), TOKEN_CH, 1, GRID, GRID), device=device, generator=generator)
                out = window_quantities(model, batch, noise, grid.expand(b * (w - 1), -1), eps)
        states.append(out.pop("state").flatten(0, 1).numpy())
        units = out["render_loss"].shape[0]
        per, lines = units // b, batch["render"].shape[0] // units   # states per window, batch rows per state
        offset = frame_offsets(raw).cpu().repeat_interleave(per, 0)
        frames = batch["render"].double()                           # 8-bit values: exact, so equal differences tie
        shown = (batch["anchor"] if "anchor" in batch else batch["render"][:, 0]).double()    # no anchor field: window frame 0
        pixel_mse = (frames - shown[:, None]).square().mean((2, 3, 4)).reshape(units, lines, w).mean(1).cpu() / 127.5 ** 2
        view = torch.arange(units) % per if lines == 1 else torch.full((units,), -1)
        per_window = dict(window=picked, episode=raw["episode"], start=raw["start"], anchor_row=raw["anchor_row"],
                          anchor_fallback=raw["anchor_fallback"])
        batch_columns = dict({k: torch.as_tensor(v).cpu().repeat_interleave(per, 0)[:, None].expand(units, w)
                              for k, v in per_window.items()},
                             view=view[:, None].expand(units, w), frame=torch.arange(w)[None].expand(units, w),
                             row=raw["rows"].cpu().repeat_interleave(per, 0), offset=offset, gap=offset.abs(),
                             anchor_future=offset < 0, **{COVARIATE: pixel_mse}, **out)
        for k, v in batch_columns.items():
            columns.setdefault(k, []).append(v.reshape(-1).numpy())
    records = {k: np.concatenate(v) for k, v in columns.items()}
    return {k: v if k in KEYS else v.astype(np.float64) for k, v in records.items()}, np.concatenate(states)


class _Records:
    """Stands in for the manifest path `Windows` reads its records from. `Windows` asserts a training manifest, so
    the chosen records (held-out ones as well) are handed over under that split name; nothing is written."""
    def __init__(self, records):
        self.text = json.dumps(dict(records=[dict(r, split="train") for r in records]))

    def read_text(self):
        return self.text


class RobocasaWindows:
    """The RoboCasa trainer's `Windows` (experiments/robocasa_stable_scaling/train.py), unchanged, plus the fields
    the gap derivation reads. `Windows` returns neither the episode nor the start of a window; both are recovered
    here by replaying the generator its `__getitem__` seeds with the index. The anchor of this family is window
    frame 0 (`prepare` returns render[:, 0]) and window frames are consecutive stored frames, so
    rows = start .. start + W - 1 and anchor_row = start: the gap of a frame is its index in the window."""
    def __init__(self, root, records, cfg):
        from experiments.robocasa_stable_scaling.train import Windows
        self.inner = Windows(root, _Records(records), cfg)

    def locate(self, index):
        """(position of the episode in the records, first stored frame) of window `index`, as `Windows` draws them."""
        rng = np.random.default_rng(np.random.SeedSequence([781021, self.inner.cfg["seed"], int(index)]))
        return int(rng.integers(len(self.inner.records))), int(rng.integers(10))

    def __getitem__(self, index):
        item = self.inner[index]
        episode, start = self.locate(index)
        return dict(item, episode=episode, start=start, rows=start + torch.arange(item["action"].shape[0]),
                    anchor_row=start, anchor_fallback=False)

    def distinct(self, cases, seed):
        """Up to `cases` window indices with distinct (episode, start), drawn with `seed`."""
        seen, picked = set(), []
        for index in np.random.default_rng(seed).integers(2 ** 31, size=50 * cases).tolist():
            if len(picked) < cases and self.locate(index) not in seen:
                seen.add(self.locate(index)); picked.append(index)
        return picked

    @staticmethod
    def model_batch(raw):
        """The trainer's `prepare(raw)` on whatever device the batch is on (theirs moves it to CUDA): the camera views
        of a window become batch rows, which share the window's actions. `episode` is repeated alongside."""
        f = raw["features"]
        b, v, w = f.shape[:3]
        batch = dict(features=f.flatten(0, 1), render=raw["render"].flatten(0, 1), episode=raw["episode"].repeat_interleave(v),
                     action=raw["action"][:, None].expand(b, v, w, raw["action"].shape[-1]).reshape(b * v, w, -1))
        if "pixels" in raw:
            batch.update(pixels=raw["pixels"].flatten(0, 1), physical_views=v)
        return batch


def load_robocasa(checkpoint, records, split="validation", root=None, cache=ROBOCASA_CACHE, device="cuda"):
    """A RoboCasa Reach3D model and windows, through the trainer's own `build` and `Windows`.

    THIN AND ONLY EXERCISED ON SYNTHETIC FILES: the real campaign directories are not available where this was
    written. The layout is read off `campaign.py`, `train.checkpoint` and `diagnostics.inspect_current`:
    `checkpoint` is the label folder <root>/training/<run>/<label> holding weights.pt, its run folder holds
    config.json, and `root` (default: two levels above the run) is what `build` receives. `records` is a JSON file
    with a "records" list (<root>/data.json, or a corpus manifest); the records of `split` are used, all of them
    when `split` is "all". A record without local_features / local_observations gets the paths `campaign.corpus`
    gives it under `cache`. Raw-pixel models need the records' own rgb fields."""
    from experiments.robocasa_stable_scaling.train import build
    folder, cache = Path(checkpoint), Path(cache)
    run = folder.parent
    root = Path(root) if root else run.parents[1]
    cfg = json.loads((run / "config.json").read_text())
    model = build(root, cfg)
    if hasattr(model, "future_decoder"):
        raise ValueError(f"variant {cfg['variant']!r} decodes fixed features and does not train the renderer")
    weights = torch.load(folder / "weights.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=True)
    chosen = [dict(r, local_features=r.get("local_features", str(cache / "features/vjepa" / (r["id"] + ".npy"))),
                   local_observations=r.get("local_observations", str(cache / "observations" / (r["id"] + ".npz"))))
              for r in json.loads(Path(records).read_text())["records"] if split == "all" or r["split"] == split]
    if not chosen:
        raise ValueError(f"no records of split {split!r} in {records}")
    return model.to(device).eval(), RobocasaWindows(root, chosen, cfg), cfg


def load_labels(data, specs):
    """Physical labels by stored row, for the probes only. Each spec is name:source[:kind[:tolerance]], the form
    `benchmark_gap.staleness_table` takes (the tolerance is not used here). On a benchmark dataset the source is
    an HDF5 key; on PushT it is a column range lo-hi of the stored state vector (0-2 agent, 2-4 block position).
    kind "position" (default) probes the values themselves; "yaw90" probes (cos 4 yaw, sin 4 yaw), which respects
    the cube's quarter-turn symmetry."""
    data.open()
    labels = {}
    for spec in specs:
        name, source, kind = (spec.split(":") + ["position"])[:3]
        if kind not in {"position", "yaw90"}:
            raise ValueError(f"unknown label kind {kind!r}")
        if hasattr(data, "file"):                                    # the loader's own handle, as staleness_table reads it
            values = np.asarray(data.file[source], dtype=np.float64)
        else:
            lo, hi = map(int, source.split("-"))
            values = np.asarray(data.states[:, lo:hi], dtype=np.float64)
        values = values.reshape(len(values), -1)
        labels[name] = np.concatenate([np.cos(4 * values[:, :1]), np.sin(4 * values[:, :1])], 1) if kind == "yaw90" else values
    return labels


def add_probes(records, states, labels):
    """Adds probe_error_<name> and probe_base_<name> to the records, in place, and returns how the frames were split.

    One ridge linear probe per label, from the standardized noisy state to the label. It is fit on every other
    episode (every other window when the records hold one episode) with all gaps pooled. For the remaining
    frames: probe_error is the squared distance between prediction and label, probe_base that between the
    fitted frames' mean label and the label. Fitted frames get NaN."""
    episodes = np.unique(records["episode"])
    by = "episode" if len(episodes) > 1 else "window"
    fit = np.unique(records[by], return_inverse=True)[1] % 2 == 0
    if fit.sum() < 2 or fit.all():
        raise ValueError("the label probes need at least two windows")
    x = states.astype(np.float64)
    z = (x - x[fit].mean(0)) / np.maximum(x[fit].std(0), 1e-12)
    gram = z[fit].T @ z[fit] + RIDGE * fit.sum() * np.eye(z.shape[1])
    for name, values in labels.items():
        y = values[records["row"]]
        centre = y[fit].mean(0)
        predicted = z @ np.linalg.solve(gram, z[fit].T @ (y[fit] - centre)) + centre
        records["probe_error_" + name] = np.where(fit, np.nan, np.square(predicted - y).sum(1))
        records["probe_base_" + name] = np.where(fit, np.nan, np.square(centre - y).sum(1))
    return dict(split_by=by, fitted_frames=int(fit.sum()), held_out_frames=int((~fit).sum()), ridge=RIDGE)


def gap_buckets(gap):
    """Bucket index of every frame, and the rule: one bucket per distinct gap when there are at most
    MAX_EXACT_GAPS ("exact"), otherwise QUANTILE_BUCKETS quantile buckets ("quantile"; frames with the same gap
    stay together, so the counts are only nearly equal)."""
    values = np.unique(gap)
    if len(values) <= MAX_EXACT_GAPS:
        return np.searchsorted(values, gap), "exact"
    edges = np.unique(np.quantile(gap, np.linspace(0, 1, QUANTILE_BUCKETS + 1)[1:-1], method="inverted_cdf"))
    return np.unique(np.searchsorted(edges, gap), return_inverse=True)[1], "quantile"


def cluster_mean(values, cluster):
    """Number of finite values, their mean, and the standard error of that mean with `cluster` (the window) as
    the sampling unit. With one value per cluster it is the usual standard deviation over root n."""
    keep = np.isfinite(values)
    v = values[keep]
    if not len(v):
        return dict(frames=0, mean=None, sem=None)
    mean = float(v.mean())
    sums = np.bincount(np.unique(cluster[keep], return_inverse=True)[1], weights=v - mean)   # per cluster: summed deviations
    g = len(sums)
    return dict(frames=len(v), mean=mean, sem=math.sqrt(g / (g - 1) * float(np.square(sums).sum())) / len(v) if g > 1 else None)


def _ranks(x):
    """Ranks from 1, tied values sharing their mean rank."""
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2.)[inverse]


def spearman(x, y, given=None):
    """Spearman correlation of x and y over the frames where every input is finite. With `given`: the partial
    rank correlation, for which the ranks of x and of y are first residualised on the ranks of `given` by least
    squares. None when fewer than three frames remain or one side has no variation left."""
    keep = np.isfinite(x) & np.isfinite(y) & (True if given is None else np.isfinite(given))
    n = int(keep.sum())
    if n < 3:
        return None
    rx, ry = (r - r.mean() for r in (_ranks(x[keep]), _ranks(y[keep])))
    if given is not None:
        rz = _ranks(given[keep])
        rz = rz - rz.mean()
        if (rz * rz).sum() > 0:
            rx, ry = (r - rz * ((r * rz).sum() / (rz * rz).sum()) for r in (rx, ry))
    xx, yy = float((rx * rx).sum()), float((ry * ry).sum())
    floor = 1e-9 * n ** 3                                            # ranks with variation have a sum of squares of order n^3
    return float((rx * ry).sum() / math.sqrt(xx * yy)) if min(xx, yy) > floor else None


def summarize(records):
    """Gap buckets, the same summary over all frames, and the rank correlations with the gap, for every quantity
    in `records` (every column outside KEYS)."""
    gap, window = records["gap"], records["window"]
    names = [k for k in records if k not in KEYS]
    index, rule = gap_buckets(gap)

    def bucket(mask):
        q = {k: cluster_mean(records[k][mask], window[mask]) for k in names}
        out = dict(gap_min=int(gap[mask].min()), gap_max=int(gap[mask].max()), gap_mean=float(gap[mask].mean()),
                   frames=int(mask.sum()), windows=len(np.unique(window[mask])), quantities=q)
        labels = [k[len("probe_error_"):] for k in names if k.startswith("probe_error_")]
        if labels:
            out["probe_r2"] = {k: 1 - q["probe_error_" + k]["mean"] / q["probe_base_" + k]["mean"]
                               if q["probe_base_" + k]["mean"] else None for k in labels}
        return out

    g, given = gap.astype(np.float64), records[COVARIATE]
    return dict(bucketing=rule, buckets=[bucket(index == i) for i in range(index.max() + 1)],
                overall=bucket(np.ones(len(gap), bool)),
                spearman={k: dict(frames=int(np.isfinite(records[k]).sum()), gap=spearman(records[k], g),
                                  gap_given_anchor_pixel_mse=None if k == COVARIATE else spearman(records[k], g, given))
                          for k in names})


TABLE = dict(render_loss="render", state_reliance="state_rel", state_reliance_time="state_rel_t", anchor_reliance="anchor_rel",
             forecast_excess="forecast_ex", surprise_nats="surprise", entry_nats="entry", anchor_pixel_mse="pixel_mse")


def table(summary):
    """The compact printed view: bucket means of the main quantities, then their two rank correlations with the gap."""
    def cell(v):
        return f"{'-' if v is None else format(v, '.4g'):>12}"
    shown = [k for k in TABLE if k in summary["spearman"]]           # forecast_excess exists for `distortion` models only
    lines = [f"{'gap':>9}{'frames':>8}" + "".join(f"{TABLE[k]:>12}" for k in shown)]
    for b in summary["buckets"]:
        span = str(b["gap_min"]) if b["gap_min"] == b["gap_max"] else f"{b['gap_min']}-{b['gap_max']}"
        lines.append(f"{span:>9}{b['frames']:>8}" + "".join(cell(b["quantities"][k]["mean"]) for k in shown))
    for label, key in (("spearman with gap", "gap"), ("given pixel_mse", "gap_given_anchor_pixel_mse")):
        lines.append(f"{label:>17}" + "".join(cell(summary["spearman"][k][key]) for k in shown))
    return "\n".join(lines)


def _plain(value):
    """JSON-ready copy: arrays become lists, numpy scalars Python ones, non-finite numbers null."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return int(value) if isinstance(value, np.integer) else value


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", required=True, help="folder with weights.pt and config.json; for robocasa the label "
                                                       "folder <root>/training/<run>/<label> (config.json is in <run>)")
    p.add_argument("--data", required=True, help="PushT data directory, a benchmark metadata directory, or for robocasa a "
                                                 "JSON file with the episode records (<root>/data.json or a corpus manifest)")
    p.add_argument("--output", required=True)
    p.add_argument("--dataset", choices=["pusht_full", "pusht", "benchmark", "robocasa"], default="pusht_full",
                   help="pusht_full: FullPushTWindows (every raw-frame start, as all_phase_data training); "
                        "pusht: PushTWindows on kept-frame data (phase0); benchmark: TwoRoom and Cube windows; "
                        "robocasa: the Reach3D trainer's Windows and build (experiments/robocasa_stable_scaling/train.py)")
    p.add_argument("--gap", type=int, help="the anchor rule's minimum distance in kept frames. Default 8 on PushT (either side); "
                                           "default 0 on benchmarks (window frame 0), above 0 GapBenchmarkWindows (past only)")
    p.add_argument("--split", help="default: test on PushT, validation on benchmarks and robocasa ('all': every robocasa record)")
    p.add_argument("--root", help="robocasa: the campaign root `build` receives (default: two levels above the run folder)")
    p.add_argument("--cache", default=ROBOCASA_CACHE, help="robocasa: where records without local paths keep their files")
    p.add_argument("--draws", type=int, default=DRAWS, help="seeds per frame for a model distortion that draws times and noise")
    p.add_argument("--cases", type=int, default=128, help="number of held-out windows")
    p.add_argument("--window", type=int, help="frames per window; default and maximum: the model's training window")
    p.add_argument("--seed", type=int, default=0, help="seeds the choice of windows and every noise draw")
    p.add_argument("--times", type=float, nargs="+", default=list(TIME_QUANTILES),
                   help="flow noise times of the grid, as quantiles of the training distribution of times")
    p.add_argument("--label", action="append", default=[],
                   help="name:source[:kind], kind in {position, yaw90}; the source is an HDF5 key (benchmark) or a "
                        "column range lo-hi of the stored state (PushT). Adds the label probes")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--fp32", action="store_true", help="FP32 instead of bfloat16 autocast, for small paired differences")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    from as_jepa_tools.evaluate import load_model
    from vforge.ops import OpsConfig, initialize_ops
    device = torch.device(a.device)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest" if a.fp32 else "high")
    initialize_ops(OpsConfig(backend="sdpa" if device.type == "cuda" else "sdpa_math", allow_fallback=False),
                   device=device, parallel_state=None)
    if a.dataset == "robocasa":
        gap, split = 0, a.split or "validation"
        model, data, _ = load_robocasa(a.checkpoint, a.data, split, a.root, a.cache, a.device)
    else:
        model = load_model(a.checkpoint, device=a.device)
    if getattr(model, "renderer", None) is None or not (hasattr(model, "distortion") or model.objective == "flow"):
        raise ValueError("the gap diagnostics need a model with a flow renderer or its own distortion")
    window = a.window or model.window
    if not 2 <= window <= model.window or not all(0 < q < 1 for q in a.times) or a.draws < 1:
        raise ValueError("--window must lie in [2, the model's window], --times in (0, 1), and --draws be positive")
    if a.dataset == "robocasa":
        if a.gap or a.label or window != model.window:
            raise ValueError("robocasa windows have the trainer's fixed length and first-frame anchor, and no label source here")
        indices = data.distinct(a.cases, a.seed)
    elif a.dataset == "benchmark":
        from as_jepa_tools.benchmark_gap import make_benchmark_dataset
        gap, split = a.gap or 0, a.split or "validation"
        data = make_benchmark_dataset(dict(data=a.data, seed=0, model=dict(window=window), gap=gap), split=split)
    else:
        from as_jepa_tools.data import FullPushTWindows, PushTWindows
        gap, split = 8 if a.gap is None else a.gap, a.split or "test"
        data = (FullPushTWindows if a.dataset == "pusht_full" else PushTWindows)(a.data, window=window, gap=gap, split=split)
    if a.dataset != "robocasa":
        indices = np.random.default_rng(a.seed).permutation(len(data))[:a.cases].tolist()
    labels = load_labels(data, a.label) if a.label else {}
    records, states = collect(model, data, indices, seed=a.seed, quantiles=tuple(a.times), batch_size=a.batch_size,
                              device=a.device, fp32=a.fp32, draws=a.draws)
    probes = add_probes(records, states, labels) if labels else None
    flow_times = draw_timesteps(len(a.times), "cpu", model.training_shift, uniform=torch.tensor(a.times)[:, None])
    by_distortion = hasattr(model, "distortion")                    # then the model draws its own flow times, if any
    report = _plain(dict(checkpoint=a.checkpoint, data=a.data, dataset=type(data).__name__, split=split, anchor_rule_gap=gap,
                         cases=len(indices), window=window, seed=a.seed, fp32=a.fp32, sigma0=model.sigma0,
                         rho=posterior_rho(model), render_loss="model.distortion" if by_distortion else "flow_loss",
                         time_quantiles=None if by_distortion else a.times,
                         flow_times=None if by_distortion else flow_times.flatten().tolist(),
                         distortion_draws=a.draws if by_distortion else None, probes=probes, **summarize(records)))
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    columns = ",\n".join(f"    {json.dumps(k)}: {json.dumps(v)}" for k, v in _plain(records).items())   # one line per column
    (out / "gap_buckets.json").write_text(json.dumps(report, indent=2)[:-2] + ',\n  "records": {\n' + columns + "\n  }\n}\n")
    print(table(report))


if __name__ == "__main__":
    main()
