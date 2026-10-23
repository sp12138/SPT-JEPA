from types import SimpleNamespace

import pytest
import torch

from vforge.evaluation.runner import _stable_sample_salt
from vforge.models.rep_world.config import CodecGeometry, ConditionerConfig
from vforge.models.rep_world.inference import RepWorldInference, RolloutConfig
from vforge.models.rep_world.model import FlowConfig
from vforge.tasks.rep_world.eval import BatchWindowEvalLoader, RuntimeInference


class Codec:
    def __init__(self, geometry):
        self.geometry = geometry
        self.inputs = []

    def encode(self, rgb):
        self.inputs.append(rgb.clone())
        geo = self.geometry
        value = rgb.mean((1, 2, 3, 4), keepdim=True)
        return value.expand(
            rgb.shape[0], geo.packed_channels, rgb.shape[2] // geo.rgb_frames, geo.grid, geo.grid
        ).contiguous()

    def stream(self):
        def step(z):
            return (
                z.mean((1, 2, 3, 4), keepdim=True)
                .expand(z.shape[0], 3, z.shape[2] * self.geometry.rgb_frames, 8, 8)
                .clone()
            )

        return SimpleNamespace(prime=lambda _: None, step=step)


class Engine:
    def __init__(self):
        self.calls = []

    def __call__(self, **kw):
        self.calls.append(kw["operation"])
        if kw["operation"] == "condition":
            x = kw["source"]
            return x.flatten(1).mean(1).view(-1, 1, 1).expand(-1, 4, 8).clone()
        z = kw["noisy"]
        value = kw["condition"].mean((1, 2)).view(-1, 1, 1, 1, 1)
        if kw.get("condition_drop") is not None:
            value = value * 0
        previous = kw.get("previous")
        return z * 0.125 + value + (previous * 0.05 if previous is not None else 0)


def make_inference(mode="no_zprev", domain="latent", cfg=1):
    geo = CodecGeometry("videorae", "test", 4, 2, 2, grid=2)
    codec, engine = Codec(geo), Engine()
    infer = RepWorldInference(
        engine,
        codec,
        geo,
        FlowConfig(mode),
        ConditionerConfig(input=domain),
        1.5,
        RolloutConfig(4, steps=3, cfg_scale=cfg, history_window_states=2),
    )
    return infer, engine, codec


def clips(count):
    generator = torch.Generator().manual_seed(901)
    return [
        dict(sample_id=f"case-{i}", rgb=torch.randn(1, 3, 16, 8, 8, generator=generator))
        for i in range(count)
    ]


@pytest.mark.parametrize(
    "mode,domain",
    [
        ("no_zprev", "latent"),
        ("zprev", "latent"),
        ("no_zprev", "rgb"),
        ("zprev", "rgb"),
    ],
)
@pytest.mark.parametrize("cfg", [1, 2])
def test_independent_seeded_rollouts_match_singletons_and_permutations(mode, domain, cfg):
    infer, _, codec = make_inference(mode, domain, cfg)
    samples, seeds = clips(3), [1047, 982, 16]
    single = [infer.infer_sample(s, seed=seed) for s, seed in zip(samples, seeds, strict=True)]
    torch.manual_seed(19)
    torch.randn(77)
    batched = infer.infer_batch(samples, seeds=seeds)
    order = [2, 0, 1]
    shuffled = infer.infer_batch([samples[i] for i in order], seeds=[seeds[i] for i in order])
    for i in range(3):
        assert batched[i]["sample_id"] == samples[i]["sample_id"]
        assert batched[i]["metadata"] == single[i]["metadata"]
        for key in ("latents", "prediction"):
            torch.testing.assert_close(batched[i][key], single[i][key], rtol=1e-6, atol=2e-6)
            torch.testing.assert_close(
                shuffled[order.index(i)][key], single[i][key], rtol=1e-6, atol=2e-6
            )
    assert all(x.shape[2] == 4 for x in codec.inputs)
    changed = [{**s, "rgb": s["rgb"].clone()} for s in samples]
    for sample in changed:
        sample["rgb"][:, :, 4:] = 1e6
    no_future = infer.infer_batch(changed, seeds=seeds)
    for actual, expected in zip(no_future, batched, strict=True):
        torch.testing.assert_close(actual["latents"], expected["latents"], rtol=0, atol=0)


def test_noise_streams_are_bitwise_independent_of_grouping_across_states():
    shape = (3, 2, 2, 2, 2)
    batched_generators = [torch.Generator().manual_seed(s) for s in (42, 83, 92)]
    single_generators = [torch.Generator().manual_seed(s) for s in (42, 83, 92)]
    for _ in range(9):
        batch = RepWorldInference._noise(
            shape, generator=batched_generators, device="cpu", dtype=torch.float32
        )
        expected = torch.cat([torch.randn((1, *shape[1:]), generator=g) for g in single_generators])
        torch.testing.assert_close(batch, expected, rtol=0, atol=0)


class Loader(list):
    @property
    def global_length(self):
        return sum(not s.get("__vforge_eval_padding__", False) for s in self)


def runtime():
    infer, engine, _ = make_inference()
    result = object.__new__(RuntimeInference)
    result.device, result.dtype = torch.device("cpu"), torch.float32
    result.inference = infer
    result._batch_key, result._batch_results = None, {}
    return result, engine


def test_window_cache_handles_skipped_prefix_partial_tail_and_padding():
    samples = clips(5)
    samples[-1]["__vforge_eval_padding__"] = True
    loader = BatchWindowEvalLoader(Loader(samples), 3)
    windows = list(loader)
    assert loader.global_length == 4 and len(windows) == 5
    assert windows[-1]["__vforge_eval_padding__"]
    assert [len(s["__repworld_eval_window__"]) for s in windows] == [3, 3, 3, 2, 2]
    infer, engine = runtime()
    # Native's collective skip can skip a whole prefix before any forward.
    for index in (1, 2, 4):
        sample = windows[index]
        seed = 29 + _stable_sample_salt(sample["sample_id"])
        actual = infer.infer_sample(sample, seed=seed)
        reference, _, _ = make_inference()
        expected = reference.infer_sample(samples[index], seed=seed)
        assert actual["metadata"]["seed"] == seed
        torch.testing.assert_close(actual["latents"], expected["latents"], rtol=1e-6, atol=2e-6)
    assert engine.calls.count("condition") == 2 * 3
    assert len(infer._batch_results) <= 3


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_batch_size_rejected(value):
    with pytest.raises(ValueError, match="batch_size"):
        RolloutConfig(4, batch_size=value)
    with pytest.raises(ValueError, match="batch_size"):
        BatchWindowEvalLoader(Loader(), value)


def test_batch_rejects_ambiguous_shapes_or_seed_counts():
    infer, _, _ = make_inference()
    with pytest.raises(ValueError, match="one seed"):
        infer.infer_batch(clips(2), seeds=[1])
    with pytest.raises(ValueError, match="one RGB"):
        infer.infer_batch([{"sample_id": "x", "rgb": torch.zeros(2, 3, 4, 8, 8)}], seeds=[1])
