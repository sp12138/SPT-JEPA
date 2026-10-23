from threading import Event
from types import SimpleNamespace

import torch

import vforge.offline_encoder.runner as runner_module
from vforge.data.cache import (
    EncodedBatchCodec,
    cache_path_for_sample,
    load_latent_bundle,
)
from vforge.offline_encoder.runner import EncodeComponents, run_encode


def test_skip_existing_requires_matching_manifest_fingerprint(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        runner_module,
        "get_parallel_state",
        lambda: SimpleNamespace(world_size=1, global_rank=0),
    )
    codec = EncodedBatchCodec(name="test", required_keys=("value",))

    def components(fingerprint, encode_condition):
        return EncodeComponents(
            dataset=[0],
            sample_ids=("sample",),
            encode_condition=encode_condition,
            cache_dir=str(tmp_path),
            encoding_fingerprint={"revision": fingerprint},
            codec=codec,
        )

    run_encode(
        components("v1", lambda _sample: {"value": torch.tensor([1])}),
        device=torch.device("cpu"),
    )
    monkeypatch.setattr(
        EncodedBatchCodec,
        "verify",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("complete cache manifests should not reload bundle files")
        ),
    )
    run_encode(
        components(
            "v1",
            lambda _sample: (_ for _ in ()).throw(
                AssertionError("matching cache should be reused")
            ),
        ),
        device=torch.device("cpu"),
    )

    run_encode(
        components("v2", lambda _sample: {"value": torch.tensor([2])}),
        device=torch.device("cpu"),
    )

    bundle = load_latent_bundle(cache_path_for_sample(str(tmp_path), "sample"))
    torch.testing.assert_close(bundle["value"], torch.tensor([2]))


def test_prepare_condition_prefetches_next_local_sample(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        runner_module,
        "get_parallel_state",
        lambda: SimpleNamespace(world_size=1, global_rank=0),
    )
    second_prepared = Event()

    def prepare(sample):
        if sample == 2:
            second_prepared.set()
        return sample * 10

    def encode(condition):
        if condition == 10:
            assert second_prepared.wait(timeout=2)
        return {"value": torch.tensor([condition])}

    run_encode(
        EncodeComponents(
            dataset=[1, 2],
            sample_ids=("first", "second"),
            prepare_condition=prepare,
            encode_condition=encode,
            cache_dir=str(tmp_path),
            encoding_fingerprint={"revision": "v1"},
            codec=EncodedBatchCodec(name="test", required_keys=("value",)),
        ),
        device=torch.device("cpu"),
    )

    first = load_latent_bundle(cache_path_for_sample(str(tmp_path), "first"))
    second = load_latent_bundle(cache_path_for_sample(str(tmp_path), "second"))
    torch.testing.assert_close(first["value"], torch.tensor([10]))
    torch.testing.assert_close(second["value"], torch.tensor([20]))


def test_compatible_resolved_config_hash_reuses_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        runner_module,
        "get_parallel_state",
        lambda: SimpleNamespace(world_size=1, global_rank=0),
    )
    codec = EncodedBatchCodec(name="test", required_keys=("value",))

    def components(fingerprint, encode_condition):
        return EncodeComponents(
            dataset=[0],
            sample_ids=("sample",),
            encode_condition=encode_condition,
            cache_dir=str(tmp_path),
            encoding_fingerprint=fingerprint,
            codec=codec,
        )

    run_encode(
        components(
            {"revision": "v1", "resolved_config_hash": "legacy"},
            lambda _sample: {"value": torch.tensor([1])},
        ),
        device=torch.device("cpu"),
    )
    run_encode(
        components(
            {
                "revision": "v1",
                "resolved_config_hash": "semantic",
                "compatible_resolved_config_hashes": ["legacy"],
            },
            lambda _sample: (_ for _ in ()).throw(
                AssertionError("compatible performance config should reuse cache")
            ),
        ),
        device=torch.device("cpu"),
    )
