"""Training loader contracts for device transfer, epochs, and resumable state."""

import weakref

import pytest
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from vforge.data.loader import (
    TrainingDataLoader,
    is_resumable_dataloader,
    supports_bit_exact_resume,
)
from vforge.data.multi_stream_loader import MultiStreamStatefulDataLoader


def _stream(loader):
    return TrainingDataLoader(loader, device=torch.device("cpu"))


@pytest.mark.parametrize("count", [1, 3])
def test_iterator_loads_on_demand_without_retaining_batches_or_reseeding_rng(count):
    events, references = [], []

    class Loader:
        def __init__(self):
            self.index = 0

        def __iter__(self):
            return self

        def __next__(self):
            assert all(reference() is None for reference in references)
            batch = torch.tensor(float(self.index))
            references.append(weakref.ref(batch))
            events.append("load")
            self.index += 1
            return batch

    source = Loader()
    stream = _stream(source)
    assert iter(stream) is stream
    assert events == []
    torch.manual_seed(999)
    reference = torch.Generator().manual_seed(999)

    for index in range(count):
        batch = next(stream)
        assert batch.item() == index
        expected = torch.rand(8, generator=reference)
        torch.testing.assert_close(torch.rand(8), expected, rtol=0, atol=0)
        events.append("backward")
        del batch
    assert events == ["load", "backward"] * count
    assert all(reference() is None for reference in references)
    assert source.index == count


def test_iterator_restores_cursor_across_epoch_boundary():
    def loader():
        return StatefulDataLoader([1, 2, 3], batch_size=None)

    original = _stream(loader())
    assert [next(original), next(original)] == [1, 2]
    saved_loader_state = original.state_dict()
    expected = [next(original), next(original)]

    resumed = _stream(loader())
    resumed.load_state_dict(saved_loader_state)

    assert [next(resumed), next(resumed)] == expected == [3, 1]
    assert resumed.state_dict()["epoch"] == original.state_dict()["epoch"] == 1
    assert resumed.state_dict()["batches_in_epoch"] == 1


@pytest.mark.parametrize("consumed", [2, 5, 7])
def test_shuffled_sampler_resume_matches_continuous_loading(consumed):
    def build():
        dataset = list(range(5))
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=1, rank=0, shuffle=True, seed=42
        )
        return _stream(StatefulDataLoader(dataset, batch_size=None, sampler=sampler))

    original = build()
    for _ in range(consumed):
        next(original)
    saved = original.state_dict()
    assert set(saved) == {"epoch", "batches_in_epoch", "state_dict"}
    expected = [next(original) for _ in range(9)]

    resumed = build()
    resumed.load_state_dict(saved)

    assert [next(resumed) for _ in range(9)] == expected


def test_sampler_epoch_advances_once_per_exhausted_loader():
    epochs = []

    class Sampler:
        def set_epoch(self, epoch):
            epochs.append(epoch)

    class Loader:
        sampler = Sampler()

        def __iter__(self):
            yield torch.ones(1)

    stream = _stream(Loader())
    for _ in range(4):
        next(stream)
    assert epochs == [0, 1, 2, 3]


def test_empty_loader_has_an_explicit_error():
    stream = _stream([])
    with pytest.raises(ValueError, match="training dataloader must not be empty"):
        next(stream)


def test_device_transfer_handles_nested_batches_and_preserves_metadata():
    batch = {"input": torch.ones(2), "condition": (torch.zeros(1), ["text", 3])}
    loader = TrainingDataLoader([batch], device=torch.device("meta"))

    moved = next(loader)

    assert moved["input"].device.type == "meta"
    assert moved["condition"][0].device.type == "meta"
    assert moved["condition"][1] == ["text", 3]
    assert batch["input"].device.type == "cpu"


def test_device_is_optional_and_preserves_the_source_batch():
    batch = {"input": torch.ones(2)}
    loader = TrainingDataLoader([batch])
    assert next(loader) is batch


def test_wrapper_does_not_claim_resume_support_for_a_stateless_source():
    loader = _stream([1, 2])
    assert not is_resumable_dataloader(loader)
    assert supports_bit_exact_resume(loader) is None
    with pytest.raises(TypeError, match="not resumable"):
        loader.state_dict()


def test_wrapper_preserves_the_source_bit_exact_policy():
    source = StatefulDataLoader([1, 2], batch_size=None)
    source._vforge_bit_exact_resume = False
    loader = _stream(source)
    assert is_resumable_dataloader(loader)
    assert supports_bit_exact_resume(loader) is False


@pytest.mark.parametrize(
    "state",
    [None, {}, {"epoch": -1, "batches_in_epoch": 0}, {"epoch": 0, "batches_in_epoch": True}],
)
def test_restore_rejects_missing_or_invalid_cursor(state):
    loader = _stream(StatefulDataLoader([1, 2], batch_size=None))
    with pytest.raises(ValueError, match="dataloader"):
        loader.load_state_dict(state)


def test_restore_requires_underlying_state_instead_of_replaying_batches():
    loader = _stream(StatefulDataLoader([1, 2], batch_size=None))
    with pytest.raises(ValueError, match="missing underlying dataloader state_dict"):
        loader.load_state_dict({"epoch": 0, "batches_in_epoch": 1})


def test_restore_sets_sampler_epoch_before_loading_its_position():
    class Sampler:
        epoch = 0

        def set_epoch(self, epoch):
            self.epoch = epoch

    class Source:
        _vforge_resumable_dataloader = True
        sampler = Sampler()

        def load_state_dict(self, state):
            assert self.sampler.epoch == 3
            assert state == {"position": 2}

        def __iter__(self):
            assert self.sampler.epoch == 3
            return iter([7])

    loader = _stream(Source())
    loader.load_state_dict({"epoch": 3, "batches_in_epoch": 2, "state_dict": {"position": 2}})
    assert next(loader) == 7


def test_multi_stream_loader_keeps_independent_epochs_and_restores_them():
    def build():
        return _stream(
            MultiStreamStatefulDataLoader(
                {
                    "generator": StatefulDataLoader([1, 2], batch_size=None),
                    "critic": StatefulDataLoader([10, 20, 30], batch_size=None),
                }
            )
        )

    loader = build()
    for _ in range(5):
        next(loader)
    saved = loader.state_dict()
    assert saved["epoch"] == 0
    assert saved["batches_in_epoch"] == 5
    assert saved["state_dict"]["epochs"] == {"generator": 2, "critic": 1}
    expected = [next(loader) for _ in range(4)]

    resumed = build()
    resumed.load_state_dict(saved)

    assert [next(resumed) for _ in range(4)] == expected
    assert resumed.state_dict()["state_dict"]["epochs"] == {"generator": 4, "critic": 2}
