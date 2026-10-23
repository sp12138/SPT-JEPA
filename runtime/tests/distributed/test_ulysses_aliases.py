"""The ulysses spelling keeps working, and says it is on the way out.

It exists for downstream repositories that pin vforge as a submodule: without it
the pin and the code reading it would have to move in one commit, which is
impossible while vforge's CI pins that repository back. The sp spelling from the
generation before is gone, and a test here holds it gone -- Megatron uses that
word for a different mechanism, which is what the rename was for.
"""

from __future__ import annotations

import dataclasses

import pytest

from vforge.distributed.parallel_state import ParallelConfig, ParallelState
from vforge.training.parity.config import ParityStepConfig
from vforge.training.parity.optimizer_step import resolve_grad_scale


def test_config_accepts_the_old_keyword():
    with pytest.warns(FutureWarning, match="usp_size"):
        config = ParallelConfig(ulysses_size=4)
    assert config.usp_size == 4


def test_config_rejects_both_spellings_at_once():
    with pytest.raises(TypeError, match="not both"):
        ParallelConfig(usp_size=4, ulysses_size=2)


def test_config_reads_back_under_the_old_name():
    config = ParallelConfig(usp_size=4)
    with pytest.warns(FutureWarning, match="usp_size"):
        assert config.ulysses_size == 4


def test_the_old_name_is_not_a_field():
    """Or it would land in every eval artifact, which serializes this config."""
    names = {f.name for f in dataclasses.fields(ParallelConfig)}
    serialized = dataclasses.asdict(ParallelConfig(usp_size=4))
    assert "usp_size" in names
    assert "ulysses_size" not in names
    assert "ulysses_size" not in serialized


def test_state_exposes_the_groups_under_the_old_name():
    state = ParallelState(config=ParallelConfig(usp_size=1))
    with pytest.warns(FutureWarning, match="usp_group"):
        assert state.ulysses_group is state.usp_group
    with pytest.warns(FutureWarning, match="usp_rank"):
        assert state.ulysses_rank == state.usp_rank


@pytest.mark.parametrize("gone", ["sp_size", "sp_rank", "sp_group"])
def test_the_sp_spelling_is_gone(gone):
    """Two generations back, and the one word Megatron means differently."""
    state = ParallelState(config=ParallelConfig(usp_size=1))
    assert not hasattr(state, gone)
    assert not hasattr(state.config, gone)


def test_parity_keeps_the_old_grad_scale_value_and_still_scales_by_it():
    """Kept verbatim so a caller reading its own config back sees what it wrote."""
    with pytest.warns(FutureWarning, match="usp_size"):
        parity = ParityStepConfig(scale_grads_before_optimizer_step="ulysses_size")
    assert parity.scale_grads_before_optimizer_step == "ulysses_size"
    assert resolve_grad_scale(parity.scale_grads_before_optimizer_step, usp_size=4) == 4.0


@pytest.mark.parametrize("rejected", ["sp_size", "dp_size"])
def test_parity_rejects_unknown_values_including_sp_size(rejected):
    with pytest.raises(ValueError, match="usp_size"):
        ParityStepConfig(scale_grads_before_optimizer_step=rejected)
