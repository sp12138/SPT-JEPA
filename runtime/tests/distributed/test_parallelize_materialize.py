from types import SimpleNamespace

import pytest
import torch
from torch.distributed.tensor import Shard

import vforge.distributed.parallelize as parallelize_module
from vforge.distributed.parallelize import (
    FSDP2Config,
    FSDPModulePlan,
    FSDPPrecisionContract,
    MixedPrecisionConfig,
    _apply_fsdp2,
    _cast_floating_buffers,
    _resolve_fsdp_leaf_modules,
    _resolve_fsdp_precision_contract,
    _resolve_mixed_precision_preserve_modules,
    _resolve_module_paths,
    materialize_meta_module,
)


class _RuntimeBufferModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(2))
        self.register_buffer(
            "runtime_table",
            torch.empty(2),
            persistent=False,
        )
        self.reset_calls = 0

    def reset_non_persistent_buffers(self) -> None:
        self.runtime_table.fill_(3.0)
        self.reset_calls += 1


class _MissingRuntimeBufferReset(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "runtime_table",
            torch.empty(2),
            persistent=False,
        )


def test_materialize_rebuilds_non_persistent_runtime_buffers() -> None:
    with torch.device("meta"):
        model = _RuntimeBufferModule()

    materialize_meta_module(model, "cpu")

    assert not model.weight.is_meta
    assert torch.equal(model.runtime_table, torch.full((2,), 3.0))
    assert model.reset_calls == 1
    assert "runtime_table" not in model.state_dict()

    materialize_meta_module(model, "cpu")
    assert model.reset_calls == 1


def test_materialize_rejects_meta_runtime_buffer_without_reset() -> None:
    with torch.device("meta"):
        model = _MissingRuntimeBufferReset()

    with pytest.raises(
        RuntimeError,
        match=r"meta non-persistent buffers.*reset_non_persistent_buffers",
    ):
        materialize_meta_module(model, "cpu")


def test_config_can_select_explicit_fsdp_leaf_module_paths() -> None:
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fp32 = torch.nn.Linear(2, 2, dtype=torch.float32)
            self.bf16 = torch.nn.Linear(2, 2, dtype=torch.bfloat16)

    model = Model()

    assert _resolve_fsdp_leaf_modules(model, paths=("bf16",)) == (model.bf16,)


def test_model_and_config_precision_contracts_append_or_replace() -> None:
    class Model(torch.nn.Module):
        _keep_in_fp32_modules = ["model_default"]

        def __init__(self) -> None:
            super().__init__()
            self.model_default = torch.nn.Linear(2, 2, dtype=torch.float32)
            self.configured = torch.nn.Linear(2, 2, dtype=torch.float32)

    model = Model()
    appended = _resolve_fsdp_precision_contract(
        model,
        config=MixedPrecisionConfig(
            preserve_module_paths=("configured",),
        ),
    )
    replaced = _resolve_fsdp_precision_contract(
        model,
        config=MixedPrecisionConfig(
            preserve_module_paths=("configured",),
            contract_mode="replace",
        ),
    )
    assert appended.preserve_module_paths == ("model_default", "configured")
    assert replaced.preserve_module_paths == ("configured",)


def test_configured_preserve_module_paths_must_exist() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 2, dtype=torch.float32))

    with pytest.raises(ValueError, match="'missing' does not exist"):
        _resolve_mixed_precision_preserve_modules(
            model,
            paths=("missing",),
        )


def test_a_preserve_path_naming_a_weight_is_skipped_and_reported(capsys) -> None:
    class Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm1 = torch.nn.LayerNorm(2)
            self.scale_shift_table = torch.nn.Parameter(torch.zeros(1, 2))

    model = Block()

    resolved = _resolve_mixed_precision_preserve_modules(
        model,
        paths=("norm1", "scale_shift_table"),
    )

    assert resolved == (model.norm1,)
    assert "scale_shift_table" in capsys.readouterr().out


def test_preserve_path_globs_expand_one_dotted_segment() -> None:
    class Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm1 = torch.nn.LayerNorm(2)

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = torch.nn.ModuleList([Block(), Block()])
            self.norm_out = torch.nn.LayerNorm(2)

    model = Model()
    assert _resolve_mixed_precision_preserve_modules(
        model,
        paths=("blocks.*.norm1",),
    ) == (
        model.blocks[0].norm1,
        model.blocks[1].norm1,
    )

    inputs = _resolve_module_paths(model, ("blocks.*",), allow_root=False)
    assert inputs == (model.blocks[0], model.blocks[1])


def test_preserve_path_globs_require_a_match() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    with pytest.raises(ValueError, match="does not match"):
        _resolve_mixed_precision_preserve_modules(model, paths=("blocks.*.norm1",))


def test_preserve_path_star_must_be_a_whole_segment() -> None:
    with pytest.raises(ValueError, match="whole segment"):
        MixedPrecisionConfig(preserve_module_paths=("blocks.norm*",))


def test_preserve_modules_keep_their_existing_dtype() -> None:
    model = torch.nn.Module()
    model.projection = torch.nn.Linear(2, 2, dtype=torch.bfloat16)

    assert _resolve_mixed_precision_preserve_modules(
        model,
        paths=("projection",),
    ) == (model.projection,)


def test_mixed_precision_preserve_module_trees_must_not_overlap() -> None:
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.container = torch.nn.Sequential(torch.nn.Linear(2, 2))

    with pytest.raises(ValueError, match="must not overlap"):
        _resolve_mixed_precision_preserve_modules(
            Model(),
            paths=("container", "container.0"),
        )


def test_taichu_compat_disables_model_precision_overrides() -> None:
    config = MixedPrecisionConfig(
        output_dtype="bfloat16",
        buffer_dtype="bfloat16",
        apply_model_overrides=False,
    )

    assert config.output_dtype == "bfloat16"
    assert config.buffer_dtype == "bfloat16"
    assert not config.apply_model_overrides


def test_global_buffer_cast_skips_keep_module_trees() -> None:
    class Keep(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("scale", torch.ones(2, dtype=torch.float32))

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.keep = Keep()
            self.register_buffer("scale", torch.ones(2, dtype=torch.float32))

    model = Model()
    _cast_floating_buffers(
        model,
        torch.bfloat16,
        excluded_modules=(model.keep,),
    )

    assert model.scale.dtype is torch.bfloat16
    assert model.keep.scale.dtype is torch.float32


def test_fsdp_wraps_fp32_keep_module_before_bf16_blocks_and_root(
    monkeypatch,
) -> None:
    class Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

    class Model(torch.nn.Module):
        _no_split_modules = ["Block"]
        _keep_in_fp32_modules = ["keep"]

        def __init__(self) -> None:
            super().__init__()
            self.keep = torch.nn.Linear(2, 2, dtype=torch.float32)
            self.block = Block()

    model = Model()
    calls = []

    def fake_fully_shard(module, **kwargs):
        calls.append((module, kwargs))

    import torch.distributed._composable.fsdp as composable_fsdp

    monkeypatch.setattr(composable_fsdp, "fully_shard", fake_fully_shard)
    monkeypatch.setattr(
        parallelize_module,
        "get_parallel_state",
        lambda: type("State", (), {"ep_enabled": False, "fsdp_mesh": object()})(),
    )

    _apply_fsdp2(
        model,
        FSDP2Config(
            mixed_precision=MixedPrecisionConfig(enable=True),
        ),
    )

    assert [module for module, _kwargs in calls] == [
        model.keep,
        model.block,
        model,
    ]
    assert "mp_policy" not in calls[0][1]
    assert calls[1][1]["mp_policy"].param_dtype is torch.bfloat16
    assert calls[2][1]["mp_policy"].param_dtype is torch.bfloat16


def test_fsdp_can_preserve_auxiliary_forward_input_dtypes(monkeypatch) -> None:
    class Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

    class Model(torch.nn.Module):
        _no_split_modules = ["Block"]

        def __init__(self) -> None:
            super().__init__()
            self.block = Block()

        def get_fsdp_precision_contract(self):
            return FSDPPrecisionContract(
                preserve_forward_input_paths=(".", "block"),
            )

    model = Model()
    calls = []

    def fake_fully_shard(module, **kwargs):
        calls.append((module, kwargs))

    import torch.distributed._composable.fsdp as composable_fsdp

    monkeypatch.setattr(composable_fsdp, "fully_shard", fake_fully_shard)
    monkeypatch.setattr(
        parallelize_module,
        "get_parallel_state",
        lambda: type("State", (), {"ep_enabled": False, "fsdp_mesh": object()})(),
    )

    _apply_fsdp2(
        model,
        FSDP2Config(
            mixed_precision=MixedPrecisionConfig(enable=True),
        ),
    )

    assert [module for module, _kwargs in calls] == [model.block, model]
    assert not calls[0][1]["mp_policy"].cast_forward_inputs
    assert not calls[1][1]["mp_policy"].cast_forward_inputs
    assert calls[0][1]["mp_policy"].param_dtype is torch.bfloat16
    assert calls[1][1]["mp_policy"].param_dtype is torch.bfloat16


def test_preserve_forward_input_path_creates_linear_boundary(monkeypatch) -> None:
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(2, 2)

        def get_fsdp_precision_contract(self):
            return FSDPPrecisionContract(
                preserve_forward_input_paths=("proj",),
            )

    model = Model()
    calls = []

    def fake_fully_shard(module, **kwargs):
        calls.append((module, kwargs))

    import torch.distributed._composable.fsdp as composable_fsdp

    monkeypatch.setattr(composable_fsdp, "fully_shard", fake_fully_shard)
    monkeypatch.setattr(
        parallelize_module,
        "get_parallel_state",
        lambda: type("State", (), {"ep_enabled": False, "fsdp_mesh": object()})(),
    )

    _apply_fsdp2(
        model,
        FSDP2Config(
            mixed_precision=MixedPrecisionConfig(enable=True),
        ),
    )

    assert [module for module, _kwargs in calls] == [model.proj, model]
    assert not calls[0][1]["mp_policy"].cast_forward_inputs
    assert calls[1][1]["mp_policy"].cast_forward_inputs


def test_preserve_parent_subsumes_preserve_input_child(monkeypatch) -> None:
    class Container(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(2, 2)

    class Model(torch.nn.Module):
        _keep_in_fp32_modules = ["container"]

        def __init__(self) -> None:
            super().__init__()
            self.container = Container()

        def get_fsdp_precision_contract(self):
            return FSDPPrecisionContract(
                preserve_forward_input_paths=("container.proj",),
            )

    model = Model()
    calls = []

    def fake_fully_shard(module, **kwargs):
        calls.append((module, kwargs))

    import torch.distributed._composable.fsdp as composable_fsdp

    monkeypatch.setattr(composable_fsdp, "fully_shard", fake_fully_shard)
    monkeypatch.setattr(
        parallelize_module,
        "get_parallel_state",
        lambda: type("State", (), {"ep_enabled": False, "fsdp_mesh": object()})(),
    )

    _apply_fsdp2(model, FSDP2Config(mixed_precision=MixedPrecisionConfig(enable=True)))

    assert [module for module, _kwargs in calls] == [model.container, model]
    assert "mp_policy" not in calls[0][1]


def test_preserve_input_parent_wraps_after_preserve_child(monkeypatch) -> None:
    class Container(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(2, 2)

    class Model(torch.nn.Module):
        _keep_in_fp32_modules = ["proj"]

        def __init__(self) -> None:
            super().__init__()
            self.container = Container()

        def get_fsdp_precision_contract(self):
            return FSDPPrecisionContract(
                preserve_forward_input_paths=("container",),
            )

    model = Model()
    calls = []

    def fake_fully_shard(module, **kwargs):
        calls.append((module, kwargs))

    import torch.distributed._composable.fsdp as composable_fsdp

    monkeypatch.setattr(composable_fsdp, "fully_shard", fake_fully_shard)
    monkeypatch.setattr(
        parallelize_module,
        "get_parallel_state",
        lambda: type("State", (), {"ep_enabled": False, "fsdp_mesh": object()})(),
    )

    _apply_fsdp2(model, FSDP2Config(mixed_precision=MixedPrecisionConfig(enable=True)))

    assert [module for module, _kwargs in calls] == [
        model.container.proj,
        model.container,
        model,
    ]
    assert "mp_policy" not in calls[0][1]
    assert not calls[1][1]["mp_policy"].cast_forward_inputs
    assert calls[2][1]["mp_policy"].cast_forward_inputs


def test_preserve_module_policy_wins_on_the_same_boundary(monkeypatch) -> None:
    class Model(torch.nn.Module):
        _keep_in_fp32_modules = ["proj"]

        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(2, 2)

        def get_fsdp_precision_contract(self):
            return FSDPPrecisionContract(
                preserve_forward_input_paths=("proj",),
            )

    model = Model()
    calls = []

    def fake_fully_shard(module, **kwargs):
        calls.append((module, kwargs))

    import torch.distributed._composable.fsdp as composable_fsdp

    monkeypatch.setattr(composable_fsdp, "fully_shard", fake_fully_shard)
    monkeypatch.setattr(
        parallelize_module,
        "get_parallel_state",
        lambda: type("State", (), {"ep_enabled": False, "fsdp_mesh": object()})(),
    )

    _apply_fsdp2(model, FSDP2Config(mixed_precision=MixedPrecisionConfig(enable=True)))

    assert [module for module, _kwargs in calls] == [model.proj, model]
    assert "mp_policy" not in calls[0][1]


def test_a_configured_path_naming_a_weight_leaves_every_group_untouched(monkeypatch) -> None:
    class TableModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(2, 2)
            self.table = torch.nn.Parameter(torch.zeros(1, 2))

    model = TableModel()
    calls: list[tuple[torch.nn.Module, dict]] = []

    def fake_fully_shard(module, **kwargs):
        calls.append((module, kwargs))

    import torch.distributed._composable.fsdp as composable_fsdp

    monkeypatch.setattr(composable_fsdp, "fully_shard", fake_fully_shard)
    monkeypatch.setattr(
        parallelize_module,
        "get_parallel_state",
        lambda: type(
            "State", (), {"ep_enabled": False, "fsdp_mesh": object(), "fsdp_group": None}
        )(),
    )

    _apply_fsdp2(
        model,
        FSDP2Config(
            mixed_precision=MixedPrecisionConfig(
                enable=True,
                preserve_module_paths=("table",),
            )
        ),
    )

    assert [module for module, _kwargs in calls] == [model]
    assert "ignored_params" not in calls[0][1]
    assert calls[0][1]["mp_policy"].param_dtype is torch.bfloat16


@pytest.mark.parametrize("precision", ["default", "preserve", "preserve_input"])
def test_extra_plan_uses_generic_fsdp_boundaries(monkeypatch, precision) -> None:
    import torch.distributed._composable.fsdp as composable_fsdp

    model = torch.nn.Sequential(torch.nn.Sequential(torch.nn.Linear(2, 2)), torch.nn.Linear(2, 2))
    mesh = object()
    extra_mesh = SimpleNamespace(size=lambda dim=None: 2)
    calls, divisors = [], []

    def fake_fully_shard(module, **kwargs):
        calls.append((module, kwargs))
        module.set_gradient_divide_factor = lambda factor: divisors.append((module, factor))

    monkeypatch.setattr(composable_fsdp, "fully_shard", fake_fully_shard)
    # No EP state or expert module is needed to consume a resolved plan.
    monkeypatch.setattr(
        parallelize_module, "get_parallel_state", lambda: SimpleNamespace(fsdp_mesh=mesh)
    )
    _apply_fsdp2(
        model,
        FSDP2Config(
            basic_modules=("Sequential", "Linear"),
            mixed_precision=MixedPrecisionConfig(
                enable=True,
                preserve_module_paths=("0",) if precision == "preserve" else (),
                preserve_forward_input_paths=("0.0",) if precision == "preserve_input" else (),
            ),
        ),
        extra_plan={model[0][0]: FSDPModulePlan(extra_mesh, Shard(1), 4)},
    )

    assert [module for module, _ in calls] == [model[0][0], model[0], model[1], model]
    assert divisors == [(model[0][0], 4)]
    kwargs = calls[0][1]
    assert kwargs["mesh"] is extra_mesh
    assert kwargs["shard_placement_fn"](model[0][0].weight) == Shard(1)
    if precision == "preserve":
        assert "mp_policy" not in kwargs
        assert "mp_policy" not in calls[1][1]
    else:
        assert kwargs["mp_policy"].param_dtype is torch.bfloat16
        assert kwargs["mp_policy"].cast_forward_inputs is (precision != "preserve_input")
    for _, kwargs in calls[1:]:
        assert kwargs["mesh"] is mesh
        assert "shard_placement_fn" not in kwargs


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_singleton_fsdp_plan_rejects_unsupported_reduction_dtype(monkeypatch, dtype):
    monkeypatch.setattr(
        parallelize_module, "get_parallel_state", lambda: SimpleNamespace(fsdp_mesh=None)
    )
    model = torch.nn.Linear(2, 2, dtype=dtype)
    with pytest.raises(NotImplementedError, match="float32 or bfloat16 gradient reduction"):
        _apply_fsdp2(
            model,
            FSDP2Config(),
            extra_plan={
                model: FSDPModulePlan(SimpleNamespace(size=lambda dim=None: 1), Shard(1), 2)
            },
        )


def test_hsdp_singleton_shard_keeps_custom_division_precision_guard(monkeypatch):
    model = torch.nn.Linear(2, 2)
    mesh = SimpleNamespace(size=lambda dim=None: 1 if dim == -1 else 2)
    monkeypatch.setattr(
        parallelize_module, "get_parallel_state", lambda: SimpleNamespace(fsdp_mesh=mesh)
    )
    with pytest.raises(NotImplementedError, match="float32 or bfloat16 gradient reduction"):
        _apply_fsdp2(
            model,
            FSDP2Config(
                mixed_precision=MixedPrecisionConfig(enable=True, reduce_dtype="float16"),
            ),
            extra_plan={model: FSDPModulePlan(mesh, Shard(1), 4)},
        )
