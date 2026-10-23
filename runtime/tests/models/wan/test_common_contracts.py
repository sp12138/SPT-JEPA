"""Structural contracts for the unified Wan model family."""

from __future__ import annotations

import ast
import importlib
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn


class _Transformer(nn.Module):
    def __init__(self, tag: str = "") -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.blocks = nn.ModuleList()
        self.config = SimpleNamespace(tag=tag)
        self.cache_events: list[tuple[str, str]] = []

    @contextmanager
    def cache_context(self, name: str):
        self.cache_events.append(("enter", name))
        try:
            yield
        finally:
            self.cache_events.append(("exit", name))

    def forward(self, value):
        return value


def test_family_import_is_lightweight() -> None:
    code = (
        "import sys; import vforge.models.wan; "
        "assert 'diffusers' not in sys.modules; "
        "assert not any(name.endswith('.transformer') "
        "for name in sys.modules if name.startswith('vforge.models.wan.'))"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).parents[3],
        check=True,
    )


@pytest.mark.parametrize(
    "module_name",
    [
        "vforge.models.wan21",
        "vforge.models.wan22_a14b",
        "vforge.models.wan22_ti2v_5b",
        "vforge.models.wan22_ti2v_5b_tf",
        "vforge.models.wan.common",
    ],
)
def test_removed_model_paths_have_no_compatibility_layer(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


def test_single_and_dual_roots_keep_variant_owned_checkpoint_prefixes() -> None:
    from vforge.models.wan.model import (
        WanModelRootBase,
        WanSingleTransformerModel,
    )
    from vforge.models.wan.v22.a14b.model import Wan22A14bModel

    single = WanSingleTransformerModel(_Transformer("single"))
    dual = Wan22A14bModel(_Transformer("high"), _Transformer("low"))

    assert isinstance(single, WanModelRootBase)
    assert isinstance(dual, WanModelRootBase)
    assert not isinstance(dual, WanSingleTransformerModel)
    assert set(single.state_dict()) == {"transformer.weight"}
    assert set(dual.state_dict()) == {"wan.weight", "wan_2.weight"}
    assert dual.fsdp_execution_roots() == (dual.wan, dual.wan_2)


def test_root_cache_context_covers_every_registered_expert() -> None:
    from vforge.models.wan.v22.a14b.model import Wan22A14bModel

    model = Wan22A14bModel(_Transformer("high"), _Transformer("low"))
    with model.cache_context("cond"):
        assert model.wan.cache_events == [("enter", "cond")]
        assert model.wan_2.cache_events == [("enter", "cond")]
    assert model.wan.cache_events[-1] == ("exit", "cond")
    assert model.wan_2.cache_events[-1] == ("exit", "cond")


def test_channel_layout_reports_whether_the_patch_embedding_grew() -> None:
    from vforge.models.wan.model import WanChannelLayout

    published = WanChannelLayout(z_dim=48, pretrained_in_channels=48, in_channels=48)
    widened = WanChannelLayout(z_dim=48, pretrained_in_channels=48, in_channels=96)
    assert not published.is_widened
    assert widened.is_widened
    with pytest.raises(ValueError, match="must be positive"):
        WanChannelLayout(z_dim=0, pretrained_in_channels=48, in_channels=48)


def test_common_rope_rebuilds_meta_runtime_buffers() -> None:
    from vforge.models.wan.transformer import WanRotaryPosEmbed

    with torch.device("meta"):
        rope = WanRotaryPosEmbed(12, (1, 2, 2), 8)
    rope.to_empty(device=torch.device("cpu"))
    rope.reset_non_persistent_buffers()

    assert not rope.freqs_cos.is_meta
    assert not rope.freqs_sin.is_meta
    assert "freqs_cos" not in rope.state_dict()


def test_variant_attention_and_blocks_share_parameter_compatible_bases() -> None:
    from vforge.models.wan.attention import WanAttention as CommonAttention
    from vforge.models.wan.transformer import (
        WanRotaryPosEmbed as CommonRotary,
    )
    from vforge.models.wan.transformer import (
        WanTransformerBlock as CommonBlock,
    )
    from vforge.models.wan.v21 import transformer as v21
    from vforge.models.wan.v22.ti2v_5b.ct import transformer as ti2v

    assert issubclass(v21.WanAttention, CommonAttention)
    assert issubclass(ti2v.WanAttention, CommonAttention)
    assert issubclass(v21.WanTransformerBlock, CommonBlock)
    assert issubclass(ti2v.WanTransformerBlock, CommonBlock)
    assert v21.WanRotaryPosEmbed is CommonRotary
    assert ti2v.WanRotaryPosEmbed is CommonRotary


def test_model_family_never_imports_task_or_training_layers() -> None:
    import vforge.models.wan

    root = Path(vforge.models.wan.__file__).parent
    forbidden = ("vforge.tasks", "vforge.training")
    violations = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name.startswith(forbidden) for name in names):
                violations.append(path.relative_to(root))
    assert violations == []
