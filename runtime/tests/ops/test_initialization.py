"""Process determinism is configured once alongside attention dispatch."""

import os

import pytest
import torch

import vforge.ops as ops
from vforge.ops._attention import dispatch
from vforge.ops._attention.flash import FA4Backend
from vforge.ops.config import OpsConfig


@pytest.fixture
def isolated_ops(monkeypatch):
    monkeypatch.setattr(ops, "_OPS_CONFIG", None)
    monkeypatch.setattr(ops, "_RESOLVED_BACKEND", None)
    for name, value in (
        ("_active_flash_backend", None),
        ("_backend_resolution_attempted", False),
        ("_flash_attn_import_error", None),
        ("_FLASH_ATTENTION_DETERMINISTIC", False),
        ("_REQUESTED_FLASH_BACKEND", None),
        ("_SDPA_PRIORITY", None),
        ("_DEBUG_ENABLED", False),
    ):
        monkeypatch.setattr(dispatch, name, value)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", False)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    yield
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("backend", ["auto", "sdpa_math", "flash_attention_4"])
def test_deterministic_controls_torch_cudnn_and_fa4(isolated_ops, monkeypatch, enabled, backend):
    torch.use_deterministic_algorithms(not enabled, warn_only=True)
    torch.backends.cudnn.deterministic = not enabled
    rng_state = torch.get_rng_state()
    loaded = []

    def load_fa4(*, deterministic=False):
        loaded.append(deterministic)
        return FA4Backend(None, None, None, None, deterministic=deterministic)

    monkeypatch.setattr(dispatch, "load_fa4_backend", load_fa4)
    monkeypatch.setattr(dispatch, "_get_gpu_kind", lambda: "b300")
    # A fake CUDA device selects the FA4 path without allocating GPU tensors.
    device = torch.device("cpu" if backend == "sdpa_math" else "cuda")
    config = OpsConfig(backend=backend, allow_fallback=False, deterministic=enabled)
    resolved = "flash_attention_4" if backend == "auto" else backend
    assert ops.initialize_ops(config, device=device, parallel_state=object()) == resolved
    assert torch.are_deterministic_algorithms_enabled() is enabled
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    assert torch.backends.cudnn.deterministic is enabled
    assert torch.backends.cudnn.benchmark is (not enabled)
    assert torch.backends.cuda.matmul.allow_tf32
    assert torch.backends.cudnn.allow_tf32
    assert torch.equal(torch.get_rng_state(), rng_state)
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == (":4096:8" if enabled else None)
    assert loaded == ([enabled] if backend == "auto" else [])
    if resolved == "flash_attention_4":
        assert ops.get_flash_attention_capability()["FLASH_ATTN_4_AVAILABLE"]
        assert loaded == [enabled]

    def unexpected_reconfiguration(*_args, **_kwargs):
        raise AssertionError("idempotent initialization must not reset process settings")

    with monkeypatch.context() as repeated:
        repeated.setattr(torch, "use_deterministic_algorithms", unexpected_reconfiguration)
        assert ops.initialize_ops(config, device=device, parallel_state=object()) == resolved
        with pytest.raises(RuntimeError, match="conflicting OpsConfig"):
            ops.initialize_ops(
                OpsConfig(backend=backend, allow_fallback=False, deterministic=not enabled),
                device=device,
                parallel_state=object(),
            )


def test_auto_backend_rejects_nondeterministic_fa3(isolated_ops, monkeypatch):
    monkeypatch.setattr(dispatch, "_get_gpu_kind", lambda: "h200")
    with pytest.raises(RuntimeError, match="runtime.deterministic=true.*FlashAttention-3"):
        ops.initialize_ops(
            OpsConfig(deterministic=True),
            device=torch.device("cuda"),
            parallel_state=object(),
        )
