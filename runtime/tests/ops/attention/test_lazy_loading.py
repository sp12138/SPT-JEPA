import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from vforge.ops._attention import dispatch as attention


def _reset_backend_resolution(monkeypatch, *, gpu_kind):
    monkeypatch.setattr(attention, "_active_flash_backend", None)
    monkeypatch.setattr(attention, "_backend_resolution_attempted", False)
    monkeypatch.setattr(attention, "_flash_attn_import_error", None)
    monkeypatch.setattr(attention, "_gpu_kind", attention._GPU_KIND_UNRESOLVED)
    monkeypatch.setattr(attention, "_detect_gpu_kind", lambda: gpu_kind)


def test_import_does_not_probe_cuda():
    script = """
import torch

def fail():
    raise AssertionError("attention import probed CUDA")

torch.cuda.is_available = fail
import vforge.ops
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONPATH": os.getcwd()},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_backend_loads_once_on_capability_query(monkeypatch):
    _reset_backend_resolution(monkeypatch, gpu_kind="b300")
    calls = []
    dense = object()
    varlen = object()
    backend = SimpleNamespace(
        backend_id="flash_attention_4",
        dense_func=dense,
        varlen_func=varlen,
    )

    def load():
        calls.append("load")
        return backend

    monkeypatch.setattr(attention, "load_fa4_backend", load)

    assert attention.get_attn_implementation() == "flash_attention_4"
    assert calls == []
    capability = attention.get_flash_attention_capability()
    assert capability["FLASH_ATTN_4_AVAILABLE"] is True
    assert capability["FLASH_ATTN_3_AVAILABLE"] is False
    assert capability["flash_attn_func"] is dense
    assert capability["flash_attn_varlen_func"] is varlen
    assert calls == ["load"]
    assert attention.get_flash_attention_capability()["backend_id"] == ("flash_attention_4")
    assert calls == ["load"]


def test_missing_backend_error_is_deferred_until_backend_use(monkeypatch):
    _reset_backend_resolution(monkeypatch, gpu_kind="b300")
    calls = []

    def load():
        calls.append("load")
        raise ModuleNotFoundError("flash_attn")

    monkeypatch.setattr(attention, "load_fa4_backend", load)

    assert attention.get_attn_implementation() == "flash_attention_4"
    assert calls == []
    capability = attention.get_flash_attention_capability()
    assert capability["FLASH_ATTN_4_AVAILABLE"] is False
    assert calls == ["load"]
    with pytest.raises(ImportError, match="B300 missing"):
        attention._require_flash_backend()
    assert calls == ["load"]
