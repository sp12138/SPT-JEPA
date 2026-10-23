from __future__ import annotations

import hashlib
import json

import torch

import vforge.utils.inference_trace as inference_trace
from vforge.utils.inference_trace import InferenceTrace


def test_inference_trace_writes_bitwise_hash_and_selected_snapshot(tmp_path) -> None:
    value = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    trace = InferenceTrace(tmp_path, "vforge", frozenset({"input.latents"}))

    trace.event("route", {"expert": "wan"})
    trace.tensor("input.latents", value)

    records = [
        json.loads(line) for line in (tmp_path / "vforge" / "trace.jsonl").read_text().splitlines()
    ]
    expected = hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    assert records[0] == {"expert": "wan", "sequence": 0, "stage": "route"}
    assert records[1]["sha256"] == expected
    assert records[1]["dtype"] == "torch.bfloat16"
    assert records[1]["shape"] == [1, 2]
    snapshot = records[1]["snapshot"]
    assert torch.equal(torch.load(snapshot, weights_only=True), value)


def test_install_inference_module_trace_hashes_normalized_module_outputs(
    tmp_path, monkeypatch
) -> None:
    class Projection(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
            self.bias = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value

    class Transformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.patch_embedding = Projection()
            self.condition_embedder = torch.nn.Module()
            self.condition_embedder.time_embedder = torch.nn.Identity()
            self.blocks = torch.nn.ModuleList([torch.nn.Identity()])
            self.norm_out = torch.nn.Identity()
            self.proj_out = torch.nn.Identity()

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            value = self.patch_embedding(value)
            value = self.condition_embedder.time_embedder(value)
            value = self.blocks[0](value)
            value = self.norm_out(value)
            return self.proj_out(value)

    class Wrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer = Transformer()

        def primary_transformer(self) -> torch.nn.Module:
            return self.transformer

    monkeypatch.setenv("WAN_TAICHU_INFERENCE_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("WAN_TAICHU_INFERENCE_TRACE_SIDE", "vforge")
    monkeypatch.setenv("WAN_TAICHU_INFERENCE_TRACE_MODULES", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(inference_trace, "_TRACE_INSTANCE", None)
    monkeypatch.setattr(inference_trace, "_TRACE_KEY", None)

    model = Wrapper()
    inference_trace.install_inference_module_trace(model)
    inference_trace.install_inference_module_trace(model)
    model.transformer(torch.tensor([1.0], dtype=torch.bfloat16))

    records = [
        json.loads(line) for line in (tmp_path / "vforge" / "trace.jsonl").read_text().splitlines()
    ]
    assert [record["stage"] for record in records] == [
        "module.weight.patch_embedding.weight",
        "module.compute_weight.patch_embedding.weight",
        "module.weight.patch_embedding.bias",
        "module.compute_weight.patch_embedding.bias",
        "module.000.patch_embedding.input.0",
        "module.000.patch_embedding.output",
        "module.000.condition_embedder.time_embedder.input.0",
        "module.000.condition_embedder.time_embedder.output",
        "module.000.blocks.0.input.0",
        "module.000.blocks.0.output",
        "module.000.norm_out.input.0",
        "module.000.norm_out.output",
        "module.000.proj_out.input.0",
        "module.000.proj_out.output",
    ]
