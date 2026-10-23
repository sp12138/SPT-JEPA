from __future__ import annotations

import pytest

from vforge.tasks.requests import EvalRequest
from vforge.tasks.wan_taichu.eval_task import build_eval_components


def test_gridnormal_workflow_routes_directly(monkeypatch) -> None:
    sentinel = object()
    observed = {}

    def build_gridnormal(request):
        observed["request"] = request
        return sentinel

    monkeypatch.setattr(
        "vforge.tasks.wan_taichu.eval_task.gridnormal.build_gridnormal_eval_components",
        build_gridnormal,
    )
    request = EvalRequest(
        extras={"variant": "wan22_ti2v_5b_ref"},
        gridnormal_enabled=True,
    )

    assert build_eval_components(request) is sentinel
    assert observed["request"] is request


def test_causal_variant_routes_without_requiring_redundant_workflow(monkeypatch) -> None:
    sentinel = object()
    observed = {}

    def build_causal(request):
        observed["request"] = request
        return sentinel

    monkeypatch.setattr(
        "vforge.tasks.wan_taichu.eval_task.causal.build_causal_eval_components",
        build_causal,
    )
    request = EvalRequest(extras={"variant": "wan22_ti2v_5b_causal_teacher"})

    assert build_eval_components(request) is sentinel
    assert observed["request"] is request


def test_router_rejects_crossed_or_redundant_execution_selectors() -> None:
    with pytest.raises(ValueError, match="not supported by the causal variant"):
        build_eval_components(
            EvalRequest(
                extras={"variant": "wan22_ti2v_5b_causal_teacher"},
                gridnormal_enabled=True,
            )
        )
    with pytest.raises(ValueError, match="only valid"):
        build_eval_components(
            EvalRequest(
                extras={"variant": "wan22_ti2v_5b"},
                rollout_mode="incremental_kv",
            )
        )
    with pytest.raises(ValueError, match="remove model.extras.wan_use_ref_token"):
        build_eval_components(
            EvalRequest(
                extras={
                    "variant": "wan22_ti2v_5b_ref",
                    "wan_use_ref_token": True,
                }
            )
        )
