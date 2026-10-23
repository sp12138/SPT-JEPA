"""Single public registry façade for Wan Taichu evaluation."""

from __future__ import annotations

from vforge.evaluation.runner import EvalComponents
from vforge.tasks.requests import EvalRequest


def build_eval_components(request: EvalRequest) -> EvalComponents:
    from vforge.tasks.wan_taichu.eval_task import build_eval_components as build_workflow

    return build_workflow(request)


__all__ = ["build_eval_components"]
