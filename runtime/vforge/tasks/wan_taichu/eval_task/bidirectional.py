"""Ordinary bidirectional Wan Taichu evaluation workflow."""

from __future__ import annotations

from vforge.evaluation.runner import EvalComponents
from vforge.tasks.requests import EvalRequest
from vforge.tasks.wan_taichu.eval_task.base import build_base_eval_components


def build_bidirectional_eval_components(request: EvalRequest) -> EvalComponents:
    return build_base_eval_components(request)


__all__ = ["build_bidirectional_eval_components"]
