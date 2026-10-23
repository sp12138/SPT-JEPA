"""Private workflow routing for Wan Taichu evaluation."""

from __future__ import annotations

from vforge.evaluation.runner import EvalComponents
from vforge.tasks.requests import EvalRequest
from vforge.tasks.wan_taichu.eval_task.bidirectional import (
    build_bidirectional_eval_components,
)


def build_eval_components(request: EvalRequest) -> EvalComponents:
    if "wan_use_ref_token" in request.extras:
        raise ValueError(
            "Wan Taichu derives ref-token capability from model.extras.variant; "
            "remove model.extras.wan_use_ref_token"
        )
    variant = str(request.extras.get("variant") or "").strip()
    is_causal = variant in {
        "wan22_ti2v_5b_causal_teacher",
        "wan22_a14b_causal_teacher",
    }
    if request.gridnormal_enabled:
        if is_causal:
            raise ValueError("evaluation.gridnormal.enabled is not supported by the causal variant")
        from vforge.tasks.wan_taichu.eval_task.gridnormal import (
            build_gridnormal_eval_components,
        )

        return build_gridnormal_eval_components(request)
    if is_causal:
        from vforge.tasks.wan_taichu.eval_task.causal import build_causal_eval_components

        return build_causal_eval_components(request)
    if request.rollout_mode:
        raise ValueError("evaluation.rollout_mode is only valid for a causal teacher variant")
    return build_bidirectional_eval_components(request)


__all__ = ["build_eval_components"]
