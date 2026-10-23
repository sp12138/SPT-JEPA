"""Optional aliases for parity diagnostics."""


def legacy_probe_aliases(metrics: dict[str, float]) -> dict[str, float]:
    aliases = {
        "diag/grad_norm_pre_clip": "probe/grad_norm_before_clip",
        "diag/deepspeed_global_grad_norm": "probe/grad_norm_before_clip",
        "diag/clip_return_norm": "probe/clip_return_norm",
        "diag/clip_coefficient": "probe/clip_coef",
        "diag/clip_active": "probe/clip_active",
        "diag/grad_norm_post_clip": "probe/grad_norm_after_clip",
        "probe/param_norm_before_step": "probe/param_norm_before_update",
        "probe/param_norm_after_step": "probe/param_norm_after_update",
    }
    return {alias: metrics[source] for alias, source in aliases.items() if source in metrics}


__all__ = ["legacy_probe_aliases"]
