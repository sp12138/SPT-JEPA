"""Explicit model dispatch shared by the existing campaign trainer and audits."""
from importlib import import_module


MODEL_TARGETS = {
    "base": "as_jepa_lewm.model.ASJepaLeWM",
    "research": "as_jepa_lewm.research.ResearchLeWM",
    "robust": "as_jepa_lewm.robust.RobustLeWM",
    "frontier": "as_jepa_lewm.frontier.FrontierLeWM",
    "carrier": "as_jepa_lewm.carrier.CarrierLeWM",
    "pathquery": "as_jepa_lewm.pathquery.PathQueryLeWM",
    "native_lewm": "as_jepa_lewm.native_lewm.NativeLeWM",
}


def model_target(config):
    selected = config.get("model_class")
    if selected is None:
        # Preserve the historical precedence exactly for all existing runs.
        selected = ("frontier" if config.get("frontier_variant") else
                    "robust" if config.get("robust_variant") else
                    "research" if config.get("research_variant") else "base")
    if selected not in MODEL_TARGETS:
        raise ValueError(f"Unknown registered model_class: {selected!r}")
    return MODEL_TARGETS[selected]


def build_model(config):
    module, name = model_target(config).rsplit(".", 1)
    return getattr(import_module(module), name)(**config["model"])


def gradient_expectations(model, config=None):
    """Frozen representation variants declare their different trainable contract."""
    method = getattr(model, "audit_expected_gradients", None)
    expectations = method() if method else {
        "encoder": "active", "predictor": "active", "renderer": "active"}
    if not isinstance(expectations, dict) or not expectations:
        raise ValueError("Gradient expectations must be a nonempty mapping")
    expectations = dict(expectations)
    expectations.update((config or {}).get("audit_expectations", {}).get("gradients", {}))
    for prefix, expectation in expectations.items():
        if not isinstance(prefix, str) or expectation not in {"active", "frozen", "optional"}:
            raise ValueError(f"Invalid gradient expectation: {prefix!r}: {expectation!r}")
    return dict(expectations)
