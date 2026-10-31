"""CPU-only algebra/AD contracts; no dataset, checkpoint, or evaluation access."""
import itertools
import json
from dataclasses import replace
import torch
from experiments.decoder_geometry_20261031.core import directional_check, spectrum, state_hash, vjp_gram


def main():
    torch.set_num_threads(1); torch.manual_seed(123)
    a = torch.randn(4, 3, dtype=torch.float64)
    signs = torch.tensor(list(itertools.product([-1., 1.], repeat=4)), dtype=torch.float64)
    u = signs @ a / 2
    torch.testing.assert_close(u.T @ u / len(u), a.T @ a / 4)
    z = torch.randn(1, 3, dtype=torch.float64)
    gram, _ = vjp_gram(lambda x:x @ a.T, z, 8)
    assert spectrum(gram)["psd"]
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    torch.testing.assert_close(torch.linalg.eigvalsh(gram), torch.linalg.eigvalsh(q.T @ gram @ q))
    derivative, _ = directional_check(lambda x:x @ a.T, z)
    assert max(x["relative_error"] for x in derivative["finite_differences"]) < 1e-8

    # Exercise the actual anchored WorldDiT hooks and math SDPA, including forward AD.
    from vforge.ops import OpsConfig, initialize_ops
    from vforge.models.as_jepa.renderer import AnchoredWorldDiT, renderer_config
    initialize_ops(OpsConfig(backend="sdpa_math", allow_fallback=False), device=torch.device("cpu"), parallel_state=None)
    cfg = replace(renderer_config(2), width=24, depth=2, heads=3, adaln_lora_dim=8)
    renderer = AnchoredWorldDiT(cfg, state_dim=2, num_slots=2).eval().requires_grad_(False)
    # Native final projection is zero at exact init. Make this synthetic AD test nonzero.
    with torch.no_grad():
        for n, p in renderer.named_parameters():
            if p.numel() > 1 and not bool(p.any()): p.normal_(std=.02)
    before = state_hash(renderer)
    x, anchor = torch.randn(1, 48, 1, 16, 16), torch.randn(1, 48, 1, 16, 16)
    z = torch.randn(1, 2, 2); t = torch.tensor([[.7]])
    def response(state):
        with renderer.anchored(anchor): return renderer(x, t, condition=state)
    derivative, _ = directional_check(response, z)
    gram, _ = vjp_gram(response, z, 2)
    assert derivative["jvp_norm"] > 0
    assert max(x["relative_error"] for x in derivative["finite_differences"]) < .03
    assert spectrum(gram)["psd"] and spectrum(gram)["numerical_rank"] <= 2
    assert state_hash(renderer) == before
    assert all(p.grad is None and not p.requires_grad for p in renderer.parameters())
    print(json.dumps(dict(passed=True, checks=["Rademacher unbiased identity", "PSD and same-spectrum rotation",
        "linear JVP finite differences", "actual anchored DiT math-SDPA forward AD and VJP",
        "frozen parameters/buffers"], tiny_renderer=derivative), indent=2))


if __name__ == "__main__": main()
