"""CPU algebra contracts for the new accumulation/factorization, no model IO."""
import json
import torch
from experiments.decoder_geometry_20261031.calibrate import finalize
from experiments.decoder_geometry_20261031.core import cpu_generator


def main():
    torch.set_num_threads(1)
    vectors = torch.randn((128, 8, 64), dtype=torch.float64, generator=cpu_generator(0, 50))
    arrays, report = finalize(vectors)
    matrix = torch.from_numpy(arrays["decoder_gram_raw"])
    sequential = sum(v.T @ v/8 for v in vectors)/128
    torch.testing.assert_close(matrix, sequential)
    assert report["raw"]["numerical_rank"] == 64 and report["raw"]["psd"]
    g, rotated = [torch.from_numpy(arrays[k]) for k in ["decoder_gram", "rotated_gram"]]
    torch.testing.assert_close(g.trace(), torch.tensor(64., dtype=torch.float64))
    torch.testing.assert_close(torch.linalg.eigvalsh(g), torch.linalg.eigvalsh(rotated))
    delta = torch.randn((20, 64), dtype=torch.float64, generator=cpu_generator(0, 51))
    for name, gram in [("decoder", g), ("rotated", rotated)]:
        factor = torch.from_numpy(arrays[name+"_factor"])
        cost = (delta @ factor).square().sum(-1)
        torch.testing.assert_close(cost, torch.einsum("bi,ij,bj->b", delta, gram, delta))
        assert (cost >= 0).all()
    # Rank deficiency must remain: QR adds no floor or implicit identity component.
    deficient = vectors.clone(); deficient[:, :, 4:] = 0
    _, low = finalize(deficient); assert low["raw"]["numerical_rank"] == low["normalized"]["numerical_rank"] == 4
    repeated, _ = finalize(vectors)
    assert all(torch.equal(torch.from_numpy(arrays[k]), torch.from_numpy(repeated[k])) for k in arrays)
    print(json.dumps(dict(passed=True, checks=["128x8 accumulation equals sequential outer mean",
        "common scalar trace normalization", "same-spectrum fixed rotation", "nonnegative QR-factor cost equals quadratic",
        "rank deficiency preserved without floor", "deterministic replay"]), indent=2))


if __name__ == "__main__": main()
