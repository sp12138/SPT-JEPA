"""Check the box optimizer against an independent constrained LS solver."""
import argparse
import json

import numpy as np
from scipy.optimize import lsq_linear
import torch

from as_jepa_lewm.predictive_kernel import PredictiveKernel


def run(output):
    torch.set_num_threads(2)
    torch.manual_seed(903)
    model = PredictiveKernel(blocks=1, width=16)
    errors = []
    for scale in [0.1, 1., 20.]:
        effect = torch.randn(17, 4, 2, 2, 2)
        drift = torch.randn(17, 2, 2, 2)
        target = drift + torch.randn_like(drift) * scale
        variance = torch.rand_like(drift) + .1
        command = model.box_projection(drift, effect, target, variance)
        for k in range(len(command)):
            weight = np.sqrt(variance[k].flatten().double().numpy())
            matrix = effect[k].flatten(1).T.double().numpy() / weight[:, None]
            rhs = (target[k] - drift[k]).flatten().double().numpy() / weight
            reference = lsq_linear(matrix, rhs, bounds=(-1, 1), tol=1e-12)
            value = np.square(matrix @ command[k].double().numpy() - rhs).sum()
            expected = np.square(matrix @ reference.x - rhs).sum()
            errors.append(float(abs(value - expected)))
    report = dict(passed=bool(max(errors) < 1e-4), problems=len(errors),
                  max_objective_difference=float(max(errors)), model_updates=0,
                  reference='scipy.optimize.lsq_linear, independent bounded LS solver')
    assert report['passed'], report
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    from pathlib import Path
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args().output)
