"""Pure-Python exact/finite-difference checks for the relative-response review.

No training, external data, NumPy, or accelerator. These are mathematical
counterexamples, not evidence of benchmark improvement.
"""
import json
import math
from pathlib import Path


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def derivative(fn, value, step=1e-6):
    return (fn(value + step) - fn(value - step)) / (2 * step)


def main():
    delta = 1 / 127.5
    floor = delta ** 2
    # P = vv^T removes both constants and linear time; rank=1, D=1.
    v = [x / math.sqrt(6) for x in [1, -2, 1]]
    time = [0, 1, 2]
    target = [0, 0, 1]
    feature = [0, 1, 4]
    target_coordinate = dot(v, target)
    energy = target_coordinate ** 2
    denominator = energy + floor
    assert abs(sum(v)) < 1e-14 and abs(dot(v, time)) < 1e-14

    def forecast_loss(z=0.3, head=0.7, output_weight=0., output_bias=0., relative=True):
        prediction = [head * (z + output_weight * h + output_bias * t)
                      for h, t in zip(feature, time)]
        raw = dot(v, [p-y for p, y in zip(prediction, target)]) ** 2
        return raw / denominator if relative else raw

    derivatives = {
        key: derivative(lambda value, key=key: forecast_loss(**{key: value}), value)
        for key, value in dict(z=.3, head=.7, output_weight=0., output_bias=0.).items()
    }
    for key in ["z", "head", "output_bias"]:
        assert abs(derivatives[key]) < 1e-8
    assert abs(derivatives["output_weight"]) > .1
    absolute_weight_grad = derivative(lambda x: forecast_loss(output_weight=x, relative=False), 0.)
    assert math.isclose(derivatives["output_weight"], absolute_weight_grad / denominator, rel_tol=1e-9)

    # Equal encoded values at theta=0 do not imply equal frame Jacobians:
    # E_theta(x_t)=z+theta*x_t; dE/dtheta=x_t. Forecast repeats initial E.
    def posterior_loss(theta, head=.7, input_feature=target):
        prediction = [head * (.3 + theta * x) for x in input_feature]
        return dot(v, [p-y for p, y in zip(prediction, target)]) ** 2 / denominator

    posterior_grad = derivative(posterior_loss, 0.)
    assert abs(posterior_grad) > .1
    zero_head_grad = derivative(lambda theta: posterior_loss(theta, head=0.), 0.)
    affine_feature_grad = derivative(lambda theta: posterior_loss(theta, input_feature=time), 0.)
    assert abs(zero_head_grad) < 1e-8 and abs(affine_feature_grad) < 1e-8

    # With denominator depending on the target, Bayes regression changes.
    ys = [1., 2.]
    weights = [1 / (y*y + floor) for y in ys]
    weighted_optimum = dot(weights, ys) / sum(weights)
    def weighted_risk(prediction):
        return sum(w*(prediction-y)**2 for w, y in zip(weights, ys)) / 2
    assert abs(weighted_optimum - 1.2) < 2e-5
    assert abs(derivative(weighted_risk, weighted_optimum)) < 1e-8
    assert weighted_risk(weighted_optimum) < weighted_risk(1.5)

    # Mixing losses retains the disagreement term; mixing predictions omits it.
    p, f = 0.2, -0.7
    half_losses = ((p-target_coordinate)**2 + (f-target_coordinate)**2) / (2*denominator)
    average_loss = ((p+f)/2-target_coordinate)**2 / denominator
    disagreement = (p-f)**2 / (4*denominator)
    assert math.isclose(half_losses, average_loss + disagreement, abs_tol=1e-14)

    # At zero forecast the norm of the output gradient is largest at energy=floor.
    def gradient_norm(e):
        return 2*math.sqrt(e)/(e+floor)
    assert gradient_norm(floor) > gradient_norm(floor/100)
    assert gradient_norm(floor) > gradient_norm(floor*100)
    examples = {
        task: dict(target_energy=e, relative_zero_prediction=e/(e+floor),
                   gradient_scalar=1/(e+floor))
        for task, e in {"tworoom_mean_energy_example": .000880527,
                        "cube_mean_energy_example": .003409041}.items()
    }
    result = dict(
        scope="Exact small-model counterexamples and finite differences only; no benchmark claim.",
        lsb_squared=floor, forecast_init_derivatives=derivatives,
        posterior_same_output_different_jacobian_gradient=posterior_grad,
        posterior_zero_head_counterexample_gradient=zero_head_grad,
        posterior_affine_jacobian_counterexample_gradient=affine_feature_grad,
        weighted_bayes_optimum=weighted_optimum, unweighted_bayes_optimum=1.5,
        mixture_loss_identity_error=half_losses-average_loss-disagreement,
        isotropic_quantization_model_after_64_pixel_average=floor/(12*64),
        floor_over_that_model_variance=12*64, examples=examples,
        assertions="all passed")
    output = Path(__file__).with_name("ideas10-relative-response-checks-20261031.json")
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
