"""Exact rejection checks for the final correction-transfer design pass.

Pure Python; no training, NumPy, remote data, or GPU. The attempted mechanism
is not admitted as a new idea after these checks and the prior-art audit.
"""
import json
import math
from pathlib import Path


def main():
    def model(z, a):
        return .7*z + .2*z*z + .3*a

    gaps = []
    for predicted in [-1., -.1, .4]:
        for observed in [-.7, .2, 1.1]:
            for action in [-.5, .5]:
                future = .4*observed + .15*observed**3 + action
                baseline = model(predicted, action)
                correction_target = future-baseline
                correction_prediction = model(predicted+(observed-predicted), action)-baseline
                correction_loss = (correction_prediction-correction_target)**2
                teacher_loss = (model(observed, action)-future)**2
                gaps.append(abs(correction_loss-teacher_loss))
    assert max(gaps) < 1e-14

    # Exact current-state correction interpolation imposes chord linearity.
    # For a perfect ReLU transition and an interval crossing contact, it gives
    # a false nonzero loss. Difference order is epsilon, squared error epsilon².
    rows = []
    for epsilon in [1., .1, .01, .001]:
        predicted, observed = -epsilon, epsilon
        lam = .5
        relu = lambda z: max(0., z)
        interpolated = (1-lam)*predicted+lam*observed
        true_value = relu(interpolated)
        invented_target = (1-lam)*relu(predicted)+lam*relu(observed)
        bias = true_value-invented_target
        assert math.isclose(bias, -.5*epsilon)
        # Replacing the physical transition by its secant makes this auxiliary
        # zero at the interpolation, despite being physically wrong there.
        chord_value = .5*(interpolated+epsilon)
        assert math.isclose(chord_value, invented_target)
        rows.append(dict(epsilon=epsilon, perfect_model_bias=bias,
                         perfect_model_auxiliary_loss=bias*bias,
                         wrong_chord_auxiliary_loss=(chord_value-invented_target)**2,
                         wrong_chord_physical_error=(chord_value-true_value)**2))

    # A smooth transition's chord error is second order, as expected.
    smooth = []
    for epsilon in [1., .1, .01, .001]:
        left, right = -epsilon, epsilon
        error = 0.**2-(left*left+right*right)/2
        assert math.isclose(error, -epsilon*epsilon)
        smooth.append(dict(epsilon=epsilon, perfect_model_bias=error))

    result = dict(
        admission="No new mechanism admitted from this pass.",
        scope="Exact identities/counterexamples only; no benchmark claim.",
        full_correction_teacher_forcing_loss_identity_max_gap=max(gaps),
        contact_interpolation_counterexamples=rows,
        smooth_interpolation_second_order=smooth,
        assertions="all passed")
    output=Path(__file__).with_name("ideas10-correction-transfer-checks-20261031.json")
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__ == "__main__":
    main()
