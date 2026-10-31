"""Primitive-vs-chunk witnesses: exact CPU arithmetic, no model training."""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import hashlib
import json


def primitive_linear(x, actions, alpha=F(1), beta=F(1)):
    for a in actions:
        x = alpha * x + beta * a
    return x


def step(state, action, noise, world):
    phase, x = state
    if phase == 0:
        return (1, noise)
    if phase < 4:
        return (phase + 1, x)
    return (5, x if world == 'retain' else action)


def trajectory(actions, noise, world):
    states = [(0, 0)]
    for a in actions:
        states.append(step(states[-1], a, noise, world))
    return states


def main():
    # The three endpoint records identify alpha^5=1 and
    # beta*(1+alpha+...+alpha^4)=5, so real alpha=beta=1 uniquely.
    train = [(0, 0), (1, 0), (0, 1)]
    for x, u in train:
        y = primitive_linear(F(x), (F(u),) * 5)
        assert y == x + 5 * u  # compatible macro model x+5*a0 also fits
    tests = list(product([-1, 0, 1], repeat=5))
    micro_errors = [primitive_linear(0, a) - sum(a) for a in tests]
    macro_errors = [5 * a[0] - sum(a) for a in tests]
    assert all(e == 0 for e in micro_errors)
    macro_mse = sum(F(e * e) for e in macro_errors) / len(tests)
    assert macro_mse == F(40, 3)

    # Shared Markov mechanisms; every intermediate state is observable.
    # Feedback chooses a4=x4, leaving the discordant state/action cells unseen.
    logged = []
    for w in [0, 1]:
        actions = [0, 0, 0, 0, w]
        p = trajectory(actions, w, 'retain')
        q = trajectory(actions, w, 'overwrite')
        assert p == q
        logged.append(dict(actions=actions, states=p))
    interventions = []
    for a4 in [0, 1]:
        actions = [0, 0, 0, 0, a4]
        laws = {world: sum(F(trajectory(actions, w, world)[-1][1], 2)
                           for w in [0, 1]) for world in ['retain', 'overwrite']}
        interventions.append({k: float(v) for k, v in laws.items()})
    assert interventions == [{'retain': .5, 'overwrite': 0.}, {'retain': .5, 'overwrite': 1.}]

    # A deterministic prefix-causal, shared primitive update z'=a has zero
    # macro endpoint MSE on the logged data, despite wrong hidden microstates.
    for row in logged:
        z = 0
        for a in row['actions']:
            z = a
        assert z == row['states'][-1][1]
    prefix_checks = 0
    for a, b in product(list(product([0, 1], repeat=5)), repeat=2):
        for j in range(1, 6):
            if a[:j] == b[:j]:
                assert a[j - 1] == b[j - 1]
                prefix_checks += 1

    # Even an accurate one-step conditional mean need not compose correctly.
    mean_then_square = ((F(-1) + F(1)) / 2) ** 2
    square_then_mean = (F(-1) ** 2 + F(1) ** 2) / 2
    assert mean_then_square == 0 and square_then_mean == 1
    result = dict(passed=True, no_training=True, no_data=True, no_GPU=True,
                  positive=dict(assumption='Correct scalar shared linear primitive class; real coefficients; deterministic',
                                endpoint_records=3, identified_alpha=1, identified_beta=1,
                                off_support_sequences=len(tests), micro_mse=0, compatible_macro_mse=float(macro_mse)),
                  failure=dict(identical_complete_logged_trajectories=logged,
                               intervened_goal_probabilities=interventions,
                               prefix_causal_endpoint_shortcut_mse=0,
                               exact_prefix_checks=prefix_checks,
                               statement='Primitive sharing and even intermediate observations do not repair missing conditional action support.'),
                  mean_composition=dict(mean_then_nonlinearity=float(mean_then_square), true_mean=float(square_then_mean)),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    Path(__file__).with_name('results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'failure'}, indent=2))


if __name__ == '__main__':
    main()
