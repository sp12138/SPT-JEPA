"""Exact finite witnesses only: no learned model, data access, environment, or GPU."""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path


def vector_add(x, y):
    return tuple(a + b for a, b in zip(x, y))


def matvec(b, a):
    return tuple(sum(x * y for x, y in zip(row, a)) for row in b)


def squared(x):
    return sum(v * v for v in x)


def main():
    # A valid joint distribution, with actions independent of initial state.
    ps = [F(1, 20), F(19, 20)]
    p_goal = [[F(4, 5), F(1, 5)], [F(86, 95), F(4, 95)]]
    py_a = [sum(ps[s] * p_goal[s][a] for s in range(2)) for a in range(2)]
    reverse = [ps[0] * p_goal[0][a] / py_a[a] for a in range(2)]
    ratio = [reverse[a] / ps[0] for a in range(2)]
    corrected = [ratio[a] * py_a[a] for a in range(2)]
    assert corrected == p_goal[0]
    assert ratio[1] > ratio[0] and corrected[0] > corrected[1]
    joint_identity_checks = 0
    py = sum(py_a) / 2
    for s in range(2):
        for a in range(2):
            joint_reverse = ps[s] * F(1, 2) * p_goal[s][a] / py
            assert joint_reverse / (ps[s] / 2) == p_goal[s][a] / py
            joint_identity_checks += 1

    # A distribution is genuinely useful relative to its mean, but this is
    # ordinary chance planning, not a new learning mechanism.
    outcomes = [[(F(-1), F(1, 2)), (F(1), F(1, 2))], [(F(1, 20), F(1))]]
    mean_cost = [sum(y * p for y, p in law) ** 2 for law in outcomes]
    success = [sum(p for y, p in law if abs(y) <= F(1, 10)) for law in outcomes]
    assert mean_cost[0] < mean_cost[1] and success == [0, 1]

    # Logged future feedback actions reveal random outcomes; conditioning on
    # that entire future action sequence is not an intervention.
    feedback_rows = [(w, w, w, F(1, 2)) for w in [0, 1]]
    obs_goal = [sum(p for w, a, y, p in feedback_rows if a == command and y == 1)
                / sum(p for w, a, y, p in feedback_rows if a == command)
                for command in [0, 1]]
    do_goal = [sum(p for w, a, y, p in feedback_rows if y == 1) for command in [0, 1]]
    assert obs_goal == [0, 1] and do_goal == [F(1, 2), F(1, 2)]

    # Same logged transitions, different effects in an unexcited direction.
    b1 = ((F(1), F(0)), (F(0), F(1)))
    b2 = ((F(1), F(2)), (F(0), F(-1)))
    s = (F(1, 3), F(-2, 5))
    support = [(F(i, 10), F(0)) for i in range(-10, 11)]
    assert all(vector_add(s, matvec(b1, a)) == vector_add(s, matvec(b2, a)) for a in support)
    delta = F(1, 5)
    contrast = [matvec(b, (0, 2 * delta)) for b in [b1, b2]]
    assert contrast[0] != contrast[1]
    # One pair cannot identify the full derivative / between-endpoint law.
    perturb = lambda a: a * (a * a - delta * delta)
    assert all(perturb(a) == 0 for a in [-delta, 0, delta])
    assert perturb(delta / 2) != 0
    # Difference supervision reweights existing pair errors, exactly.
    checks = 0
    for i in range(-4, 5):
        for j in range(-4, 5):
            e1, e2 = F(i, 7), F(j, 9)
            assert (e1 - e2) ** 2 == 2 * (e1 * e1 + e2 * e2) - (e1 + e2) ** 2
            checks += 1

    payload = dict(
        passed=True, execution='stdlib CPU exact rational arithmetic; no training or data',
        reverse_candidate=dict(
            success_probability=list(map(float, p_goal[0])),
            action_conditional_goal_frequency=list(map(float, py_a)),
            reverse_ratio=list(map(float, ratio)),
            corrected_probability=list(map(float, corrected)),
            wrong_selected_action=1, correct_selected_action=0,
            joint_reverse_identity_checks=joint_identity_checks),
        distribution_vs_mean=dict(point_cost=list(map(float, mean_cost)), goal_mass=list(map(float, success))),
        future_feedback=dict(observational_goal_probability=list(map(float, obs_goal)),
                             intervened_goal_probability=list(map(float, do_goal))),
        controlled_pair=dict(same_logged_transitions=len(support),
                             identified_direction_response=[[float(v) for v in c] for c in contrast],
                             remaining_midpoint_ambiguity=float(perturb(delta / 2)),
                             secant_loss_identities=checks,
                             interpretation='A real branch adds directional support; differencing alone adds no new observations.'),
        admission='No new complete distribution/reverse-cost mechanism admitted. Controlled forks merit a same-data point-loss control; no GPU gate authorized.',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    dest = Path(__file__).with_name('distribution-boundary-results.json')
    dest.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
