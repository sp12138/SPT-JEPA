"""CPU-only exact witnesses; no data, model calls, fitting, or benchmark claim."""

from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path


def flow(state, action, heisenberg=True):
    x, y, z = state
    a, b = action
    return x + a, y + b, z + ((x * b - y * a) / 2 if heisenberg else 0)


def loop(epsilon, heisenberg=True):
    state = (F(0), F(0), F(0))
    for action in ((epsilon, 0), (0, epsilon), (-epsilon, 0), (0, -epsilon)):
        state = flow(state, action, heisenberg)
    return state


def main():
    origin = (F(0), F(0), F(0))
    actions = [(F(a, 4), F(b, 4)) for a in range(-4, 5) for b in range(-4, 5)]
    assert all(flow(origin, a, True) == flow(origin, a, False) for a in actions)
    loops = []
    for k in range(1, 9):
        e = F(1, 2**k)
        h, c = loop(e), loop(e, False)
        assert h == (F(0), F(0), e**2)
        assert c == origin
        assert loop(2 * e)[2] == 4 * h[2]
        loops.append({"epsilon": str(e), "vertical_displacement": str(h[2]),
                      "square_control_length": str(4 * e),
                      "optimal_control_length": 2 * math.sqrt(math.pi) * float(e)})

    horizontal, vertical = 0.04, 0.001
    cc_horizontal, cc_vertical = horizontal, 2 * math.sqrt(math.pi * vertical)
    assert vertical < horizontal and cc_horizontal < cc_vertical

    # For f=z, Xf=-y/2 and Yf=x/2: zero at the origin, not on a neighborhood.
    # The exact coefficient calculation gives X(Yf)-Y(Xf)=1 everywhere.
    bracket_on_z = F(1, 2) - F(-1, 2)
    assert bracket_on_z == 1
    result = {
        "passed": True,
        "scope": "stdlib CPU exact algebra plus analytic-distance evaluation; no learned model/data",
        "constant_actions_identical_at_origin": len(actions),
        "square_loop_checks": loops,
        "ranking_witness": {
            "A": [horizontal, 0, 0], "B": [0, 0, vertical],
            "euclidean_A_B": [horizontal, vertical],
            "control_length_A_B": [cc_horizontal, cc_vertical],
            "euclidean_prefers": "B", "control_length_prefers": "A"},
        "pointwise_invariance_is_insufficient": {
            "Xf_at_origin": 0, "Yf_at_origin": 0, "bracket_f": int(bracket_on_z)},
        "limitations": ["Global invariance theorem and isoperimetric optimality are proved in the note, not inferred from finite examples.",
                        "No evidence that Cube satisfies smooth reversible bracket-generating assumptions."],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output = Path(__file__).with_name("lie-geometry-witness.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": True, "output": str(output), "exact_loop_checks": len(loops),
                      "constant_action_checks": len(actions)}))


if __name__ == "__main__":
    main()
