"""Toy check of the per-window value condition (A5) in as-jepa-story.md, Section 5.6.

One dynamics component x_t = x_0 + D_t moves deterministically under known actions.
Var(x_0) = v (spread across the data); D_t has variance r^2 * t when the component moves
in the window, 0 otherwise (it moves in a fraction p of windows). The code is
s_t = A x_t + sigma0 * xi_t against a unit-variance first-state prior. From Proposition 1
at beta = 1, the objective change relative to not encoding the component is

    dJ(A) = A^2 v / 2 - sum_t 0.5 * log[(A^2 Var(anchor-to-frame displacement) + sigma0^2)
                                       / (sigma0^2 + P_t)],
    P_t = A^2 v sigma0^2 / (sigma0^2 + t A^2 v)   (posterior variance of A x_0 from t noisy states).

dJ < 0 at the best A means encoding the component beats dropping it. A distant anchor adds an
independent displacement with variance v_far between the anchor and the window.
Run: python3 a5_value_condition.py
"""
import math


def dJ(A, *, r, s0, T, p=1.0, v=1.0, v_far=0.0):
    total = A * A * v / 2
    for t in range(1, T + 1):
        post = A * A * v * s0 * s0 / (s0 * s0 + t * A * A * v)
        moving = 0.5 * math.log((A * A * (v_far + r * r * t) + s0 * s0) / (s0 * s0 + post))
        still = 0.5 * math.log((A * A * v_far + s0 * s0) / (s0 * s0 + post))
        total -= p * moving + (1 - p) * still
    return total


def best(**kw):
    grid = [10 ** (k / 200) for k in range(-800, 401)]
    A = min(grid, key=lambda a: dJ(a, **kw))
    return A, dJ(A, **kw)


def kappa(T):
    """High-precision threshold factor: keep if geometric-mean sd(D_t)/sqrt(v) > kappa(T) * sigma0."""
    return math.exp((1 - math.log(T)) / 2 + math.log(T + 1) / (2 * T))


if __name__ == "__main__":
    print("Always-moving component; r = per-frame displacement sd / spread. dJ = 0.00 means dropped.")
    for s0 in (0.05, 0.1, 0.3):
        for T in (3, 7, 15):
            cells = [f"r={r:<5} dJ={best(r=r, s0=s0, T=T)[1]:7.2f}" for r in (0.01, 0.02, 0.05, 0.1, 0.2)]
            print(f"sigma0={s0:<4} T={T:<2} kappa={kappa(T):.2f} | " + " | ".join(cells))
    print("\nCollapse barrier (sigma0=0.1, T=7, r=0.2): dJ along the amplitude A")
    print("  " + "  ".join(f"A={A}: {dJ(A, r=0.2, s0=0.1, T=7):.3f}" for A in (0.01, 0.1, 0.5, 1.0, 3.0)))
    print("\nComponent moving in a fraction p of windows (sigma0=0.1, T=7, r=0.1)")
    for v_far, label in ((0.0, "first-frame anchor"), (0.25, "distant anchor v_far=0.25")):
        cells = [f"p={p:<4} dJ={best(r=0.1, s0=0.1, T=7, p=p, v_far=v_far)[1]:7.2f}" for p in (0.05, 0.2, 0.4, 1.0)]
        print(f"{label:<26} | " + " | ".join(cells))
