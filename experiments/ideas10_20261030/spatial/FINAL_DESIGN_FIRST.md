# Final design pass, before nearest-work search

2026-10-31. Two proposed mechanisms are being examined, not admitted as innovations. The latest 24-run gate separates observed-state learning from causal prediction: Cube's action-null mixed route improves observed projected reconstruction and block-height readout, while causal forecast remains essentially the zero-response baseline and native P0 stays 2/2%. Operator variants are not additional ideas.

## Attempt A: preserve the grounded current state through each transition

Replace an unconstrained overwrite predictor by a residual transition

    F_a(z) = z + g_theta(z,a),       Lip_z(g_theta) <= c < 1.

Retain the existing sample-level world-model objectives and native planner. No new observation/history, expert label or control policy is introduced. The intended difference is structural: an untrained action transition cannot map every grounded initial code to the same state. A small norm-bounded residual MLP is the minimal implementation; this is a transition constraint, distinct from the existing proposal to constrain the encoder's carrier chart.

For every fixed action a,

    ||F_a(z)-F_a(z')|| >= (1-c)||z-z'||.

The proof is the reverse triangle inequality. After H transitions the lower bound becomes (1-c)^H times the initial distance. This is a true finite-horizon initial-information bound, but it can become weak at large H and does not guarantee physical accuracy or controllability.

An immediate falsification matters more than the attractive bound. Suppose the true observed transition clamps x in [0,1] to the same wall point, and the encoded next observation is the same vector z_g. Any m-co-Lipschitz predictor F has

    E ||F(X)-z_g||^2 >= m^2 Var(X) = m^2/12,  X uniform[0,1].

To prove this, draw X' independently, use Var(F(X)) = .5 E||F(X)-F(X')||², then the lower distance bound. Mean squared error about any fixed target is at least that variance. A deterministic contact transition can legitimately erase information about initial distance. Preserving it in hidden memory is possible only by changing the state/goal readout contract; if the native goal metric and target encoder read all dimensions, that escape is unavailable. A contact gate would add another learned mechanism and overlap known contact models, not automatically rescue this candidate.

The CPU check will verify the lower bound and a concrete wall example. If this reduces to established invertible residual networks and an inappropriate contact prior, it will be rejected rather than renamed an information-preserving world model.

## Attempt B: ground the clean rollout with a continuous path-area target

Let y_t be a fixed low-dimensional projection of addressed RGB features. Decode the same y coordinates from the clean causal latent rollout. For increments d_t=y_{t+1}-y_t, use the endpoint displacement and signed area

    r = sum_t d_t,
    A = .5 sum_{i<j}(d_i d_j^T - d_j d_i^T).

Add a per-clip loss ||r_hat-r||²+||A_hat-A||², with a matched pointwise feature-prediction control and the original objective retained. For a first specification, a fixed projection of 192 pooled RGB values to 8 coordinates gives 8 displacement and 28 independent area coordinates; it adds no semantic detector, optimal-action target, branch collection or query bank. Extra head/training computation must be counted. It is not S1's action-affine linear annihilator: the area is quadratic in the observation path and contains no action-conditioned projection. Its closeness to S3's compositional path target is nevertheless a serious admission issue.

The exact concatenation identity is

    r(XY)=r(X)+r(Y),
    A(XY)=A(X)+A(Y)+.5(r(X)r(Y)^T-r(Y)r(X)^T).

Two L-shaped paths can have identical endpoints but opposite signed areas. A loop around an obstacle can have a nonzero area while its endpoint displacement vanishes. This is classical path-signature algebra, not a new theorem.

The key failure check is optimization at carry initialization. If every predicted frame is identical, every predicted increment is zero. The area map is quadratic in the increments, so its full Jacobian is zero there. On a target closed loop, the displacement target is also zero. Therefore the proposed displacement-plus-area loss has positive loss but exactly zero gradient to **all predicted frames**, not merely to their common initial code. A framewise objective has a nonzero gradient in the same construction. Adding the shared initial anchor does not teach the missing loop from this stationary point.

The CPU check will verify concatenation and this stationary failure. It will then be compared with primary conditional-signature generative modeling and controlled differential-equation work. If it is established signature matching with the same carry failure, it will not count as another new method.
