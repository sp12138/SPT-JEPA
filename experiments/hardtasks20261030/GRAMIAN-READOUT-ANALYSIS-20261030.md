# Can controllability normalize the latent planning cost?

2026-10-30. The motivating Cube audit reports strong association of AS-JEPA's true-endpoint latent goal cost with end-effector error, even after controlling object pose, and very weak remaining association with cube position. Predicted and true latent costs agree well. These observations motivate a frozen readout intervention; they do not establish a new representation-learning algorithm.

**Answer:** inverse controllability weighting is principled for local minimum-energy reachability. It can correct scale distortions and discount easy-to-change directions. It does not identify which goal-image factors matter to a task, and singular or inaccurate Jacobians can make it do exactly the wrong thing for contact-mediated object motion.

## Exact result and coordinate invariance

For linear time-varying dynamics, stack the control inputs into u and write the terminal state as

\[
 z_H=\Phi_H z_0+G_Hu.
\]

For input energy `u^T R u`, with R positive definite, define `W=G_H R^{-1} G_H^T` and `delta=z_goal-Phi_H*z0`. If delta lies in the reachable range of G, the minimum input energy is `delta^T W^dagger delta`; it is infinite outside that range. When W is full rank, the pseudoinverse becomes the usual inverse. The displacement must be relative to the unforced terminal state, not blindly relative to z0. This is classical finite-horizon minimum-energy control. [Boyd, EE263 Lecture 19](https://see.stanford.edu/Course/EE263/73), [Kia, controllability/reachability notes](https://solmaz.eng.uci.edu/Teaching/MAE270A/Lec/ControllabilityReachability_StateFeedback.pdf).

For an invertible linear latent change `z'=Tz`, the consistently transformed quantities satisfy `delta'=T delta` and `W'=T W T^T`. The full-rank cost is exactly invariant:

\[
 \delta'^T(W')^{-1}\delta'=\delta^TW^{-1}\delta.
\]

For rank-deficient W this invariance still holds for feasible displacements, because both formulations solve the same minimum-input-energy constraint. It does not generally hold for arbitrary infeasible displacements if one substitutes a pseudoinverse cost and silently drops the infeasible residual.

For a nonlinear model F that maps an action sequence to its terminal latent, linearize at a nominal sequence a0: `F(a0+du) ~= F(a0)+J du`. Then `W=J R^{-1} J^T` describes local corrective control energy around a0. It is not automatically the total energy of reaching the goal from the start. For nonlinear coordinate changes, `J'=D h(F(a0))*J`, but `h(z_goal)-h(F(a0))` need not equal that Jacobian times the finite displacement. Thus invariance is only first order locally. A globally invariant path metric would require integrating a varying geometry or solving a control problem, not simply applying one matrix to a long goal displacement.

## Why it could help our Cube cost

If robot-motion directions have high controllability eigenvalues and object-motion directions have smaller, reliably estimated nonzero eigenvalues, inverse weighting assigns less cost to a given robot discrepancy and more to the object discrepancy. A mere rescaling of latent coordinates cancels out under the full-rank transformation above. This directly addresses one plausible failure of Euclidean distance.

However, easy to move does not mean irrelevant, and difficult to move does not mean task relevant. The end effector is the goal in Reacher. Lighting may be difficult or impossible to control but irrelevant to Cube. A minimum-energy metric does not infer a task's permissible goal set from one full-scene goal image. It still tries to match every represented difference according to an effort criterion. Removing robot-goal constraints entirely requires identifying a task equivalence or goal set; controllability alone does not provide that identification.

## Rank deficiency is a substantive problem, not just numerical conditioning

Take `W=diag(1,0)` and `delta=(0,1)`. The second direction is unreachable in the local model, yet `delta^T W^dagger delta=0`. The pseudoinverse therefore makes a pure unreachable discrepancy free. In Cube, before a contact event, the object's local action derivative can vanish while the robot derivative is large. Unqualified pseudoinverse weighting could ignore the cube rather than prioritize it.

For an explicit coordinate counterexample outside the reachable range, take the same W, `delta=(1,1)`, and `T=[[1,1],[0,1]]`. W remains unchanged but delta becomes `(2,1)`, so the pseudoinverse cost changes from 1 to 4. Full linear coordinate invariance must not be claimed for this infeasible case.

Ridge weighting `(W+lambda I)^{-1}` penalizes null-space residuals, but introduces a new metric and scale. It is not invariant under a general invertible T if one inserts I again after transformation. A reference covariance C that transforms as `C'=T C T^T` permits `(W+lambda C)^{-1}` to remain linearly invariant when positive definite. This does not remove the need to define C and lambda independently of held-out success, nor make the resulting cost exact minimum energy.

## Contact, horizon, action bounds and model ignorance

1. **Contact modes:** a cube can be reachable by a finite action change even when a no-contact trajectory's first-order Jacobian has no cube direction. Local differential controllability is weaker than nonlinear reachability. Increasing horizon can help only if the nominal trajectory and derivatives expose the relevant contact transition; averaging Jacobians from several trajectories is a useful approximation but not the original local Gramian theorem.
2. **Action bounds:** finite energy does not imply feasibility under bounded actions, collision constraints or gripper limits. The unconstrained minimum-energy formula omits those constraints.
3. **Model ignorance:** a small model Jacobian can mean genuine weak controllability, missing contact dynamics, encoder insensitivity, or learned saturation. The Gramian alone cannot distinguish them. High predicted-versus-true *scalar cost* rank is insufficient, especially when that cost is dominated by the robot.
4. **Inverse amplification:** for a Gramian error E, inverse error obeys an identity involving `W^{-1} E (W+E)^{-1}`. Errors are most amplified along the smallest eigenvalues—the very directions the inverse proposes to emphasize. Damping limits amplification but trades it for a reference metric; it cannot create missing causal knowledge.
5. **Candidate-dependent weights:** recomputing a different inverse for every candidate changes both geometry and outcome during selection. A model can favor a candidate with a large predicted Jacobian because its residual appears cheap to correct. That is a different claim from ranking terminal goal proximity. A fixed per-start metric is the cleaner first diagnostic.

These failure modes mean the proposed quantity measures *modeled* local control energy. It can mistake ignorance for difficulty unless local action responses are independently checked.

## One frozen discriminating experiment

Before a new planning benchmark, use the existing held-out Cube starts and nonoracle candidate bank for one **model-versus-real local-response audit**:

- Freeze both temporal-forecast seeds and released LeWM. Preselect a small fixed subset of nonoracle reference trajectories without using physical success; include several trajectories per start so the experiment is not restricted to zero-action/no-contact derivatives.
- At each reference trajectory, apply the same small, normalized action-sequence perturbations in fixed directions. Compare model endpoint changes against the encoder applied to simulator endpoint changes. A second perturbation amplitude checks whether a nominally local result immediately breaks at contact; this is a derivative-validity check, not a tuned performance grid.
- Report agreement of response directions and magnitudes, effective reachable rank, the goal residual inside/outside the empirically supported response subspace, and whether directions that would receive large inverse weights actually change cube pose rather than only robot pose. Simulator pose is used only to interpret the directions, never to construct a deployable cost.
- Keep the action bank and candidate outcomes unchanged. If Jacobian agreement is strong precisely in the goal-residual directions and the inverse weights reduce robot dominance, a subsequent readout-only ranking test is justified. If high-weight directions have poor real-response agreement or are outside the reachable subspace, the hypothesis is falsified before a regularization choice can conceal the problem.

This experiment asks whether the proposed metric's physical premise holds. It deliberately avoids selecting a damping parameter from task success. A positive result would justify a separate planner-readout improvement, not a claim that world-model learning itself improved.

## Prior work and novelty boundary

Minimum-energy Gramians are classical control theory. Learned latent spaces designed for locally linear control also have a substantial history; **Predictive Coding for Locally-Linear Control** develops a decoder-free representation for that purpose. [Shu et al., ICML 2020](https://proceedings.mlr.press/v119/shu20a.html).

**Learning Koopman Representations with Controllability Guarantees** explicitly uses controllability-preserving latent dynamics and Gramian-based regularization. Its conditions and learned-dynamics construction differ from applying an inverse readout to our existing encoder, but Gramian-based latent control is already established prior art. [Miao et al., ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/e3781fb5d2c17ea7283cc2a543793bdb-Abstract-Conference.html).

**Decision-Metric Alignment in Latent World Models** already separates probe quality from action-ranking quality, measures latent/real ranks and analyzes encoder distortion, rollout error and candidate margins. Its improvement uses inverse-dynamics and demonstration-conditioned action objectives, which differ from our primary supervision setting. The diagnostic distinction itself must not be presented as our invention. [Wang et al., 2026](https://arxiv.org/abs/2608.18746).

The LeCun-coauthored **When Does LeJEPA Learn a World Model?** relates orthogonal identifiability to planning with consistently transformed, rotation-invariant costs, under specified latent distribution/transition assumptions. This supports taking coordinate geometry seriously, not assuming arbitrary invertibility is sufficient. Its planning demonstrations and assumptions do not settle contact-rich Cube control. [Klindt, LeCun and Balestriero, 2026](https://arxiv.org/html/2605.26379v1).
