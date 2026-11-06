# What the successful action target may actually supply

The previous 78.75% result used an action prediction target. It did not establish that image reconstruction is necessary, or that action labels are the only route to its useful information. The current strict experiment tests a different explanation: a distant goal needs to specify a *local transition*, while a forward model needs to specify how controls can realize that transition.

## A population equivalence, with explicit assumptions

Let x denote a sufficient observation history, g a future observation, z' the next observed feature, and u the already-executed current-transition control. Assume

1. The conditional mean of the next feature is control-affine: E[z' | x,u] = z(x) + b(x) + G(x)u.
2. After conditioning on x,u, the residual has zero conditional mean even when g is specified. This is stronger than an ordinary Markov assumption: conditioning on a future endpoint can select transition noise.
3. G has full column rank, and the relevant behavior controls lie in the allowed box.

Then the law of total expectation gives

    E[z' | x,g] - z(x) - b(x) = G(x) E[u | x,g].

Both the left-hand expectation and the forward dynamics can be learned using **only next-observation prediction targets**. Solving the four-dimensional forward inverse gives

    argmin_u ||b + Gu - (E[z' | x,g] - z)||_W^2 = E[u | x,g]

for every positive definite W. Thus the population conditional-mean controller learned by action MSE has an observation-target counterpart. This is a statement about sufficient statistics and identifiable dynamics, not a claim that the finite networks are equivalent or that the action is optimal. The controls in this equation are factual executed controls, never future action targets.

This suggests a concrete abstraction: **query one next-observation variable with two information sets, and reconcile the resulting predictions**. The goal query supplies local progress; the executed-control query supplies physical realizability. A terminal appearance distance conflates these roles and need not represent either one correctly.

## Why this can fail

Partial observation, contact discontinuities, stochastic endpoint selection, or behavior-dependent hidden state break the equivalence. An expert corpus may cover only one control near each state. For any on-support model F, F(x,u) + H(x,u)(u - pi(x)) agrees on expert data u=pi(x) while implying arbitrary off-support control effects. Low factual prediction error cannot distinguish them. Even deterministic dynamics do not fix inadequate action coverage.

Gaussian prediction can describe ambiguity but cannot manufacture information. A Gaussian boundary mean can average incompatible routes. Diagonal uncertainty also depends on the feature coordinates; it is not a coordinate-invariant geometry. Fixed random features avoid target collapse and pretrained priors, but may still represent the relevant object poorly. The experiment must separate these failure modes rather than equating small loss with understanding.

## The present controller

The shared network predicts next fixed-random observation features in two modes. The control mode reads only the current transition's known executed command. The boundary mode reads the current history and a later goal image, with all controls zeroed. Its target is strictly between the two observed endpoints, so it cannot copy a provided target frame.

For affine Gaussian predictions, their overlap induces the projection weight (Sigma_forward + Sigma_boundary)^-1. The controller enumerates all 81 active faces of a four-command box and solves the resulting small quadratic programs. There is no learned action head, action reconstruction, action retrieval, old controller distillation, or pixel decoder. The nonlinear control arm tests whether the affine restriction, rather than the information factorization, is responsible for an outcome.

## Evidence needed

- Full-state restore, positive held-out gradients, frozen-coordinate equality, sample independence, and actual physics are technical gates.
- Boundary versus forward-only control tests the value of learned local progress under the same observation coordinates.
- Point versus Gaussian targets tests whether predictive ambiguity improves actual control.
- Affine versus nonlinear dynamics tests the analytic inverse restriction, with different inference compute disclosed.
- Factual prediction versus copy, sensitivity to current executed controls, the projection residual, and physical response to self-chosen commands diagnose failures. Model-scored alternatives alone are not physical counterfactual evidence.
- A performance claim requires common physical cases, independent training seeds, and complete success traces. The primary reference is the old 78.75% six-manipulation result, with its different model, target and budget disclosed. A few screening cases cannot establish a substantial improvement.

## Prior art boundary

Goal-conditioned observation prediction and learned visual servoing are established. GCP generated visual plans and executed them using action-supervised inverse models; this experiment instead inverts an observation-supervised forward model. This distinction meets the current supervision constraint, but alone does not establish a publishable novelty claim.

Primary sources: [GCP project and paper](https://orybkin.github.io/video-gcp/), [Learning Visual Servoing with Deep Features and Fitted Q-Iteration](https://arxiv.org/abs/1703.11000), [Analyzing Neural Jacobian Methods](https://arxiv.org/abs/2106.06083).
