# What the successful action target may actually supply

## Updated mechanism test, November 7 Beijing time

The objective is still unmet. The old action-supervised score is only a scalar
comparison; no inverse learner,its weights,outputs,or traces enter this study.
The following separates mathematical statements from empirical hypotheses.

Let a known executed control word `u` produce an observed image path
`Y_H = F_H(h,u)`,where `h` includes information inferred from observation and
own-control history. Near a reference word,an image prediction error induces
the control-space quadratic form

`delta_u^T G_H delta_u`,with
`G_H = sum_k J_k^T Sigma_k^{-1} J_k`,

where `J_k` is the derivative of the k-th future observation with respect to
the control word. This is a local statement,not a claim that image MSE is an
action loss. If `G_H` has a small eigenvalue,excellent image prediction gives
a weak bound on a control error in that direction. An action target instead
measures errors directly in the logged actuator coordinates. It also trains
a goal-conditioned controller,so its gain cannot be attributed solely to
representation quality. Logged demonstrations need not be optimal actions.

A preparation action can principally change *which later transitions are
possible*. It need not change the immediate image substantially. Increasing
the horizon can reveal that effect,but is not sufficient in general: for a
bilinear delayed effect `y_next = y + hidden_force * later_motion`,the local
derivative at the all-zero word can remain zero. Ordered,finite interventions
can distinguish these systems even when a zero-control linearization cannot.
This motivates latent memory,compositional observation likelihood,and symmetric
experiments over all control axes. It does not motivate a hand-coded gripper
rule,an action decoder,or pseudo-action labels.

There is a separate identification problem. If the demonstrations satisfy
`u = pi(h)`,then every model

`F_tilde(h,u) = F(h,u) + A(h,u) * (u - pi(h))`

has the same factual predictions on that support. A state-only predictor may
therefore fit well while exposing no useful control dependence. This does not
prove imitation through a forward model impossible: the demonstrated control
can remain a valid solution. It shows why fitting the observations alone need
not make the planning solution unique or make its derivatives physically right.
Same-context interventions constrain this otherwise unconstrained family.

The present evidence is narrower than this interpretation:

- Native image forecasting improved substantially over copying,yet the best
  native20k arm achieved only32/480=6.67% manipulation success.
- A CPU physical diagnostic of eligible own-forward trajectories found42/576
  signed control-order pairs differing by more than1cm at nearest-TCP contexts,
  versus0/576 at initial contexts. These are repeated pairs within shared
  physical contexts,not576 independent environments. Some failures occur before
  contact;some include contact/lift without completion. No unique failure cause
  follows from these observations.
- A matched original-data recurrent experiment,with the same five future RGB
  targets and paired initial weights,scored9/48=18.75% on six manipulation tasks
  at2500 updates when belief was composed. Resetting it on each observed
  transition scored0/48. The gain was door4/8 and drawer5/8;the other four tasks
  remained0. This is a screening result on four shared physical cases/task and
  two trained parents,not confirmation or an all-task solution.
- Frozen original-data models are now tested on exactly50 common,previously
  unseen own-intervention contexts. This audit separately measures factual
  forward error,odd control-response agreement,and whether a correct achieved
  RGB goal predicts better than a different branch goal from the same context.
  No audit optimizes a learner or predicts an action label.

The new corpus contains11600 own-command branches from400 contexts across50
tasks. All commands were self-generated experimental inputs;no expert command
was read/replayed,and no reward/state/success/task-ID tensor is a learner input.
Simulator state is used only to restore a branch and for separate diagnostics.
All branches of each exploration parent share one train/validation split and
benchmark physical cases are excluded. It is a separate extra-data track.
Original-data experiments never silently consume it. A comparison using it must
give the same corpus to its baselines and count collection/full pretraining
cost. Neither the data nor these results are equivalent to the gated official
JEPA-WMs public setting.

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
