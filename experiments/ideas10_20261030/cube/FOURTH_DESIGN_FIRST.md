# Focused final pass: synchronous driven-memory geometry

2026-10-31. Written before the targeted prior search. One candidate, not a quota fill. No performance or novelty claim.

## D9 — compare what initial-state differences survive the same future commands

The specific failure is not a lack of predictability: TF predicts its own Cube distance well, but that distance strongly follows actuator configuration. A new intrinsic bias could distinguish an immediately overwritten actuator state from a persistent configuration without declaring an object class. The test is not how late an action first affects a coordinate (D7), nor whether one can reversibly travel between two states (D4). It is whether two states driven by the *same* future commands preserve their difference.

For a fixed common open-loop action sequence A, define the diagonal action operator

    D_A(s,t) = (F_A(s), F_A(t)).

For observation feature psi and H-step action distribution nu, use

    d_H²(s,t) = E_A ||psi(F_A(s)) - psi(F_A(t))||².

The proposal is to distill this synchronized endpoint distance into the existing state metric through a paired, sample-level objective, using genuinely executed paired futures, not model-generated targets. Fit d_theta(s,t)² to an empirical fixed-A target average; train the same action-conditioned world model jointly. No optimal-action labels, physical state labels, goal success labels or changed task evaluator. All added controlled interaction goes to the baselines too. A comparison head alone is a controller/readout extension, so compare native latent-distance distillation against that variant rather than calling both world-model improvements.

The diagonal coupling is essential: independent random futures can have identical marginal distributions while preserving every initial relative displacement under common commands. For circular navigation x_next=x+u mod1 with uniform u, both next-state marginals are uniform (a distributional distance is zero), but under common u their circle-distance is unchanged. For an overwrite actuator r_next=u, any initial r difference disappears in one step. A persistent object b_next=b+v retains its difference, as does a sole navigation agent. Hence the same rule can remove actuator mismatch while retaining a navigation goal without an explicit task switch.

The contracting variant r_next=alpha*r+u gives d_H² = alpha^(2H) delta_r² + delta_b² in a decoupled linear example. For alpha=.5 and two candidates relative to goal with errors (0,1) versus (2,0), Euclidean selects the former, while H>=2 selects the latter. Unlike the response-order proposal, both output coordinates can have first-order action response; the distinction is memory survival, not delayed controllability.

The exact target is a positive-semidefinite kernel distance because it is an L2 distance between functions A->psi(F_A(s)); finite-dimensional latent distillation can approximate it, with unavoidable rank/truncation error. It changes the demanded representation geometry, not the CEM budget. All futures used as targets must be observed; fitting the metric to its own predictions would only certify the model's current bias.

Major falsifiers already visible: relative/delta-control actuators may preserve position under common commands just like navigation, in which case this does nothing for Cube. Contact can cause different initial hands to move the object differently, so apparent motor differences become persistent consequences and must remain. Random commands can destroy both goal configurations, producing aliasing; absorbing walls can erase navigation position. Static background differences also survive, unless observations share the same environment; the mechanism does not solve distractor invariance. Infinite H is not automatically useful; H is an actual intervention horizon and must be one declared global value, not tuned per task. This may be known incremental-stability / synchronizing-distance representation learning; inspect prior art before admission.

No second proposal is retained at design stage: goal-set completion, action repair energy and terminal inverse solves collapse toward old Gramian/reachability/inverse-planning mechanisms. The present pass will spend its CPU check on the distinct synchronized-versus-marginal property rather than add another renamed candidate.
