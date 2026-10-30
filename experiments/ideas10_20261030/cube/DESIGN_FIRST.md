# Independent design record: Cube / TwoRoom mechanisms

2026-10-30. This draft was written before the literature search for these candidates. It records hypotheses, including a rejected one; it is not a novelty or performance claim. No GPU experiment is launched. Readout coordinates are diagnostic only. All proposed training targets below come from observations and executed actions, not an optimal next action.

## D1 — A minimax contract for predictable observation queries

Replace area-averaged distortion by the worst *relative explanatory deficit* among an object-agnostic family of addressed observation queries. A query is a rectangle/address and a bounded image-feature functional, selected within one clip. Let b_q predict the future query from the permitted bypass alone, t_q predict it from the full permitted past observation/actions, and d_q predict it from the compressed state and actions. Estimate risks on separate ordinary interaction samples. Define the predictable advantage A_q = R(b_q)-R(t_q), and use only query scales for which this advantage is estimably positive. Learn the state under

    min rate(z), subject to R_q(d)-R_q(t) <= epsilon A_q for every q.

The sample-level surrogate is a within-clip maximum of normalized, stop-gradient teacher-referenced excess losses; a smooth maximum and a single global tolerance are possible implementations. The teacher has no extra observation or optimal-action information. A same-parameter full-state teacher or a charged EMA teacher is required, and its computation must be counted. The risk constraints are population statements; the noisy sample maximum is only a surrogate and can be biased by aleatoric outliers.

The essential difference from old I07 is a coverage constraint over query-relative deficit, not heteroscedastic weighting of an averaged loss. The same query family can define a planning discrepancy by taking the worst normalized mismatch to the goal's observed query answers. This cannot infer whether goal-hand pose is semantically irrelevant; it merely prevents a small predictable object from being priced only by its pixel area.

Property: if the constraints hold and a downstream task observable belongs to the query family, its MSE is at most teacher risk plus epsilon times the available predictive advantage. A capacity counterexample with two independent bits shows area-averaged distortion may discard the small object completely while a feasible coverage-constrained code cannot. If capacity is insufficient, the constraints honestly become infeasible; no trick avoids the required information rate.

Hypotheses: TwoRoom must retain an addressed small-agent signal that exists before QueryPool. Cube must retain block response even when the larger robot is easier to predict. Kill if the teacher does not explain the missing object, if a few stochastic outliers monopolize the max, or if coverage/probes improve without nonoracle planning.

## D2 — Action-conditioned bridge support at multiple trajectory cuts

Train an action-conditioned backward *distribution*, not a mean inverse, from ordinary tuples (x_k, a_k:H, x_H), alongside the forward predictor. For each candidate action sequence A, generate forward interior states f_k. Ask whether each f_k belongs to the learned predecessor support of the goal under the exact suffix A_k:H. A cut-local score is a calibrated backward nonconformity statistic, and the candidate is judged by its worst violated cut in addition to the unchanged forward goal score. The support thresholds are fitted on true held-out training-split trajectories. Train using within-sample worst-cut proper losses so a long prefix cannot hide a locally impossible suffix.

The mechanism is a *family of trajectory feasibility witnesses*, not a new goal embedding and not an action policy. A contact transition can have multiple valid predecessors; a Gaussian inverse mean can be physically impossible and is explicitly forbidden as a support witness. Conditional density is also not automatically a physical support score: behavior-policy occupancy and action dependence must be audited. For a first implementation, retain a set of sampled predecessors and report coverage at each cut rather than calling its raw NLL a transition likelihood.

Property under exact deterministic dynamics and exact predecessor sets: a candidate is feasible iff its actual forward state at any cut belongs to the predecessor set of the goal under that suffix. Thus an exact cut witness detects a false endpoint prediction even if a scalar latent distance is optimistic. For approximate models, correctness of *at least one witness on the relevant invalid path* is needed. Shared model errors can defeat all cuts. Under held-out exchangeability, per-cut coverage with alpha/H union correction controls rejection of real data paths; it does not certify adaptive CEM candidates or an unseen policy.

Hypotheses: TwoRoom door/wall passages and Cube contact/release are the locations where false endpoint success should disagree with at least one suffix witness. This is useful only after the state preserves the needed observations; it cannot repair the TwoRoom collapsed state alone. Kill if an adequately fitted backward support accepts the same false candidates, if occupancy confounding rejects valid rare paths, or if equal-compute ensembles/multiple shooting achieve the same improvement.

## D3 — Learn signed, goal-directional planning regret, then calibrate the selection rule

For an executed ordinary action sequence, freeze a target encoder for one calibration stage and record delta = z_true-z_pred. For any observed goal g,

    C_true-C_pred = 2 (z_pred-g)^T delta + ||delta||^2.

Learn the conditional distribution/quantile of this scalar with a proper sample-level loss. Inputs are permitted past observations, candidate actions, predicted state and goal image; the label is observed outcome regret, never an optimal action. The planner minimizes predicted cost plus a learned upper regret quantile. Distinguish signed directional error from an undirected latent variance penalty. A frozen staged version is a controller/readout extension; joint training that prioritizes optimistic residuals must be a separate experiment.

Calibration must be on *outputs of the whole frozen CEM selection rule*, not random candidate pairs. A candidatewise interval is not automatically valid after searching 9,000 correlated predictions. Use additional ordinary executed planner trials, record the selected-plan residual, and conformalize that frozen rule. When the scorer/model/search changes, the previous selected-plan guarantee does not transfer automatically. Data must also be provided to matched baselines.

Property: the displayed regret identity is exact. Conditional expectation of regret gives the Bayes-optimal corrected expected squared encoded-goal cost. A calibrated upper bound applies only to the selected-rule distribution and still says nothing about physical goal alignment. A counterexample with equal error norm but opposite goal-projected sign shows a generic uncertainty penalty cannot distinguish useful from harmful model error.

Hypotheses: Cube's high overall true/predicted rank but optimistic CEM winners suggests selection-conditioned directional error is more relevant than average latent MSE. TwoRoom can benefit only after a usable spatial state exists, and should be treated as a delayed second-stage test. Kill if directional regret is not predictable out of sample, if it is just a monotone transform of existing cost, or if it fails nonoracle selection after equal-data/compute accounting.

## D4 — Reversible-action orbit quotient (reject as a universal method)

Tempting idea: identify states reachable from one another by short, reversible action loops, then train a goal representation invariant along those orbits. This could identify many free-space gripper configurations while preserving the cube pose.

Counterexample: a free point agent's positions in TwoRoom are also mutually reachable by reversible moves. The transitive orbit quotient maps all positions in a connected room to one state, erasing the goal. With arbitrary reversible manipulation it may merge object positions too. Restricting the quotient to robot-only motion requires additional semantics; neither a single goal image nor reversibility supplies that semantics universally.

Therefore do not spend GPU budget on this as a cross-task algorithm. A local orbit or finite-horizon version only weakens the failure and introduces a scale; it does not resolve identifiability. It is a useful falsification of an attractive abstraction, not a tenth method to fill a quota.
