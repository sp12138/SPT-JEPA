# No optimal-action label is necessary; missing action support still matters

The primary research contract uses observations, executed actions and their outcomes. It does not require an oracle identifying the next optimal action. That is a meaningful distinction from goal-to-action imitation. It does not, by itself, make every action-conditioned conditional prediction a causal counterfactual model.

## A simple limit no new loss can remove

Let X take values−1,+1. A logged behavior always executes A=X. Consider two deterministic worlds:

\[
F_1(X,A)=X+A,\qquad F_2(X,A)=1.5XA^2+0.5A.
\]

Both produce Y=2X on every logged sample. Thus the complete observed training distributions are identical, even though X is fully observed and neither world uses hidden action labels. For the goal Y=0, world1 needs A=−X, while world2 has solutions A=0 or A=−X/3. Their exact successful actions do not overlap. A learner restricted to the common logged data cannot know which world it is controlling; for a single proposed action, the sum of success probabilities across these two worlds is at most1. One of the worlds therefore has success probability at most1/2 for any randomized learner. More parameter capacity or a different batch-independent loss cannot identify the missing fact.

This is a deliberately simple indistinguishability argument, not a new causal-identification theorem and not a claim that our actual benchmarks have this worst-case ambiguity. Physics priors, broader action support, repeated interventions, stronger state sufficiency or known structural restrictions can remove the ambiguity. If one candidate gets new re-roll observations, those observations and their cost must also be offered to the matched comparison.

With a fully sufficient Markov state and supported actions, modeling executed transitions is a legitimate way to learn dynamics from ordinary interaction. With a partial image state, a multi-step recorded action sequence can also reveal hidden state or later policy feedback. A predictor can use that correlation while fitting observed trajectories, then fail on a CEM sequence that does not arise from the behavior policy. Neither a good probe nor a low on-data rollout error excludes this mechanism.

## What the current data establish

The episode-disjoint action-only diagnostic finds Cube initial block height R² about.95 and TwoRoom initial position about.48/.49 from recorded action sequences. This establishes predictive dependence, not a quantified lack of support or proof of causal confounding for every transition. It does invalidate the convenient assumption that actions are exogenous and reveal only the actuator, which one proposed residual objective relied on.

The relevant primary [TDWM project report](https://td-worldmodel.github.io/TD-WorldModel/) also separates representation quality from planning on perturbed actions and adds noisy re-roll observations in its Scene study. Its scores cannot be copied into our Cube table. Our own fixed-bank Cube diagnostic similarly finds that model-cost prediction can be accurate while the actual goal metric emphasizes the wrong entity. These are separate failure modes and require separate tests.

## Consequence for the ten-idea request

The requested standard is a concrete, non-equivalent mechanism with theoretical or short-experiment evidence before counting an idea. An idealized theorem does not establish benchmark gains when its required information is absent from the actual learning graph or dataset. The admission records therefore separate a mechanism property, its real-data applicability, whether its gradient can train the intended component, and its measured task benefit. The standard stays strict even when that leaves fewer than ten qualified proposals.
