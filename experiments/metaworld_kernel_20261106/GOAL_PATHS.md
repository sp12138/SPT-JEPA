# Test a persistent observation path

Objective remains unmet. Native20k world training is complete on16learners;
formal evaluations continue. At5000, six-task success was4.17% free,3.125%
linear,3.125% signed,8.33% stencil over96episodes/arm. These are four trained
seeds crossed with four shared physical cases/task. They are not96independent
physical cases. Image-domain own-intervention prediction is substantially better
than copy, while gripper response remains weak. Better factual prediction alone
has not produced reliable manipulation.

A separate read-only optimizer check at linear-seed2/5000 reduced the rendered
future-image objective on all8initialcases. Exact bilinear derivatives agreed
with autograd within3.7e-5; independent multistart L-BFGS-B gave nearly the same
solutions. Objective improvements were modest,approximately0.16–4.3%,not evidence
that solver precision alone explains manipulation failure. No training/physics
updates occurred in this check.

The next hypothesis is path consistency. One-step conditional image prediction
can fit a tiny immediate change but behave inconsistently when repeatedly used
to guide the agent. Train an observation generator whose repeated local image
transformations explain a later observed endpoint. Its supervision is a single
future RGB likelihood. It does not read command arrays,predict action coordinates,
use inverse/action teachers,or consume future action references. All generator
weights start randomly. Its four horizon controls are point1,jump5,compose5,
compose15. Jump5 predicts a per-native-time image displacement but applies it
once at5times magnitude for training;compose5 repeatedly recomputes and composes
five native transforms. This distinguishes a longer visual target from temporal
composition. Compose15 tests the same mechanism over a longer interval.

All variants reuse the same frozen own stencil world for each training seed.
That world originated randomly and received20000future-RGB updates on the same
original data;its full cost counts. New observation-generator training adds at
most10000updates. The world is frozen and never called as a neural network in
the generator loss. Only its deterministic image-warp operation is reused there.
The generator's dataset opens RGB arrays only. Targets are strictly between
current and goal frames,with the same45/5trajectorysplit and matched eligible
starts across all four variants.

At execution,a nonlearned optimizer projects a goal-conditioned future-image
prediction onto the frozen world's controllable future-image family. No inverse
learner is optimized. It uses a unit-scale own-previous-command prior and bounded
self-generated command iterates. Observation-space optimization avoids treating
unidentifiable internal displacement coordinates as physical ground truth.
Replanning every5native steps uses the existing5-step RGB cadence,but increases
decision count/compute relative to old replan15. Therefore each original frozen
model first receives a matched observation-projection/replan5pilot on the same
cases. A strong candidate must also receive expanded matched controls before a
large superiority claim. This is an internal LeRobot protocol,not publicSOTA.

The two forward models can be interpreted as an uncontrolled goal-directed
observation path and a controlled causal transformation family;planning aligns
their observable consequences. GCP,visual foresight,neural flow models,and visual
servoing are relevant prior art. This design has not established novelty or
performance. Technical gates and actual goal completion decide its value.
