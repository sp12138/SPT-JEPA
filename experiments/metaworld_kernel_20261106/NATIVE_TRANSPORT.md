# Native forward image transformations

The current objective has not been achieved. The old inverse-action Flow score
of 78.75% over six manipulation tasks is a scalar comparator only. No weights,
action predictions, demonstration commands as proposals, or excluded inverse
learners are used by this experiment.

The working hypothesis separates three issues: action-relevant representation,
counterfactual forward identification, and sequential control composition.
Random-feature affine predictors failed actual own-command interventions, and
adding signed control effects alone finished with 0/480 manipulation success.
This motivates a change of observation coordinates rather than more controller
parameter tuning.

The native model predicts the next RGB image by transporting the current image.
Known already-executed commands condition only forward prediction. A second
query of the same observation likelihood predicts the next image given a
strictly later goal image, with the current command input zeroed. There is no
action output during training and no inverse objective. Losses are sample-level
future RGB likelihoods; all parameters start randomly. This is image prediction,
which the latest clarification permits; it is not inverse-action reconstruction.

Four paired modes test free drift, mandatory linear control, signed control,
and signed control increments relative to the agent's own previous command.
All use the same globally shared architecture and hyperparameters, without task
IDs or object rules. Planning projects predicted goal-conditioned image motion
onto forward control fields through the image's photometric metric. A unit-scale
Gaussian prior on the agent's own previous command retains force in directions
with no visual evidence. This is an inference calculation, never a training
target. Fifteen native commands are generated sequentially; displacement maps
compose before rendering from the original observed image to avoid repeated
texture blurring. Actual observations remain every five native steps and
replanning every fifteen. No recorded future command is a reference.

All native frames and current inputs are reassembled from the same original
LeRobot corpus, with the original 45/5 trajectory split. Five temporal phases
are interleaved. Phase-terminal macro input rows were zero-padded in the original
packed representation; the four remaining native inputs are recovered from the
last valid overlapping macro interval. Exact overlap equality passed for all
2500 episodes. This is input alignment, not an action reconstruction objective.
There are 204806 frames and 202306 native pairs, no new training trajectories.

Sixteen learners span four modes and four training seeds on dev13/dev15 only.
The initial finite budget is 20000 updates, with full native restore, FP32
gradient, information, sample independence, and actual 20-step physics gates
after two updates. Formal screens occur at 1000/5000/20000 updates. Frozen
evaluation uses the same cached eight-task physical cases, RGB goal images,
and 500-native-step any-success criterion as earlier internal experiments.
Training representation, native temporal sampling, and compute differ, so these
are not equal-FLOPs comparisons or public SOTA/MT50 results.

Transport/visual servoing and goal-conditioned observation prediction have
precedents. Novelty and superiority are not established by this design or by
passing technical gates. Successful manipulation on shared and fresh physical
cases, forward intervention accuracy, and matched ablations must decide whether
this is a useful replacement for inverse supervision.
