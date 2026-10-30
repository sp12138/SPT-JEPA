# S3 bounded gate: conditional path transformations

This is a proposed sample-level auxiliary for the existing temporal-forecast
world model, not a novelty or superiority claim. Three fresh matched 600-update
arms test what predicting a compositional path transformation adds beyond frame
classification and scalar completion. Native training, renderer, rate, observed
data and planning interfaces remain intact. No expert optimal-action labels are
used. The proposed loss does not couple examples; native baseline objectives
are preserved.

## Fixed query and target

For a window of eight normalized RGB frames, let psi be fixed 8x8 adaptive
average pooling, flattened to 192 coordinates. At each optimizer step a private
CPU generator (seed 20261031 + step) draws eight unsorted triples of distinct
frame indices. The same index draws apply to all clips at that step, so splitting
a batch leaves each clip's loss unchanged. Every prototype value comes from its
own clip. This shared index draw is a declared implementation refinement of the
previous independently sampled per-window diagnostic, not new data access.

Each frame receives its nearest prototype symbol A/B/O using squared Euclidean
distance. Exact ties select the first prototype; degenerate queries are retained.
Prototype indices and timestamps are never inputs to the head. Future prototype
images are legitimate auxiliary target/query data, but can reveal future
appearance and permit shortcuts; the objective does not guarantee state use.

The monitor's progress states are 0 (A not seen), 1 (A seen), and 2 (A then B
seen). A advances 0 to 1; B advances 1 to 2; O preserves progress. A segment
induces exactly one of five maps, in implementation class order:

| Class | Map on starting progress (0,1,2) |
|---|---|
| I | (0,1,2) |
| A | (1,1,2) |
| B | (0,2,2) |
| AB | (2,2,2) |
| BA | (1,2,2) |

Adjacent segments compose by function composition. Scalar completion from zero
is the indicator of class AB. For example I and B have equal negative completion
from zero, but after an A prefix only B completes: the full map retains a
response to incoming progress that scalar completion loses. This elementary
automaton property is established prior mathematics, not claimed as ours.

The full window and its first/second halves are supervised. When prototypes are
distinct and ties absent, the full window contains A and B, so its map is only AB
or BA; halves are required to expose all five possibilities. Directly predicting
the joint map avoids multiplying marginal event probabilities, which would
incorrectly predict 0.25 ordered-completion probability for a 50:50 mixture of
AA and BB (the actual probability is zero).

## Matched head, causal route and objective

The causal trajectory starts at the observed initial encoding. All seven clean
native predictor calls use only the generated state prefix and recorded action
prefix. Future observed encodings and query prototypes never enter this route.
Predictor dropout retains train/eval semantics but runs under a forked RNG with
seed forecast_seed + 2000003 + step, preserving the native RNG stream.

Every arm evaluates the same 11 contexts: eight one-frame masks, full window,
first half, second half. Concatenate the flattened masked causal trajectory,
eight-entry mask and three flattened psi prototypes, then apply
Linear(1096,64), GELU, Linear(64,5). The active head has 70,533 parameters. Its
initialization is forked after the original model constructor, preserving all
old tensors and outer RNG. All three arms share identical head initialization
and produce B x 8 x 11 x 5 logits in one head call.

| Arm | Contexts entering loss | Active logits | Target |
|---|---|---|---|
| symbol | eight frame contexts | first 3 | A/B/O |
| completion | full and both halves | first 2 | complete from progress 0 |
| transform | full and both halves | all 5 | joint map |

The auxiliary is the mean cross entropy divided by log(K), K=3/2/5. This fixed
class-count formula gives uniform predictions loss one for every arm; it does
not equalize gradient geometry or class imbalance. Global auxiliary weight is
fixed at one. All three raw and normalized CE/accuracy diagnostics are computed
in every arm, but only the selected objective contributes gradients. Native TF
loss is unchanged and added once. No classification output or query is used by
the native planner.

Parameter count and forward computation match across arms; active output rows
do not: symbol has 130 inactive final-layer scalars, completion 195, transform
zero. Some contexts receive no selected-loss gradient in each arm. These are
explicit consequences of the learning target, not hidden capacity equality.
The added head budget relative to TF is reported separately (about 0.14% for the
full experimental model). The head is absent in default `path_query_mode=none`.

The subclass captures the one native `prepare`/`encode` result during loss, then
clears the transient capture in `finally`. It does not rerun the encoder or
replace Native Trainer/checkpoint logic. Public encode/predict signatures are
unchanged. Active path and response-query auxiliaries cannot be combined in
this gate.

## Evidence and limits before the gate

Previous episode-disjoint within-clip CPU readouts gave adding initial psi to
query+actions RBF readouts +2.31 percentage points in TwoRoom and +1.43 in Cube.
Linear readouts did not show a consistent increment. Queries can already explain
much of the target; this experiment tests conditional usefulness, not a theorem
that the auxiliary forces a useful state. Independent-bank controls changed
both class support and state gains, and are not mixed into this gate.

Recent action-null auxiliary gates learned more observed Cube information but
did not transfer it to causal forecasts or planning. S3 is therefore evaluated
on clean forecast versus observed/teacher-forced/carry query responses,
initial-state/action interventions, native planning, and native full-gradient
health. Classification progress alone is insufficient. All arms stop at 600 for
review; no automatic long run follows.

Closest context includes predictive-state representations and finite automata,
Neural Event ODE (event-time dynamics), C-learning (goal reachability), and TDWM
(within-clip time-kernel regression). None makes global originality established.
The narrow proposed experimental difference is supervising query-conditioned
joint segment transformations of a clean native latent rollout while preserving
the model/planner contract, with scalar-completion and frame-symbol controls.

## Audit interface

`PathQueryLeWM` in `as_jepa_lewm.pathquery` accepts `path_query_mode`,
`path_query_weight=1`, and `path_query_seed=20261031` plus Robust kwargs.
`path_query_targets(render, step)` returns detached prototypes/symbols/maps and
context masks; render is normalized [-1,1]. `path_query_outputs(flat, act_emb,
render=None, step=0, targets=None)` returns `forecast`, `logits`, `targets`.
Pass the same `targets` for interventions. `path_query_logits_from_states(states,
targets)` reuses the exact readout on observed, teacher-prefix or carry paths;
those are diagnostic paths with explicitly different observation access.
`audit_expected_gradients()` declares the native groups and `path_query_head`
active (head optional at weight zero), for the post-two-step preflight. Both the
native carry output projection and AdaLN-zero action modulation start zero.
CPU contracts explicitly check that two ordinary SGD updates open the initially
blocked action route. No initialization is changed to make the test pass.
