# RoboCasa kitchen generalization pilot

This historical issued-history pilot is complete and STOP; its unfinished old
reference queue is retired. Corrected comparison and later authorized work are
in [the current protocol](../robocasa_refine/PROTOCOL.md). Do not resume the queue
from the original launch instructions below.

The user authorizes a finite pilot on dev13/dev15/dev6, with one SPT-JEPA route
per host. Old campaigns, goals, and recurring LLM monitors remain stopped.

Training uses six official RoboCasa365 pretrain-human atomic datasets: OpenDrawer,
CloseDrawer, CounterToCabinet, CabinetToCounter, CounterToSink, and SinkToCounter.
Original archives, checksums, episode metadata and all errors stay on EFS. The
actual downloaded episode counts, eligibility and kitchen identities must be
read from the inventory, rather than inferred from nominal 100-demo descriptions.

Kitchens are grouped by `(layout_id, style_id)`. The ten official target pairs
`(1,1)` through `(10,10)` must be absent from training and validation. The pinned
RoboCasa registry defines both held-out layout/style groups as IDs1–10, and the
pretraining groups as11–60. Layout ID zero is invalid. The released environment
correctly maps scene seeds0–9 to kitchen IDs1–10; both its mapping and base-pose
cache must use scene seed `kitchen_id-1`. A fixed hash
selects validation kitchens from the remaining source kitchens. No model selection
uses target-kitchen scores. This is scene generalization; it does not establish
unseen-task or cross-embodiment generalization.

All three candidates start from random weights, use the same source episodes and
common valid start population, and receive RGB plus factual transition controls.
No external teacher, proprioception, reward, success, task/language token, expert
path target, action policy target or retrieval bank enters a candidate. Future
observations are prediction targets only. The known-input convention is the
published 14-D four-native-step state displacement. Absolute poses are used only
by preprocessing to compute this input and never returned to a learner. Its
quaternion-sign limitation is disclosed and not silently repaired.

The side and wrist cameras are both used. SPT applies its shared model independently
to each view and averages the two view costs; it has no added cross-view teacher.
Source videos have a square camera field of view. Candidate inference crops the
central square from the public320x240 observation before the same OpenCV-area224
resize used during training. Replaying the source XML/state verified this geometry:
square-vs-crop MSE9.6/6.1 versus3150/1673 when stretching the full wide frame.
RGB orientation also matches the recordings. This common correction is derived
from source data, not target-score-selected cropping.
The observed first frame is the renderer anchor. Every temporal phase is eligible.
All candidates share starts valid for a 16-frame window; no fabricated padded
future or unknown terminal action contributes a target.

| Host | Candidate | Difference from temporal-forecast |
| --- | --- | --- |
| dev13 | temporal-forecast | Existing 64-D carrier, rho .8, forecast fraction .5 |
| dev15 | spatial | Sixteen addressed image-grid slots with16 channels each;256-D state, no CLS shortcut |
| dev6 | long-rollout | 16-frame context, causal forecast prior in the rate term |

Two independent training seeds per route. The initial cap is 5,000 updates;
entry/early/final checkpoints are 2/500/5000. The learning-rate horizon stays
5000 throughout. Standard arms use global batch128; the 16-frame arm uses64 to
approximately match observed-image volume. This does not equalize FLOPs,
parameters, target-image count or optimizer noise. Those differences and complete
training ancestry/costs must accompany results. No automatic longer run follows.
An entry-check recovery replayed one unsaved update per model; this is retained
in the remote recovery ledger and counts toward actual compute, beyond the
nominal checkpoint step count.

The500-update cohort was cancelled after the camera correction arrived during
its startup; it cannot support a matched cross-route or scaling claim. All
final5000-update candidate comparisons use the corrected preprocessing. Early
logs and explicit cancellation records remain; these are not method negatives.
All six500-update models remain and continue without repeating completed SGD.

Formal continuation requires actual real-data CUDA backward, positive required
gradients, full CPU model/AdamW/scheduler/RNG/cursor restoration, sample-independent
model components, and a real simulator/planner entry. Checkpoint and source hashes
are recorded. Failed interface attempts are retained separately from method
outcomes. Workers have bounded deadlines and stop markers.

Evaluation starts with Reach and ObjectPush, two common physical seeds in each of
ten target kitchens. Every model uses the same cached case identities, goal images,
observations, blocking controller, action search budget, execution cadence and
task metric. Reach has5 planner actions and H1; Push18 and H5. Common CEM uses1024
candidates, cold/warm15/10 iterations,20 elites, context2, execute1. This shared
single-GPU budget is fixed for the pilot; a numerical reproduction of the
published table is not asserted. The
normalized progress/return and task completion are separate fields. Goal creation
failures remain in the case ledger; no reward/state enters a planning cost.
For the acyclic SPT tensor-state adapter, redundant explicit Python GC between
candidate batches is omitted through a private function namespace; pinned public
CEM bytecode, sampling and math stay identical. A frozen500-step H5/64-candidate/
two-iteration test found
all rewards/means/stds bit-identical and12.6x faster (GC consumed2.20/2.65s).
The public reference retains its original cache collection. This changes runtime,
not the candidate search budget or method scores.

Released RoboJEPA checkpoints are isolated frozen external references. Their
external V-JEPA encoder, full multi-dataset training, input modalities and costs
must be disclosed. They never initialize or teach a candidate. A public-reference
gap is not an equal-data/from-scratch algorithm ranking. No published SOTA claim
is authorized by this pilot; a full matching-data scratch RoboJEPA retraining is
not silently substituted by an unrelated baseline.

The RGB-only reference uses the released action-only embodiment. The separate
state-delta reference gets native arm-base sensors, matching recorded14-D columns.
Source-frame replay identified a0.7m origin offset and a quaternion-sign difference
in the released manually composed pose helper; that helper is not used for the
state reference. No native state enters an SPT candidate or its planning cost.

Canonical remote root:
`/efs/prescottsun/SPT-JEPA/runs/robocasa_compare_20261110`.
Only core code, this compact protocol, conclusions and verified best configurations
belong in Git; raw traces, weights, caches and operational records remain remote.
