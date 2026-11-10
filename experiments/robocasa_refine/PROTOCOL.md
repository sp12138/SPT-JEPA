# RoboCasa mechanism tests

Latest authorization: diagnose, fix and rerun toward exceeding RoboJEPA on
dev13/dev15/dev6/dev6-2c. The goal is not achieved. Old campaigns/goals and
recurring LLM monitors stay stopped. Controllers are finite and owned. Raw cases,
weights, traces and operational records stay under `/efs/prescottsun/SPT-JEPA/runs/`.

## Completed evidence

`robocasa_compare_20261110` has six scratch5000-step models,642 human trajectories
from six atomic tasks, kitchen-grouped splits, no target kitchens in training.
TF/spatial/long progress is21.80/18.33/20.55% Reach and4.59/3.68/11.36% Push;
strict completion is zero. These are progress, not success rates. Issued history
differs from the released achieved-motion replay. This pilot is STOP; unfinished
old references are explicitly cancelled, not method negatives.

`robocasa_refine_20261110` forks two long parents into control/clean/highnoise,
1000 updates each, full Adam/rank RNG/cursor, common cosine restart and10-step
warmup. Clean adds recursive clean Gaussian calibration; highnoise uses DiT t=1.
No extra learned parameters, inverse, action labels, reward or teacher. All six
6000-step restores/backwards pass. Control-seed2 recovers5250→6000 with zero replay;
the original scratch ancestry still includes one unsaved update replay per parent.
Clean forecast/copy0.91/0.92 improves while control worsens; factual/zero ratios
0.92/0.95 versus controls0.62/0.61. Own latent MSE is not a shared physical unit;
unsimulated zero inputs are not counterfactual causal evidence.

## Corrected reference

`robocasa_official_20261110` verifies all143/613/1093 predictor parameters of
50M120k/2B450k/8B300k, per-camera encoding, achieved history, KV cache, prediction
and short public CEM on an actual transition. Encoder FP16 and predictor BF16
follow the paper table; literal-client predictor FP16 differs and is disclosed.
RGB zero-absolute-pose, released hand-composed proprio, and training-aligned native
sensor tracks stay separate. Their0.7m origin/quaternion difference was verified
against source recordings. References never initialize or teach candidates.

## Ordinary-command experiment

`robocasa_intervention_20261110` attempts two16-command random walks in each of128
parent-TRAIN kitchens: isotropic directions, radius uniform[0,.3]. Only RGB, issued
commands and realized robot displacements are recorded; no goal, reward, success,
absolute state or policy targets. Public init resets are used without recording
goal observations or expert paths.34 initialization geometry failures are retained.

Final corpus:222 trajectories/111 kitchens/3552 transitions. A fresh salted
kitchen split gives102 training and9 new-log validation kitchens. All54 original
validation and ten target kitchens are excluded. Minimum pilot coverage is128
valid episodes/80 kitchens. Every arm uses the identical remaining corpus; no
model score selects logs. The parent split hash must not be reused, since all
source kitchens already belong to parent TRAIN. Raw split metadata is preserved.

| Host | Input | Image generation |
| --- | --- | --- |
| dev13 | Realized14-D displacement | Original SPT |
| dev15 | Issued translation mapped to14-D | Original SPT |
| dev6 | Same issued command | Pure-noise DiT |

Each arm restores the two original long5000-step parents, all weights/Adam/RNG/
cursor, batch64, identical1000-step restart. No new learned parameters, inverse,
future/optimal action targets, reward, external teacher or expert-path learning.
Unknown terminal action is zero. Losses remain sample-level. This added-data
recipe is not an identical-data RoboJEPA pretraining reproduction. Realized
orientation channels are a competing explanation until a translation-only
control isolates command feasibility from quaternion artifacts.

Before SGD, eleven constructible source cases undergo two fresh resets on each
host:62/66 strict passes; four unreliable Reach-v1 replays remain technical
failures. The first two/task passing all six are Reach-v2/v3 and Push-v2/v3,
seed7401. Their24 selected checks pass qpos/qvel/time1e-8 and RGB gates without
relaxing tolerances. Fixed scene RNG alone does not explain every old failure.
Selection uses no neural scores; never mix old and fresh case results.

Audits at5000/5002/6000 require actual full restoration, sample independence,
terminal-action poisoning, finite FP32 backward and positive required gradients.
Four source cases × two parents are a short screen, not eight independent cases
or a SOTA claim. No automatic long training follows this cap.

All six models and24 source episodes are complete. Reach/Push progress averages:
achieved34.62/7.29%, issued40.27/11.00%, highnoise13.14/9.26%; zero control has
0.057/9.09%. All strict completions are zero. Issued-input gains are inconsistent
between parents. `results/source-summary.json` records NO_LONG_PROMOTION.

## Full reference collection

The new target cohort is kitchens1–10, seeds8101/8102, raw640×480 and released
random flour_bag variants.39/40 constructions pass; one invalid goal remains.
Public CEM:1024 candidates,15/10 iterations,20 elites,H1/5,context2,execute1,
Reach5/Push18 commands,maxnorm/std.3,equal views. Report Reach mean and Push final
progress separately from strict completion and zero control. Passive movement
is not learned success. Candidate square-FOV cropping remains source-verified.

Before any target endpoint, three waiting controllers transfer ownership to
dev6-2c:50M GPUs0–1,2B GPUs2–3,8B GPUs4–7. Both proprio conventions and50M RGB are
reported. The extra host runs model-free physical/RGB preflights before loading;
invalid interfaces are not negatives, and comparisons require common valid cases.
Same evaluation does not equalize pretraining/data/FLOPs. Allocated slots do not
establish continuous GPU utilization.

## Paired structural tests

`local_model.py` confines every encoder layer to the same spatial regions; CLS
cannot receive image content. Original parameters, global dynamics and loss are
retained. Spatial pooling alone has no such guarantee because ViT tokens already
mix the image. First require an actual outside-region perturbation check in
`robocasa_local_20261110/LOCALITY_APPROVED.json`. This synthetic test is neither
physical performance nor a novelty claim. Actual TRAIN and simulator perturbation
checks pass: original outside-region max change0.1344, regional encoder exactly0.

`analytic_model.py` gives coding residuals an explicit AR Gaussian prior around
the deterministic controlled mean trajectory. The exact conditional mean error
is e_t−rho*e_(t−1), where e=encoded mean−initial-only rollout. Learned dynamics
therefore need not learn artificial noise propagation. Retained entry regularizer
does not establish a joint ELBO. Independent distribution-KL checks pass within
5.2e-6; future-code and unused-terminal-action gradients into rollout are zero.

`robocasa_local_20261110` initializes six exact full5000-step parent restores:
dev13 global-spatial, dev15 regional-spatial, dev6 analytic-noise, two parents
each. Regional/global pairs share every parameter, optimizer state, data and
sample-level loss. Analytic shares its long parent with the prior issued arm.
Spatial windows8/batch128 and analytic windows16/batch64 each observe2048 camera
frames/update; sequence targets and FLOPs differ. No additional neural parameters.
Every arm has a1000-update restart/10warmup cap, restore/backward audit and an
actual5002-step simulator entry check before continuing. Uniform versus fixed
initial-goal support costs are paired only for spatial models. These are source
development tests; long training requires a physical-control gain.

Old R3 source-reference processes use an obsolete replay bank and are explicitly
retired, retaining partial results, to free full paired-training hosts. The main
fourth-host reference queue continues. New physics-ceiling diagnostics may use
protected state to test contact attainability under the same18 commands/.3 norm;
their state-dependent traces cannot enter training, teachers or proposal memory.

All six structural models finish6000. Forty source episodes (four physical cases,
two ancestors, five controller/model combinations) give the following progress:

| Model / cost | Reach | Push |
| --- | ---: | ---: |
| Global spatial / uniform |45.14%|9.24%|
| Global spatial / support |18.76%|34.88%|
| Regional spatial / uniform |32.08%|4.65%|
| Regional spatial / support |13.67%|9.28%|
| Analytic noise / uniform |40.32%|0%|

All strict completions remain zero. Both global-spatial parents improve Push
with support, while Reach regresses. Frozen direction-only projection on these
and the prior issued models adds24 episodes and does not resolve the tradeoff.
No long promotion. One final audit originally reported1.43e-4 size-changing
predictor error; FP32 math SDPA gives2.38e-7 and fixed-shape other-sample poisoning
gives exactly0. Original failure remains; full6000 state, all gradients, weights
and controls are preserved with zero SGD replay.

Protected state-assisted pushes do make gripper/object contact and approach the
goal, but do not finish stably. Contact versus free-motion latent MSE shows no
large consistent jump; this aggregate latent statistic does not establish correct
object dynamics or identify a unique failure cause.

## Repeated-state interventions

`robocasa_forks_20261110` selects64 original TRAIN kitchens before model scores,
two initial seeds and eight independent16-command programs/state. Maximum1024
trajectories/16,384 transitions. First commands use a scrambled spherical Sobol
design with the same uniform-radius bound; suffixes are independent isotropic
commands. No goal, reward, success, object state or expert path selects controls.
This is additional task adaptation data, not identical public pretraining.

Every branch must replay the same initial physical state within1e-8 and raw RGB
MSE<5. The actual branch0 observation is then the common root image. Physical
state stays in separate technical reset proofs. Only full eight-branch groups
enter the corpus; incomplete technical groups remain archived. Existing original
validation and target kitchens stay excluded; the fresh log split is inherited.
Dataset consumers return RGB and current factual controls only.

Three arms restore both prior issued6000-step parents: ordinary-flat on the
previous corpus, fork-flat on the new grouped corpus, fork-shared on identical
grouped data. All use the analytic conditional prior,2048 camera images/update,
the same1000-update restart/10warmup and no added neural parameters. The ordinary
control can train while acquisition finishes. Grouped batch8×8 branches replaces
flat batch64; the cumulative sampler cursor is converted exactly by8, with all
model/Adam/rank-RNG states preserved. This is a changed data unit/corpus, not an
identical old sample order. Uniform/direction source controllers are paired.

Shared-root charges the initial-code entry once per physical experiment while
retaining every conditional transition/image term. Algebraically this is the
existing branch loss with entry scaled1/8 on matched data; it is not by itself a
novelty or joint-ELBO claim. A genuine TRAIN eight-branch loss identity agrees
within3.1e-9; terminal poisoning changes loss0; all required gradients are positive
without SGD or parameter changes. The objective remains a sum of trajectory terms,
with no statistics/negatives coupling independent worlds.

Finite acquisition, proof, restoration, actual6002 MPC entry,7000-step training
and48 source episodes are owned by `phase-launch-*-v2.json`. No automatic long
training follows. Acquisition ownership is now `acquisition-launch-parallel-*`:
at most48 independent initial states/host, same GPU and unchanged eight-program
sequence per state. Original live programs are adopted, completed SHA-verified
records reused, original acquisition deadline/corpus cap retained. This uses the
otherwise idle160-core CPU quota; render/physics need not saturate GPU arithmetic.

Generation-one/full-GC cadence checks pass for50M/2B/8B: all output and RNG errors0,
H5/128 candidates/two refits, plus1024-candidate cache lifetime. Repeated allocated
memory is constant; measured speedups1.34/1.11/1.25. Reference-only execution is
owned by `robocasa_reference_adopt_20261110/launch.json`. Eight original chunks
continue unchanged after retiring their parent PIDs; only subsequent chunks use
the cadenced collector. Same model, sampler, CEM bytecode/math, cases, inputs and
original24-hour deadline. Results retain which GC implementation produced them;
the new root's finite collector is authoritative. No current episode is replayed.
