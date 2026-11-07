# MetaWorld: inverse of the executed forward input

The user authorizes dev6, dev6-2c, dev13 and dev15, candidate exploration and evidence-gated long training. Published baseline results are references only: this campaign trains and evaluates **zero baseline models**. Old goals and recurring LLM monitors remain off. Old disqualified goal-conditioned action policies, weights, outputs and traces remain excluded.

## Supervision

For a recorded transition `(o_t, a_t, o_{t+1})`, forward prediction consumes `a_t`. The auxiliary inverse reconstructs **that exact supplied input** from the observed latent pair. In this corpus one forward input is a 20-dimensional block of five already-executed native 4D controls between the observed frames. The last padding action is neither a target nor a usable transition. There is no `(current RGB, desired goal RGB) -> next action` training target, action sequence reference, action teacher or policy decoder.

All learner parameters start random. The learner receives RGB and corresponding executed actions only. No reward, success, state, task ID, language, pretrained backbone or external teacher enters learning/planning. Losses are separable across clips; batch-wide diagnostics are logging only. At deployment the inverse head is unused: the encoder, forward model and common CEM select their own controls. Removing the inverse head must leave actual encoding/prediction exactly unchanged.

The native checkpoint audit actually restores model, optimizer and scheduler on CPU, and checks that RNG states are present and the consumed-sample cursor equals the saved step times batch size. Actual training continuations use the unchanged native RNG/cursor restoration path. The audit checks a held-out FP32 backward pass, poisons the last padded action, checks cross-sample inverse gradients, and removes the inverse head during actual inference. It does not establish bitwise equality to an uninterrupted training trajectory. A real 20-native-step physics/MPC run follows the entry audit. These are correctness checks, not task-success evidence.

## Hypotheses and candidates

The established inverse-dynamics auxiliary is not claimed as our invention. The working hypothesis is that forecast/render losses can preserve image predictability while assigning too little learning pressure to the effect of actions. Reconstructing the supplied action may preserve controllable differences, but robot-motion cues alone might still miss object/contact information.

Four families share the temporal-forecast model:

| Family | Training change | What it tests |
| --- | --- | --- |
| observed | MSE inverse from the observed adjacent latent pair | Whether preserving executed-action information suffices |
| highnoise | Same inverse, image DiT training noise odds multiplied by four | Whether reducing the noisy-image shortcut improves conditional forecasting |
| generative | Conditional flow inverse, training-only, with high-noise time sampling | Whether multimodal/partially unidentifiable actions make MSE a poor auxiliary |
| predictive | Average observed-pair and forward-predicted-pair inverse losses, same head and same input target | Whether the auxiliary must train the forward action pathway directly |

Each family crosses four/16 latent slots (64/256 dimensions) with global inverse weights 10/100, all shared across tasks. Two independent scratch seeds per configuration give 32 initial trainings. Shared parent modules use the same CPU initialization seed stream where dimensions permit; changed-width models are not identical initial tensors. Inverse head CPU initialization and added per-step training noise use forked RNG contexts. Exact GPU-noise pairing with an earlier no-inverse run is not claimed. Inverse MSE and flow losses have different target units; their magnitudes are not comparative control scores.

The predictive auxiliary can itself admit an action watermark shortcut. A larger action-encoder gradient is therefore a diagnostic, **not evidence of physically correct control**. Only closed-loop manipulation outcomes decide continuation. No novelty or broad superiority is established by the candidate list.

## Data and internal evaluation

The independent ungated corpus is `lerobot/metaworld_mt50@b46a1c185a2368660650a82ac5a5083a56722f48`: 2,500 expert/scripted episodes, 204,806 RGB frames, 50 tasks; trajectory-grouped 45/5 train/test episodes per task; all five temporal phases retained. This is not the gated JEPA-WMs mixed-quality corpus. All 250 packed HDF5 shards are copied to each host's local SSD with byte SHA equality, then 64 actual windows are compared on RGB, action and raw-row fields. Only the file location changes. Existing native stages finish before cache adoption; completed SGD is retained.

The development screen has eight tasks and four previously inspected physical cases per task: Bin picking, Door, Drawer, Peg insertion, Pick place, Push, Reach and ReachWall. It is not untouched confirmation. Shared RGB goal MPC uses 300 CEM candidates, 15 iterations, 10 elites, horizon six macro transitions, execute three, context two, frameskip five, warm tail, 500 native any-success. It supplies no simulator state or expert action to planning. All expert-goal failures remain in the main denominator and are reported separately.

Native stages are 2 -> 1,000 -> 5,000, with the learning-rate horizon fixed at 41,200. At 5,000 the screen runs 32 episodes per model (1,024 total). At most four configurations proceed if **each seed** has six-manipulation success >=35% and Bin/Peg/Pick/Push mean >=20%. This is compute allocation from a small development cohort, not a matched-baseline significance test.

Selected parents continue their exact native states to 20,000 and 41,200, with the existing 160-case eight-task evaluation at each stage. Six additional independent scratch seeds (3..8) per selected configuration run the same stages and long budget, without a second performance filter. Freed GPU slots run these candidates; up to 32 long learners are possible when four configurations qualify. Fewer qualifying configurations need not keep all GPUs continuously busy.

A fresh 50-task cohort is fixed before candidate outcomes: 20 cases/task, seeds 6042..6051 and 7042..7051, 1,000 physical cases. Official scripted experts construct terminal RGB goals only. Exact physics/RGB replay gates and all failed goals are retained. The same native CEM evaluates final 41,200 checkpoints in ten five-task shards per model. Each selected configuration gets up to 8,000 episodes across its eight independently trained models. Initial two selected seeds and six fresh replicas are reported separately. This is an internal RGB image-goal benchmark, not canonical task-ID MT50.

## Public comparisons and limits

`public_baselines.json` records author-published numbers and primary sources. LaLa-World's image-goal comparison reports LeWM, reward-free adaptations of DreamerV3/TD-MPC2/IRIS, and pretrained DINO-WM/JEPA-WMs/LaLa-World on two reaching tasks. Its corpus, camera, goal construction and horizon differ from our setup. Those numbers do not establish general-manipulation leadership.

The SmolVLA paper reports full-50-task MetaWorld results on LeRobot data, including Diffusion Policy, TinyVLA, pi0 and SmolVLA. Those methods supervise policies with action labels; VLAs additionally use pretrained backbones, language/proprioception, and a different data split/controller/budget. These are context, not an equal-information ranking. No SOTA claim is permitted by comparing our eight-task mean against a public 50-task mean or by combining results from different models.

## Execution and evidence

Canonical root: `/efs/prescottsun/SPT-JEPA/runs/metaworld_input_inverse_20261107_v1`.

Immutable learner/trainer source: `reports/metaworld_input_inverse_20261107/source-faf40148`, full digest `faf40148e12de2fad8f6af3a0d83f1c412db328fc735c0b796d1e2ab74ce48b7`. The local-SSD binder and finite follow-up queue are separately hashed operational helpers; neither changes this learner source.

Read `launch-io-v2-{host}.json`, `collector-launch-io-v2.json`, `followup/launch-{host}.json`, status and actual endpoints for current ownership/progress. Original parent launch records are historical after adoption. No unrelated CPU jobs on dev6-2c are affected. Finite deadline is the original root plan's five-day deadline. Failed runs remain explicit; no hidden retries, completed-SGD replay, new baseline or automatic subsequent research round.

The completed5,000-update screen failed the long-training gate:32models/1,024episodes, best complete paired six-manipulation score1/48=2.083%, one Push success. One original cleanup error and one repair-attempt metadata collision were retained and repaired within the original recipe; one unsaved update was replayed, no checkpointed updates repeated. No long/fresh-replica/full50performance run was started. Additional frozen main/inverse gradient decomposition covered31models,0SGD/0simulation/0baseline execution. Results and scope are in RESULTS.md. The preceding qualified original-corpus best remains six-manipulation25% on a different learned model and budget. Withdrawn policy-target scores remain inadmissible.
