# DIAGNOSTIC_ONLY_NOT_FOR_BENCHMARK

**Decision: no demonstrated gain; stop this fixed pilot.** A directly learned action–outcome compatibility score acquired transferable fork-identification information, but identified heldout outcomes less accurately than the existing TF prediction distance and did not improve the selected physical-goal outcomes. No hyperparameter, fold, seed, label, checkpoint or continuation was selected by these results.

## Exact scope and information conditions

The two frozen TF12000 checkpoints supplied saved64D initial/actual endpoint representations for50 source episodes, each with39 already executed raw25×5 action sequences. All50 episodes belong to the original validation split. Sorted episode IDs alternate between two folds; each head trains on25 episodes and is evaluated on the other25. Both encoder seeds and all branches from an episode use the same fold assignment. Thus this is **cross-fitted diagnostic learning using additional validation-source fork interaction**, not an improvement under the original training split or standard evaluation. There was no encoder/dynamics update, new interaction, new case, GPU, physical-label fit or optimal-action supervision.

The86,017-parameter GELU head scores each253D `[initial, goal−initial, action sequence]` input independently. The goal during fitting is a uniformly sampled **actual branch endpoint**, and the39-way within-fork CE identifies its action branch. No negatives from other examples, candidate IDs, whole-bank features, physical success or actual evaluation goals enter fitting. Actions retain the original dataset training standardizer; a single training-fold latent RMS scales initial/delta. This is a learned compatibility/controller readout, not a new world model or a novelty claim. Full specifications preceded fitting in PROTOCOL.md.

## Complete identification results

Each CE/accuracy below evaluates every39 possible target branches per episode. Chance is2.56%, uniform CE3.664. The final classifier is used; there was no early stopping. Prediction-distance accuracy is the same old frozen model's predicted endpoint nearest to the actual target endpoint, without a new fit. It is not scored using physical labels.

|Encoder seed|Heldout fold|Train CE|Heldout CE|Train accuracy %|Heldout accuracy %|Old predicted-MSE heldout %|
|---|---:|---:|---:|---:|---:|---:|
|1|0|1.406|3.010|54.05|35.59|56.92|
|1|1|1.143|2.367|61.85|39.38|54.15|
|2|0|1.320|3.403|55.08|37.64|57.54|
|2|1|1.138|2.269|61.85|41.64|51.59|

The train–heldout gap is material. Heldout average identification accuracy is37.49%/39.64% for the two seeds, compared with55.54%/54.56% for predicted MSE. True-encoded nearest-endpoint index accuracy is97.44% in each fold; exact-index accuracy can be affected by coincident/tied endpoint codes. No branch was removed, relabeled or merged. Stronger fork classification would still not by itself identify which goal factors the physical evaluator ignores.

## Original actual goal:22 nonoracle candidates

Each row combines the50 heldout predictions from the two heads for that seed. These are cached25-step first-segment outcomes, **not50-step closed-loop planning**. Pose cost is `max(position/.03, cube24-symmetry angle/15°)`; joint and any-step sustained hold are distinct.

|Seed|Selector|Pose cost|Position cm|Angle degrees|Joint %|Hold %|
|---:|---|---:|---:|---:|---:|---:|
|1|Compatibility head|2.964|8.39|12.57|20|14|
|1|Original predicted MSE|3.421|9.58|12.96|22|16|
|1|True encoded MSE|3.631|10.37|11.25|20|14|
|2|Compatibility head|3.272|9.54|11.33|16|14|
|2|Original predicted MSE|3.126|8.78|10.79|26|14|
|2|True encoded MSE|3.076|8.72|10.82|24|18|

Two-seed case-average head−predicted-MSE changes: pose cost−.155,95% descriptive paired interval[−.626,+.301]; joint−6pp[−15,+3]; hold−1pp[−9,+7]. Head−true-encoded-MSE pose is−.235[−.730,+.227], joint−4pp[−15,+6], hold−2pp[−12,+7]. These do not establish a gain. Intervals condition on the fixed two seeds/fold fits and recorded pool; shared training folds create dependencies, and ordinary case bootstrap does not account for refitting uncertainty, environment repeats or fresh-case generalization.

## Complete39-candidate appendix

This larger library includes17 recorded-action neighbors. It must not replace the primary nonoracle analysis.

|Seed|Selector|Pose cost|Position cm|Angle degrees|Joint %|Hold %|
|---:|---|---:|---:|---:|---:|---:|
|1|Compatibility head|1.932|5.59|7.65|46|28|
|1|Original predicted MSE|2.556|7.05|10.18|44|26|
|1|True encoded MSE|.014|.027|.16|100|56|
|2|Compatibility head|1.830|5.27|8.72|48|24|
|2|Original predicted MSE|1.169|2.91|7.78|70|26|
|2|True encoded MSE|.040|.043|.56|98|56|

Head−predicted-MSE averages to0pp hold[−10,+10],−10pp joint[−25,+5], and+.019 pose[−.702,+.699]. The opposite seed directions are retained. The mean head−true-encoded-MSE hold difference is−30pp[−42,−18]. These measurements provide no justification for promoting a head using this extra data.

## Integrity, arithmetic failure and budget

All100 input hashes,50 independent episode IDs, disjoint folds, config/card paths and input shapes were checked. The head's fitting structure contains no actual goal, physical coordinates, quaternion, hold or success fields. All four fits completed before any physical outcome was opened. Each fold of an encoder seed starts from exactly the same head tensors and resets the same seed20261101+encoderseed. RMS values by seed/fold are .835743739/.817719084 and .835493869/.818156396.

Attempt1 completed seed1/fold0 at600 updates but then failed an overly strict **batched bitwise** contract before physical evaluation and checkpoint saving. Its code, protocol, error log and output remain intact. An independent untrained-head reproduction found1.49e−8 batched differences from row arithmetic; fixed1×253 single-candidate evaluation was exact. Root approved the contract-only recovery, with no training or scoring change. The first fold's seven recorded loss values were reproduced exactly.

All four recovered fits pass exact permutation and zero-action equality when each candidate is evaluated with the same1×253 shape. Batch permutation maxima are1.53e−5/3.81e−6/7.63e−6/3.81e−6; identical zero-action candidate maxima9.54e−7/7.63e−6/0/1.91e−6. These are recorded and pass the declared FP32 relative1e−5/absolute1e−6 check. No claim of batched bitwise equality is made, and no dependence on bank identity or order is introduced.

The successful complete run, including input checks, four fits and full evaluation, took **13.486 seconds** of measured main-process wall time. Its four fit/save intervals were2.237/3.090/2.099/3.100 seconds. The failed attempt's protocol-to-error-log timestamp span is3.874 seconds; this is a filesystem-derived span, not a captured exact full process duration. The transparent total update count is **3000:2400 prescribed updates plus600 same-seed failure-recovery repeats**. GPU/model/encoder/renderer/simulator calls are0. No checkpoint is eligible for deployment or benchmark submission.

## Artifacts and stopping boundary

Small stable artifacts: protocol, pilot.py, this report, complete results.json, successful execution log, and attempt1 code/protocol/log/arithmetic explanation. EFS results and diagnostic-only head/optimizer states: `/efs/prescottsun/SPT-JEPA/reports/next-mechanism-matching-pilot-v2-20261101`. Original failed output: `/efs/prescottsun/SPT-JEPA/reports/next-mechanism-matching-pilot-20261101`. Large weights remain on EFS; no Git index or shared instructions were edited.

This screen rejects neither all compatibility learning nor all outcome-based objectives. Its fixed short head failed the specific feasibility gate: it did not outperform existing predicted outcome matching or produce a consistent physical-goal selection gain. No additional fitting, variant, longer run, GPU activity or resumed periodic monitor follows.
