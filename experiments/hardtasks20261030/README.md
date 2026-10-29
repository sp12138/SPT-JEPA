# Harder-task research, 2026-10-30

User redirected research from additional PushT expansion to harder tasks based on temporal-forecast, prioritizing effectiveness, a self-contained mechanism, simplicity, substantive novelty and causal understanding. Use dev2/dev13/dev15. Stop unnecessary work with resumable state; long training is authorized once focused gates justify it. Avoid frequent token-consuming polling.

First targets: official OGBench Cube manipulation and TwoRoom navigation. Download only into this project EFS root. Freeze source/data/evaluation identities before comparisons. Released LeWM is an external reference with different parameter/training budgets, not a matched-size baseline. Matched AS-JEPA original and lower-beta controls remain required. All new method objectives are sample-level.

Hypothesis: stronger corruption of the future image reduces the renderer's ability to solve denoising from that image alone and increases useful gradient through the causal forecast. Counterhypothesis: extremely high corruption removes precision or worsens dynamics. Test a single global logit-noise shift, across tasks; measure state-use and action-dependent prediction as well as control. Sampling-noise changes alone are not an academic novelty claim.

Existing flow convention is t=1 pure noise, with logit-normal mean .8/std .8 and automatic shift. Preserve inference sampler and parameters while changing only training corruption. Do not equate losses under different timestep distributions.

Start with 600-update health gates, then task evidence at a predeclared learning budget. Early unlearned control alone is inconclusive. No broad hyperparameter search, seed dropping, or benchmark selection based on favorable results. Record both original benchmark and any stricter evaluations separately.

Sources: https://le-wm.github.io/ ; https://github.com/lucas-maes/le-wm ; https://github.com/galilai-group/stable-worldmodel . Novelty overlaps to examine: PlaNet latent overshooting (https://arxiv.org/abs/1811.04551), Diffusion Autoencoders (https://arxiv.org/abs/2111.15640), generative latent world models and diffusion noise weighting. Do not claim that multi-step generation supervision or denoising representations are new in isolation.

## Frozen execution

`runs/hardtasks_gate600_20261030/plan.json`: 20 training runs (5 arms x2 seeds x2 tasks), each stops exactly at600 updates; 20 full-state/gradient/fixed-noise diagnostics; 4 external random/LeWM strict-validation references. Ten TwoRoom slots use dev2 GPUs2–7 and dev13 GPUs0–3. Cube and reference slots wait for a reviewed closure marker from the remaining PushT controls. No automatic longer training.

Each task has9,000 training,500 validation and500 test episodes; no training labels/state inputs. Fixed50-query validation sets use50 distinct episodes. Cube test100queries use100episodes; TwoRoom test100queries contain60/61distinct episodes after geometry filters, so future uncertainty must cluster by episode and cannot pretend there are100 independent trajectories. The official raw training data are unchanged; the corrected navigation evaluator creates a documented dynamics mismatch when source trajectories contain invalid transitions. This is shared across all matched arms and must be revisited before a generalization claim.

The high-noise arm adds0.8 to the training corruption logit. At fixed noisy input and t→1, the optimal velocity regresses the conditional clean-image mean minus noise. This gives a concrete hypothesis for stronger dependence on predicted state, but also a reason to watch for loss of precise/multimodal dynamics. This limiting argument is an interpretation, not evidence of task improvement or a novelty theorem. Measure the full fixed-noise curve and closed-loop performance.

Source dependencies: stable-worldmodel remains the existing project pin; CLEAR scoring is pinned to14acb4924f6cb9a658b62c9be63efb78e59f1987. Its CLI targets an older planner interface; our adapter uses the existing project planner and only its task/geometry rules. The result is a custom campaign protocol, not an official CLEAR leaderboard submission. HDF5 sorted-index compatibility is corrected in our manifest adapter without changing sampled row identities/order. Original preparation failures remain in EFS.
