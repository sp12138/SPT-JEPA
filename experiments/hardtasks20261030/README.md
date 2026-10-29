# Harder-task research, 2026-10-30

User redirected research from additional PushT expansion to harder tasks based on temporal-forecast, prioritizing effectiveness, a self-contained mechanism, simplicity, substantive novelty and causal understanding. Use dev2/dev13/dev15. Stop unnecessary work with resumable state; long training is authorized once focused gates justify it. Avoid frequent token-consuming polling.

First targets: official OGBench Cube manipulation and TwoRoom navigation. Download only into this project EFS root. Freeze source/data/evaluation identities before comparisons. Released LeWM is an external reference with different parameter/training budgets, not a matched-size baseline. Matched AS-JEPA original and lower-beta controls remain required. All new method objectives are sample-level.

Hypothesis: stronger corruption of the future image reduces the renderer's ability to solve denoising from that image alone and increases useful gradient through the causal forecast. Counterhypothesis: extremely high corruption removes precision or worsens dynamics. Test a single global logit-noise shift, across tasks; measure state-use and action-dependent prediction as well as control. Sampling-noise changes alone are not an academic novelty claim.

Existing flow convention is t=1 pure noise, with logit-normal mean .8/std .8 and automatic shift. Preserve inference sampler and parameters while changing only training corruption. Do not equate losses under different timestep distributions.

Start with 600-update health gates, then task evidence at a predeclared learning budget. Early unlearned control alone is inconclusive. No broad hyperparameter search, seed dropping, or benchmark selection based on favorable results. Record both original benchmark and any stricter evaluations separately.

Sources: https://le-wm.github.io/ ; https://github.com/lucas-maes/le-wm ; https://github.com/galilai-group/stable-worldmodel . Novelty overlaps to examine: PlaNet latent overshooting (https://arxiv.org/abs/1811.04551), Diffusion Autoencoders (https://arxiv.org/abs/2111.15640), generative latent world models and diffusion noise weighting. Do not claim that multi-step generation supervision or denoising representations are new in isolation.
