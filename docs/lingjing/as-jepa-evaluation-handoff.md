# Measuring how latent world models scale: story, protocol and benchmark plan

Written by Lingjing on 2026-11-16 for the collaborator working in the RoboJEPA code base, and for the agents they run. Revision 2 adds three things: the story and contribution, a stage 2 with frozen and fine-tuned options, and a probe of the representation alone.

Literature facts were read on 2026-11-15 and 2026-11-16 from full HTML text, through model-extracted reads. Recheck every number against the PDF before it enters the paper. Claims marked "snippet" were seen only in search results.

## 0. Story and contributions

### 0.1 The problem

Latent world models are said to scale, but there is no agreed way to measure whether a larger or longer-trained world model is a better one. The four measurements in use each fail in a specific way:

- **Planning success with CEM** (V-JEPA 2-AC, DINO-WM, LeWM, RoboJEPA) is slow. V-JEPA 2-AC needs 16 s per action. In the collaborator's run, RoboJEPA's 1B predictor needed 14 minutes for one 18-action Push episode on 32 H200s. CEM success also mixes four things: the encoder, the predictor, the hand-chosen latent distance, and the sampler's noise.
- **Latent prediction error** (RoboJEPA's scaling law) is measured in each model's own latent space, so it cannot be compared across models.
- **Video-quality suites** (WorldScore, EWMBench and similar) apply only to models that generate pixels. They score appearance, not usefulness for control.
- **Policy success of world action models** (Cosmos Policy, V-JEPA Policy, the LeWAMs) mixes the world model with action supervision during pretraining. These systems are no longer pure world models.

The scaling evidence that exists is narrow. RoboJEPA scales its predictor (20M to 8B) on top of one frozen V-JEPA 2.1 encoder and measures planning. V-JEPA 2 scales its encoder and measures video-classification probes, not control. We found no published curve of control quality against pretraining compute under a fixed evaluation for latent world models. Our checks were focused, not an exhaustive search.

### 0.2 The protocol

The protocol is a two-stage evaluation, modelled on how self-supervised vision is evaluated.

**Stage 1, pretraining.** Each model trains with its own world-model objective. Actions may be inputs, and nothing in the objective may predict actions; this is what makes it a world model. A model that predicts actions during pretraining is an action model or a WAM, and enters only as a reference row.

**Stage 2, adaptation.** On fixed, task-specific labelled data, one of two fixed regimes is applied:

- **Frozen:** the pretrained model is frozen and only a probe is trained.
- **Fine-tuned:** the model and the probe are trained together.

This mirrors MAE's protocol (2111.06377), which reports linear probing, end-to-end fine-tuning, and partial fine-tuning of the last `k` blocks. The two regimes can disagree, and the disagreement is itself informative. In MAE's Figure 9, MoCo v3 beats MAE under linear probing, but "all of its partial fine-tuning results are worse than MAE". For ViT-L, MAE goes from 73.5% with linear probing to 81.0% after fine-tuning one block. The authors conclude that "linear separability is not the sole metric for evaluating representation quality."

The adapted model is then measured three ways:

| Measurement | What runs | What it isolates |
| --- | --- | --- |
| **E1: policy alone** | an action head on the model's representation, executed closed-loop in the simulator | whether the representation supports control |
| **E2: representation alone** | probes that predict physical state, from encoded states (E2a) and from the predictor's predicted states (E2b) | what the representation contains, and how accurate the predictions are, measured in physical units that every model shares |
| **E3: policy plus latent planning** | the head proposes 16 action chunks, the predictor rolls each forward, and the one closest to the goal is executed, closed-loop | whether the world model picks actions that make progress; the gain E3 − E1 is the world model's own contribution |

**Reference row: Gaussian CEM** on a small task subset, in the standard JEPA setting. It has two jobs. It connects our results to the published planning literature. It also checks that the cheap measurements rank models the way the expensive standard does.

### 0.3 Contributions

1. **A scientific protocol for measuring the scalability of world models.** It is defined by four properties:
   - a definition-respecting stage 1 (no action prediction);
   - two stage-2 regimes with fixed probe contracts;
   - three measurements that separate the representation, the predictions and planning;
   - a validation against CEM.

   It applies to latent and generative world models alike, because every measurement ends in control success or physical-state error, both of which all models share. It is cheap: no iterative search, except the small CEM subset.
2. **A scaling study using the protocol.** It covers two released series that need no training: RoboJEPA predictors from 50M to 8B, and released V-JEPA 2 / 2.1 encoders of several sizes. It also covers our controlled arms (AS-JEPA, a SIGReg JEPA, a generative video world model and a no-pretraining baseline), all trained on one corpus at several sizes and data fractions.
3. **AS-JEPA,** evaluated within the protocol. Its priced, appearance-free state makes specific predictions the protocol can test. Examples: E2a should find little appearance information but full dynamics information, and the state should keep its control value at small data scales.

### 0.4 Questions the protocol can answer

These are hypotheses to test, not results:

1. **Do representation and prediction scale separately?** In the RoboJEPA series, the encoder is fixed while the predictor grows. E1 and E2a should therefore stay flat while E2b and the E3 gain grow, if the predictor's scaling is real. In our arms, both components scale together, and the protocol shows which one improves.
2. **Do the frozen and fine-tuned regimes disagree?** This is the MAE question asked of world models. For example, generative video features may be weak frozen but strong fine-tuned, and JEPA features the reverse.
3. **Does planning value appear only above a compute threshold?** In RoboJEPA's CEM results, Reach appears near 1e20 FLOPs and Push near 1e22. Does the E3 gain show the same threshold, more cheaply?
4. **Does pretraining reduce the demonstrations needed?** This is tested as E1 success against the fraction of stage-2 labels used.
5. **Do latent and generative world models scale differently per FLOP?**
6. **Do the conclusions depend on the probe?** Readouts are known to flip rankings in representation benchmarks (Section 3.5).

### 0.5 Assessment of the contribution claim

I think the protocol is a real contribution, under three conditions.

**First, a protocol is judged by what it reveals.** Reviewers accept an evaluation protocol when it changes a conclusion or exposes something non-obvious. Examples: the frozen and fine-tuned regimes rank models differently; representation and planning value scale separately; a cheap measurement agrees with CEM at a fraction of the cost. "We defined a protocol" alone is weak. The study must include at least three model families, and the paper should lead with the findings.

**Second, the protocol must be shown to be neutral, because the authors also propose a method.** Mitigations:

- Fix and publish the probe contract before running our own arms.
- Include external checkpoints we did not train (RoboJEPA, V-JEPA 2 / 2.1).
- Release the harness.
- Report cases where AS-JEPA loses.

**Third, the cheap measurements need validating against the accepted standard.** That is the second job of the CEM subset. If E3's ranking of models agrees with CEM's, measured by rank correlation across checkpoints, the protocol can replace CEM for scaling studies. If not, that is a finding to report.

**Framing risk.** One paper with two primary contributions, a protocol and a method, can read as diluted. Leading with the protocol and the scaling findings is more robust: the paper stands even if AS-JEPA does not win. AS-JEPA is then the method whose specific properties the protocol is able to reveal. The alternative is to lead with AS-JEPA and present the protocol as the evaluation that makes its scaling claim checkable. This choice is listed in Section 7.

## 1. The map: dichotomies we discussed

### 1.1 What the pretraining objective outputs

This is the main dividing line. Whether a model sees action labels cannot be the line, because action-conditioned world models all consume actions as inputs during pretraining.

| Family | Pretraining predicts | Actions in pretraining | Members |
| --- | --- | --- | --- |
| Latent world models (JEPA family) | future latent states | input, or none | V-JEPA 2-AC, DINO-WM, LeWM, PLDM, RoboJEPA, **AS-JEPA** |
| Generative video world models | future frames, in pixels or VAE latents | input, or none | Cosmos-Predict, NWM, IRASim, UniSim, Ctrl-World; the video stages of GR-1, VPP, Video Policy and Genie Envisioner |
| Latent generative models with reconstruction | latent states and frames | input | RSSM / Dreamer |
| Action models (policies, VLAs) | actions | output | pi0, pi0.5, OpenVLA, GR00T, Diffusion Policy |
| World action models (WAMs) | frames or latents **and** actions | output | Cosmos Policy, UVA, UWM, Seer, DreamZero, both LeWAMs, V-JEPA Policy, La-JEPA |

AS-JEPA is a latent world model. Its renderer places it near the RSSM row, which matters for positioning but not for evaluation.

### 1.2 Latent versus generative world models

Both kinds predict the future, but in different spaces. A generative video model predicts pixels, so it can be scored on video quality, and its frames can be inspected. A latent world model predicts embeddings, and these cannot be compared across models, because each model learns its own space. Our protocol therefore ends every measurement in a space that all models share: control success (E1, E3) or physical state (E2).

### 1.3 Foundation versus task-specific models

A foundation model is one pretrained checkpoint that serves many tasks, scenes and embodiments with light adaptation. Single-environment models, such as LeWM on PushT, are task-specific. RoboJEPA (23 public datasets, about 15,000 video hours) and the planned AS-JEPA corpus (about 2,000 hours of static-camera robot video) aim to be foundation models. The two-stage protocol tests this directly: one frozen checkpoint with one small probe has to work across RoboCasa365 tasks, LIBERO-Plus perturbations and goal-image tasks.

### 1.4 Planning versus an action head

These are the two ways to get behavior from a world model, and they complement each other. The head proposes plausible actions, and the world model judges which of them leads toward the goal; that pairing is E3.

| | Planning with the predictor | Action head |
| --- | --- | --- |
| Task-specific labels | none; any goal image works | demonstrations |
| Inference cost | CEM: seconds to minutes per action; ranking a few policy proposals: one batched pass | tens of ms per decision |
| What it exercises | encoder, predictor and the latent goal distance | the representation |
| Task specification | goal image | language, goal image or task ID |

The head also covers tasks where planning cannot run. RoboCasa365 and LIBERO-Plus specify tasks in language, with no goal image to score against.

## 2. How each family is evaluated today

### 2.1 Latent world models: planning toward a goal image

V-JEPA 2-AC, DINO-WM, LeWM, PLDM and RoboJEPA all run the same closed loop:

1. Encode the current frame and the goal image.
2. Propose candidate action sequences.
3. Roll each candidate through the predictor.
4. Score each candidate by the latent distance from its predicted final state to the goal.
5. Execute the first actions of the best candidate, observe the next real frame, and repeat.

The candidates come from CEM or MPPI.

| Model | CEM budget | Time |
| --- | --- | --- |
| V-JEPA 2-AC | 800 samples × 10 iterations | 16 s per action |
| DINO-WM | 100 × 10 | 53 s per plan |
| LeWM | 300 × 30 on PushT | |
| RoboJEPA | 1,024 samples, 15 iterations cold and 10 warm, top 20, no policy proposals | |

Terver et al. (2512.24497) compared planners for JEPA world models. They found CEM with an L2 cost best overall, and L2 better than L1 in every setup.

Frozen-feature probes also appear in this literature:

- V-JEPA 2 uses 4-layer attentive probes, not linear ones, for video tasks.
- PSG-JEPA reports linear and MLP probes on frozen latents, scored by Pearson r.
- The collaborator's runs probed physical facts from predicted states against a copy baseline, which copies the last state forward (`experiments/CONCLUSIONS.md`).

### 2.2 Generative video world models: four modes

1. **The same search loop, in pixels or video latents.** NWM runs CEM for navigation, at 30 s per trajectory. In V-JEPA 2's real-robot comparison, Cosmos was planned with CEM at 4 min per action.
2. **Generate a video, then recover actions by inverse dynamics.** UniPi works this way and is weak on manipulation: 0% on LIBERO-10 in Video Policy's comparison.
3. **Feature source for an action head.** This two-stage protocol dominates manipulation benchmarks:
   - VPP puts a diffusion transformer on frozen SVD features. It reaches CALVIN 4.33, against 1.63 without the video pretraining.
   - Video Policy puts a Diffusion Policy head on frozen SVD. On RoboCasa it scores 0.63 two-stage versus 0.57 trained jointly.
   - mimic-video uses a flow head on frozen Cosmos-Predict2 2B.
   - GR-1 and Genie Envisioner follow the same pattern.
4. **Policy evaluator.** WorldGym and Ctrl-World run a policy inside the model to estimate its real success. WorldGym reports per-task Pearson r = 0.78 against real success.

### 2.3 Policies combined with a world model

Ranking or refining a policy's proposals with a world model is established practice.

| Work | What it does | Result |
| --- | --- | --- |
| TD-MPC2 | mixes 24 policy samples into an MPPI population of 512 | |
| NWM | ranks 16 or 32 NoMaD proposals | trajectory error 1.93 → 1.78; standalone CEM was better at 1.13 |
| UVA | ranks 100 diffusion-policy proposals | success 38% → 60%; ranking by the simulator itself reaches 75% |
| Cosmos Policy | picks the best of 8 policy samples by a learned value | +12.5 points on its hardest tasks, at 4.9 s versus 0.61 s |
| LeWAM (Fu et al.) | refines policy samples by gradient steps | 32× faster than CEM |
| LeWAM (Hegde et al.) | plans in the policy's noise space | success on robomimic Can falls (82.8 → 78.8) |

Ranking is far cheaper than CEM, but it can leave performance on the table, and planning around a policy can hurt. Reporting E1 next to E3 makes both effects visible.

### 2.4 Where WAMs and VLAs fit

Action models and WAMs report standard closed-loop success, with a fresh real observation at every decision. A world model inside a policy is used **within** a decision, to predict features or rank proposals, never as a substitute for the next observation.

Because our stage 2 trains a head on benchmark demonstrations, our systems compete on the same leaderboards. WAMs and VLAs therefore appear as labelled **reference rows**, with their published numbers:

- V-JEPA Policy: LIBERO-Plus 91.5.
- La-JEPA: LIBERO-Plus 88.0.
- Cosmos Policy: RoboCasa 67.1, in the original 24-task setting.
- pi0.5.

Scaling conclusions rest only on the controlled arms and the released series of Section 5.

## 3. The protocol in detail

### 3.1 Stage 1

Each model trains with its own objective on its own data:

- **Released series:** RoboJEPA's and V-JEPA 2 / 2.1's published training.
- **Controlled arms:** one shared corpus, with identical resolution, frame rate, action normalization and FLOP accounting. FLOPs count the encoder, predictor, renderer and any tokenizer.

No arm has an action-prediction loss. Save checkpoints at common FLOP points, so that one run gives several points on each curve.

### 3.2 Stage 2 regimes

| Regime | What trains | Applies to | Reason |
| --- | --- | --- | --- |
| **Frozen** | the probe only | E1, E2, E3 | the main regime for every scaling curve; it measures the pretrained model as it is |
| **Partial fine-tuning** | the probe plus the last `k ∈ {1, 4}` encoder blocks | E1 | MAE's bridge between the two endpoints: in MAE, one block already recovers most of the gap |
| **Full fine-tuning** | the probe plus the whole encoder | E1 | the practical ceiling; it shows whether the information is present but not readable by a frozen probe |

**E2 and E3 stay frozen-only**, for two reasons:

- **E3:** fine-tuning the encoder with action labels moves its outputs away from what the frozen predictor was trained to predict, so planning with the fine-tuned encoder is no longer defined. Fine-tuning the predictor as well would need the model's own objective on the task data, which is continued pretraining, not adaptation. That would be a separate study.
- **E2:** a probe on a fine-tuned encoder measures the fine-tuning, not the pretraining.

### 3.3 What the probes read

Every probe reads the model's own representation of a frame:

| Model | Probe input |
| --- | --- |
| AS-JEPA | the 4 state tokens of 16 dims ([design doc](anchored-state-jepa-design.md), encoder row). Learned queries produce them by cross-attending to the ViT's patch tokens, and the predictor predicts them. |
| LeWM / SIGReg arm | its embedding, a single CLS-derived vector in LeWM |
| RoboJEPA, V-JEPA 2 / 2.1 | frozen V-JEPA patch tokens |
| Generative video world model | features of one fixed mid-depth block at one fixed noise level, from a single forward pass |

For AS-JEPA the state tokens are primary: they are the object of the claim, and E3 plans with them. A diagnostic reruns E1 and E2a on the ViT's patch tokens:

- If the patch tokens score much higher, the information price is discarding detail the task needs. That points to a method change (more state tokens, or a spatial grid), not an evaluation change.
- If the scores are about equal, the compact state loses nothing the policy or the probe needs.

### 3.4 Probe contracts

**The policy head (E1, E3)**

Readouts change results by large factors and can reverse rankings:

- Voltron: MVP ViT-S scores 7.90 with a CLS readout and 26.73 with attention pooling, while R3M does better with mean pooling.
- Patch Policy (2607.18236): one encoder scores 1.73 on Cube from patch tokens and 0.21 from a pooled vector.
- DynaMo: the policy head changes which encoder ranks higher on PushT.

The current pattern for world-model policies is a flow-matching action transformer that cross-attends to the frozen tokens, trained with one shared budget (V-JEPA Policy, mimic-video, GE-Act). V-JEPA Policy's encoder comparison keeps "the same predictor and action-expert backbones, downstream demonstrations, global batch size of 128, and 21360 optimizer updates." Cross-attention accepts 4 state tokens, a single vector or a patch grid without pooling.

| Constant | Value | Reason |
| --- | --- | --- |
| token interface | per-model linear projection to the head width, plus a learned positional embedding | V-JEPA Policy |
| `HEAD` | flow-matching DiT, `HEAD_LAYERS = 8`, `HEAD_WIDTH = 384`, cross-attention in every block, AdaLN on flow time | between Patch Policy's head (8 layers, width 256) and V-JEPA Policy's action expert (about 0.1B) |
| `FLOW_STEPS` | 10 Euler steps; noise seeded per episode and per proposal, shared across arms | V-JEPA Policy, VPP |
| `CHUNK`, `EXECUTE` | predict 16 actions, execute 8; DROID-style tasks may need 32 | V-JEPA Policy uses 16 on RoboCasa |
| conditioning | proprioception tokens; language from a frozen T5; goal image through the same frozen model, as appended tokens | mimic-video, V-JEPA Policy, Patch Policy |
| `HISTORY` | current frame plus `HISTORY_FRAMES = 1` earlier frame | identical for every arm |
| training | same demonstrations, batch 128, fixed `HEAD_UPDATES`, same optimizer | V-JEPA Policy's budget rule |
| learning rate | one value, chosen once on validation tasks with frozen V-JEPA 2.1 as the reference, then frozen; check ×0.3 and ×3 on two arms at the largest scale | no per-model tuning (V-JEPA Policy, Voltron, SPA) |
| fine-tuning regimes | same head and budget; encoder learning rate `ENC_LR_SCALE = 0.1` × head rate | to be confirmed on the reference model in Step 1 |
| checkpoint, seeds | final checkpoint; `HEAD_SEEDS = 3` | best-checkpoint selection on test episodes inflates results |

**The state probe (E2)**

| Constant | Value | Reason |
| --- | --- | --- |
| readout | one learned query cross-attending to the model's tokens, then a 2-layer MLP (width 256) | the attentive probe of V-JEPA 2, reduced to one block; handles any token format |
| target | normalized simulator state per factor: robot end-effector pose, object poses, gripper state; PushT agent and block pose; Cube block poses | the same physical units for every model |
| training | real encoded states from the training episodes, fixed updates, fixed learning rate; `PROBE_SEEDS = 3` | |
| E2a metric | R² per factor on held-out episodes | what the representation contains |
| E2b metric | the **same trained probe** applied to predicted states at horizons `h ∈ {1, 4, 16}` along the held-out episodes' logged actions; error per factor, divided by the copy baseline's error (the state at `t`, carried forward) | prediction accuracy in physical units; below 1 means the predictor beats "nothing moves" |

E2b is the one prediction metric that can be compared across models, because its target is physical state rather than each model's own latent. It also penalizes predicted states that drift off the distribution the probe was trained on, which a policy acting on predictions would also suffer from.

### 3.5 Probe variety

A small, deliberate variety, not a zoo of probes. Each variant asks whether a conclusion depends on the probe:

- **Policy head capacity:** at the endpoints of every series, rerun E1 with a lighter head, the E2 readout (one attentive query plus an MLP) feeding the same flow decoder.
- **State probe capacity:** rerun E2a with a linear readout on mean-pooled tokens next to the attentive one.

Report the rank agreement between probe variants across checkpoints (Kendall's tau).

- If the rankings agree, the conclusions are robust to the probe.
- If they disagree, that is an MAE-style finding to report: it says what kind of information each model carries.

### 3.6 E1: policy alone

At every decision:

1. Encode the current real frame.
2. The head outputs a 16-action chunk.
3. Execute 8 actions and observe again.

Continue until the task succeeds or the benchmark's time limit is reached. This is the standard closed-loop evaluation. It runs on every benchmark, in all three regimes.

### 3.7 E3: policy plus latent planning

At every decision:

1. Encode the current real frame and the goal image.
2. The head proposes `N_PROPOSALS = 16` chunks, using 16 fixed seeds.
3. The frozen predictor rolls each full chunk forward, at the predictor's native action grouping.
4. Score each proposal by L2 distance between its predicted end state and the encoded goal.
5. Execute the first 8 actions of the lowest-cost chunk, then observe again.

```
z, g = encode(real_frame), encode(goal_image)
chunks = [head(z, history, g, seed=k) for k in range(16)]
cost = [l2(predictor.rollout(z, c), g) for c in chunks]      # batched, one pass
env.execute(chunks[argmin(cost)][:8])
```

E3 reports three things:

- its success rate;
- **the world-model gain, E3 − E1**, on the same episodes and seeds;
- milliseconds per decision.

Choosing a proposal at random reproduces E1 in expectation, so the gain is exactly what the predictor's ranking adds. A negative gain means the ranking hurts.

Each arm uses its own head and its own predictor, so no latent spaces are compared across arms. For the generative video arm, the rollouts are video predictions, scored on its fixed block's features. Its higher per-decision cost is part of the result.

E3 requires goal images, so it runs on goal-image tasks only.

### 3.8 The CEM reference row

Gaussian CEM, run on RoboJEPA's Reach and Push tasks only, with Terver et al.'s settings:

- `CEM_SAMPLES = 300`, `CEM_ELITES = 10`, `CEM_ITERS = 15`;
- horizon 3 at frameskip 5;
- L2 latent cost;
- replanning after every executed step.

It runs for AS-JEPA, the SIGReg arm and the RoboJEPA series, and for the video arm if the budget allows. RoboJEPA's own checkpoints additionally run under RoboJEPA's published settings, as a reproduction check.

The row has two jobs:

1. connecting our results to the planning literature;
2. **validating the protocol**: the rank correlation, across checkpoints, between CEM success and the E3 gain (and E2b).

### 3.9 Diagnostics and generation

**Within-arm rollout error.** The open-loop latent error along held-out trajectories, as in RoboJEPA's scaling law. It may be plotted against compute within one arm, but never compared across arms.

**The collaborator's linear readout.** `W(E(x_next) − E(x_current))` predicts the action taken from two consecutive real frames; at control time the goal image replaces the next frame. It is a linear inverse-dynamics probe on the encoder alone. Its results so far fit a probe too weak for contact: Reach returns were in CEM's range, strict 2 mm success was 0/8, and Push stayed near 9%. Keep it as the cheapest check. Never give any head the time remaining to the goal: GC-IDM (2605.08732) uses that privileged input (snippet, from a NeurIPS 2026 submission).

**Generation is auxiliary.**

- *Renderer track:* report FVD only in the test that render quality and control success are independent (story prediction 9). Rendered E3 candidates help diagnose failures.
- *Decoder-free track:* a pixel decoder trained on frozen states in stage 2 serves the same diagnostic role, without any pixel loss in pretraining.
- We do not compete on video-quality suites.

## 4. Benchmarks

| Benchmark | Measurements | Stage-2 data | Role | Notes |
| --- | --- | --- | --- | --- |
| PushT and OGBench Cube (LeWM / PSG-JEPA setups) | E1, E2, E3 | published demonstrations; simulator state for E2 | small-scale pilot; comparison with LeWM, PSG-JEPA and both LeWAMs | single-task |
| RoboJEPA RoboCasa goal-image suite: Reach, ObjectReach, ObstacleReach, Push | E1, E2, E3; CEM on Reach and Push | hindsight-relabeled interaction logs (a later frame of the episode is the goal; the action taken is the label); simulator state for E2 | the main world-model test at scale; continuity with RoboJEPA | hold the evaluation kitchens out of relabeling; Push is so far unsolved by everyone |
| RoboCasa365, official leaderboard protocol | E1 in all regimes; E2 on validation episodes | `pretrain_human300`, unaltered, as the leaderboard requires | main generalist benchmark | language tasks, so no E3 |
| LIBERO-Plus | E1 frozen and fully fine-tuned | original LIBERO demonstrations | robustness to camera, lighting, layout and language perturbations | report categories separately; never mix instruction-patch versions |
| LIBERO | E1 | original demonstrations | sanity check | saturated: Cosmos Policy reaches 94.6% from random initialization |
| DROID / RoboArena | E2 on proprioceptive state; RoboArena later | DROID episodes | stretch check in the real world | not a dependency of the core claim |

Freeze the simulator version, task list, horizons, scorer and number of trials before any comparison. Make every choice on validation tasks, keep the final test episodes sealed, and report confidence intervals.

## 5. Model series, arms and figures

### 5.1 Released series (no training needed)

| Series | What scales | Measurements |
| --- | --- | --- |
| RoboJEPA predictors: 50M, 300M, 1B, 2B, 8B | the predictor; the encoder is fixed (V-JEPA 2.1-G) | E1, E2, E3, CEM |
| Released V-JEPA 2 / 2.1 encoders (enumerate the available sizes in Step 2) | the encoder | E1, E2a; E3 only where a released action-conditioned predictor exists (V-JEPA 2-AC) |

### 5.2 Controlled arms (one corpus, several sizes and data fractions)

| Arm | Stage 1 | Question |
| --- | --- | --- |
| AS-JEPA | our objective | the proposed method |
| Scratch | none: the encoder and head are trained end to end on stage-2 data only (E1 only) | does pretraining help at all? |
| SIGReg JEPA | the LeWM / LeWAM objective | the nearest end-to-end JEPA world model |
| Generative video world model | action-conditioned video prediction from random weights | the generative member of the family (VPP and Video Policy are the two-stage precedents) |

The reference rows (Section 2.4) carry their published numbers, each labelled with what it inherits.

### 5.3 Figures and tables

1. **E1 success against stage-1 FLOPs**, frozen and fully fine-tuned, per arm and series. Disagreement between the two regimes is reported as a finding.
2. **E2a R² and E2b error ratio against FLOPs**, per series and arm.
3. **E3 gain and E3 success against FLOPs**, on the goal-image suite.
4. **Partial fine-tuning curve:** E1 against `k` (0, 1, 4, all blocks) at the largest checkpoint of each arm, as in MAE's Figure 9.
5. **Label efficiency:** E1 against the stage-2 label fraction (10%, 30%, 100%).
6. **Protocol validation:** rank correlation of CEM success with the E3 gain and with E2b; probe-variant rank agreement.
7. **Cost table:** ms per decision for E1, E3 and CEM, next to the published per-action times of V-JEPA 2-AC and RoboJEPA.

**Evidence rules:**

- Scaling means a rise across at least four compute points, with confidence intervals separating the endpoints.
- "Scales better" requires a different slope or a better compute frontier; a constant offset is an efficiency gain.
- Latent losses are never compared across arms.

## 6. Execution plan for the agents

Each step lists what it produces, how to tell it worked, and what to do if it did not. Steps 1 and 2 need no new pretraining and can run in parallel after Step 0.

### Step 0: Freeze the contract

**Owners:** Lingjing and the collaborator. Agents do not choose these.

Confirm every constant in Section 3, the benchmark versions in Section 4 and the decisions in Section 7. Write them into one config that every run reads, and publish it before training our own arms.

### Step 1: Stage-2 harness

Build it in the RoboJEPA code base (`github.com/facebookresearch/robo_jepa`, as used on branch `as-jepa/lingjing-fullspec-20261102`).

**What to add:**

- a model adapter with `encode(frames) -> tokens` and `predict(tokens, actions) -> tokens`;
- the head trainer, covering the frozen, partial and full regimes;
- the state-probe trainer;
- the E1 and E3 loops, with shared seeds;
- the E2b rollout probe;
- the CEM row;
- a policy mode next to the CEM mode in the RoboCasa evaluation client.

Reuse the frozen-feature caching and per-case replay in `experiments/robocasa_action_probe/`.

**Success checks:**

- E1 with frozen V-JEPA 2.1 reaches the high 90s on plain LIBERO. V-JEPA Policy reports 98.7, with a larger stage 2.
- E3 with `N_PROPOSALS = 1` reproduces E1 exactly.
- E2b at `h = 0` equals E2a.
- The CEM row with RoboJEPA's 300M checkpoint, under RoboJEPA's settings, reproduces the collaborator's earlier Reach result: 0.887 average normalized return over four held-out kitchens.
- `ENC_LR_SCALE` gives stable full fine-tuning on the reference model.

If any check fails, debug the harness before measuring anything else.

### Step 2: First scaling results from released series

Run every applicable measurement on the RoboJEPA predictor series and the V-JEPA encoder series.

**Expected:**

- In the RoboJEPA series, E1 and E2a stay flat (the encoder is fixed), while E2b, the E3 gain and CEM success improve with predictor size.
- In the V-JEPA series, E1 and E2a improve with encoder size.

Also run E1 on the time-indexed AS-JEPA checkpoints on EFS (`runs/robocasa_action_scaling_20261113`).

**If the expectations fail:**

- *E1 varies across RoboJEPA sizes:* only head noise differs between them, so use that variation to set the confidence intervals.
- *The E3 gain stays flat while CEM rises:* 16 proposals from one head are too similar. Raise `N_PROPOSALS` to 32 for every arm before going on.
- *E3 and CEM rank checkpoints differently:* report it, and investigate before relying on E3 for scaling claims.

This step already produces a reportable result on how released latent world models scale.

### Step 3: Controlled pilot

Train every controlled arm at the two smallest sizes on a nested 10% slice of the stage-A corpus. Then run:

- E1 (frozen and full), E2 and E3 on PushT, Cube and the goal-image suite;
- E1 and E2 on a RoboCasa365 validation subset;
- the patch-token diagnostic of Section 3.3 and the probe variants of Section 3.5.

**Success:** the arms separate, and no benchmark sits at its floor or ceiling.

**If not:**

- *Scratch matches the pretrained arms:* switch to the 10% label fraction.
- *AS-JEPA's patch tokens beat its state by a wide margin:* Lingjing decides whether to change the state before any scaling.

### Step 4: Scaling grid

Four model sizes by three nested data fractions (provisionally 10%, 30% and 100% of unique episodes). Widths and depths come from the throughput measured in Step 1. Expand only once Step 3 is interpretable.

- Save checkpoints at common FLOP points.
- On every checkpoint, run frozen E1 (3 seeds), E2 and E3.
- Run full fine-tuning and the partial fine-tuning curve at the endpoints only.

### Step 5: Headline benchmarks

Run E1 (frozen and full) under the official RoboCasa365 and LIBERO-Plus protocols. Add the reference rows, the CEM validation and the cost table.

### Failure responses that change the story

- **The loss improves with scale, but E1 and E3 do not:** rerun the endpoints with a larger head, and check full fine-tuning, before concluding anything.
- **Frozen E1 is flat, but fine-tuning scales:** report it as an MAE-style finding. The claim becomes "pretraining is a good initialization".
- **E1 and E2a improve, but E2b and the E3 gain do not:** the representation scales but the predictor does not. That is itself a finding about latent world models. Decide the AS-JEPA narrative before Step 4 spends compute.

## 7. Decisions still open

1. **Which contribution leads.** Either the protocol and the scaling findings, with AS-JEPA as the method studied within them (recommended, because it survives if AS-JEPA does not win), or AS-JEPA first, with the protocol as its evaluation.
2. **Headline wording.** It needs approval before use. A protocol-first proposal:

   > We introduce a two-stage protocol for measuring how latent world models scale. Each model is pretrained with its own objective, without predicting actions, and then adapted under fixed probe contracts, frozen or fine-tuned. Three measurements follow: representation probes in physical units, a policy, and policy-guided latent planning, all validated against CEM. Applying it to released JEPA series and to matched JEPA and generative video world models, we find [findings]. Within it, AS-JEPA [result].

3. **Stage-2 data for the goal-image suite:** hindsight-relabeled interaction logs (proposed) or scripted demonstrations.
4. **Compute envelope for Steps 3 and 4,** set from the throughput measured in Step 1.

## Sources

**Evaluation protocols:** MAE [2111.06377](https://arxiv.org/abs/2111.06377); VC-1 / CortexBench [2303.18240](https://arxiv.org/html/2303.18240); Voltron [2302.12766](https://arxiv.org/html/2302.12766); SPA [2410.08208](https://arxiv.org/html/2410.08208); DynaMo [2409.12192](https://arxiv.org/html/2409.12192); Patch Policy [2607.18236](https://arxiv.org/html/2607.18236).

**JEPA world models and planning:** V-JEPA 2 [2506.09985](https://arxiv.org/html/2506.09985); DINO-WM [2411.04983](https://arxiv.org/html/2411.04983); LeWM [2603.19312](https://arxiv.org/html/2603.19312); RoboJEPA [2610.10515](https://arxiv.org/html/2610.10515); Terver et al. [2512.24497](https://arxiv.org/html/2512.24497v3); PLDM [2502.14819](https://arxiv.org/html/2502.14819); GC-IDM [2605.08732](https://arxiv.org/html/2605.08732); PSG-JEPA [2608.06799](https://arxiv.org/html/2608.06799v1).

**Policies with world models:** TD-MPC2 [2310.16828](https://arxiv.org/html/2310.16828); NWM [2412.03572](https://arxiv.org/html/2412.03572); UVA [2503.00200](https://arxiv.org/html/2503.00200); Cosmos Policy [2601.16163](https://arxiv.org/html/2601.16163); LeWAM [2610.12407](https://arxiv.org/html/2610.12407) and [2610.12818](https://arxiv.org/html/2610.12818); WorldGym [2506.00613](https://arxiv.org/abs/2506.00613); Ctrl-World [2510.10125](https://arxiv.org/abs/2510.10125); UniPi [2302.00111](https://arxiv.org/abs/2302.00111).

**Two-stage policies:** VPP [2412.14803](https://arxiv.org/html/2412.14803); Video Policy [2508.00795](https://arxiv.org/html/2508.00795); mimic-video [2512.15692](https://arxiv.org/html/2512.15692); Genie Envisioner [2508.05635](https://arxiv.org/html/2508.05635); GR-1 [2312.13139](https://arxiv.org/abs/2312.13139); V-JEPA Policy [2609.37250](https://arxiv.org/html/2609.37250); La-JEPA [2611.08156](https://arxiv.org/html/2611.08156).

**Collaborator results:** `experiments/CONCLUSIONS.md` and `experiments/robocasa_action_probe/` on branch `as-jepa/lingjing-fullspec-20261102`.
