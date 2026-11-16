# AS-JEPA evaluation handoff: world-model framing, protocols and benchmark plan

Written 2026-11-16 by Lingjing for the collaborator working in the RoboJEPA code base and for the agents they run. This is the agreed evaluation design. It explains how AS-JEPA and its competitors will be evaluated, why, and in what order to build and run it.

Literature facts were read from the papers' full HTML text on 2026-11-15 and 2026-11-16, through reads extracted by a model. Recheck any number against the PDF before it enters the paper. Claims marked "snippet" were seen only in search results.

## 0. Summary

**Decisions** (settled on 2026-11-16):

1. **AS-JEPA is a world model.** During pretraining it consumes actions as inputs and never predicts them. It is not presented as a world action model (WAM).
2. **Evaluation is two-stage.** Stage 1 pretrains each method with its own objective. Stage 2 freezes the pretrained model, trains one fixed action head on it, and evaluates control. The analogy to linear probing in self-supervised learning is only an analogy: the head is a nonlinear policy, but it is identical for every competitor.
3. **Two protocols, both closed-loop in the real simulator,** where a fresh real observation arrives at every decision, as in every standard policy evaluation:
   - **E1, policy alone:** the head acts directly. This tests the representation, and it runs on every benchmark.
   - **E2, policy plus world-model selection:** at every decision the head proposes 16 action chunks, the frozen predictor rolls each one forward, and the chunk whose predicted end state is closest to the goal is executed. This tests whether the world model picks the action that makes progress, which is how JEPA world models are used. It runs on goal-image tasks.
4. **One Gaussian CEM row** on one or two tasks only, for direct comparison with V-JEPA 2-AC and RoboJEPA. CEM is not used anywhere else.

**The claim we want to support:** scaling stage 1 improves E1 success, and improves the gain that the world model adds in E2. No published work shows this curve for a JEPA world model. RoboJEPA's scaling law covers CEM planning only, and V-JEPA 2 scales video-classification probes, not control.

**Reading order:**

| Section | Content |
| --- | --- |
| 1 | the map of model families |
| 2 | how each family is evaluated today |
| 3 | the protocols in full detail |
| 4 | benchmarks |
| 5 | the controlled comparison and the figures |
| 6 | step-by-step plan for the agents |
| 7 | open decisions |

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

Both kinds predict the future, but in different spaces. A generative video model predicts pixels, so it can be scored on video quality (FVD and similar) and its frames can be inspected. A latent world model predicts embeddings. These cannot be compared across models, because each model learns its own space, so latent models are judged by what their latents let you do. Both of our protocols score models by **control success in the real simulator**, which is comparable across both kinds.

### 1.3 Foundation versus task-specific models

A foundation model is one pretrained checkpoint that serves many tasks, scenes and embodiments with light adaptation. Single-environment models, such as LeWM on PushT, are task-specific. RoboJEPA (23 public datasets, about 15,000 video hours) and the planned AS-JEPA corpus (about 2,000 hours of static-camera robot video in stage A) aim to be foundation models. The two-stage protocol tests this claim directly: one frozen checkpoint with one small trained head has to work on RoboCasa365 tasks, LIBERO-Plus perturbations and goal-image tasks.

### 1.4 Planning versus an action head

These are the two ways to get behavior from a world model. They are complementary, not competing.

| | Planning with the predictor | Action head |
| --- | --- | --- |
| Task-specific labels | none; any goal image works | demonstrations |
| Inference cost | CEM: seconds to minutes per action; ranking a few policy proposals: one batched pass | tens of ms per decision |
| What it exercises | encoder, predictor and the latent goal distance | the representation |
| Task specification | goal image | language, goal image or task ID |

The head proposes plausible actions, and the world model judges which of them leads toward the goal. E2 is exactly this pairing. The head also covers tasks that planning cannot. RoboCasa365 and LIBERO-Plus give tasks in language, with no goal image to score against, so a world model with only a latent goal distance cannot plan there.

## 2. How each family is evaluated today

### 2.1 Latent world models: planning toward a goal image

V-JEPA 2-AC, DINO-WM, LeWM, PLDM and RoboJEPA all run the same closed loop at every decision:

1. Encode the current frame and the goal image.
2. Propose candidate action sequences.
3. Roll each candidate through the predictor.
4. Score each candidate by the latent distance from its predicted final state to the goal.
5. Execute the first actions of the best candidate, observe a new real frame, and repeat.

The candidates come from CEM or MPPI.

| Model | CEM budget | Time |
| --- | --- | --- |
| V-JEPA 2-AC | 800 samples × 10 iterations | 16 s per action |
| DINO-WM | 100 × 10 | 53 s per plan |
| LeWM | 300 × 30 on PushT | not recorded here |
| RoboJEPA | 1,024 samples, 15 cold / 10 warm iterations, top 20, no policy proposals | 14 min for one 18-action Push episode in the collaborator's measurement (1B predictor, 32 H200s) |

Terver et al. (2512.24497) compared planners for JEPA world models and found CEM with an L2 cost best overall, and L2 better than L1 in every setup they tested. V-JEPA 2 also reports frozen-feature probes, using 4-layer attentive probes rather than linear ones.

### 2.2 Generative video world models: four modes

1. **The same search loop**, in pixels or video latents. NWM uses CEM in navigation, at 30 s per trajectory. In V-JEPA 2's real-robot comparison, Cosmos was planned with CEM at 4 min per action.
2. **Generate a video of the task, then recover actions with inverse dynamics.** UniPi works this way, but it is weak on manipulation: 0% on LIBERO-10 in Video Policy's comparison.
3. **Feature source for an action head**, the two-stage protocol. This is the dominant mode on manipulation benchmarks:
   - VPP: frozen SVD features feed a diffusion transformer. CALVIN 4.33, and 1.63 without the video pretraining.
   - Video Policy: frozen SVD with a Diffusion Policy head. RoboCasa 0.63 two-stage versus 0.57 joint.
   - mimic-video (frozen Cosmos-Predict2 2B with a flow head), GR-1 and Genie Envisioner.
4. **Policy evaluator.** WorldGym and Ctrl-World run a policy inside the model to estimate its real success. WorldGym reports a per-task Pearson r of 0.78 against real success.

### 2.3 Policies combined with a world model

Ranking or refining a policy's proposals with a world model is established practice:

| Work | What it does | Result |
| --- | --- | --- |
| TD-MPC2 | mixes 24 policy samples into an MPPI population of 512 | |
| NWM | ranks 16 or 32 NoMaD proposals | trajectory error 1.93 → 1.78; standalone CEM was better at 1.13 |
| UVA | ranks 100 diffusion-policy proposals | success 38% → 60%; ranking by the simulator itself reaches 75% |
| Cosmos Policy | picks the best of 8 policy samples with a learned value | +12.5 points on its hardest tasks, at 4.9 s versus 0.61 s |
| LeWAM (Fu et al.) | refines policy samples by gradient steps | 32× faster than CEM |
| LeWAM (Hegde et al.) | plans in the policy's noise space | lowered success on robomimic Can (82.8 → 78.8) |

Two lessons follow. Ranking policy proposals is far cheaper than CEM but can leave performance on the table. Planning around a policy can also hurt. Reporting the policy alone next to the policy plus selection makes both effects visible.

### 2.4 Where WAMs and VLAs fit

All action models and WAMs report standard closed-loop success: a fresh real observation at every decision, for the whole episode. A world model inside a policy is used **within** a decision, to predict future features or to rank proposals. It never replaces the next observation.

Once our stage 2 trains a head on benchmark demonstrations, our system is a policy on the same leaderboards. So WAMs and VLAs appear as labelled **reference rows**:

- V-JEPA Policy: LIBERO-Plus 91.5.
- La-JEPA: LIBERO-Plus 88.0.
- Cosmos Policy: RoboCasa 67.1, in the original 24-task setting.
- pi0.5.

The scaling claim rests only on the controlled arms of Section 5, because the reference systems differ from ours in data, initialization and objective.

## 3. The protocols

### 3.1 What the head reads

The head reads each model's own representation of a frame:

| Model | Head input |
| --- | --- |
| AS-JEPA | the 4 state tokens of 16 dims (see [the design doc](anchored-state-jepa-design.md), encoder row). Learned queries produce them by cross-attending to the ViT's patch tokens, and the predictor predicts them. |
| LeWM / SIGReg arm | its embedding (a single vector derived from the CLS token in LeWM) |
| RoboJEPA | frozen V-JEPA 2.1-G patch tokens |
| Generative video world model | features from one fixed mid-depth block at one fixed noise level, computed in a single forward pass on the frame |

For AS-JEPA, the state tokens are primary because they are the object of the claim, and because E2 plans with them, so E1 and E2 test the same representation. The ViT's patch tokens are also available. A diagnostic in Step 3 reruns E1 with the head reading them:

- If the patch tokens score much higher, the information price discards control-relevant detail. Changing that would be a method decision (more state tokens, or a spatial grid), not an evaluation change.
- If they score about the same, the compact state loses nothing the policy needs.

### 3.2 The action head contract

The way a head reads backbone features changes results by large factors and can reverse rankings:

- In Voltron, MVP ViT-S scores 7.90 from its CLS readout and 26.73 with attention pooling, while R3M does better with mean pooling.
- In Patch Policy (2607.18236), one encoder scores 1.73 on Cube from patch tokens and 0.21 from a pooled vector.
- In DynaMo, changing the policy head changes which encoder ranks higher on PushT.

The older default, a small MLP on a single vector (VC-1 / CortexBench, R3M), favors single-vector models. The current pattern for world-model policies is a **flow-matching action transformer that cross-attends to the frozen backbone's tokens**, with one shared budget (V-JEPA Policy, mimic-video, GE-Act). Cross-attention accepts any number of tokens, so 4 state tokens, one vector and a patch grid all enter without pooling.

The closest precedent is V-JEPA Policy's encoder comparison. It keeps "the same predictor and action-expert backbones, downstream demonstrations, global batch size of 128, and 21360 optimizer updates" and changes only the projection specific to each encoder.

| Constant | Value | Reason |
| --- | --- | --- |
| token interface | per-model linear projection to the head width, plus a learned positional embedding | V-JEPA Policy; nothing is pooled away |
| `HEAD` | flow-matching DiT, `HEAD_LAYERS = 8`, `HEAD_WIDTH = 384`, cross-attention to the backbone tokens in every block, AdaLN on flow time | between Patch Policy's head (8 layers, width 256) and V-JEPA Policy's action expert (about 0.1B) |
| `FLOW_STEPS` | 10 Euler steps; initial noise seeded per episode and per proposal, shared across competitors | V-JEPA Policy, VPP; shared seeds remove sampling differences between arms |
| `CHUNK`, `EXECUTE` | predict 16 actions, execute 8, then decide again; DROID-style tasks may need 32 | V-JEPA Policy uses 16 on RoboCasa; Diffusion Policy-style heads use 8 to 10 |
| conditioning | proprioception as tokens; language from a frozen T5 (a frozen text encoder is allowed); goal images through the same frozen backbone, appended as tokens | mimic-video, V-JEPA Policy, Patch Policy |
| `HISTORY` | the current frame plus `HISTORY_FRAMES = 1` earlier frame | the minimum that shows velocity; identical for every arm |
| training | same demonstrations, batch 128, a fixed `HEAD_UPDATES`, same optimizer | V-JEPA Policy's rule for matching budgets |
| learning rate | one value, chosen once on validation tasks with frozen V-JEPA 2.1 as the reference model, then frozen for everyone; a ×0.3 / ×3 sensitivity check on two arms at the largest scale | no per-model tuning (V-JEPA Policy, Voltron, SPA); the check catches a rate that suits only one arm |
| checkpoint and seeds | final checkpoint; `HEAD_SEEDS = 3` | selecting the best checkpoint on test episodes inflates results |
| fine-tuned variant | E1 only: the same head and budget with the model unfrozen, reported separately | shows the practical ceiling; it does not carry the scaling claim, because fine-tuning washes out pretraining differences |

### 3.3 E1: policy alone

At every decision:

1. Encode the current real frame.
2. The head outputs a 16-action chunk.
3. Execute 8 actions, observe the next real frame, and repeat until the task succeeds or the benchmark's time limit is reached.

This is the standard closed-loop evaluation that every policy paper reports. It runs on every benchmark and carries the main scaling curve. It measures the representation. For arms whose encoder was trained jointly with a predictor (AS-JEPA, SIGReg JEPA, the video arm), it also reflects what world-model training did to that representation. It never uses the predictor's forecasts, so it cannot see predictor quality. For example, all RoboJEPA sizes share one frozen encoder, so E1 scores their predictors identically.

### 3.4 E2: policy plus world-model selection

At every decision:

1. Encode the current real frame and the goal image with the frozen encoder.
2. The head proposes `N_PROPOSALS = 16` chunks of 16 actions, using 16 fixed noise seeds.
3. The frozen predictor rolls each full chunk forward from the current state, using the predictor's native action grouping. This gives one predicted end state per proposal.
4. Score each proposal by the L2 distance between its predicted end state and the encoded goal, in the model's own latent space.
5. Execute the first 8 actions of the lowest-cost proposal, observe the next real frame, and repeat.

```
z, g = encode(real_frame), encode(goal_image)
chunks = [head(z, history, g, seed=k) for k in range(16)]      # 16 x 16 actions
cost = [l2(predictor.rollout(z, c), g) for c in chunks]         # batched, one pass
env.execute(chunks[argmin(cost)][:8])                          # then observe again
```

**What E2 reports:**

- E2 success.
- **The world-model gain, E2 minus E1**, on the same episodes and seeds. Picking a proposal at random is E1 in expectation, so the gain is exactly what the predictor's ranking adds. It also exposes harm: a negative gain means the ranking picks worse actions than the policy alone does.
- Milliseconds per decision.

**Why this is standard, fair and efficient:**

- **Standard.** It is the JEPA world-model planning loop of Section 2.1, with candidates supplied by the policy, as in NWM, UVA and Cosmos Policy.
- **Fair.** Each arm uses its own head and its own predictor, under identical seeds, budget and cost. The gain is measured within each arm, so no latent space is compared with another.
- **Efficient.** It costs 16 batched predictor rollouts per decision, with no iterations.

For the generative video arm, the rollouts are video predictions, and the cost is computed on the features of its fixed block. Its higher per-decision time is reported as part of the result.

E2 needs a goal image, so it runs on the goal-image tasks of Section 4. Language tasks use E1 only.

### 3.5 The CEM continuity row

Run Gaussian CEM with Terver et al.'s settings on RoboJEPA's Reach and Push tasks only:

- `CEM_SAMPLES = 300`, `CEM_ELITES = 10`, `CEM_ITERS = 15`;
- horizon 3 at frameskip 5;
- L2 latent cost;
- receding horizon: replan after each executed step.

Run it for AS-JEPA, the SIGReg arm and RoboJEPA, and for the video arm only if the budget allows. Report success and seconds per decision. This row makes our numbers directly comparable to the published planning results of V-JEPA 2-AC and RoboJEPA. For RoboJEPA's own checkpoints, also report its published settings as a reproduction check.

### 3.6 Diagnostics (not headline results)

- **Within-arm rollout error.** The L1 or L2 error of open-loop predictions along held-out trajectories, as in RoboJEPA's scaling law. It can be plotted against compute within one arm, but never compared across arms.
- **The collaborator's linear readout `W(E(x_next) − E(x_current))`.** It predicts the action taken from two consecutive real frames, and at control time the goal image replaces the next frame. It is a linear inverse-dynamics probe on the encoder; it never calls the predictor. The collaborator's results fit a probe too weak for contact: Reach returns were in CEM's range, but strict 2 mm success was 0/8, and Push stayed near 9%. Keep it as the cheapest check, which E1 supersedes.

The nonlinear version of that readout is GC-IDM (2605.08732). One caveat (snippet, a NeurIPS 2026 submission): GC-IDM is given the true time-to-goal, which is privileged information. Never give our head that input.

### 3.7 Generation

Generation is auxiliary. For the renderer track:

- Report matched-case videos and FVD only inside the test of whether render quality and control success are largely independent (story prediction 9).
- The renderer can also show the predicted end states that E2 chose between, which helps diagnose failures.

For the decoder-free track, a pixel decoder trained on frozen states in stage 2 serves the same purpose, with no pixel loss in pretraining. We do not compete on video-quality suites.

## 4. Benchmarks

| Benchmark | Protocols | Stage-2 data | Role | Notes |
| --- | --- | --- | --- | --- |
| PushT and OGBench Cube (LeWM / PSG-JEPA setups) | E1, E2 | published demonstration sets | small-scale pilot; direct comparison with LeWM, PSG-JEPA and both LeWAMs | single-task, so not foundation evidence |
| RoboJEPA RoboCasa goal-image suite: Reach, ObjectReach, ObstacleReach, Push | E1, E2, CEM row (Reach and Push) | hindsight-relabeled interaction logs: a later frame of the same episode is the goal, and the action taken is the label | the world-model test at scale; continuity with RoboJEPA | hold the evaluation kitchens out of relabeling; never give the head the time to goal; Push is so far unsolved by everyone |
| RoboCasa365 (official leaderboard protocol) | E1 frozen and fine-tuned | `pretrain_human300` demonstrations, unaltered, as the leaderboard requires | main generalist benchmark for the scaling curve | tasks are in language, so E1 only |
| LIBERO-Plus | E1 | original LIBERO demonstrations | robustness to camera, lighting, layout and language perturbations | report categories separately; never mix instruction-patch versions |
| LIBERO | E1 | original demonstrations | sanity check only | saturated: Cosmos Policy reaches 94.6% from random initialization |
| DROID / RoboArena | offline action error first; RoboArena later | DROID episodes | stretch check in the real world | not a dependency of the core claim |

For every benchmark, freeze the simulator version, task list, horizons, scorer and number of trials before comparisons start. Use validation tasks for every choice, keep the final test episodes sealed, and report confidence intervals.

## 5. The controlled comparison and the figures

### 5.1 Arms

All controlled arms train on the same corpus with the same:

- resolution and frame rate;
- action normalization;
- FLOP accounting, which includes the encoder, predictor, renderer and any tokenizer.

All are evaluated with the identical E1 and E2.

| Arm | Stage 1 | Question it answers |
| --- | --- | --- |
| AS-JEPA | our objective | the proposed method |
| Scratch | none: the same encoder and head trained end to end on the stage-2 data only; E1 only, since it has no predictor | does pretraining help at all? |
| SIGReg JEPA | the LeWM / LeWAM objective, with the same backbone and corpus | the nearest end-to-end JEPA world model |
| Generative video world model | action-conditioned video prediction from random weights, on the same corpus | the generative member of our family under matched conditions; VPP and Video Policy are the published two-stage precedents |
| RoboJEPA released checkpoints | frozen V-JEPA 2.1-G with predictors from 50M to 8B, on RoboJEPA's own corpus | external pretrained JEPA reference, with inherited data disclosed |
| Reference rows | published numbers of V-JEPA Policy, La-JEPA, Cosmos Policy and pi0.5, plus the planning results of V-JEPA 2-AC and RoboJEPA | practical context only, each labelled with what it inherits |

### 5.2 Figures and tables

1. **E1 success against stage-1 training FLOPs:** one curve per arm, one point per saved checkpoint and model size, on RoboCasa365 subsets and LIBERO-Plus categories.
2. **E2 success and the world-model gain (E2 minus E1) against stage-1 FLOPs,** on the RoboCasa goal-image suite. Together with figure 1, this separates the two halves of the claim: a better representation (figure 1) and a better predictor (the gain).
3. **Label efficiency:** E1 success against the fraction of stage-2 demonstrations (10%, 30%, 100%), for a few fixed checkpoints.
4. **Cost table:** milliseconds per decision for E1, E2 and the CEM row per arm, next to the published per-action times of V-JEPA 2-AC and RoboJEPA.

What counts as evidence:

- **Scaling:** success, or the gain, rises across at least four compute points, with confidence intervals separating the endpoints, and the gap to the scratch arm does not shrink.
- **"Scales better than" another arm:** a different slope or a better compute frontier. A constant offset is an efficiency gain, not better scaling.
- **Latent losses are never compared across arms.**

## 6. Execution plan for the agents

Each step says what it produces, how to tell it worked, and what to do if it did not. Steps 1 and 2 need no new pretraining and can run in parallel once Step 0 is done.

### Step 0: freeze the contract

Owners: Lingjing and the collaborator; agents do not choose these. Confirm every constant in Section 3, the benchmark versions in Section 4 and the decisions in Section 7. Write them into one config file that every run reads.

### Step 1: build the stage-2 harness

Build it in the RoboJEPA code base (`github.com/facebookresearch/robo_jepa`, as already used on the collaborator's branch `as-jepa/lingjing-fullspec-20261102`).

- Add a model adapter with two calls, `encode(frames) -> tokens` and `predict(tokens, actions) -> tokens`, so that every arm plugs in unchanged.
- Add the head trainer, the E1 loop, the E2 loop (proposals, batched rollouts, L2 selection, shared seeds) and the CEM row, with Terver et al.'s settings as defaults.
- Add a policy mode next to the existing CEM mode in the RoboCasa evaluation client.
- Reuse the frozen-feature caching and per-case replay from `experiments/robocasa_action_probe/`.

Success checks:

- E1 on frozen V-JEPA 2.1 reaches the high 90s on plain LIBERO. V-JEPA Policy reports 98.7, with a larger stage 2.
- E2 with `N_PROPOSALS = 1` reproduces E1 exactly, which catches bugs in the loop.
- The CEM row with RoboJEPA's 300M checkpoint reproduces the collaborator's earlier Reach result: an average normalized return of 0.887 over four held-out kitchens, under RoboJEPA's settings.

If any check falls short, debug the harness before measuring anything else, because every comparison inherits its bias.

### Step 2: check that E1 and E2 separate the encoder from the predictor

Use existing checkpoints. Run E1, E2 and the CEM row on the released RoboJEPA predictors (50M, 300M, 1B, 2B, 8B) on the goal-image suite. Also run E1 on the time-indexed AS-JEPA checkpoints already on EFS (`runs/robocasa_action_scaling_20261113`).

- **Expected:** E1 stays flat across RoboJEPA sizes, since they share one encoder and so one head input, while the E2 gain grows with predictor size, as RoboJEPA's CEM success does.
- **If E1 varies across sizes:** only the head seed differs, so the spread is head noise. Use it to set the confidence intervals.
- **If the E2 gain stays flat while CEM success rises:** 16 proposals from one head are too similar for ranking to matter. Raise `N_PROPOSALS` to 32, or raise the head's sampling temperature, once for every arm, before using E2 on our arms.

### Step 3: small controlled pilot

Train every controlled arm at the two smallest sizes on a nested 10% slice of the stage-A corpus. Run E1 and E2 on PushT, Cube and the goal-image suite, and E1 on a RoboCasa365 validation subset. Include the diagnostic of Section 3.1: E1 with AS-JEPA's patch tokens.

- **Success:** the arms separate, and no benchmark sits at its floor or ceiling.
- **If scratch matches the pretrained arms:** stage-2 data is too plentiful for pretraining to matter. Move to the 10% demonstration fraction.
- **If AS-JEPA's patch tokens beat its state by a wide margin:** the information price removes control-relevant detail. Lingjing decides, before any scaling, whether to change the state.

### Step 4: the scaling grid

Four model sizes by three nested fractions of the stage-A corpus (provisionally 10%, 30% and 100% of unique episodes). The widths and depths follow the throughput measured in Step 1. Expand from the two-size pilot only once Step 3 is interpretable.

- Save checkpoints at common FLOP points.
- On every checkpoint, run E1 with three head seeds, plus E2.
- Record FLOPs, unique hours and frames seen.

### Step 5: headline benchmarks and reference rows

Run E1, frozen and fine-tuned, under the official RoboCasa365 and LIBERO-Plus protocols. Add the reference rows and the cost table.

### Failure responses that change the story

- **E1 and E2 do not improve with scale while the pretraining loss does:** the head may be the bottleneck. Rerun the endpoints with a larger head, and check the fine-tuned variant, before concluding anything.
- **E1 is flat while fine-tuning scales:** the information is present but not readable by a frozen head. Report this; the claim weakens to "pretraining is a good initialization".
- **E1 improves with scale but the E2 gain does not:** pretraining improves the representation but not the predictor's judgement of actions, which removes the world-model half of the claim. Decide the narrative before Step 4 spends compute.

## 7. Decisions still open

1. **Headline wording**, which needs Lingjing's approval before use. A proposal:

   > AS-JEPA is a world model whose visual encoder, latent dynamics and renderer are trained end to end from random initialization on public robot video. With each pretrained model frozen and the same action head trained on it, both the policy's success and the gain from letting the world model choose among the policy's proposals rise with pretraining compute, more efficiently than for a generative video world model and a regularized JEPA trained on the same data.

2. **Stage-2 data for the goal-image suite:** hindsight-relabeled interaction logs (proposed) or scripted demonstrations.
3. **Compute envelope for Steps 3 and 4,** set by the executor from the throughput measured in Step 1.

## Sources

JEPA world models and planning: V-JEPA 2 [2506.09985](https://arxiv.org/html/2506.09985); DINO-WM [2411.04983](https://arxiv.org/html/2411.04983); LeWM [2603.19312](https://arxiv.org/html/2603.19312); RoboJEPA [2610.10515](https://arxiv.org/html/2610.10515); Terver et al. [2512.24497](https://arxiv.org/html/2512.24497v3); PLDM [2502.14819](https://arxiv.org/html/2502.14819); GC-IDM [2605.08732](https://arxiv.org/html/2605.08732); PSG-JEPA [2608.06799](https://arxiv.org/html/2608.06799v1).

Policies combined with world models: TD-MPC2 [2310.16828](https://arxiv.org/html/2310.16828); NWM [2412.03572](https://arxiv.org/html/2412.03572); UVA [2503.00200](https://arxiv.org/html/2503.00200); Cosmos Policy [2601.16163](https://arxiv.org/html/2601.16163); LeWAM [2610.12407](https://arxiv.org/html/2610.12407) and [2610.12818](https://arxiv.org/html/2610.12818); WorldGym [2506.00613](https://arxiv.org/abs/2506.00613); Ctrl-World [2510.10125](https://arxiv.org/abs/2510.10125); UniPi [2302.00111](https://arxiv.org/abs/2302.00111).

Two-stage policies and action heads: VPP [2412.14803](https://arxiv.org/html/2412.14803); Video Policy [2508.00795](https://arxiv.org/html/2508.00795); mimic-video [2512.15692](https://arxiv.org/html/2512.15692); Genie Envisioner [2508.05635](https://arxiv.org/html/2508.05635); GR-1 [2312.13139](https://arxiv.org/abs/2312.13139); V-JEPA Policy [2609.37250](https://arxiv.org/html/2609.37250); La-JEPA [2611.08156](https://arxiv.org/html/2611.08156); VC-1 [2303.18240](https://arxiv.org/html/2303.18240); Voltron [2302.12766](https://arxiv.org/html/2302.12766); SPA [2410.08208](https://arxiv.org/html/2410.08208); DynaMo [2409.12192](https://arxiv.org/html/2409.12192); Patch Policy [2607.18236](https://arxiv.org/html/2607.18236).

Collaborator results: `experiments/CONCLUSIONS.md` and `experiments/robocasa_action_probe/` on branch `as-jepa/lingjing-fullspec-20261102`.
