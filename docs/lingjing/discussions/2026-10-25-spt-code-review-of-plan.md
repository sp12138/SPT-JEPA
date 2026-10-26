# Discussion log: reviewing the AS-JEPA plan against the SPT code (2026-10-25)

The user asked for a review of the implementation plan now that the real SPT code is in this repository, and for the plan to be updated where the code makes it necessary.

What was read: `runtime/vforge/models/rep_world/` (model, split conditioner, World DiT, pixel codec and refiner, config), `runtime/vforge/tasks/rep_world/` (data, train, EMA, optimizer, shared config), `jobs/configs/pretrain_50k.yaml`, the hash checks in `jobs/worker.py` and `tests/test_portability.py`, and the PushT tooling in `~/code/sparse_jepa`.

Main findings (details in [as-jepa-spt-implementation-plan.md](../as-jepa-spt-implementation-plan.md), Section 1):

- SPT already splits its encoder into a per-state ViT-E and a state-causal ViT-C, so the AS-JEPA encoder and prior map onto existing components.
- SPT masks 90% of patches before ViT-E, not after the ViT; reusing ViT-E as the encoder requires masking off.
- SPT's DiT cross-attends to spatially aligned tokens of the latest history state, which is the recent-anchor bypass AS-JEPA avoids.
- A chunk is a multi-frame state (16 frames in RGB16, 4 in RGB4); the pixel refiner hard-codes 4 frames per plane at 256x256.
- SPT's EMA is an evaluation-only weight average, not a target network; its noise schedule is already high-noise biased; it already has condition dropout for CFG.
- No actions exist anywhere in SPT; the RGB16 worker refuses to run if pinned runtime files change.
- SPT's flow loss is not an exact likelihood, so `beta = 1` has no special meaning here.

Plan changes made: new file `as-jepa-spt-implementation-plan.md`; design doc edits to the objective caveat, the PushT starting-point pointer, the noise-weighting note, the masking paragraph, the SPT relation, and the decisions list.

Open for the user: compute placement (this host is a login node; dev3/dev8 belong to the SPT maintainers), RGB4 versus a new one-frame `rgb1` geometry, and wrapper-subclass versus editing pinned files for the anchor route.

## Revision after the user's follow-up (same day)

The user pointed out that the plan had not considered all major points of the design session's final answer (Happier session `cmvmnyra9gqc1qq5775z6t4j9`, seq 144). Re-reading it against the plan showed these gaps, now fixed in the plan:

- I had switched chunks to 4-frame RGB4 states for code convenience. That broke the design's argument that renderer noise is only the within-frame residual, the identifiability step that renders frames independently given state and anchor, and single-image goal encoding. It was also unnecessary: `WorldDiT` with its plain output layer renders one 64 px frame packed as 4x4 patches (48 channels per token) without touching pinned files. Reverted to one-frame chunks at frameskip 5.
- I had folded entry and surprise into one `beta`; the design's three-line form has separate prices for entry and surprise. Restored.
- The alternative codes (learned-variance Gaussian, FSQ) were missing; the learned-variance code is needed for the component-wise identifiability bonus. Added a `StateCode` interface.
- Missing: the guidance-scale reliance diagnostic, the swap test, rate-based leak and collapse signals, the dot and contact checks, the no-anchor and MSE-renderer ablations, generation with prior samples and both continuation modes, data conditions required by the theory (colours sampled independently of initial state, dot i.i.d.), and the camera-motion anchor route. Added, with a design-to-plan traceability table (plan Section 2).
- The masking disagreement was left implicit; the plan now proposes one measured comparison.

## Story and first experiments (user question, same day)

User asked whether an "image editing is world prediction" framing changes the design, how SPT, editing architectures and the theory fit together, whether to start with PushT planning or SPT's generation/FVD evaluation, and which benchmarks JEPA world-model papers use.

Answer given (not yet written into the design doc):

- The framing fits if stated as "the world model predicts the edit instruction; the editor executes it". The predictor stays in latent space (JEPA), and reconstruction only grounds what an instruction means. Recent editing-as-world-model papers (ImageWAM, Orca, ChronoEdit, EditWorld) edit from the *current* frame with the action as instruction, which is pixel-space prediction and the recent-anchor bypass; AS-JEPA edits from a fixed first frame with a learned, priced instruction. A literal "Image Editing is World Prediction" title would invite that confusion.
- Recommended order: PushT with the variation space, generation and editing visualizations first (rendered latent rollouts, counterfactual actions, anchor/state swap), then LeWM-protocol planning (PushT, TwoRoom, Reacher, OGBench-Cube, all present in `~/code/sparse_jepa/stable_worldmodel/envs`). SPT's SolarWM/FVD evaluation later, as a scale and generation-cost check, since it is action-free with camera motion.
- New consideration from the search ("Frozen in a Frame", arXiv 2610.04585): single-frame encoders miss velocity. With one-frame chunks the state is not Markov in velocity; the prior can infer it from history, but the identifiability assumption that the frame determines the state weakens. Bears on the chunk-length knob (two-frame chunks are the minimal fix).
- Benchmarks: LeWM suite (TwoRoom, Reacher, PushT, OGBench-Cube), extended suites (Wall, PointMaze, Scene, PushObj, Ant-U-Maze), V-JEPA 2 (SSv2, Epic-Kitchens, DROID real-robot planning), intuitive physics (IntPhys, IntPhys 2, InfLevel; Physion). The per-step surprise rate is a natural violation-of-expectation score for IntPhys-style tests.

## Consolidation: information story first, editing as interpretation (user request, same day)

The user prefers the information-theoretic narrative from the design session, especially that the JEPA latent prediction loss emerges from the KL chain, and asked whether the editing framing changes the implementation, how SPT, JEPA and editing pieces integrate, and to re-consolidate story, implementation, experiments and theory while keeping block identifiability.

Decided and written down:

- Story: design doc Section 4 now leads with the derivation (4.1): predictive coding of noisy states with the first frame as decoder side information; with fixed variances the per-step code length is LeWM's attached-target squared error; the entry term contains SIGReg's role (marginal matching) plus the information price SIGReg omits; the renderer's distortion is what prevents collapse. Editing is Section 4.3, an interpretation, valid only with a fixed reference frame.
- Implementation: the editing framing adds no module and no pretrained weights. Stages 1-2 use LeWM's harness, encoder (ViT-tiny plus query head) and predictor (residual mean), with SPT's `WorldDiT` as the renderer (runs outside SPT's trainer after `initialize_ops`, as SPT's CPU tests do). The paper's comparison becomes a literal diff from LeWM. Stage 3 uses SPT's own ViT-E, causal blocks, pixel DiT and Native trainer. LeWM's BatchNorm heads must be replaced on `mu` and `delta`, since batch normalization would prevent the entry price from switching dimensions off.
- Theory: block identifiability unchanged; design 5.1 now allows lag-`L` transitions so velocity lives in the prior's context and A1 holds per frame.
- Experiments: same data and order; Section 9 of the design doc is now a ladder LeWM -> LeWM + anchored renderer (SIGReg kept) -> AS-JEPA -> AS-JEPA without anchor -> AS-JEPA with MSE renderer, which isolates the price as what removes appearance.

## Story document and theory check (2026-10-26)

The user approved the plan, story and design and asked to document the whole story carefully, theory first, and commit before discussing the empirical plan. Written: `as-jepa-story.md`, now the canonical statement (README item 0; design notes point to it).

Checking the identifiability sketch step by step while writing changed the theory, which the user should review:

- The design notes' matching step does not hold: with an anchor-blind state chain the model must pay for the first state and can at best match the frames' conditional distribution given the appearance, not given the anchor.
- Replacement (exact, Proposition 1): with optimal prior and renderer, the objective equals the entry cost minus the sum over frames of `I(s_t; past states, actions) - I(s_t; anchor)`. The objective rewards information that the past predicts and the anchor does not, which formalizes the user's original intuition from the first design round.
- Appearance is excluded for every `beta > 0` with an exact gap (Proposition 2); transient detail for `beta >= 1`, strictly at `beta = 1` because the shared encoder also encodes the never-rendered anchor (Proposition 3).
- Dynamics inclusion needs a per-window value condition (A5): the entry cost is paid per training window, so window length, contact frequency and anchor distance matter. Static dynamics-relevant factors such as block shape are kept only through their value for prediction. This raises an open design question: an anchor drawn from a distant time in the same episode would strengthen the value of slow components.

## Empirical plan decisions (2026-10-26)

Context: the plan will be handed to a collaborator on another system with the same setup as the original SPT code base and access to SPT and the public LeWM (stable-worldmodel). The collaborator will not review the design, so the plan must be a written spec detailing every component.

Settled with the user:

1. Two branches as two ladders meeting at AS-JEPA: L (LeWM -> LeWM + anchored renderer with SIGReg -> AS-JEPA, plus ablations) and S (SPT with actions -> + first-frame anchor -> AS-JEPA from SPT modules). Shared PushT data and one shared evaluation package. Run independent rungs in parallel within each branch.
2. SPT base experiments: environment smoke test only; natural video only after PushT works.
3. PushT only for now; move on (shape/scale variation, other environments) only after decent behaviour on PushT. (Whether round one includes the appearance-variation and dot variants of PushT is being confirmed.)
4. Round-one ablations: window length and anchor choice.
5. Video evaluation: pose-estimator action following, same-state counterfactual actions, LPIPS/SSIM, FVD secondary; also save successful and failure cases for human inspection.
6. Planning: PushT planning from the start, reported but not gating; full LeWM suite only after PushT works.
7. No external baselines for now; the LeWM rung is the comparison.
8. Written specs only, detailed component by component so the implementation cannot drift from the design.

Theory follow-up: the user asked what the unreduced assumption is and whether block identifiability depends on it. Answer recorded in `as-jepa-story.md` Section 5.6: purity (no appearance, no transient information) is proved without it; completeness needs the per-window value condition A5, whose wording was corrected. A toy Gaussian calculation (`theory_checks/a5_value_condition.py`) reduces A5 to "the component moves within a window by more than about the code noise relative to its spread", shows that a rarely pushed block is dropped with a first-frame anchor but kept with a distant anchor, and shows collapse is a local minimum (motivating the `beta` ramp).

## Handoff spec written (2026-10-26)

Final answers from the user: no experiments designed to test the theory's assumption or the collapse-barrier finding (focus is benchmark success); variation and dot variants of PushT stay in round one; the distant anchor is a co-primary arm; the main criteria are generation and planning (planning algorithm may need variants), with probes, purity and rates for understanding only. The user asked whether the collapse-barrier finding is negative for theory or practice: it is neither (the theory concerns the global optimum, which the toy confirms; in practice it is the standard posterior-collapse risk, already handled by the beta ramp, and the distant anchor weakens or removes the barrier in the toy).

Written: `as-jepa-experiment-spec.md`. An independent reviewer ran the spec's code against the real vforge and stable-worldmodel sources and found 4 errors and 6 risks, all fixed before commit: 64-pixel targets now rounded to uint8 in both branches; tests account for WorldDiT's zero-initialized output layer; rollout context defaults to the predictor's num_frames (the window-4 ablation would have crashed); state tokens get slot embeddings (cross-attention without positions sees an unordered set); sampling in eval mode; S0/S1 rollout context 9; PushT install extras; meta-device initialization for S2's pooling; LeWM scheduler-interval check; seed and signature details.

## Spec review: knobs, native priors, priorities (2026-10-26)

The user reviewed the handoff spec and asked: is fixed `sigma0 = 0.1` too rigid, and should it and "the 0.5" be tested; should constants be named and derived; why keep the residual prior; why the `beta` ramp; why two anchor arms rather than one distance hyperparameter. They also asked to prioritize L2 and S2 with the other rungs as ablations, to use the latest stable-worldmodel, to parallelize by available compute (L before S if compute is short), and to finish standard PushT before anything else.

Decided and written into `as-jepa-experiment-spec.md` (rewritten) and `as-jepa-story.md` (Sections 3.2, 4.1, 4.2, 4.4, 5.6, 6.4, 8, 9):

1. **Code noise.** The 1/2 factors and the log constant are the Gaussian KL itself, not choices. Only `sigma0` relative to the unit entry prior matters (the prior and renderer read the state through linear layers), and in noise units `u = mu / sigma0` the rate is `(sigma0^2/2)||u_0||^2 + (1/2) sum_t ||u_t - m_t/sigma0||^2`: `sigma0^2` is exactly the entry price relative to surprise. So separate entry/surprise weights duplicate `sigma0` and are dropped (one `beta`, exact code length), the prior's spread stays tied to `sigma0` (another value only rescales `beta`), and `sigma0` is searched in {0.03, 0.1, 0.3}. In the story's worked example (T = 7) a component is kept when its per-frame displacement exceeds about 0.9%, 3.1% or 9.2% of its spread at these values, so the grid spans where the block switches from dropped to kept. A learned `sigma0` was considered and deferred: at the optimum it gives `sigma0^2 = 1/(1 + mean SNR^2)`, but near a collapsed code it drifts toward 1 and prices information about 100 times higher than `sigma0 = 0.1`, deepening the collapse basin.
2. **Constants.** A named-quantities section: derived quantities as formulas (`FRAME_ELEMS = C * RENDER_PX**2`, `TOKEN_CH`, `GRID`, `ACTION_DIM`, SPT's automatic flow shift computed from the renderer grid), a hyperparameter table, and evaluation and decision constants. The rate is normalized by `y.numel()`.
3. **Prior.** Native predictors with direct output: LeWM's `Predictor` (its output is already direct; no BatchNorm `pred_proj`) and SPT's state-causal blocks with a linear head. The residual identity carry only changed where training starts; it is kept as collapse fallback 1.
4. **No `beta` ramp.** The optimum does not depend on the path. In the toy, the collapse barrier is small (about 0.34 nats at `beta = 1` for a fast component), sits at a fixed signal-to-noise ratio (about 1.4 noise widths, so smaller `sigma0` makes random initial features clear it), shrinks about threefold with a distant anchor, and vanishes as `beta` goes to 0. Constant `beta` throughout; a fixed 5% ramp is collapse fallback 2, never tuned.
5. **Anchor gap `g`** replaces the two arms (`g = 0` is the old `first`). Episodes are short (18,685 episodes, median 25 kept frames), so gaps of 12 or more leave 30% or more of 8-frame windows without an eligible anchor; the grid is {0, 8} in Stage A and {0, 2, 4, 8} in Stage B, skipping `W = 16, g = 8`. Test-time anchors follow each model's gap (a past frame), with context frame 0 reported as a secondary check.
6. **Branch L.** L2 first, with a staged search: A (`beta` x `sigma0` x `g` in {0, 8}, 36 runs, 40% budget), B (`g` x `W`, 9 runs), C (top two, 3 seeds). Selection: generation score within 10% of best, then planning if it separates by more than 10 points, then lowest rate. L1 redefined as L2's exact architecture trained with LeWM's objective (prediction loss plus SIGReg), so L1 versus L2 isolates the price; it gets a 3-point SIGReg-weight screen. L0 is LeWM as published.
7. **stable-worldmodel at `21446f1` (2026-10-02).** Barely moved since the user's fork, but the spec now: inherits LeWM's `rollout` with its `action_history` contract (required when `H > 1`, which `history_len=3` triggers); trains from `pusht_expert_train.lance` (HF `galilai-group/lewm-pusht`, revision `ea321e3`, JPEG frames) and plans from `pusht_expert_train.h5` (from `quentinll/lewm-pusht`); builds the index from `get_col_data`; zeroes NaN end-of-episode actions; runs L0 and L1 on one GPU (SIGReg is single-GPU and the batch is per device); drops the scheduler stop-and-ask (stable-pretraining steps per optimizer step); notes CEM's population std, P1's context-skip and iCEM's clamp. The full budget is large: about 1.25 million steps at `W = 8`.
8. **Branch S.** S2 first (full search in parallel if compute allows; otherwise a 3-point `beta` check around L2's selection), S0 and S1 as comparisons. Priorities when compute is short: L2, S2, L0 and L1, S0 and S1, no-anchor ablations; only branch L if compute suffices for one branch; the collaborator allocates compute.
9. **`D-std` only**, then a report and stop. `D-var` and `D-dot` moved to an appendix; on `D-std` the purity claims cannot be tested, only generation, planning and which dynamics components are kept.

## Simplified empirical design: one searched knob, 10 epochs (2026-10-26)

The user asked whether `sigma0` and `beta` must both be tuned, whether 100 epochs is necessary (with a search of LeWM's work for whether 10 epochs gives enough signal), accepted the identity carry and the ramp only if they reduce difficulty and search while keeping the story, and asked that empirical details be frozen in an empirical section while the method and theory sections keep the canonical design.

Evidence gathered:

- An execution check of the spec found that, with a direct predictor and constant `beta`, the starting rate per rendered value is 1.32, 0.119 and 0.0132 times `beta` at `sigma0` = 0.03, 0.1 and 0.3 on real PushT windows, against 0.0026 times `beta` with the identity carry (the untrained encoder barely changes between consecutive frames: mean squared change 0.00013 against 3.8 for the state). With the flow loss starting near 1.3 to 1.8, the direct predictor starts rate-dominated in about two thirds of the earlier 36-run grid, the carry only at `beta = 1024`.
- LeWM's paper (arXiv 2603.19312, Appendix E) trains every PushT world model for only 10 epochs and states that "10 epochs are sufficient to reach the best performance" (96.0 +/- 2.83% PushT success, three seeds, 50 episodes). The public config's 100 epochs is not the published setting.
- The user's October LeWM campaign (`sparse_jepa/experimental_results/lewm_campaign`) compared every arm at the 10-epoch checkpoint (123,920 updates, 200 tasks per seed): unmodified LeWM 86.0, 81.0 and 86.5%; collapsing arms 1.5 to 9%; LeWM from 10 to 20 epochs 43/50 against 40/50.

Decided and written (`as-jepa-story.md` new Section 8 "Empirical design of the first experiments", method and theory sections restored to the canonical design; `as-jepa-experiment-spec.md` updated):

1. `beta` is the only searched quantity ({1, 4, 16, 64, 256}, one seed each at the full budget, then two more seeds at the chosen value). `sigma0` is frozen at 0.1: `beta` has no calibrated value because SPT's flow loss is not in nats, while `sigma0` only sets the entry-to-surprise price ratio and its effect on slow components is shared with the window and anchor, frozen at values that favour keeping them.
2. Identity carry is the default prior parametrization in both branches (native predictor computes the change, last layer zero-initialized; in branch L, `predict` adds its input and `predictor.transformer.output_proj` is zero-initialized, so LeWM's `rollout` works unchanged).
3. No `beta` ramp; it is the fallback only if every price collapses. Collapsed prices are excluded, not rescued.
4. Frozen anchor gap 8 and window 8; the first-frame anchor is one ablation run; `sigma0 = 0.03` and 16-frame windows (gap 4) are fallbacks if the block is left out. The no-anchor ablations moved to the appearance-varied round.
5. Budget: 10 LeWM epochs for every run (about 125k steps at window 8), L0 at the paper's `max_epochs=10`; continue rule (20 epochs once if the generation score improves more than 5% between epochs 8 and 10); runs that fail the collapse test at 3 epochs are stopped.
6. L1 at LeWM's published SIGReg weight 0.09 (LeWM reports success above 80% for 0.01 to 0.2), screened at 0.03 and 0.27 only if L2's advantage is within twice the seed spread. The full plan is 27 runs of 10 epochs.
