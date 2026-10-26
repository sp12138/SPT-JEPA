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
