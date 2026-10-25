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
