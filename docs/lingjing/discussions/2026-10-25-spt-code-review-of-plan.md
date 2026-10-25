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
