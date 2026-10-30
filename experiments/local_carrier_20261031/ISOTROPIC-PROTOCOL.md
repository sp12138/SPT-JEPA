# Reviewed isotropic repair

The original diagonal-calibrated preflights remain archived and held. The read-only diagnosis found TwoRoom local near-zero coordinate-variance amplification on real training windows; all numerical floors were untriggered. Root approved replacing every1024-dimensional measurement's scale by `sqrt(mean(original128-frame coordinate variances))`, preserving all original coordinate means. Apply the same rule to local/dense, both tasks and both seeds. This introduces no new images, labels, optimization, tuned floor or task-specific rule.

Use the newly committed immutable source descriptor; the model/Robust/Trainer/factory paths must match original `96aca13` byte-for-byte. `prepare_isotropic.py` reuses the unchanged preparation implementation with three new roots:

- `runs/local_isotropic_calibration_20261031`: executed CPU derivation of four joint records. Each records its old calibration SHA256, original encoder/projection identity, transformation, new helper/model source hashes and CPU execution provenance. There are deliberately no fabricated calibration jobs, plan or exits.
- `runs/local_isotropic_preflight_20261031`: eight real fresh native2-step jobs at batch128 plus eight fullstate/gradient/frozen/paired-initial checks on dev15 GPUs0–7.
- `runs/local_isotropic600_20261031`: eight fresh native600 jobs, each followed by health, information, causal-transition and original50-case planning. Created only after all16 real preflights pass and root writes the new preflight `review.json` with `passed: true`.

Commands, executed from the new frozen source:

```text
prepare_isotropic.py derive --source-record <new descriptor>
prepare_isotropic.py preflight --source-record <new descriptor>
# Root reviews all actual preflight results and writes the new review.
prepare_isotropic.py training --source-record <new descriptor>
```

No helper launches a worker. The same native batch, data, optimizer, scheduler, model initial parameters, sample-level TF loss, beta=.25, sigma=.1,600 cap, checkpoint200/400/600 and original planner remain. The normalization changes the representation metric, and comparing with old64-D systems still involves capacity/state-dimension differences. Local8×8 and dense16×16 also differ in averaging/projection subspace. No pose/planning success or novelty follows from normalization health alone.

Three new stdlib CPU contracts passed: exact common scalar/unchanged means and immutable inputs; rejection of clipped/zero/nonfinite source variances; and orthogonal metric preservation versus a counterexample for diagonal whitening. Existing real16 production preflights validate the unchanged native code path, but **new normalization-specific real16 preflights remain required**. Do not reuse their success as approval for this new gate.
