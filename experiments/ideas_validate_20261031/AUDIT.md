# Frozen diagnostic contract for the 2026-10-31 short validation

These independent scripts use `as_jepa_lewm.factory.build_model(cfg)` and the existing `BenchmarkWindows`; they do not implement a trainer. Root owns source freezing, native training, launch, and the task evaluator. No script in this document starts training automatically.

## Two-step preflight

After the native `as_jepa_tools.train_stage --stop-step 2` finishes, run:

```bash
python -m experiments.ideas_validate_20261031.preflight \
  --run RUN --output RUN/preflight --expected-step 2
```

`preflight.py` checks the native full state: exact update, finite model/optimizer/scheduler, RNG fields, consumed-sample cursor, and portable/native tensor equality. It compares every declared frozen group with `initial-model/weights.pt`. For `model_class=carrier`, immutable projection/mean/scale buffers are included even when the chart learns. It checks optimizer parameter count against actual trainable tensors.

It then loads the frozen model in **evaluation mode**, sets a declared seed, evaluates the real loss on two fixed held-out windows and backpropagates once. There is no optimizer or parameter update. Expectations come from `model.audit_expected_gradients()` (a dict attribute is also supported), overridden by `cfg.audit_expectations.gradients`:

```json
{
  "audit_expectations": {
    "gradients": {"action_encoder": "active"},
    "initial_unchanged_prefixes": []
  }
}
```

Accepted expectations are `frozen`, `active`, and `optional`; `trainable_active`/`trainable_optional` are aliases. Frozen means no trainable parameters and no gradients, with unchanged initial tensors. Active means a positive finite aggregate gradient on trainable parameters; it does **not** demand every tensor have a nonzero gradient. Optional means trainable and finite if used, with zero/absent gradient permitted.

For F, encoder/pool are frozen in all three arms. The identity chart is frozen; bounded/unconstrained charts are active. Predictor and renderer are active. The campaign explicitly requires the action encoder active at step 2, overriding its model default of optional; exact step-0 zero predictor initialization can otherwise block that gradient.

For S3, encoder/pool/predictor/renderer/action_encoder/path_query_head are active in the nonzero-weight arms. Failed checks are saved in `preflight.json` before a failing assertion.

To verify matched initial native components against another arm, additionally pass `--reference-initial OTHER/initial-model/weights.pt`. The default comparison prefixes are encoder, pool, predictor, action_encoder, renderer. Repeat `--reference-prefix` to specify a different reviewed set. This is distinct from checking frozen tensors did not change during training.

## Fixed-window state and forecast audit

```bash
python -m experiments.ideas_validate_20261031.audit \
  --run RUN --output RUN/state-audit --cases 64 --seed 1042
```

Use 32 or 64 according to the frozen plan. Selection matches the existing information audit: one window from each distinct validation episode, chosen using one model-independent NumPy seed. The manifest records dataset indices, authoritative episode-array indices/IDs, starts, raw rows, stride, and hashes of metadata and split files. It verifies episode-disjoint train/validation splits. No privileged HDF5 fields are read.

For an explicit shared artifact, first use `--manifest-only`, then give every matched model `--manifest PATH/windows.json --cases 64`. Without that flag the same data/window/count/seed still generate identical canonical manifest bytes and SHA-256. A manifest mismatch fails; no per-model window resampling occurs. Other-initial/action pairings rotate the entire fixed case list, so chunking cannot turn the intervention into a self-swap.

The primary outputs are `state-audit.json`, `state-audit.npz`, and `windows.json`. The small NPZ saves observed states, actions, and every forecast/control sequence for paired review without re-encoding.

| Diagnostic | Real observation access | Meaning |
|---|---|---|
| `observed_future` | All actual frames | Tautological zero-error reference, not learned prediction |
| `clean_rollout` | Only frame 0 | Free causal rollout using clean states and executed actions |
| `teacher_forced_prefix` | Every real prefix through the current step | One-step prediction with fresh observations; no future frames |
| `shifted_observed_carry` | A fresh real encoding each step | Copies the previous observed state; no learned dynamics |
| `initial_carry` | Only frame 0 | Constant initial-state baseline |
| `zero_prediction` | No state | Zero-vector baseline; for S3 also query-only shortcut control |
| `other_initial` / `zero_initial` | Replaced initial state | Action inputs and target stay fixed |
| `wrong_action_order` | Correct frame 0 | Cyclically shifts only valid action blocks inside each window |
| `other_episode_actions` | Correct frame 0 | Uses another fixed episode's valid action blocks |

`BenchmarkWindows` supplies W action entries for W images, but the last action is padding. It is never rolled into the W−1 valid transitions. The first returned rollout state predicts observed frame 1; the initial state is not accidentally scored against that target.

Teacher-forced outputs are computed prefix by prefix. A separate diagnostic compares them with one vectorized teacher call; nonzero disagreement warns that the latter may use future observed states or otherwise differ. Prefix-only results remain authoritative.

MSE is reported raw and normalized by **actual** (a) displacement from frame 0 and (b) one-step temporal variation, with per-horizon and per-window values. Aggregate normalization is ratio of sums, not mean of unstable per-window ratios. Exact zero denominators produce JSON `null`, never an epsilon floor that makes a collapsed representation look healthy. Latent geometries differ across methods; raw MSE alone is not a cross-model quality comparison.

For interventions, report both changed target error and changed predictions. An intervention that changes output but does not improve correct-input accuracy demonstrates sensitivity, not useful control knowledge. Wrong-action input MSE is recorded because nearly constant actions can make an order permutation ineffective.

## F and S3 additions

F records the chart's model-owned norm certificate, raw matrix norms, and whether spectral normalization is active. This separates an unactivated bound from an actually constrained learned chart; the certificate itself is not a control result.

S3 reuses `path_query_targets(render, step=0)` once and the model-owned `path_query_logits_from_states` hook. Queries, future-image prototypes, masks and targets remain fixed under every state/action intervention. It evaluates observed states, clean forecast, prefix teacher, shifted carry, other/zero initial, wrong actions, reversed/other observed trajectories, and an all-zero state sequence. It reports CE, CE/log(K), accuracy, label counts, per-case CE and per-frame/segment CE. Symbol scores additionally exclude the already observed initial frame. Only the selected trained mode is objective-matched; other output interpretations are descriptive.

The all-zero state sequence retains prototypes and masks: good performance here exposes a query-only shortcut. It is not evidence that the world model predicts. Observed/reversed branches also see actual future encodings and are not forecasts. The audit captures render during the normal data pass and does not repeat encoder work.

Existing `benchmark_information_audit` remains responsible for frozen physical-label probes; this audit intentionally does not duplicate them. Existing factory-aware `benchmark_health` remains the 600-step full-state check. Neither latent/query scores nor numerical health replace the fixed strict task evaluator.

## Executed contract tests

`test_audit.py` uses tiny CPU analytic models, no GPU or real environment. Contracts cover exact additive rollout alignment, padding exclusion, chunk-independent intervention pairing, shifted-observation vs free-rollout access, detection of a future-leaking vectorized predictor, zero temporal-variation handling, frozen/active gradient distinctions, checkpoint corruption detection, and fixed-query initial replacement. Root should retain the CPU result with the frozen source.

```bash
CUDA_VISIBLE_DEVICES= python -m unittest experiments.ideas_validate_20261031.test_audit -v
```
