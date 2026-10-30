# Native LeWM objective, matched short baseline

This directory prepares four fresh runs: TwoRoom/Cube × seeds1/2. It launches nothing. Root must freeze a committed source, review the CPU evidence, then allocate proposed dev13 GPUs4–7. Current6000 source8e0f821 remains untouched.

`NativeLeWM` instantiates the official ViT-tiny/CLS, predictor, action embedder and BN projectors. Cube preserves the official `smoothed_dim=10`. Native loss is prediction MSE +.09 SIGReg (17 knots,1024 projections), including target gradients, BatchNorm and native autocast behavior. It has no renderer, carry residual, privileged training labels or optimal-action targets. The native baseline is intentionally batch coupled; proposed method objectives remain sample-level.

The explicit matched adaptation uses history7/window8, same training-only episode split, RGB224/action rows, normalization, sampler, batch128, seed and124,910-update schedule horizon as TF/F. It differs from the literal official history3/window4, random clip split, whole-dataset scaler and default100-epoch budget. The native scheduler starts at lr0 and advances per optimizer update; warmup is1,249. Only the new model opts into a three-line optimizer configuration hook in the existing Lightning module. Other model code paths are byte-identical after removing that hook/factory target, checked against8e0f821. Trainer, checkpoint manager and data sampler are reused.

Exact capacities: TwoRoom18,035,246 and Cube18,035,396 trainable parameters, versus approximately51.1M total for TF/F. Equal data/update exposure is not equal capacity or compute. The short gate lies inside warmup and cannot establish mature baseline performance.

CPU contracts call the pinned official `scripts/train/lewm.py:lejepa_forward` itself. For each task, train/eval mode losses and all297 gradient tensors match exactly under fixed tensors/RNG; BN state also matches. They use the full production architecture and eight frames at28px for bounded CPU algebra, with RGB224 preprocessing checked separately. Official optimizer-factory defaults, native scheduler rates, export loading, optimizer/scheduler roundtrip and continued next update are checked. These do not replace the actual production-batch GPU preflight.

The roundtrip test performs disposable CPU optimizer updates using those synthetic-input gradients and checks populated Adam state for all297 parameter tensors. Its report's “no world-model training” means no dataset training/campaign, not an absence of optimizer calls inside this test. No model from the test is reused by an experiment.

After root freezes `as_jepa_lewm`, `as_jepa_tools`, this directory, and `experiments/ideas_validate_20261031` (including `audit.py`, `preflight.py`, both audit-window manifests) into an immutable source:

```sh
python -B -m experiments.native_baseline_20261031.prepare prepare --source-record SOURCE_RECORD.json --contracts /efs/prescottsun/SPT-JEPA/reports/native-lewm-preparation-20261031/cpu-v3/contracts.json
```

This produces:

- `/efs/prescottsun/SPT-JEPA/runs/native_lewm_preflight_20261031/plan.json`: four real two-step training jobs plus four fullstate/gradient audits, batch128 retained because SIGReg depends on batch size.
- `/efs/prescottsun/SPT-JEPA/runs/native_lewm600_20261031/candidate-plan.json`: four600-update jobs, each followed by fullstate/gradient review, common64-window clean-forecast interventions and unchanged50-case strict planning (16 endpoints total).
- Separate configs/protocol/hashes; no600 `plan.json` yet.

After all eight preflight endpoints pass **and root reviews them**, `prepare activate` verifies exits/audits and creates the600 plan without launching a worker. It does not clear STOP markers or resume any historical experiment.

The generic preflight audits native encoder/predictor/action_encoder/projector/pred_proj gradients and complete saved state. At600 the same audit is reused with `--expected-step 600`, followed by generic clean rollout diagnostics. Existing AS-JEPA flow/rate/renderer audit mains are not applicable and are not called. Physical readout expansion is deferred; no dummy renderer, posterior sigma or rate is added to appease those interfaces. Planning exports keep the `as_jepa_lewm.` target so the unchanged evaluator correctly selects training-only action normalization and retains its fixed planner history3.

Expected learning rates after updates: step0=0;1=4.00320256205e-8;2=8.00640512410e-8;600=2.40192153723e-5;1249=5e-5;2000=4.99954500047e-5. The initial lr0 no-op is native behavior. Preserve all seeds and review actual runtime/memory; there is no automatic longer continuation.
