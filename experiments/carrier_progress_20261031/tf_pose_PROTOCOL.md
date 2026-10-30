# Approved short TF6000 pose/position diagnostic

Prepared only. Root approved the single information-versus-dynamics question; root still owns source freeze and the two GPU launches. All14 training jobs remain untouched. No training polling, new model objective, sweep, physical-label model training, extra intervention assay or figure is included.

Exactly two jobs, one per original Cube TF6000 training seed, on reserved dev2 GPUs4/5. `tf_pose.py run` first extracts1536 RGB frames (128 training clips and64 validation clips,8 frames each) with the original8e model/data/forecast code, then performs CPU linear+RBF readout analysis. Each reads `runs/carrier_learn6000_20261031/cube-temporal-forecast-seed{1,2}/model-step0006000` explicitly; no `last.ckpt` or active training result is used.

Extraction checks source bytes against8e, checkpoint/config hashes against completed full-state audits, exact original F cohort rows/episodes/metadata, eval+FP32/highest precision and strict state loading. No physical label enters extraction. Each TF is encoded once; clean rollout receives only frame0 and recorded actions. Four diagnostic references are observed future, teacher prefix, initial carry and shifted observed carry. Final padded action is excluded from forecasts. All serialized weights/buffers must remain bytewise tensor-equal after extraction; no BN calibration occurs.

F600 observed/training64-D embeddings are reused; they must equal saved F6000 observations exactly. Its already-saved6000 causal/teacher/carry arrays are reused. There are zero old-F model calls. CPU fits reuse the original unchanged spatial/pose helpers: coordinate standardization from1024 training frames, ridgeλ=.001, train-median centered RBF, cube24-invariant tensor81+xyz targets and training quaternion codebook. One fitted readout per model/type applies to all five conditions. No validation selection or tuning. Recomputed F observation readouts must reproduce the completed RP64 readouts within2e−5 in reported scalars (equivalent affine feature normalization).

Large arrays and full frame errors remain under `/efs/prescottsun/SPT-JEPA/reports/tf6000-pose-20261031/seed{1,2}-{features,readouts}`. Each readout directory writes a compact `summary.json` with full-result SHA. Retain both seeds and both readouts. Observed readability, its degradation under causal forecast and actual planning success are distinct. Finite128-clip support and probe inductive bias prevent absence-of-information conclusions. Framewise3cm/15° accuracy does not establish three-step sustained task success.

After root freezes the helper, use the normal immutable analysis-source/runtime/overlay `PYTHONPATH`; the script asserts original8e model/forecast arithmetic. One command per reserved GPU, with CUDA visibility managed by the existing dispatcher:

```sh
python -B -u /FROZEN_SOURCE/experiments/carrier_progress_20261031/tf_pose.py run --seed 1 --device cuda --batch-size 4 --output /efs/prescottsun/SPT-JEPA/reports/tf6000-pose-20261031
python -B -u /FROZEN_SOURCE/experiments/carrier_progress_20261031/tf_pose.py run --seed 2 --device cuda --batch-size 4 --output /efs/prescottsun/SPT-JEPA/reports/tf6000-pose-20261031
```

CPU `contracts` tests use no real model or GPU: future observations cannot alter clean rollout; padded actions do not enter it; valid actions do; joint condition processing preserves the original primal linear/RBF formula and never fits on validation rows. CPU source/reference checks verify original8e helper bytes and exact reused F observations before any launch. Outcomes will determine whether a new mechanism is warranted; this protocol claims no expected winner.
