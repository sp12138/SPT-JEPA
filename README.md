# SPT-JEPA

当前在 `research/ideas-20261028` 继续原始全量实验和十个候选方法验证。用户已授权 dev2/dev13/dev15 持续运行，没有本轮时间上限。运行与恢复入口见 [研究恢复](RECOVERY_RESEARCH_20261028.md) 和 [追加实验状态](experiments/completion20261028/README.md)。

追加实测见[生成质量、规划几何及固定32例图](experiments/completion20261028/results/README.md)和[六个25k基线终点](experiments/completion20261028/results/BASELINES_25K.md)。轻量L2seed0的256例FVD=278.21（copy=1628.62）；I10修复缓存后的Gramian规划均值8.7%（GoalMSE=71.3%），是明确负面结果。三训练种子与原始全量方案仍在推进。

已发布的12小时探索结果保留在 [关键结论](experiments/insights12h/results/INSIGHTS.md)；28个当时选定终点完成不等于原始全量方案完成，缺项见 [任务审计](experiments/insights12h/TASK_AUDIT_20261028.md)。新一轮正在补三训练种子、Native精确25k终点、完整生成指标、原始全数据曝光量与三epoch坍塌筛选，并验证 [十个候选](experiments/ideas20261028/README.md)。

权威代码、环境、数据、检查点、日志、媒体和恢复状态均在 `/efs/prescottsun/SPT-JEPA`。Git只发布代码、配置、完整数值和关键科学图。以下是继承的SPT基线文档，历史机器与目录不覆盖当前授权。

Initialized from [sp12138/SPT](https://github.com/sp12138/SPT) at commit
[`d43a452`](https://github.com/sp12138/SPT/commit/d43a4528a6aaaaf76f10a92a67a294d91f1996b6) on 2026-10-24.
The original SPT baseline documentation follows.

SPT is a self-contained source repository for the RGB16 baseline from
`rgb16_scale_budget_20261010`. It includes the Native training/runtime/model
implementation, inference, FVD detector and evaluator, launchers, rendering,
checkpoint comparison and reproducibility audit tools. No entry point imports
code from another user's experiment directory or another checkout.

The baseline has 423,579,328 trainable parameters: RGB16 lossless pixel packing,
Split ViT E=3/C=18 at width 1024, Patch DiT depth 8/width 768, pixel refiner
depth 2/width 16, and cross-attention at blocks 0 and 4. Pretraining uses context
mask 0.9, distant-target probability 0.5, Muon with auxiliary Adam, BF16, FA4,
EMA 0.99, local batch 8/global batch 64, seed 42 and 50,000 updates on 8 H200s.
Runtime source is preserved byte-for-byte; portability changes are in launchers.

## Repository and persistent assets

| Directory | Contents |
| --- | --- |
| `runtime/` | Bundled Native implementation and package metadata |
| `jobs/` | Training, demo, FVD preparation/workers and audit tools |
| `jobs/fvd_reference/detector/fvd.py` | Bundled, hash-pinned I3D/Frechet implementation |
| `provenance/` | Original configurations, launchers, hashes and FVD sources |
| `environment/` | Versions recorded for the validated baseline |
| `tests/` | Portability and evaluator checks |

Data, manifests, weights, logs, videos, caches and checkpoints are external assets
under `/efs/prescottsun/SPT`, outside Git. Code is kept under
`/threed-code/prescottsun/SPT`; Git metadata is `/efs/prescottsun/SPT/git/SPT.git`
on machines with a separate Git directory. A normal Git clone works too.
Set `SPT_DATA_ROOT` or pass `--data-root` to change asset storage. Shared videos
and supplied checkpoints may be used as read-only inputs.

## Environment

Use Linux, Python 3.12+, CUDA 13 and 8 H200s. The validated baseline used Python
3.12.3, torch 2.9.1+cu130, FlashAttention-4 4.0.0b12 and CUTLASS DSL 4.5.0.
Recorded versions are in `environment/reference_versions.json`. The upstream
`runtime/uv.lock` describes a different general-purpose environment and is not
the RGB16 reproduction lock.

```bash
python3.12 -m venv /efs/prescottsun/SPT/venvs/rgb16
PY=/efs/prescottsun/SPT/venvs/rgb16/bin/python
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r requirements-rgb16.txt
"$PY" -m pip install -e ./runtime
```

The requirements pin the numerical backend and video/data dependencies. Some
optional libraries in the recorded environment are development builds; an
identical fresh environment requires those matching builds. Reuse the validated
EFS interpreter for exact experiment reproduction. All project code is in this
repo; standard Python/CUDA libraries are installed as dependencies.

## Training and recovery

Put the JSONL manifests under `$SPT_DATA_ROOT/data`, or pass `--train-manifest`
and `--val-manifest`. Video paths must be accessible. See `docs/assets.md`.
Commit the source before preparation; it archives the revision into an independent
EFS run and records versions, hashes and resolved paths.

```bash
export SPT_DATA_ROOT=/efs/prescottsun/SPT
PY=/efs/prescottsun/SPT/venvs/rgb16/bin/python
"$PY" -B jobs/prepare_run.py rgb16_run \
  --train-manifest "$SPT_DATA_ROOT/data/train_windows_820343.jsonl" \
  --val-manifest "$SPT_DATA_ROOT/data/test_windows_2048.jsonl"
R="$SPT_DATA_ROOT/runs/rgb16_run"
cd "$R/source/runtime"
SPT_RUN_ROOT="$R" PYTHONDONTWRITEBYTECODE=1 nohup "$PY" -u \
  "$R/source/jobs/worker.py" pretrain > "$R/workers/pretrain/launcher.log" 2>&1 &
```

Return to the SPT checkout before running the following repo-relative commands.

The worker checks source hashes and GPU occupancy, uses the Native Trainer, and
saves full asynchronous DCP every 2500 updates, including optimizer, scheduler,
RNG/data cursor and EMA. Inspect `workers/pretrain/status.json`, `pretrain.log`
and `output/train_steps.jsonl`.

Recover on another machine by checking its GPUs and launching the same frozen
worker with the same `SPT_RUN_ROOT`. It resumes the latest complete checkpoint.
Keep existing frozen source/config/plan intact. Loss before the first checkpoint
restarts from scratch. Current allocation: dev3 for main training, dev8 for
demos/FVD; dev0 is not permitted.

## EMA demos

Run these commands from the repo root:

```bash
"$PY" -B jobs/checkpoint_demo.py "$R" 10000 "$R/samples/step10000"
"$PY" -B jobs/render_demo.py "$R" "$R/samples/step10000" \
  --previous "$R/samples/step7500"
```

The demo reads the frozen config: 16 observed +128 generated frames, 50 Euler
steps/state, CFG=1, EMA and no context mask. Rendering compares eight matching
cases/seeds against the previous demo.

## FVD-2048

Supply the fixed RGB bank, its manifest and pinned I3D weights. All evaluator and
detector code is bundled; these paths contain data/weights only.

```bash
"$PY" -B jobs/prepare_fvd.py rgb16_step10000 \
  --checkpoint "$R/output/checkpoints/10000.dcp" \
  --config "$R/source/jobs/configs/pretrain_50k.yaml" \
  --manifest "$SPT_DATA_ROOT/data/fvd/eval_2048.jsonl" \
  --bank "$SPT_DATA_ROOT/data/fvd/bank" \
  --weights "$SPT_DATA_ROOT/weights/i3d_torchscript.pt"
E="$SPT_DATA_ROOT/evaluations/rgb16_step10000"
nohup "$PY" -B -u "$E/jobs/fvd_worker.py" "$E" \
  > "$E/launcher.log" 2>&1 &
```

Use dev8 with 8 idle H200s. The worker exits after validating 2048 cases and
`score.json`; inspect `status.json` and `evaluation.log`. Sampling uses independent
case seeds (base 20260920), batch 8/GPU, 16 observed +128 generated frames,
50 Euler steps and CFG=1. I3D extracts 400 pre-softmax FP32 features from generated
frames only. FVD uses float64 and sample covariance (`ddof=1`); EMA loading is strict.

Portable sources preserve the sampling/detector math and record actual hashes.
Their protocol identity differs from historical results because code paths
changed; the historical hash is recorded separately. Original FVD sources are
under `provenance/fvd_reference/`. Historical reports remain immutable.

## Audits

```bash
"$PY" -B jobs/audit_alignment.py ORIGINAL_REPORT "$R" AUDIT_JSON
"$PY" -B jobs/audit_reproducibility.py ORIGINAL_REPORT "$R" \
  ORIGINAL_FVD_REPORT SPT_FVD_REPORT AUDIT_DIRECTORY
"$PY" -B jobs/compare_checkpoint_weights.py ORIGINAL_DCP SPT_DCP COMPARISON_JSON
PYTHONPATH=runtime "$PY" -B -m unittest discover -s tests
```

Original-report paths are audit inputs, not training/inference dependencies.
Historical measurements and run status are in `HANDOFF.md`. Component licenses
and provenance are preserved in `runtime/`; vendored components are not relicensed.
