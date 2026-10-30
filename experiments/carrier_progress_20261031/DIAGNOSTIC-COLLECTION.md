# Complete diagnostic collection

This analysis runs only after the parent confirms all 34 jobs have completed. It does not poll or read partially completed evaluation outputs. All expected exit files must exist and contain returncode zero before any result is opened.

Run from the analysis checkout, which contains the reused prior pairing helper. The frozen running source deliberately does not bundle the earlier analysis scripts, so do not use its Python path alone:

```bash
cd /efs/prescottsun/SPT-JEPA/code
CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  PYTHONPATH=/efs/prescottsun/SPT-JEPA/code \
  /efs/prescottsun/project_jepa/envs/lewm/bin/python -B \
  -m experiments.carrier_progress_20261031.collect_diagnostics \
  --root /efs/prescottsun/SPT-JEPA/runs/carrier_diagnostics_20261031 \
  --output /efs/prescottsun/SPT-JEPA/reports/carrier-diagnostics-20261031
```

Outputs are `results.json` and `RESULTS.md`. All 34 models/metric evaluations, both training seeds, original normalization and calibration information are retained. Large candidate cost/order arrays remain in their original EFS output directories. Hashes, descriptive rank statistics and first-bank checks are included in the small aggregate.

Required checks: the exact 7 fresh TwoRoom, 12 primary Cube metric, 3 fresh Cube native baseline and 12 fresh Cube metric grid; common frozen source; checkpoint/config, source-input and calibration hashes; native case/solver/plan/protocol budgets; 50 binary case outcomes; exact per-case primary identity replay against all four original F600 evaluations. An identity replay mismatch fails collection rather than silently treating the modified adapter as a valid control.

Metric comparisons pair only the same checkpoint and manifest. Primary and fresh sets have different exact pairs and policy seeds; they are not paired across sets. Their overlap in validation episodes is disclosed, and neither is called an independent final test set. Shared-case bootstrap intervals condition on two trained seeds and do not estimate training-seed variation or correct exploratory multiple comparisons. Released LeWM retains its different capacity, training exposure and full-data normalization. F/TF also differ in frozen parameters and calibration access.

Each recorded bank contains all three metric costs on identical predictions. Spearman correlations use average ranks for ties; constant banks are undefined rather than assigned correlation zero or one. Stable-order/top1/elite comparisons explicitly remain diagnostic because native CEM uses `torch.topk` tie handling. Rank summaries remain separate by the metric that generated each search path; dependent CEM banks receive no inferential bootstrap.

Across selected-metric runs, only the first CEM optimization iteration for each initial case is aligned. Under the pinned native CEM loop these are calls `case_index × n_steps` in the first solve. Actions, observation/history, goal and terminal-difference hashes must match before corresponding cost arrays are treated as the same bank. Missing observation hashes prevent a verified-input claim. Later diverged populations and physical trajectories are never silently aligned by call index.

The collector changes no objective, model or experiment. Evaluated metrics are controller diagnostics; the underlying sample-level world-model learning loss and checkpoint weights remain unchanged. Ranking changes do not themselves demonstrate better physical outcomes.
