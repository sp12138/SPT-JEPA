# S3 endpoint summarizer

`summarize.py` reads the main protocol's 12 pathquery runs and corresponding
planning/transition/health/exit reports, plus the six seed-1 native preflights.
It writes `results.json`, `REPORT.md`, `planning.png`, and
`query-dependence.png`. The two seed outcomes remain separate. Planning
comparisons require identical case lists, manifests, evaluator/planner settings
and strict loaded-checkpoint equality. Query comparisons require identical
audit windows, query indices, step, and aggregate target labels within each task.

Example on the authorized CPU environment (set PYTHONPATH to the main root's
`source-record.json` value; keep `CUDA_VISIBLE_DEVICES=''`):

```sh
python experiments/ideas_validate_20261031/pathquery/summarize.py \
  --root /efs/prescottsun/SPT-JEPA/runs/ideas_validate600_20261031 \
  --output /efs/prescottsun/SPT-JEPA/reports/ideas_validate_20261031/pathquery/review600 \
  --require-complete --derive-support
```

The script lives outside the frozen training source when newly copied to
`reports/ideas_validate_20261031/pathquery/summarize.py`; pass that path until the
next documentation/source commit. The main running `source-v1` is unchanged.

Without `--require-complete`, pending endpoints are explicitly marked and no
missing value becomes zero. `--no-plots` is useful for a provisional read-only
review. `--derive-support` performs a single independent CPU target reconstruction
per task: it reads only the exact 64 already-declared audit windows, pools the
same render RGB, applies saved query indices, and obtains full/half monitor
classes. It verifies metadata, episode identities and exact aggregate-count
agreement with the frozen audit. It does not encode images, load model weights,
read privileged physical labels, train, or run a GPU. This is necessary because
the frozen audit records aggregate label support but not its full/half split.

The native preflight reports compare 547 tensors under the five named model
prefixes, not all 550 state-dict tensors. All six S3 reports passed; recorded
action-gradient norms range from 7.70e-17 to 1.59e-14. These establish a nonzero
gradient path after two updates, not functional action sensitivity. The report
retains actual tensor counts and full gradient records.

Different trained targets have different class entropy, balance, contexts and
active output rows. Neither raw CE nor CE/log(K) across arms selects a winner.
Within-head zero/observed/wrong-state/action comparisons diagnose conditioning;
the native 50-case task metric evaluates control. Interventions can be outside
training support, and changed-condition losses against fixed original targets
are not labeled counterfactual dynamics errors. At 600 steps all arms are still
inside the scheduled warmup.
