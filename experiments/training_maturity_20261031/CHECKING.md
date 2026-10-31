# One scheduled read, six fixed slots

Use the local `check.py --all` entry once at each authorized half-hour review. It starts three independent SSH reads, restricted to this campaign and GPU0/1 on dev2/dev13/dev15, then aggregates saved EFS JSON. It has no polling loop, model call, training, fix/restart/stop action or planning-success read. Do not rerun a label: output creation is exclusive. If a network/process-boundary race occurs, retain the evidence and review it before a narrowly scoped follow-up.

```sh
python3 experiments/training_maturity_20261031/check.py --all --label 123026
```

The immutable remote copy is `/efs/prescottsun/SPT-JEPA/reports/training-maturity-20261031/checker-v3/check.py`. Each invocation verifies its bytes against the local script SHA passed through SSH. Full evidence goes to `runs/cube_maturity23342_20261031/review<label>/{dev2,dev13,dev15,summary}.json`; stdout is short. Root can copy only summary.json locally if useful. Native's snapshot utilization can be0 during normal data/copy gaps; judge progress across scheduled reads, not one utilization sample. The earlier v1 performed only CPU aggregation of the existing12:00 records and remains preserved. v2 additionally retains NaN/Inf as strings in failure evidence and explicitly flags steps beyond23342.

For recovery after host reads already succeeded, aggregate without another GPU/process read:

```sh
ssh dev13 '/efs/prescottsun/project_jepa/envs/lewm/bin/python -B /efs/prescottsun/SPT-JEPA/reports/training-maturity-20261031/checker-v3/check.py --aggregate --label 123026 --expected-sha <recorded SHA>'
```

The archive check reuses actual completed audit/checkpoint-hash records and verifies the archive exists; it does not re-read multi-GB checkpoints at every poll. Module trace positivity is not a new per-tensor backward audit. Final scientific review still requires all68 endpoints and both seeds/controllers/case sets. Training/config/source and old HOLD files remain untouched.

At13:00 v2 misclassified the two completed native statuses: final summaries omit gradient fields. v3 falls back only for finished summaries to the last actual running gradient record, reporting its step/time separately (23300 in this incident). Missing running gradients or missing historical evidence still fail; zero/NaN/Inf active gradients still fail. The complete trace finite check and actual final fullstate endpoint remain separate requirements. Raw v2 snapshots and the clarification are preserved; no GPU check or training was repeated.
