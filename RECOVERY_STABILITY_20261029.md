# Same-size robustness followup

The primary multi-seed P0 replication is complete and does not meet the prespecified stable89% target. See [active recovery](RECOVERY_ROBUSTNESS_20261029.md); useful new conclusions are being published. Old snapshots stay immutable.

---

# Active stability reproduction — 2026-10-29

Latest user authorization: reproduce the full-data beta=.25 three-epoch89.33% P0 configuration across training seeds, then pursue useful Lingjing experiments if stable or evidence-driven method improvements at the same model size if unstable. Publish useful new numerical conclusions and key figures without another approval.

Authoritative root:/efs/prescottsun/SPT-JEPA; campaign:runs/stability_20261029; branch:research/ideas-20261028. Read protocol.json, plan.json, workers/, exits/ and actual GPU processes. Never launch duplicate workers.

Primary new seeds1-4 change only random seed and output versus frozen seed0; preserve37473 optimizer updates and the124910-step schedule. Training math/source is the original committed fa658a6 archive. Old completion_20261028 source/config snapshots and HOLD_DISPATCH stay intact.

Original checkpoint:/efs/prescottsun/SPT-JEPA/runs/completion_20261028/full-L2-beta0.25-stage3-seed0/model-step0037473. Use paired P0 seeds42/43/44 and fresh1042/1043/1044. Bad seeds must remain in the report.

Python:/efs/prescottsun/project_jepa/envs/lewm/bin/python. Reuse the existing Lightning trainer/checkpoints. Do not alter a launched config or source. If a host is lost, inspect the saved full checkpoint and source before resuming in a fresh recovery directory; the old trainer bypasses stop_after when resuming, so enforce the37473-step cap with an explicitly validated operational wrapper and retain the original schedule. Do not accidentally continue to ten epochs.

The previous campaign is a completed historical main-conclusion report, not authorization to relaunch its1453 held entries. Runtime outputs, checkpoints and logs stay EFS. Only small code/configs and permitted complete numeric results/figures enter Git.
