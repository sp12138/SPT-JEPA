# Completed insight campaign — 2026-10-27T20:56:14.635676+00:00

Original full-spec completion was audited separately: [remaining tasks](experiments/insights12h/TASK_AUDIT_20261028.md). The original full spec is incomplete; the 28 selected endpoint finalizers are complete.

The selected 28 main finalizers and all late diagnostics are complete. Read
[the final report](experiments/insights12h/results/INSIGHTS.md) and
[the numeric/figure index](experiments/insights12h/results/README.md).
Final verification: complete=true, 28/28 models, no pending entries or errors.

Key result: tiny L2 beta=.01, sigma=.1, one-time init std=.1, 50k, seed0:
noisy agent/block R2=.868/.971; h7 agent/moving-block error over copy=.234/.234;
four sampled latent rollouts=.265/.261; P0 seeds42/43/44=74/76/60%, mean70%.
One training seed; original three-seed and manual32-case gallery gate are incomplete.
Same 50k-run 24k->50k improves agent .860->.234 and P0 24->70.
All numerical observations and key scientific PNG/SVG figures are packaged in Git.

Authoritative assets: /efs/prescottsun/SPT-JEPA.
Final review: runs/insights12h_20261027/review/final_2100.
Report source: committed 0533159, source-report-v33; main/diagnostic/verifier bytes
are identical to v32. Training/config snapshots were not changed.
All 903 inherited runtime files remain unchanged; nine CPU contracts passed.

Campaign-owned idle workers/evaluator on dev2/dev13/dev15 were terminated after
completion. Final live GPU-process queries were empty on all three hosts.
Lost-host historical statuses are preserved and may still say Running; they are
not evidence of live work. Do not restart held or historical jobs.
This does not authorize new training or inherited SPT/FVD work.

Publication destination is https://github.com/sp12138/SPT-JEPA.git,
branch insights/12h-20261027. Remote push verification is recorded outside Git
in /efs/prescottsun/SPT-JEPA/reports/publication.json.
The following entries are historical and superseded by this completed report.

---


Recovery requires EFS assets plus this Git branch, not a surviving development
host. No compute remains to resume for this campaign. Check publication.json
and remote HEAD before attempting a duplicate publication.
The user's publication authorization includes complete numerical results and
insightful figures; do not ask for the same authorization again.
