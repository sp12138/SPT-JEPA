# 全数据 beta=.25 五训练种子复现已完成

[结果、完整数值与关键图](results/FULL-DATA-STABILITY.md)。新增四种子均值84.00±6.28%、最差74.67%；原先89.33%未稳定复现。

后续见[同规模稳健性对照](../robustness20261029/README.md)。以下为执行前冻结协议记录。

---

# Full-data beta=.25 stability campaign

User authorized multi-training-seed reproduction of the 89.33% P0 result, followed by evidence-directed research. This campaign is active; no new replication result is available yet.

Training seeds1-4 exactly copy the original seed0 model, data, batch128, noise=.1 and optimizer/scheduler configuration, changing only the output and random seed. Stop at37473 optimizer updates while retaining the124910-step learning-rate schedule. The original single-seed result remains an exploratory reference.

The frozen plan and numerical decision criteria are in [protocol.json](protocol.json). Old exhaustive grids remain held.

Evaluate original P0 seeds42/43/44 and prespecified fresh seeds1042/1043/1044 on common heldout start/goal cases. Report across-training-seed statistics separately from evaluation-seed variation, retain bad seeds, and distinguish practical stability checks from formal benchmark claims.

Authoritative runtime: /efs/prescottsun/SPT-JEPA/runs/stability_20261029. All useful new conclusions will be committed and pushed with complete numerical results and key figures.

首个已完成检查：[seed0原案例精确复跑与新案例](results/SEED0-REEVALUATION.md)。这仍只是同一训练种子的评测，不是跨训练种子稳定性证明。
