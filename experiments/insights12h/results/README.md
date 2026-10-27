# SPT-JEPA insight campaign：完整数值与关键图

28 组所选主终点评估全部完成，完整性核验通过。本轮核心正面结果：轻量 L2 β=.01/50k 在单训练种子上，P0 均值 70%，h7 agent 和移动物块误差约为复制基线的 23%。详见[关键结论](INSIGHTS.md)。

原始全量计划尚未全部完成，见[逐项任务核对](../TASK_AUDIT_20261028.md)。28/28 仅指本轮所选终点评估。

- [统一主表](RESULTS.md)、[CSV](summary.csv)、[完整终点 JSON](summary.json)：28 组的探针、均值/噪声状态生成、P0 和反事实。训练种子分开列出。
- [预算对照](budget-comparison.csv)、[训练耗时与损失](training-summary.csv)：同一 50k 轨迹的 24k/50k 检查点与独立短跑；耗时、吞吐和训练损失独立于生成质量。
- [全部聚合记录 CSV](all-aggregate-measurements.csv)、[完整 JSON](all-aggregate-measurements.json)：包含历史、终点、诊断、定位器控制和梯度检查，附 tier、flags、路径和版本；记录数量不代表独立实验数量。
- [补充诊断](diagnostics.json)、[晚期探针](late-probes.csv)：精度、Native 反馈、真未来后验、颜色几何干预和预算对照。
- [配置与运行索引](campaign-configurations.json)：140 个调度条目的配置、实际源版本、resume 来源、生成 launcher 及哈希、训练摘要和可移植检查点。保留的旧 status 是历史记录；当前计算均已结束。
- [环境](environment-versions.json)、[独立控制](evaluation-controls.json)、[资源释放记录](resource-release.json)：软件/驱动、数据、定位器、9 个 CPU 合约、跨机恢复及最终 GPU 进程检查。
- [完整性验证](verification.json)、[报告源版本](report-source-manifest.json)、[文件哈希](artifact-manifest.json)、[图表审阅记录](figure-review.json)。

**关键图索引**

| 图 | 直接回答的问题 |
| --- | --- |
| [训练预算](figures/training-budget-evidence.png) / [SVG](figures/training-budget-evidence.svg) | 同一条轨迹继续训练，表征、生成、规划是否共同改善？ |
| [主要终点](figures/final-comparison.png) / [SVG](figures/final-comparison.svg) | 各配置的规划、agent 预测与物块运动能否同时成立？ |
| [外观分工](figures/color-mechanism.png) / [SVG](figures/color-mechanism.svg) | 背景配色来自 anchor 还是 state，两个训练种子是否一致？ |
| [物块几何](figures/color-geometry-interventions.png) / [SVG](figures/color-geometry-interventions.svg) | 换 anchor 保留原几何、换 state 转到 donor 几何了吗？ |
| [Native 真实历史](figures/native-context-intervention.png) / [SVG](figures/native-context-intervention.svg) | 有单步能力的模型为何在自由 rollout 中失稳？ |
| [后验与预测](figures/posterior-forecast.png) / [SVG](figures/posterior-forecast.svg) | 25k 的 agent 重建缺口是否在 50k 改善？ |
| [随机点](figures/dot-information.png) / [SVG](figures/dot-information.svg) | flow 是否比 MSE 更会去除随机细节？ |
| [精度](figures/precision-diagnostics.png) / [SVG](figures/precision-diagnostics.svg) | 较高 BF16 方差是否代表有用状态？ |
| [Native 自由 rollout](figures/native-rollout-drift.png) / [SVG](figures/native-rollout-drift.svg) | 首步改善是否能延续？ |
| [历史裁剪](figures/native-feedback-clipping.png) / [SVG](figures/native-feedback-clipping.svg) | 裁回训练范围是否修复漂移？ |
| [50k 固定前 16 例](figures/rollout-50k-first16.png) | 未按误差筛选的生成结果长什么样？ |
| [25k 有利样例](figures/positive-selected-beta001-25k.png) | 早期局部成功长什么样？ |

50k 图采用固定 256 例顺序的前 16 例，附 [case 元数据](rollout-50k-first16-cases.json)，没有按误差筛选。25k 有利样例按 h7 agent<40px、block<25px 且两者都优于复制筛选，256 例中共 10 例满足，图中展示 3 例；见[选择说明](selected-cases.json)。它们均不构成原方案的 32 例人工 gate。

位置误差使用 512×512 模拟器坐标，图像为 64×64。P0 为 3 个评估种子各 50 局，均值±SD 不代表训练种子区间。生成区间是按测试 episode 重采样的条件 95% 区间；噪声状态结果平均每例全部 4 次采样。大 renderer 的噪声采样是 64 例，其他 L2 终点为 256 例。L0 没有 renderer。

Native 真实历史和 L2 真未来后验都使用额外真实信息，仅供诊断。颜色干预使用同一批 128 例、同一反向 donor 配对与相同噪声，分别只更换 anchor 或 state。原方案三训练种子统计尚未完成，LPIPS/SSIM/FVD 未测，旧 dot pose-v1 分数排除。

数值包含 1,040 条聚合记录。发布报告源为提交 0533159aa3e6efb55d5e99f0b52ced25ce1801c6；主表、诊断和核验使用与此提交字节相同的 v32 脚本，聚合导出使用 v33。源文件哈希见 manifest。复查可在 EFS 的冻结报告源目录运行：

```bash
cd /efs/prescottsun/SPT-JEPA/runs/insights12h_20261027/source-report-v33
C=/efs/prescottsun/SPT-JEPA/runs/insights12h_20261027
PY=/efs/prescottsun/project_jepa/envs/lewm/bin/python
OUT=/efs/prescottsun/SPT-JEPA/reports/review-new-directory
mkdir -p "$OUT"
"$PY" -B -m as_jepa_tools.summarize_final --campaign "$C" --output "$OUT"
"$PY" -B -m as_jepa_tools.summarize_diagnostics --campaign "$C" --output "$OUT"
"$PY" -B -m as_jepa_tools.export_campaign --campaign "$C" --output "$OUT"
"$PY" -B -m as_jepa_tools.verify_campaign --campaign "$C" --output "$OUT/verification.json" --require-complete
```

权重、数据、逐 case 数组、原始日志和视频保存在 /efs/prescottsun/SPT-JEPA/runs/insights12h_20261027。Git 只发布代码、聚合数值、元数据与展示图。无需恢复或重启历史任务。
