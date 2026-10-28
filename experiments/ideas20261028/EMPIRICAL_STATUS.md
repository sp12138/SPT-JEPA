# 十候选的实际验证状态

核查UTC：2026-10-28T07:44:04.391953+00:00。这里逐项给出已实现和正在运行的对照，不能把代码存在或训练不崩写成方法有效。原始完整预算实验同时继续。

| ID | 假设 | 实际运行状态/更新步 | 效果判定 |
| --- | --- | --- | --- |
| I01 | 将生成收益与状态KL统一到nats | running @40900；running @7600 | 待最终指标和新case确认 |
| I02 | 按decoder几何分配状态噪声 | running @37700 | 待最终指标和新case确认 |
| I03 | 相关后验噪声与真实jointKL | running @40800 | 待最终指标和新case确认 |
| I04 | 让anchor定义颜色坐标 | running @38100；queued | 待最终指标和新case确认 |
| I05 | 直接训练forecast状态的生成条件 | running @34300 | 待最终指标和新case确认 |
| I06 | 保护FM梯度的encoder更新 | running @23400 | 待最终指标和新case确认 |
| I07 | 按因果可预测性加权FM | running @39700 | 待最终指标和新case确认 |
| I08 | 分离语义记忆与图像反馈 | running @19500 | 待最终指标和新case确认 |
| I09 | 用真实动作fork约束状态 | queued | 待最终指标和新case确认 |
| I10 | 局部可控能量规划 | exit0；exit0；exit0 | 负面，已完整配对验证 |

[结构化记录](EMPIRICAL_STATUS.json)包括每个方法的真实实现差异、控制组、源提交、配置hash、机器与退出证据。I01的负ELBO、I04的anchor颜色变换与最初surrogate分别报告；I08预训练及finetune成本分开；I10旧缓存结果作无效历史保留。

确认集使用新seed并移除和旧诊断16帧clip共享帧的窗口；反事实及规划也用事先声明的新case种子。仍是同一test episode split，不能冒充独立数据集或多训练种子。完整数值和图见[追加结果](../completion20261028/results/README.md)。
