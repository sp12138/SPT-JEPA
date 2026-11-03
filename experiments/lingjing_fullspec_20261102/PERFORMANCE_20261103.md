# Lingjing 新提案的最新性能：尚未超过旧方案

最终状态已由[2026-11-03收尾报告](CLOSEOUT_20261103.md)和[最终逐条覆盖](results/closeout20261103/coverage-final.json)更新；本文下方保留原时间范围的证据。

2026-11-03 北京时间零点后的一次只读复核；精确快照UTC时间见JSON。两个远程分支仍为 anchored-state `43962c5`、decoder-free `fbb3efa`，没有新的上传提交。

**目前没有已经验证、性能超过旧方案的新配置。** 最新变化是18条 decoder-free 已完整完成37473更新/3个loader epoch与全部六个固定P0评测，其规划性能仍很弱。这里使用完整阶段终点，包含两条保存故障恢复后的结果，没有把执行错误当成算法失败。

下表全部使用训练seed1/2均值；primary为42/43/44，fresh为1042/1043/1044，每项50回合。已逐例核对旧TF与全部新输出的episode/start一致。没有为不完整配对选好种子。

| PushT配置 | Primary成功率 | Fresh成功率 |
|---|---:|---:|
| 旧temporal-forecast，37473步 | **94.00%** | **92.33%** |
| DF A，β=.1 / λ=0 | 4.00% | 4.00% |
| DF A，β=.1 / λ=.5 | 2.00% | 2.00% |
| DF A，β=.1 / λ=1 | 3.00% | 3.33% |
| DF A，β=.3 / λ=0 | 4.00% | 2.33% |
| DF A，β=.3 / λ=1 | 2.67% | 2.00% |
| DF A，β=1 / λ=0 | 2.33% | 2.33% |
| DF B，λ=1 / open-loop sample | 2.00% | 2.00% |

旧TF逐seed为94.00/94.00% primary、93.33/91.33% fresh。七个完整双seed DF配置的均值均在2–4%范围。另有四条已完成的单seed DF（B一条、dot域E三条），逐条保留在JSON，不把不完整配对当作胜出或稳定性证据。colour L0控制也已完成，但其域/训练曝光不同，不拿其约2.67%的原域P0去替代标准域成熟L0基线。

这里的A/B完成组使用pooled logdet，是批耦合对照；sample-level window目标的D组和新的forecast-scored因素配对尚未收齐。不能据此说全部后续提案均已失败，也没有把pooled版本算成用户要求的sample-level新算法。

旧TF与新DF更新数、实际heldout案例及P0协议相同，但模型、训练FLOPs不同；旧TF是指定历史归档，未挑选同模型的最佳重跑。对比不涵盖simulator-repeat不确定性，完成3epoch不等于证明收敛。

Cube和TwoRoom的结论未改变：Cube12条12000步/48终点均未达到续训门槛，例如K1 white为14/16%，同预算Native24/29%；TwoRoom26个主要提案未通过状态使用/可读性门槛并按规则结束，没有更优控制证据。不能把旧local1024+NGC的TwoRoom高分归到这轮新上传方案。

当前106条主队列快照：12条Cube继承完成，19条PushT名义完成（18条DF+1条colour L0），5条首epoch联合失效早停，70条未完成。此计数已通过CORRECTIONS读取补充终点；没有尚未恢复的执行错误终点。70条包括已启动和待启动，不等于70条正在训练。仍按既有有限队列运行，本次未启停训练或追加配方、预算、监控。

下一项判断等待现有forecast-scored P2及DF B/D的完整配对；目前没有依据启动DF Phase2或新的Cube长训。此前标准L0的90.89/87.33%来自10epoch，必须与这里的3epoch、DF损失类型分开标注。

完整证据：[所有完成的新输出](results/performance20261102T1610Z/performance.json)、[旧TF同案例参考](results/performance20261102T1610Z/old-temporal-forecast.json)、[全部配对汇总与分支SHA](results/performance20261102T1610Z/summary.json)。前一轮Cube与成熟L0分析见 [REVIEW_20261102_1229.md](REVIEW_20261102_1229.md)。
