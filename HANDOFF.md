# 当前交接

最新目标是超过 RoboJEPA，尚未达成。用户现授权 dev13、dev15、dev6、dev6-2c 四台，尽量用于有效实验。旧 MetaWorld／PushT 等研究、goal 和定时 LLM 监控不恢复。协议见 [robocasa_refine/PROTOCOL.md](experiments/robocasa_refine/PROTOCOL.md)。不把有限实验控制器当作新 goal 或周期监控。

六个原始5000步模型已完成240回合：TF／spatial／long 的 Reach 平均进度21.80／18.33／20.55%，Push 最终进度4.59／3.68／11.36%；严格完成均0。这是旧 issued-history pilot，不能当官方客户端复现。其根目录 `runs/robocasa_compare_20261110` 已STOP，未完的旧参考评测已明确取消，勿重启。源图像视场裁剪修复保留；原始随机初始化、5000步祖先与一次未保存更新的重放均保留。

`runs/robocasa_official_20261110` 的修正参考链已实际验证：50M120k、2B450k、8B300k 的全部预测器参数均与下载权重一致；逐相机编码、实际运动历史回放、KV缓存、下一步预测及短CEM与直接公开调用相同。编码FP16、预测BF16按论文表格；发布客户端的FP16预测有差异，不声称逐项完整论文复现。RGB零绝对状态、released手工状态、native训练对齐状态分别报告。外部权重只作冻结参考，不进入候选初始化／教师。

`runs/robocasa_refine_20261110`：两个原long父模型各续训control／clean／highnoise1000步，六个6000步恢复／梯度检查已完成。control-seed2从真实5250完整状态跨机恢复750步，0重放，原错误保留。clean的预测／copy误差比降至0.91／0.92，但旧共同可重放Reach案例控制变差；factual／zero预测误差比0.92／0.95，control约0.61／0.62。不同表示的裸MSE不可直接比较，未模拟的零动作也不是反事实因果证据。源验证Reach-v1不稳定，相关技术失败不计模型负结果。

`runs/robocasa_intervention_20261110` 已完成：在128个原TRAIN厨房尝试普通随机控制，222条有效轨迹／111厨房／3552转移，34个初始化几何失败保留。独立加盐分组得到102训练厨房、9新日志验证厨房，旧54验证厨房和十个测试厨房全部排除。三组同用RGB日志，各两个同源5000步父模型，完整Adam/RNG/游标续训1000步。六个6000步模型／24个源闭环回合完成，实际位移、发出指令、高噪声的Reach／Push平均进度为34.62／7.29%、40.27／11.00%、13.14／9.26%；零动作Push已有9.09%。严格完成全0，两个父模型改善不一致，NO_LONG_PROMOTION。无inverse、最优／未来动作标签、奖励、专家路径或外部教师。完整结论在该根目录 `results/source-summary.json`。

该轮源案例先在三台各两个新进程严格重放；选定Reach-v2/v3、Push-v2/v3，seed7401，24次选定案例检查全部通过，状态容差仍1e-8。全部66次检查及4次不可靠Reach-v1失败保留。目标图像只用于规划。不能混用旧源案例分数。当前入口：`training/*/{progress,audit-006000,ERROR}.json`、`source-evaluation/`、`REPLAY_APPROVED.json`。

dev6-2c 已接通，8×H200；八路冻结诊断完成，模拟器资产校验／缓存完成。全量50M／2B450k／8B300k冻结参考队列已从三台等待控制器转交此机，前三台可继续候选训练。正式参考先对新主机共同目标案例做物理／RGB预检，入口 `NEW_HOST_ENTRY.json`、`public-launch-dev6-2c.json`、`public-evaluation/`。同评测不等于同训练预算；新增随机交互属于另一数据配方，不能据此直接宣称同数据方法SOTA。

当前新实验为 `runs/robocasa_local_20261110`，六个完整5000步父模型恢复已通过：dev13 global-spatial，dev15 regional-spatial，dev6 analytic-noise，各两个父模型、1000步续训上限。局部编码在训练和模拟环境都实际验证区域外扰动响应为0，原全局骨干为0.1344；解析噪声条件KL核对误差<5.2e-6，未来编码不进入学习 rollout。二者均无新增参数，保留sample-level目标。局部／全局使用完全相同spatial父权重与日志；analytic与上一轮issued共享long父模型。短窗8×batch128和长窗16×batch64均每步2048相机图像，FLOPs及目标数不同。完整恢复、5002梯度及真实MPC入口检查后继续998步，再在共同源案例比较uniform／support代价。不能把结构证明当性能突破。入口 `launch-dev*.json`、`training/*/`、`source-evaluation/`；冻结源码只在该根目录独立experiments副本内，旧R4不改写。

旧R3源参考仍占卡的进程已明确退役（`retirement-dev13/dev15.json`），保留部分结果及旧重放失败；完整官方参考继续由第四台负责。新 `physics_ceiling.py` 仅作受保护的状态辅助接触／可达性诊断，与公开18指令、0.3范数约束一致，轨迹禁止进入训练、教师或proposal memory。

2026-11-09 最新 causal-inverse 研究已结束：8 个模型各新增 10,000 步，384 个控制评测回合，无执行错误。六项困难 manipulation 在每种方法、每种控制器下均为 0/12；尚无可靠性能突破。每组复用六个物理案例及两个原训练父模型，不能把总回合数当独立案例数。

- [核心结论](experiments/CONCLUSIONS.md)
- [较优配置与适用边界](experiments/best-configs.json)
- [原始记录及清理归档](docs/experiment-archives.md)
- [项目约定](AGENTS.md)

源码保留在 `as_jepa_lewm/`、`as_jepa_tools/`、`jobs/`、`runtime/` 和各 `experiments/` 方法目录。历史实验脚本可能需要从归档恢复配置／数据才能重放；不据旧运行状态自动启动。
