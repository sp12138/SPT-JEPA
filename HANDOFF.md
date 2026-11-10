# 当前交接

最新目标是超过 RoboJEPA，尚未达成。用户现授权 dev13、dev15、dev6、dev6-2c 四台，尽量用于有效实验。旧 MetaWorld／PushT 等研究、goal 和定时 LLM 监控不恢复。协议见 [robocasa_refine/PROTOCOL.md](experiments/robocasa_refine/PROTOCOL.md)。不把有限实验控制器当作新 goal 或周期监控。

六个原始5000步模型已完成240回合：TF／spatial／long 的 Reach 平均进度21.80／18.33／20.55%，Push 最终进度4.59／3.68／11.36%；严格完成均0。这是旧 issued-history pilot，不能当官方客户端复现。其根目录 `runs/robocasa_compare_20261110` 已STOP，未完的旧参考评测已明确取消，勿重启。源图像视场裁剪修复保留；原始随机初始化、5000步祖先与一次未保存更新的重放均保留。

`runs/robocasa_official_20261110` 的修正参考链已实际验证：50M120k、2B450k、8B300k 的全部预测器参数均与下载权重一致；逐相机编码、实际运动历史回放、KV缓存、下一步预测及短CEM与直接公开调用相同。编码FP16、预测BF16按论文表格；发布客户端的FP16预测有差异，不声称逐项完整论文复现。RGB零绝对状态、released手工状态、native训练对齐状态分别报告。外部权重只作冻结参考，不进入候选初始化／教师。

`runs/robocasa_refine_20261110`：两个原long父模型各续训control／clean／highnoise1000步，六个6000步恢复／梯度检查已完成。control-seed2从真实5250完整状态跨机恢复750步，0重放，原错误保留。clean的预测／copy误差比降至0.91／0.92，但旧共同可重放Reach案例控制变差；factual／zero预测误差比0.92／0.95，control约0.61／0.62。不同表示的裸MSE不可直接比较，未模拟的零动作也不是反事实因果证据。源验证Reach-v1不稳定，相关技术失败不计模型负结果。

`runs/robocasa_intervention_20261110` 已完成：在128个原TRAIN厨房尝试普通随机控制，222条有效轨迹／111厨房／3552转移，34个初始化几何失败保留。独立加盐分组得到102训练厨房、9新日志验证厨房，旧54验证厨房和十个测试厨房全部排除。三组同用RGB日志，各两个同源5000步父模型，完整Adam/RNG/游标续训1000步。六个6000步模型／24个源闭环回合完成，实际位移、发出指令、高噪声的Reach／Push平均进度为34.62／7.29%、40.27／11.00%、13.14／9.26%；零动作Push已有9.09%。严格完成全0，两个父模型改善不一致，NO_LONG_PROMOTION。无inverse、最优／未来动作标签、奖励、专家路径或外部教师。完整结论在该根目录 `results/source-summary.json`。

该轮源案例先在三台各两个新进程严格重放；选定Reach-v2/v3、Push-v2/v3，seed7401，24次选定案例检查全部通过，状态容差仍1e-8。全部66次检查及4次不可靠Reach-v1失败保留。目标图像只用于规划。不能混用旧源案例分数。当前入口：`training/*/{progress,audit-006000,ERROR}.json`、`source-evaluation/`、`REPLAY_APPROVED.json`。

dev6-2c 已接通，8×H200；八路冻结诊断和模拟器缓存校验完成，50M／2B450k／8B300k参考队列由此机负责。共同目标物理／RGB预检在R4 `NEW_HOST_ENTRY.json`，结果仍在R4 `public-evaluation/`。当前调度入口已改为 `runs/robocasa_reference_adopt_20261110/launch.json`（PID2920，须核对实际存活）；原八个chunk2158–2165原样接管，当前案例未终止或重跑，旧父控制器1548/1549明确退役。后续chunk只调整Python GC节奏；三种模型输出/RNG完全一致，H5/128候选两次refit及1024候选缓存测试通过，重复显存不增长，提速1.34／1.11／1.25倍。原24小时截止不变。最终collector在新根 `results/summary.json`，不再等旧父进程的完成标记。同评测不等于同训练预算；新增随机交互不是同数据SOTA证明。

`runs/robocasa_local_20261110` 已完成六个6000步模型与40个源闭环回合。global-spatial的uniform／support分别为Reach45.14／18.76%、Push9.24／34.88%；regional-spatial为32.08／13.67%、4.65／9.28%；analytic-noise为40.32／0%。严格完成全0。Support对两个global父模型的Push均改善，但损害Reach，NO_LONG_PROMOTION。源物理案例仍只有四个。局部结构扰动／条件KL证明不能当控制改善。global-seed2原终检1.43e-4数值差已用FP32 math SDPA复核至2.38e-7，固定形状其他样本污染0，完整6000状态及所有梯度通过，无SGD重放；原错误在CORRECTIONS中保留。该根目录 `results/source-summary.json` 为结论入口。

旧R3源参考仍占卡的进程已明确退役（`retirement-dev13/dev15.json`），保留部分结果及旧重放失败；完整官方参考继续由第四台负责。新 `physics_ceiling.py` 仅作受保护的状态辅助接触／可达性诊断，与公开18指令、0.3范数约束一致，轨迹禁止进入训练、教师或proposal memory。

当前有限队列为 `runs/robocasa_forks_20261110`。三台采集64个原TRAIN厨房×两个初态×八个普通16指令分支，上限1024条／16,384转移；不读目标、奖励、成功或对象状态选动作。分支先严格重放同初态，再共用实测branch0初始RGB；物理状态只在独立技术证明中。TREE_APPROVED已在真实训练八分支上通过：目标代数误差3.1e-9、终端动作污染0、所有梯度为正，0SGD／0新参数。数据仍在采集，不能声称已收齐。

三组各两个原issued6000步父模型：dev13 ordinary-flat用旧普通日志先跑对照，两个6002完整审计／真实MPC入口均通过，已实际续训；dev15 fork-flat和dev6 fork-shared等完整分支语料后再训。都是analytic条件先验、每步2048相机图像、1000步上限；分组batch8×8对应flat64，游标按8换单位并披露，不宣称样本顺序不变。shared只把共同根entry计一次（等价该项乘1/8），属于有依据的消融，尚不是创新或性能证明。6000/6002/7000完整审计及48源回合后结束，不自动长训。当前所有权 `phase-launch-{coordinator,worker-dev*}-v2.json`；旧v1控制器无SGD时明确退役。

采集现由 `acquisition-launch-parallel-dev*.json` 接管，旧父控制器退役但各八条在跑的程序保留。每台实际CPU配额160核，原八进程利用不足；新调度最多48个独立初态并行，每个初态仍在原GPU顺序执行八个完全相同的程序。案例、动作、episode函数、语料上限、原截止均不变；完成文件逐个验SHA后复用，不重跑。操作helper为根外冻结 `acquisition-parallel.py`，SHA及原子接管记录都在EFS。物理仿真阶段GPU利用率会低于训练，不宣称持续32卡满算。结果入口为 `results/summary.json`。

同根目录 `results/direction-summary.json` 的24次冻结方向投影评测也未解决Reach／Push取舍。三种GC节奏测试是执行基准，不是性能进步；当前参考chunk仍为原实现，后续chunk才用经核对的新回收节奏。

2026-11-09 最新 causal-inverse 研究已结束：8 个模型各新增 10,000 步，384 个控制评测回合，无执行错误。六项困难 manipulation 在每种方法、每种控制器下均为 0/12；尚无可靠性能突破。每组复用六个物理案例及两个原训练父模型，不能把总回合数当独立案例数。

- [核心结论](experiments/CONCLUSIONS.md)
- [较优配置与适用边界](experiments/best-configs.json)
- [原始记录及清理归档](docs/experiment-archives.md)
- [项目约定](AGENTS.md)

源码保留在 `as_jepa_lewm/`、`as_jepa_tools/`、`jobs/`、`runtime/` 和各 `experiments/` 方法目录。历史实验脚本可能需要从归档恢复配置／数据才能重放；不据旧运行状态自动启动。
