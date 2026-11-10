# 当前交接

最新目标是匹配条件下超过 RoboJEPA，尚未达成。最新用户明确授权1000→5000→10000步续训、在本会话每小时检查；若性能差，先复现官方RoboJEPA，再逐个融合／修正SPT组件。四机dev13/dev15/dev6/dev6-2c可用，旧任务／goal／旧监控不恢复。当前检查为持续会话等待，没有创建不存在的托管定时任务。

新有限根 `runs/robocasa_encoder_budget_20261111` 已实际启动：8个原1000步完整状态，零NN／优化器动量／rank RNG／游标改动，统一重启9000步cosine／90步warmup。相同数据、窗口、batch、NN、loss、控制案例；不是从零固定LR日程scaling law。新stages1002（恢复＋真实MPC入口）／5000／10000，每个正式点32回合，共64新回合＋32继承回合，四物理案例。八个实际恢复和1002入口均已通过、继续SGD，`STARTUP_APPROVED.json`，NN及输入定义与v7的AST相同。源 `sources/encoder-budget-20261111-v1`，核心sha dbcfc09b；启动 `launch-dev*.json`，finite collector `collector-launch.json`，结果 `results/{step-001000,step-005000,step-010000,summary}.json`。六小时cap，旧根HOLD保持。外部骨干预算不匹配、无未来／最优动作target、无proprio／奖励输入。

当前每小时检查计划在新根 `SESSION_CHECKS.json`，首次2026-11-10 18:31:46UTC（Nov11 02:31北京时间），之后每小时；只在当次实际检查后记录结果。若10k仍差，官方公开权重评测复现与同数据训练复现分别标注，不把自建dense头冒称官方RoboJEPA。先建立同输入／图像预处理／历史动作／CEM的参考锚点，再一次只换一个SPT组件；额外预训练或本体输入只能分轨，不能冒称公平算法胜利。第四台的旧官方目标厨房队列仍在dev6-2c独立运行。

最新用户明确授权试用 V-JEPA2.1，现追问预算是否不足。有限根 `runs/robocasa_encoder_20261110` 已收齐8个1000步新头、32个源闭环回合，严格final／any成功全0，HOLD。Own／SPT、own／dense、V-JEPA／SPT、V-JEPA／dense的Reach／Push进度为17.32／9.33%、22.07／5.52%、11.03／9.33%、10.11／9.26%。这些是进度；同案例旧零动作Push已有9.09%。同1024随机日志，RGB／当前指令、无动作target或proprio；只导入外部视觉编码器，非同预算从零SOTA比较。Own骨干继承7000步，但所有预测头重新初始化，不能当成熟模型额外续训1000步。只有四个物理源案例和两个新头seed，不是32独立案例。

只读预算复核已完成24项：现有500／1000检查点、相同128次train／validation窗口抽样，每seed身份一致，新增SGD／模拟0。验证7步预测／复制误差为own-SPT1.35→1.34、own-dense0.95→0.91、V-JEPA-SPT1.67→1.51、V-JEPA-dense0.90→0.89；1000步SPT训练片段仍为1.29／1.46。预算不足值得检验，不能凭短训0成功否定编码器／扩展性；SPT目标表示也在变化，且预测误差不是控制成功率。正式头训练约4–7分钟，尚未启动5000／10000步续训。三主机的本轮训练与只读预算队列均结束，第四台原官方参考继续；不恢复旧goal或LLM监控。

有效代码v7，入口 `launch-training-v7-*.json`；梯度预检v4、缓存v2和8个原4步完整状态沿用，已完成SGD重放0。启动错误与修正保留；`STARTUP_APPROVED-v7.json` 为实际恢复／MPC证明。权威结果 `results/summary-v7.json`，预算复核 `budget-review/summary.json`；早期incomplete summaries/HOLD是历史。核心源码 `experiments/robocasa_refine/encoder_ablation.py`，协议见末节。

六个原始5000步模型已完成240回合：TF／spatial／long 的 Reach 平均进度21.80／18.33／20.55%，Push 最终进度4.59／3.68／11.36%；严格完成均0。这是旧 issued-history pilot，不能当官方客户端复现。其根目录 `runs/robocasa_compare_20261110` 已STOP，未完的旧参考评测已明确取消，勿重启。源图像视场裁剪修复保留；原始随机初始化、5000步祖先与一次未保存更新的重放均保留。

`runs/robocasa_official_20261110` 的修正参考链已实际验证：50M120k、2B450k、8B300k 的全部预测器参数均与下载权重一致；逐相机编码、实际运动历史回放、KV缓存、下一步预测及短CEM与直接公开调用相同。编码FP16、预测BF16按论文表格；发布客户端的FP16预测有差异，不声称逐项完整论文复现。RGB零绝对状态、released手工状态、native训练对齐状态分别报告。外部权重只作冻结参考，不进入候选初始化／教师。

`runs/robocasa_refine_20261110`：两个原long父模型各续训control／clean／highnoise1000步，六个6000步恢复／梯度检查已完成。control-seed2从真实5250完整状态跨机恢复750步，0重放，原错误保留。clean的预测／copy误差比降至0.91／0.92，但旧共同可重放Reach案例控制变差；factual／zero预测误差比0.92／0.95，control约0.61／0.62。不同表示的裸MSE不可直接比较，未模拟的零动作也不是反事实因果证据。源验证Reach-v1不稳定，相关技术失败不计模型负结果。

`runs/robocasa_intervention_20261110` 已完成：在128个原TRAIN厨房尝试普通随机控制，222条有效轨迹／111厨房／3552转移，34个初始化几何失败保留。独立加盐分组得到102训练厨房、9新日志验证厨房，旧54验证厨房和十个测试厨房全部排除。三组同用RGB日志，各两个同源5000步父模型，完整Adam/RNG/游标续训1000步。六个6000步模型／24个源闭环回合完成，实际位移、发出指令、高噪声的Reach／Push平均进度为34.62／7.29%、40.27／11.00%、13.14／9.26%；零动作Push已有9.09%。严格完成全0，两个父模型改善不一致，NO_LONG_PROMOTION。无inverse、最优／未来动作标签、奖励、专家路径或外部教师。完整结论在该根目录 `results/source-summary.json`。

该轮源案例先在三台各两个新进程严格重放；选定Reach-v2/v3、Push-v2/v3，seed7401，24次选定案例检查全部通过，状态容差仍1e-8。全部66次检查及4次不可靠Reach-v1失败保留。目标图像只用于规划。不能混用旧源案例分数。当前入口：`training/*/{progress,audit-006000,ERROR}.json`、`source-evaluation/`、`REPLAY_APPROVED.json`。

dev6-2c 已接通，8×H200；八路冻结诊断和模拟器缓存校验完成，50M／2B450k／8B300k参考队列由此机负责。共同目标物理／RGB预检在R4 `NEW_HOST_ENTRY.json`，结果仍在R4 `public-evaluation/`。当前调度入口已改为 `runs/robocasa_reference_adopt_20261110/launch.json`（PID2920，须核对实际存活）；原八个chunk2158–2165原样接管，当前案例未终止或重跑，旧父控制器1548/1549明确退役。后续chunk只调整Python GC节奏；三种模型输出/RNG完全一致，H5/128候选两次refit及1024候选缓存测试通过，重复显存不增长，提速1.34／1.11／1.25倍。原24小时截止不变。最终collector在新根 `results/summary.json`，不再等旧父进程的完成标记。同评测不等于同训练预算；新增随机交互不是同数据SOTA证明。

`runs/robocasa_local_20261110` 已完成六个6000步模型与40个源闭环回合。global-spatial的uniform／support分别为Reach45.14／18.76%、Push9.24／34.88%；regional-spatial为32.08／13.67%、4.65／9.28%；analytic-noise为40.32／0%。严格完成全0。Support对两个global父模型的Push均改善，但损害Reach，NO_LONG_PROMOTION。源物理案例仍只有四个。局部结构扰动／条件KL证明不能当控制改善。global-seed2原终检1.43e-4数值差已用FP32 math SDPA复核至2.38e-7，固定形状其他样本污染0，完整6000状态及所有梯度通过，无SGD重放；原错误在CORRECTIONS中保留。该根目录 `results/source-summary.json` 为结论入口。

旧R3源参考仍占卡的进程已明确退役（`retirement-dev13/dev15.json`），保留部分结果及旧重放失败；完整官方参考继续由第四台负责。新 `physics_ceiling.py` 仅作受保护的状态辅助接触／可达性诊断，与公开18指令、0.3范数约束一致，轨迹禁止进入训练、教师或proposal memory。

`runs/robocasa_forks_20261110` 已全部完成。实际采集64个原TRAIN厨房×两个初态×八个普通16指令分支，共1024条／16,384转移，128完整组，116训练组／12验证组（58／6厨房），0采集错误；不读目标、奖励、成功或对象状态选动作。分支严格重放同初态，再共用实测branch0初始RGB；物理状态只在独立技术证明中。TREE_APPROVED的目标代数误差3.1e-9、终端动作污染0、所有梯度为正，是正确性证明，不能当性能改善。

三组各两个原issued6000步父模型都到7000，完整恢复／梯度／6002真实MPC入口检查和48源回合全部通过。Uniform的Reach／Push平均进度：ordinary-flat41.28／6.17%，fork-flat38.61／9.30%，fork-shared35.77／10.34%。Direction下分别45.20／9.31%、32.55／9.29%、30.94／27.29%。全部严格完成0；shared-direction的Push两父模型为45.34／9.25%，改善不稳定，且未超过旧global-support的源验证均值34.88%。NO_LONG_PROMOTION。结果入口 `results/{summary,review}.json`。四个源物理案例及两个从零祖先重复评测，不是48个独立案例，不是留出厨房SOTA比较。

三个方法同用analytic条件先验、每步2048相机图像、1000新增步；分组batch8×8对应flat64，游标按8换单位，旧数据对照与分支语料不同，fork-flat／fork-shared数据和采样相同。shared仅把共同根entry计一次（等价该项乘1/8），没有新增参数或未来动作标签。结论不支持直接长训，也不支持把该缩放称为新范式。原有限控制器均结束；2026-11-10 13:09UTC实测dev13/dev15/dev6本轮控制器已退出、24卡0MiB，这是当时快照。

采集提速曾由 `acquisition-launch-parallel-dev*.json` 接管，保留当时在跑程序；48个独立初态/host并行，案例、动作、episode函数、上限及截止不变。该队列现已全部退出，原始日志／SHA／接管证据仍在EFS。不要据这些旧启动记录自动恢复任务。

同根目录 `results/direction-summary.json` 的24次冻结方向投影评测也未解决Reach／Push取舍。三种GC节奏测试是执行基准，不是性能进步；当前参考chunk仍为原实现，后续chunk才用经核对的新回收节奏。

dev6-2c的公开参考仍在跑。13:09UTC，released-proprio轨道的20个共同Reach测试案例已齐：50M91.00%、2B93.40%、8B96.12%平均进度；严格final为0／3／1次。Push未齐，不汇报部分案例均值作排名。其测试厨房、外部预训练和绝对本体输入不同于上述源验证RGB候选；不能直接把两张表当同条件差距。

最新用户追问失败本质／关键点。完成 `runs/robocasa_geometry_review_20261110` 的冻结诊断，0SGD／0新模拟：新shared两个模型和官方视觉编码器在同两物理Push案例四路径上均把第1步判为图像目标距离最小，所以全图距离偏离物体进度是共有问题，不能单独归因成我们输官方的原因。12个交互验证初态×八个真实当前a0，分支模型辨识对应下一帧为77%–82%，否定“完全没学动作”的笼统描述；它仍可能主要利用机械臂／视角变化，不能证明对象接触或长程控制。官方动作诊断仅mode2/action-key＋已裁切图像，不是主mode3轨道，不能据此正式排名。几何与动作诊断已经结束，原模型均未变；入口 `SUMMARY.json`、`review.py`、`action-review.py`，源码SHA在launch记录。下一项有区分力的问题是固定机器人／相机姿态后，物体状态信息是否保留、后果能否预测；尚未启动这类新模拟或训练。

2026-11-09 最新 causal-inverse 研究已结束：8 个模型各新增 10,000 步，384 个控制评测回合，无执行错误。六项困难 manipulation 在每种方法、每种控制器下均为 0/12；尚无可靠性能突破。每组复用六个物理案例及两个原训练父模型，不能把总回合数当独立案例数。

- [核心结论](experiments/CONCLUSIONS.md)
- [较优配置与适用边界](experiments/best-configs.json)
- [原始记录及清理归档](docs/experiment-archives.md)
- [项目约定](AGENTS.md)

源码保留在 `as_jepa_lewm/`、`as_jepa_tools/`、`jobs/`、`runtime/` 和各 `experiments/` 方法目录。历史实验脚本可能需要从归档恢复配置／数据才能重放；不据旧运行状态自动启动。
