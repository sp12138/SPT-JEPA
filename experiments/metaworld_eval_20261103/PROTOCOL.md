# Meta-World：无需授权的独立数据上的方法评测

2026-11-03用户授权评测我们的方法，并明确要求寻找无需HF授权的数据。选用LeRobot独立发布的`lerobot/metaworld_mt50`，固定revision `b46a1c185a2368660650a82ac5a5083a56722f48`。原`facebook/jepa-wms`数据继续保留访问失败证据，不绕过gate、不把本数据称为它的镜像或等价复现。

实际无token全量下载已完成，497个文件共45,864,127,230 bytes。全语料逐行核对2500条episode、204806帧和50个实际task_id，所有PNG均已解码。当前注册表50项与task_id分组一致；不用历史版本合并过的push/push-back文本分组。本数据来自expert policies，与JEPA-WMs的TD-MPC2早期混合质量数据分开报告。

## 固定的比较边界

- 主方法是已有temporal-forecast，保留原sample-level目标、ρ=.8、forecast_fraction=.5，不以新数据分数调noise/anchor。记录与原PushT/Cube版本相比必需的接口适配。
- 主线RGB-only，已执行4D动作作为条件；不把动作作为最优动作预测标签。不提供39D环境真值、reward/success/任务语言给学习器。若另跑RGB+4D proprio，所有方法同轨道输入一致。
- 轨迹级、按task_id分层划分训练/验证，固定同一份清单；相邻窗口不能跨划分。任务名和task_id只用于采样/分组和环境选择。先检查历史push-back/push文本合并问题；当前50项注册不能替代全帧核验。
- 核对数据camera、翻转、原生版本、动作裁剪语义和state/action时序。使用上游公开相机与环境处理，不能靠逐任务视觉调参匹配。图像、动作scaler只用训练数据拟合并对所有方法固定。
- 基线必须在相同独立数据、输入、图像预处理、目标和评测预算下比较。现成JEPA-WM/DINO-WM权重只作明确标注的外部参考，不冒充同数据重训。预训练资源/参数/FLOPs差异单列。
- 先2-update原生训练/完整model–optimizer–scheduler恢复与GPU反传，再短程训练和完整规划；技术通过不等于任务成功。预先保存数据/模型/控制器合同后才启动正式训练。
- 第一组评估预先保留Reach/Reach-Wall及push、pick-place、drawer-open、door-open、peg-insert-side、bin-picking；逐任务报告，不只挑成功任务。之后扩展全部可验证task_id。
- 原JEPA-WMs的100原始步final-success及Farama/LeRobot的500步any-success分开，若扩展两套规则，所有方法同样评估。原published Reach分数不直接与本数据/相机结果做优越性结论。

## 数据与训练合同

将所有5个时间相位都保留，原始t处输入图像与执行动作a[t:t+5]对应t+5图像。每个动作块为20D；storage stride=1、physical frameskip=5。最后一帧零填充动作不参与有效转移。所有图像使用CPU torch bilinear/antialias缩为224、round uint8，规划使用相同处理。动作按原生Box[-1,1]裁剪后拼接，mean/std只由训练轨迹拟合。

按task_id各保留5条原始轨迹为test，45条train；seed=20261103+task_id。一个原始episode的全部相位属于同一划分。完整train有105486个window8窗口，batch128时每轮824 updates。训练HDF5只向学习器提供pixels/action；raw_row仅用于审计。动作因果对齐的合成跨文件回归已通过。

预先固定4组×2 seeds：temporal-forecast（rho=.8、forecast_fraction=.5）、原AS-JEPA（0/0）、forecast-render（0/.5）、native LeWM。前三组同一随机初始化、51.1M参数body和原sample-level目标，必须实际比较所有初始张量；native LeWM保留原生BN/SIGReg和optimizer，是不同容量的batch-coupled参照。全部随机初始化，不加载旧任务学习权重。

最初配对门禁实际拒绝seed1的一项encoder projection权重（最大差1.49384e-6），其余549项一致；seed2全部550项一致，原因未定。原始8个技术训练和失败门禁保留。正式比较用plan-paired-v2：六个AS训练从头开始，统一通过现有native trainer加载原seed temporal-forecast在任何SGD前保存的全部随机encoder；pool/predictor/renderer和optimizer仍fresh。两个未改动的native控制复用完整2-step状态，不重复完成的SGD。必须再次逐张量验证配对后才能放行。这是比较初始化一致性修复，不使用学习过的权重，也不改变NN或目标。

原生Lightning训练先到2 updates，必须实际恢复CPU model/AdamW/scheduler，并做真实held-out loss GPU反传。所有必要分组finite且有梯度、原生与portable权重一致、sample cursor正确，再通过小预算原生规划接口检查，才可放行短程训练。LR总horizon固定41200 updates（50轮），阶段为600、2472（3轮）、8240（10轮）；有限训练上限8240，不依照某一个任务的分数选择checkpoint。完整原生状态和所有结果保留。

## 环境与评估合同

所有方法共用MetaWorld3.1.1、MuJoCo3.3.0、Gymnasium1.4.0和公开LeRobot camera convention（corner2、固定cam_pos、双轴翻转）。这是独立评测轨道，没有证明与历史数据生成器完全同版本。MetaWorld3.0/Gym组合的失败记录保留。当前原生版本忽略reset(seed=...)，必须调用seed(seed)，并用_encode_task/set_task明确绑定rand_vec。

环境起点比较39D诊断观测及qpos/qvel/mocap/ctrl，全部要求逐元素精确重放；这些物理量不输入学习器或规划器。原生OpenGL重放中少量像素出现1–2灰度级舍入差异，因此记录max、MSE及变化像素比例，统一要求RGB MSE<=.001（原始8-bit单位）、变化比例<=.001。规划第一帧用同一缓存图像，各方法输入完全相同。此接口修复在任何方法任务成绩之前完成。

评测任务预定8项，各primary42–51、fresh1042–1051，160个case。goal由脚本专家在500原生步内首次成功图像构造；未成功则保留末帧并显式标记。当前156个goal构造成功、4个失败，所有case及goal-valid子集分别报告；专家构造失败不等于任务不可能。专家动作不进入学习器或在线planner。

共用原生CEM300×15、top10、H6宏步、每次执行3宏步、最多2图像context、warm start及合法Box域投影。主指标是500原生步any-success。2472阶段各任务primary前5个case；8240阶段primary/fresh各10个。100步final-success若另测必须单独标注，不能从早停500步的轨迹推断。公共JEPA/DINO成绩或权重仅作明确外部参考，尚非同数据重训结果。

## 运行状态与数据读取修复

2026-11-03复核后的统一3轮（2472 updates）结果已完整收集：64个有效端点、320次episode。每任务固定5个primary case、两个训练种子。8任务均值TF15%、AS-JEPA5%、forecast-render10%、本地native LeWM8.75%；Reach分别30/30/30/40%，Reach-Wall90/10/50/30%，其余6个操作任务全部0%。仅是初步小样本结果。TF−LeWM均值差+6.25pp的条件case-cluster 95%CI=[0,12.5]，不代表模型种子/总体不确定性；Native18.0M与AS系列51.1M容量不同。完整数值见results/epoch3/epoch3-summary.json。

评测中抓到3个首帧RGB错误。仅预热的27a339e在连续案例验证中失败，原始失败保留；最终114213e统一预热一次无动作渲染，并在所属OpenGL context中先释放MjrContext，再释放EGL context。160个完整case逐一通过：物理状态、模拟时间及NumPy/Torch CPU/CUDA RNG不变，原RGB门槛不变。原13个有效LeWM端点和3个报错端点全数重测，13个有效端点逐步动作与success序列均逐位相同，3个错误已补齐。其余48个AS系列端点均使用修复后流程；没有模型/数据/目标/规划预算变化。

当前有效汇总必须按RENDERER_REPAIR_REQUESTS.json解析16个LeWM短训端点；原始13个完成和3个报错仍完整保留。完整有限队列的新collector位于renderer-v2-completion，最终results/summary.json为权威索引。原dev2 collector及原始summary属于旧渲染流程历史。

dev2在11:55:29UTC后无新训练心跳，SSH入口502/连接关闭，内部SSH亦不可达，原因未确认；TF/AS四个长训练接续受阻，但2472步完整model/optimizer/scheduler/RNG及初始权重均已归档。它们的父控制器在渲染流程升级时暂停，若机器被重建不可盲用旧PID。已询问用户最新SSH入口。dev13/dev15控制器已恢复，Native两次8240训练完成、forecast-render继续，评测后台按原预算执行。这里没有恢复旧goal或周期LLM监控。

初始启动记录（现已由上述3轮结果更新）：全部门禁通过，四个AS配对比较各550个初始张量完全一致。8个原生恢复训练实际到达step3，原生恢复日志和runtime源文件校验通过；status.json在第100步前显示原2-step检查的finished，不能当作600训练完成。当时尚无正式任务成功率。固定deadline仍为2026-11-06T09:12:24Z。

启动吞吐检查发现BenchmarkWindows每窗口临时打开HDF5 VDS Dataset，使句柄/源文件缓存反复释放。原生真实数据ABBA读取比较支持保留Dataset对象；这是读取耗时实验，不是端到端训练加速倍数。修复只延长句柄生命周期。64次真实窗口逐字段比较全部相等，24个NN模块及原生sampler/training源码逐字节不变；证据DATA_IO_APPROVED.json。

固定runtime60c887e接续：8个旧训练父进程由新有限runner接管，实际native600子进程原样保留并验证存活。它们完成同一个600预算后，由新runner恢复/反传/归档，再以缓存读取继续2472/8240；不重复SGD，不修改config、NN、目标、优化器、LR、数据划分、scaler、planner、种子或预算。16个已有评测worker和有限collector不变。当前训练PID须查training-io-adoption-dev2/dev15.json；原launch文件保留已退休父进程的历史PID。

旧PushT/TwoRoom/Cube所有HOLD/STOP和paused goal保留；不重启旧队列或周期LLM监控。自动化仅限已固定的有限训练/评测队列，72h结束，不自动重试、扩参或开启下一轮研究。

公开来源：[数据](https://huggingface.co/datasets/lerobot/metaworld_mt50)、[作者环境与评测说明](https://github.com/huggingface/lerobot/blob/main/docs/source/metaworld.mdx)、[历史任务文本映射问题](https://github.com/huggingface/lerobot/issues/4409)。

## 2026-11-04 两台机器接续

用户明确要求后续只用dev13、dev15。新有限队列`runs/metaworld_eval_twohost_20261104`接管；不再连接dev2或等待其新SSH入口。4个TF/AS learner从不可变2472步完整检查点迁入新输出目录，唯一配置变化是output路径；新实际CPU模型/AdamW/scheduler恢复和GPU反传审计通过后，以原生Trainer继续8240。之前2/600/2472的归档与审计保留原来源。无法恢复的未保存更新可能被重做，不能宣称完全零重复SGD。

Native和forecast-render的4次长训练已完成，128个已完成主评测端点全部复用，余64个TF/AS成熟端点仍按原案例、预算和原deadline执行。每台分配2卡续训、6卡统一评测；空闲评测卡先做12个已有checkpoint的50任务只读诊断，每任务最多32个不重复heldout窗口，不足则全部取一次。不为占卡新增训练种子或模型网格。训练完成的卡自动加入同一有限评测队列。

诊断比较真实未来图像latent上的预测/复制误差，以及事实动作、时间反序动作、同任务其他episode动作和物理零动作的模型排序。事实动作不被假设为最优，替代动作未在模拟器执行，故不把排序当作真实反事实控制能力。task ID仅用于报告分组，不进入模型。第一版缓存一致性断言失败且保留全部12个失败记录；修订版直接使用MPC同一原生rollout接口、原生增长的想象历史，并在ops初始化后设置highest FP32，与主评测一致。没有改动训练、模型或主评估器。当前状态须读新root的activation与startup记录；分配16个slot不等于每个时刻16卡均计算。

诊断v2在task8遇到有效窗口不足32的采样错误，12个失败及已输出的任务行均保留。统一min规则已逐任务核对全部50项：task8/23/24分别只有10/31/9个有效窗口，最终每模型1554个窗口，所有模型使用同一索引和动作对照。v3 helper为2aceec1，当前评测父进程见diagnostic-activation-v3-dev13/dev15.json；续训父进程和原生子进程未改动。单次训练启动检查已通过：dev13在4000/4100步、dev15在2700/2700步，损失有限且5个必需梯度组均为正；这不是任务性能结论。
