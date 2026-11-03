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

旧PushT/TwoRoom/Cube所有HOLD/STOP和paused goal保留；不重启旧队列或周期LLM监控。当前2-update技术检查已经开始，尚无Meta-World任务性能成绩。后续自动化仅限已固定的有限训练/评测队列，72h结束，不自动重试、扩参或开启下一轮研究。

公开来源：[数据](https://huggingface.co/datasets/lerobot/metaworld_mt50)、[作者环境与评测说明](https://github.com/huggingface/lerobot/blob/main/docs/source/metaworld.mdx)、[历史任务文本映射问题](https://github.com/huggingface/lerobot/issues/4409)。
