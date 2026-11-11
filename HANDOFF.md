# 当前交接

目标是匹配条件下超过 RoboJEPA，尚未达成。用户授权在本会话每小时检查，性能差则复现参考、逐组件融合／改进。仅用dev13/dev15/dev6/dev6-2c；旧goal／旧监控不恢复。没有创建托管定时任务。以下成绩均为进度，严格成功另报；源验证只含Reach-v2/v3、Push-v2/v3四个物理案例（seed7401），重复评测不增加独立案例数。

## 当前推进：先确认官方基线

用户最新追问训练／评测复现，已明确承认尚未复现论文操纵效果；优先官方基线，不新增算法分支。RoboJEPA离线学习事实轨迹中的未来视觉编码，仿真用于闭环控制评测。目前工作是使用公开权重复现规划，不是重新训练全部23数据混合。SPT仍用自己的普通转移日志，不能叫同训练数据比较；完整同数据／同编码器／同预算训练对照尚未完成。

**当前规范入口**`runs/robocasa_literal_minimal_v2_20261111`，dev6-2c，launch.json PID9360。执行未修改的上游policy_client.py与robot_env_server.py；字体／绘图视频依赖已提供，本地仿真包与资产清单核对，未加入adapter或记录钩子。公开2B450k、完整240×320、embodiment3/proprio、客户端FP16、默认action_repeat2；target kitchen4/taskseed0。先前单卡18步技术测试已完整执行、官方NPZ／视频输出、客户端exit0；仅4候选／2次CEM，不作性能结果。新根继承该technical结果不重复运行，formal为官方15/10次CEM、top20、1024全局候选，8rank×16sample×8batch分担**同一个**物理案例，不算8个案例。绝对deadline沿用上一minimal根plan，不重新延长。当前正式结果未齐，不能声称性能复现。

旧`robocasa_literal_client_20261111`的启动错误全部保留：启动器漏掉本地仿真PYTHONPATH导致EFS资产os.listdir扫描；依赖已改节点本地；upstream --help有未转义百分号，只影响help格式，实际参数解析通过。随后我加的记录器对代理env直接赋reset，触发__setattr__转发到内层并递归，8例均在首次reset失败、0CEM动作，属于启动错误而非模型负结果。新minimal根完全移除这些钩子。旧minimal技术18步实际成功，外层glob没找到官方嵌套task输出路径而误报AssertionError；原错保留，递归定位并验证18有限reward后继承。不可恢复旧失败控制器／将其计成性能0。

官方脚本与论文表格有差异：example沿用action_repeat2默认、客户端FP16；论文Table14列BF16 eager，未明确区分该repeat参数。旧适配repeat1/BF16是单列诊断；不能称literal-client复现。官方本身执行CEM最后均值。Paper Fig15的Push成功描述仅为物体向目标移动，精确二值阈值尚未验证；本地2cm严格完成不得当作同一指标。

现已把三台空闲机器也转到同一官方流程：dev13的`robocasa_official_k4_s1_20261111`、dev15的`robocasa_official_k8_s0_20261111`、dev6的`robocasa_official_k8_s1_20261111`，各8rank、一个Push案例、同1024全局候选／repeat2／完整输入／FP16。共同入口`runs/robocasa_official_cohort_20261111/{plan.json,collect.py,summary.json}`。四物理案例、两厨房、同一个公开checkpoint，不是32个案例或论文完整评测群。三新根继承技术证据，全部沿用最初minimal的绝对deadline。运行环境Python3.12.3／Torch2.9.1+cu130／MuJoCo3.3.1；官方README为Python3.11／Torch2.7.1+cu126，未做逐版本环境重建。差异在RUNTIME.json，不冒称完整环境严格复刻。

已有有限根`runs/robocasa_current_inverse_20261111`，dev13/control、dev15/observed、dev6/paired，各两个四卡模型已到50000，24源回合全部完成、0执行错误、HOLD，不追加分支／长训。Control／observed／paired Reach90.98／67.72／68.68%，Push9.31／9.16／9.66%，严格final／any均0。父模型为global16/40000，1120TRAIN／96旧验证、batch32/window8，再10000步，兼容361个Adam状态／四rank RNG／1280000游标保留，共同cosine重启。Observed仅从q(x0),q(x1)重建实际输入a0；paired另从F(q(x0),a0)经同头重建**同一个a0**。辅助项原beta=.25乘单位方差Gaussian每坐标NLL，不是新ELBO。无未来动作target、专家路径或独立世界间统计。三结构私有后向、终端动作污染、全部未来动作改变不改inverse项、六模型实际恢复／梯度／Push-H5入口均通过。Paired自预测inverse很低仍无控制收益，不能当真实后果已学好的证据。

`runs/robocasa_rescored_cem_20261111`已收齐16冻结源回合、0错误，HOLD。Global16／grid16-noise重评分后的Reach91.42／81.70%、Push9.15／9.23%，原uniform91.88／81.32、9.16／9.20，严格全0。搜索／精英／均值更新不变，仅多两条NN rollout比较最后均值与最后一代最佳样本；预测评分更优没有转换成可靠Push改善。源四案例／两共享父模型的边界保留。

最近实际每小时检查12:51:25UTC，下一次13:51UTC，canonical helper仍为`runs/robocasa_robojepa_fusion_20261111/session-check.py`，已更新至minimal_v2入口；配置在`runs/robocasa_encoder_budget_20261111/SESSION_CHECKS.json`。没有托管定时任务／第二套LLM监控／旧goal恢复。12:51快照三台24卡在计算、dev6-2c当时尚无GPU进程；不宣称持续满载。以新的实际GPU和进程为准。

## 40000步空间轮及冻结诊断已收齐

`runs/robocasa_spatial_20261111`完成六模型／48回合、0有效错误，HOLD。Global16／grid16／grid16-noise的uniform Reach为91.88／84.35／81.32%，Push9.16／9.22／9.20%；support Reach91.74／91.83／87.39%，Push9.15／9.25／9.19%。严格final／any全0，不长训。三组同256维容量与初始张量；结构性lift保留旧确定性核的数值证明不能证明物理收益。

冻结公开2B RGB square224的patch对照`runs/robocasa_reference_patch_v3_20261111`也收齐8回合：uniform Reach67.78／Push9.51%，support39.55／2.19%，均严格0；四rank动作／奖励／RGB及NN不变证明齐。旧v1/v2资源路径错误仍保留。这不是完整官方输入／literal配置的性能复现。冻结V-JEPA dense10000两父模型的16回合`runs/robocasa_dense_patch_20261111`，uniform38.85／9.14，support30.41／11.30%，均严格0，未构成可靠操纵突破。

自然留出回位日志的生成分解`runs/robocasa_decoder_decomposition_20261111`完成八模型，96日志／12初态组／6厨房、0SGD／0新控制。共享解码噪声时，H7真实未来编码的两分支图像差误差／真实差能量约.54–.76，动作预测编码约1.20–2.03；预测缺口明显，但前者也非完美重构。H16超出训练窗、仅保留7帧想象历史。对比包含机器人／相机变化，不是对象信息的独立证明；oracle未来编码只用于诊断，不进入训练或规划。原始图像NPZ和每模型summary留EFS。

## 30000步条件代码轮已结束

`runs/robocasa_conditional_20261111` 完成8模型／32源回合，0有效执行错误，HOLD，不继续长训。Global／delta／delta-noise／forecast的Reach为91.66／57.48／69.77／81.46%，Push9.18／9.18／9.15／9.22%，严格final／any均0。约9.2%仍是同案例零动作水平，未超过RoboJEPA的操纵能力。六个非forecast头共同训练到30000，forecast只冻结表示与生成器学原创新残差；它在旧验证上的7步prediction/copy从父模型.794/.890降至.555/.614，仍没有控制收益。不能以总体预测改善代替物体预测或成功率。

每组来自同两个spatial20000完整父模型，新增10000步；数据是旧1024自然指令日志＋192普通闭合／反馈回位日志，共1120TRAIN／96旧验证。兼容Adam／四rank RNG／累计样本游标保留、统一10000步cosine／100warmup；数据混合改变了记录取样，非原数据顺序逐位续训。仅使用公开冻结视觉编码器，不导入公开动力学或策略；不与scratch主线混报。Delta在pool前作实际初帧差并减去首代码，推理时目标随真实历史重编码，目标不进入动力学；该实现已被本轮控制结果否定收益。

新验证 `runs/robocasa_loops_val_20261111` 已完成96日志／1536真实转移和冻结特征缓存、0错误。六个与TRAIN不交叠的厨房×两个预定seed×八分支；相同普通闭合／反馈回位，无物体／任务目标／奖励或模型分数挑选，全部有效日志保留、0SGD，不自动训练。这是内部验证，不是论文留出目标厨房控制评测。

## 空间读出的机制证据与边界

四条旧受保护路径、两物理Push案例，只读NN／物理标签事后评分，0SGD／0新控制。密集冻结V-JEPA的initial-goal feature support在两条h0路径上选中真实物体最近点（step3／5），全局距离都选step1；两条h.03路径物理最近点代价排名17→9、12→2。RGB支持另有局部改善，但未作该控制器的性能声明。压缩query上的support仍与uniform同排序，direction也不稳健；不因此启动方向代价长试。入口本轮patch-geometry.json和goal-geometry/summary.json。

六个本轮联合头的纯噪声t=1生成器读出（两个固定噪声）／目标区域加权／生成器核诊断，也未给出跨父模型稳定修复；不启动更贵的decoder规划实验。该量不是已验证的log-likelihood。入口decoder-goal-probe，各源与NN哈希不变。

query-redundancy.json给出更具体的可检验问题：两个global30000读出在这些路径上，槽变化相对总变化RMS仅.00883／.00926，4槽参与秩约1.00013／1.00017，前16主成分解释64维方差99.99962%／99.99943%。平均注意力有效patch119／131（共196）；四槽大致重复相同变化，不是四个独立空间地址。这只限开发路径，不等同于全数据的真实信息维度或对象信息缺失。新空间轮用同容量控制来检验它是否影响实际控制。

## 已结束的条件先验与回位诊断

`runs/robocasa_context_prior_20261111` 六模型全部20000步、24回合、无有效错误。Plain／pooled／spatial Reach67.74／78.66／67.02%，Push9.17／9.16／9.18%，严格final／any全0；未超过旧15000步plain Reach86.10%，不长训。旧context初始零输出、匹配参数和392个key的pooled控制、跨世界隔离／目标隔离等证明及实际15002恢复、梯度、Push-H5入口均通过。六项最终完整model／Adam／scheduler／四rank RNG和累计游标恢复复核全部通过。

`runs/robocasa_loops_20261111` 的192条普通TRAIN日志／3072真实转移已采集并缓存完成，0错误。机器人反馈只在底层执行回位使用，学习档案为RGB／已执行命令／实测位移，不含机器人绝对状态。166/192末端平移回位误差<1mm；这不等同于世界坐标的完整关节／相机姿态闭合。新数据没有专门富集接触，不能这样宣传。

固定盐8个新TRAIN组×八条普通日志的只读重放诊断收齐64回合／1024转移，物理／RGB／实测位移误差全0；9/64有接触、12/1024动作步有接触。受保护物体／接触／世界姿态字段仅用于事后诊断，绝不进入学习、教师或proposal。诊断选择不改变原192条自然日志的学习资格，也不是留出控制评测。

世界EEF位置≤1mm、完整旋转≤.004rad、夹爪≤.5mm时有112个同时间pair，其中20对物体XY差>1cm（两端点和八组被重复使用，不是112个独立案例）。该子集上dense／两个15000步SPT读出的物体XY秩相关为.352／.410／.298，未支持压缩特有的信息丢失；同时EEF残差仍有较强相关，关节也匹配后仅38对、其中2对物体差>1cm，不能据此唯一归因。固定距离不证明信息不存在，也不证明目标几何已足够。入口 `coverage/geometry/{summary.json,pairs.npz}`。八个旧模型的冻结事实分支预测诊断已齐：当前spatial两父模型H1误差／copy为.412／.421，H3为.529／.554，H7为1.889／1.999。H7按代码最近事实分支匹配的物体XY误差为1.43／3.30cm，copy约1.22cm；这只是八分支候选上的诊断评分，不是NN直接预测物体位置，不可冒称控制结果。64条日志对旧模型是未见日志，加入下一轮后不再是留出；受保护物理只在NN预测后评分，无probe或控制proposal库。入口coverage/forecast/summary.json。

原公开参考已按原截止07:05:18UTC结束，CAP_RELEASE_VERIFIED.json实际核对全部旧PID不存在、八卡无CUDA进程／0MiB；未完成案例保留为预算截断，无续跑或延长。

## 已收齐的关键结果

所有下表源回合严格final／any成功均0；Reach为平均归一化进度，Push为最终归一化进度。同案例旧零动作Push已有9.09%，约9.2%不是操纵突破。

| 对照／有限根 | Reach | Push | 结论 |
| --- | ---: | ---: | --- |
| own-SPT 10000步 |15.73%|9.29%|未解决操作 |
| V-JEPA-SPT 10000步 |64.77%|9.25%|Reach随预算明显提高，Push不动 |
| own-dense／V-JEPA-dense 10000步 |20.11%／38.85%|9.31%／9.14%|换dense头也没有操作收益 |
| 官方2B RGB全视场 |77.19%|11.25%|非论文完整输入，非同训练预算 |
| 同2B square224 |70.67%|9.29%|全视场有差异，未解决操作 |
| 官方8B RGB全视场 |43.51%|9.18%|该源队列未显示更大模型收益 |
| 官方2B released proprio |92.02%|28.33%|Push-v2为55.39%，v3为1.26%；仍0严格成功 |
| 官方2B native proprio |93.14%|4.54%|训练式本体输入并未更好 |
| 官方8B released proprio |95.44%|9.15%|Reach高不代表Push完成 |
| SPT续训plain／self／cross 11000步 |75.67%／77.75%／80.05%|9.20%／9.19%／9.17%|跨相机消息未改善操纵，不长训 |
| 官方2B动力学＋SPT读出 |86.69%|10.89%|低于原92.02／28.33；读出额外适配预算和矩形分布偏移须披露 |

预算根 `runs/robocasa_encoder_budget_20261111` 已完成八模型／64新增回合，保留32继承回合。原1000→5000→10000完整状态续训，统一9000步cosine重启／90warmup；不是固定LR的纯scaling law。V-JEPA-SPT Reach为11.03→47.58→64.77%，说明短训会遮蔽骨干收益。所有头1000步前重新初始化；own骨干继承7000步，外部视觉骨干预算不同。

公开RGB／预处理诊断 `runs/robocasa_robojepa_fusion_20261111`、公开主输入 `runs/robocasa_public_inputs_20261111`、跨视角 `runs/robocasa_crossview_20261111`、读出融合 `runs/robocasa_readout_20261111` 均已完成，HOLD，无有效执行错误。各正式回合前实际入口、NN不变和分布式物理／动作一致检查通过。模式2到3同时改变proprio、embodiment和动作编码，不是纯proprio消融。公开预训练／状态参考与RGB候选分轨，不作同预算SOTA排名。

## 目标／表示短训已结束

`runs/robocasa_target_refinement_20261111` 六个模型全部15000步，24回合完成，无有效执行错误。Plain／fixed／features的Reach为86.10／86.21／66.25%，Push为9.21／9.20／9.26%，严格final／any均0，不长训。六项最终CPU完整model／Adam／scheduler恢复及四rank RNG结构／游标通过；fixed读出与原10000父模型仍精确相同。固定表示未显示额外收益，未来特征失真也未解决操纵；它们的损失单位不同，不能横比裸loss。结果和 `FINAL_RESTORE_AUDIT.json` 留在该根。

只读 `runs/robocasa_decoder_geometry_20261111` 也完成八模型×四既有路径共32行，0新SGD／模拟。比较代码L1、生成器输出核距离及目标流匹配能量，MC四次，同训练时刻分布。所有生成器核距离仍选第1步，物理最近点为3／16／5／18；流匹配能量的改变也未稳定匹配最近点，不自动启动该度量的规划实验。这些路径的机器人姿态不同于目标，故不能据此证明对象信息丢失或直接认定规划必败。只有两个物理Push案例，绝非32独立案例。

## 采样器假设已被物理结果否定

`runs/robocasa_endpoint_20261111` 已完成24回合、0SGD、严格成功全0。`endpoint_cem.py`在相同1024候选／20elite／原refit数下比较发布式径向分布、初始矩匹配Gaussian和终点／路径分解协方差。CPU小实验确实发现分布／高斯方差refit不一致会在无信息目标下冷却，分解协方差可保留路径探索；这没有转成控制收益。公开2B released轨的Push从原28.33%降为Gaussian0%、endpoint1.72%。SPT只有一个父模型的单Push-v2病例有Gaussian收益，另一父模型不成立；不能叫稳健改进。无长训晋级。实际源、CPU证明和分family结果留在该根及同级 `robocasa_endpoint_{gaussian,endpoint}_20261111`。

## 表示／数据诊断的边界

固定盐抽8个TRAIN初态组×8分支，64条旧日志／1024事实转移全部严格重放，初始物理、RGB、实测位移误差均0。16/64条有机器人接触，36/1024动作步有接触；8/8组跨动作物体XY位移差>1cm。这否定完全没有对象干预数据，不能排除接触稀疏、目标失配或泛化问题。对象／接触字段仅诊断，不进训练、教师或proposal。

固定距离统计（无拟合probe、无SGD／模拟）3584同时间跨分支pair：dense V-JEPA的物体XY／机械臂EEF相关为.122／.740，SPT两读出为.090/.910、.092/.921；控制EEF的部分秩相关分别.100、.054、.060。没有符合阈值的固定EEF匹配pair，所以只能说该度量明显跟随机械臂运动，不能证明物体信息不存在或找到了唯一原因。新增同pair的RGB224／64固定距离统计中，物体XY相关为.117／.121，EEF为.579／.580，未显示明显分辨率退化；这削弱了单独归因64像素的解释，不是生成器或控制能力证明。入口 `robocasa_robojepa_fusion_20261111/coverage/{protected,distance-statistics}`。

## 第四台公开目标厨房参考

已按24小时原cap停止：`runs/robocasa_reference_adopt_20261110`。实际结果在 `runs/robocasa_intervention_20261110/public-evaluation/*/gpu*/*/summary.json`。Released轨20个Reach案例全部收齐，50M／2B／8B平均进度91.00／93.40／96.12%，严格final为0／3／1。50M全部19个Push为0.49%、严格0；2B只收齐7个、8B只收齐4个，不能按部分均值排名，未齐项不算失败。50M native Reach20个93.62%、严格0，Push5个为部分结果。目标厨房／flour_bag、完整图像／预训练／proprio与上表源案例／flourbag001不同，不能拼成同条件性能差距。BF16表格设置与发布客户端FP16差异仍披露。

## 历史轮次

旧 `robocasa_compare/refine/intervention/local/forks_20261110` 均已结束；没有可靠严格操纵成功，均未获长训晋级。局部目标support和共享根direction曾提高个别源Push进度，但牺牲Reach或跨父模型不稳定。冻结几何／动作辨识诊断在 `runs/robocasa_geometry_review_20261110`；分支预测77%–82%可辨识动作，不证明对象接触已学好。全部原始错误、修复、祖先与结果仍在EFS。旧R3参考已STOP，不恢复。旧MetaWorld困难六任务均未可靠突破；历史分任务正结果及边界见结论。

- [关键结论](experiments/CONCLUSIONS.md)
- [较优配置](experiments/best-configs.json)：本轮无新配方晋级
- [原始记录归档](docs/experiment-archives.md)
- [稳定约定](AGENTS.md)
