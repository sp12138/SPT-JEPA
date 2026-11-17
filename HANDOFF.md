# 当前交接

目标：从零SPT-JEPA后果学习＋冻结表示的独立当前动作头，验证可靠稳定的实际控制scaling，并超过参照。**尚未达成**；不能把粗回报、loss或零星成功当达标。时间优先算法和闭环结果，避免重复审计与细碎记录。

只用dev13/dev15/dev6，禁用dev6-2c/grid16/dev2。同会话每小时检查，按证据改进，不自行有限HOLD，不恢复旧goal/监控或第二套监督。R=`/efs/prescottsun/SPT-JEPA/runs/robocasa_reach3d_scaling_20261114`，实际状态查R/MASTER.json与SESSION_CHECKS.json，不能按旧ACTIVE恢复。

| 主机/卡 | 当前run；根后缀加robocasa_reach3d_ | 训练参数 / 头参数 |
|---|---|---|
| dev13 0–3 | spt-relative-one-step-pixel-world；relative_one_step_pixel_world_20261117 | 40.888M / 384 |
| dev13 4–7 | spt-joint-full-covariance-profile；joint_full_covariance_profile_20261117 | 22.224M / 384 |
| dev15 0–3 | spt-joint-relative-full-covariance-profile；joint_relative_full_covariance_profile_20261117 | 22.224M / 384 |
| dev15 4–7 | spt-joint-profile-rate；joint_profile_rate_20261117 | 22.224M / 384 |
| dev6 0–3 | spt-one-step-pixel-world；one_step_pixel_world_20261117 | 40.888M / 384 |
| dev6 4–7 | spt-joint-action-scale-rate；joint_action_scale_rate_20261117 | 22.224M / 384 |

最近实际检查UTC10:12:09三台均8训练rank，瞬时利用率83–95/84–98/84–97%，六组无ERROR，不称持续满载。新一步RGB两组实际8→16和主训已确认，初始NN同配对短测2ee15f35。下一整点UTC11:12:09。

## 当前结果与决策

均同64反复使用的DEV、双224RGB、最多5命令/norm .3/native2mm，32确认封闭。以下是活跃后果训练小时，不是墙钟、等FLOPs或独立种子。

- 联合尺度前向1/2/3/4/5/6h：69.52/83.03/83.52/88.23/89.36/87.29%，69.82/30.30/29.14/21.84/20.03/24.01mm，final5/4/7/5/7/5、any5/7/10/8/11/6。粗指标后退，严格成功仍波动；继续固定配方长曲线。
- 联合相对等方profile1/2/3h：76.75/80.14/81.71%，51.08/41.72/40.30mm，final3/6/12、any3/8/13，出现连续成功率增长。3h较同预算未profile的7/64多5例，四厨房区组95%差[3.125,12.5]个百分点，但粗距离较29.14mm更差，一厨房明显退步；这是筛选后的DEV信号，不是独立确认或稳定scaling。最细20%事实输入RMSE1.93/1.75/1.80mm，3h对照2.07mm（含分位边界308条）。1→3h原成功仅保留1/3、失2例，负结果保留，固定配方继续后续预算。依据analysis/relative-isotropic-first-three-hours-paired-control-and-factual-readout.json。
- 相对等方profile4h：87.10%/25.48mm、final8/any13；粗指标改善但final从12降8，不能称稳定scaling。4h有5例某次观测低于2mm而最终离开；一例0.57mm时下一命令11.46mm，另有低于物理1mm执行死区的命令。仅作受保护DEV只读解释，不据此拟合头、增益或成功停机规则。原始analysis/relative-isotropic-any-to-final-readonly-trajectory-summary.json。
- 完整协方差绝对1/2/3h：82.08/78.70/85.46%，54.70/57.86/41.97mm，final4/3/3、any5/6/4；相对1/2/3h72.17/78.41/76.40%、67.87/54.56/65.96mm、final6/6/8、any11/9/13。尚未统一粗定位与精度，保留后续预算和负结果。
- 原联合绝对前向1/2/3/4/5h：91.43/90.23/90.70/91.16/88.84%，21.02/26.33/23.34/25.30/29.60mm，final1/0/0/0/0；完整5h及部分6h全状态保全后交给一步RGB绝对组。绝对等方profile1/2h91.24/89.54%、19.99/23.96mm、final均0，完整2h及部分3h保全后交给一步RGB相对组。后继均继承0更新。
- 新一步RGB首小时绝对/相对组74.67/25.13%、54.28/205.44mm，final/any均0。使用当前RGB作已知条件，非线性F预测自然下一码，固定Gaussian未来码后验只用于下一帧RGB解码，解析KL＋单位RGB方差NLL，按双相机像素数归一。无初始码率、AR噪声、forecast解码混合、额外熵或固定命令效果长度；相对版仅用已执行当前命令长度加权整个单步风险。动作不是输出目标，原384参数头仍完全detach。
- 普通1536验证转移：新RGB绝对/相对一步码预测误差分别为copy的.449/1.010、零动作查询的.461/1.196；绝对组确实使用输入，但最细20%仍为copy的2.062倍，相对组码标准差仅.0306（绝对.457）。不能把全局预测改善当控制成功；冻结F局部Jacobian＋9参数当前输入校准的两次拟合，整体验证绝对38.40→43.22mm、相对54.53→58.57mm，虽细组改善但总体负，0新控制收尾。训练与Policy均相同中心裁剪/AREA224，排除所疑缩放差异，无预处理修改。原始analysis/conditional-rgb-world-first-hour-factual-mean.json。
- 新两组完整初始NN同2ee15f35，视觉E初值同50486878；与22M无像素组不是同参数/同函数比较。3独立Gaussian似然值/梯度、零命令权重及物理配对合同通过；绝对短测共8废弃更新（首4后测试fixture误对int调用clone，修fixture未改冻结源码），相对4，主祖先0。实际生产点读出=训练码，最后未来帧不能改变任一prior输入或解码器当前条件，样本隔离及E/pool/F/actionEncoder/D梯度已过。源code-one-step-pixel-world-v1，入口one-step-pixel-world-20261117及relative-one-step-pixel-world-20261117。

输入尺度噪声候选已过4个Gaussian值/梯度/单位测试与4废弃GPU更新（4.69s），主训尚未启动；准备接替dev13 4–7完整协方差绝对组的下一完整预算。新源code-input-scaled-pixel-world-v1，初始NN同2ee15f35、40.888M/384头、祖先0。只把q/p码标准差同时改为sigma0乘已执行当前输入的归一化长度，零输入保留sigma0；像素方差1不变、不整体加权。依据是8条普通验证日志内16个大小输入的配对解码诊断：绝对组posterior/copy大动作.309、小动作.944，相对整体加权组约.935/.870且未来码依赖微弱；小动作诊断不是因果证明或控制进步。原始analysis/conditional-rgb-world-first-hour-within-sample-decoder-reliance.json。

## 机制insights与Lingjing

- 最新已读anchored-state-jepa 484f10f3、decoder-free-as-jepa cd03863b，增量主要是区分表示/预测器及冻结读出的评测协议，不是有真实RGB控制证据的新胜出配方。原AnchorReader beta .3/1码头仅15.36/15.47%，相对锚点1/2h16.09/15.40%，严格全0，完整/部分状态保全；未跑作者PushT门，不称完成PartP/PartR。未来动作chunk分支不采用。
- 当前条件熵的相对原点修复使匹配首小时56.27→85.28%，但4h回退75.01%/final0，未可靠超过无熵组。固定正交Q已约束动作单位；理想可实现时条件熵由动作分布决定，额外reader未必增加约束，不能从此否定其他F/任务。
- 普通1536验证转移：全局Q前向误差81–92%是长度、8–19%是方向；最细20%方向占约一半。只旋转动作基不能解决主要残差，不启动该分支。前7个独立随机命令的14个微型正向拟合中，增加已执行上一输入仅改善.27–.45%，不支持增加延迟核。它们均不是实测物理位移/唯一因果解释。原始analysis/joint-{forward-radial-angular-residual-by-budget,coded-transition-current-past-input-regression}.json。
- 固定读出W→WQ_XYZQ_XYZᵀ的1/2h结果final5→7/4→5，粗回报近乎不变，投影自身7→5，已有限收尾不晋级。128模型—案例记录仍64物理DEV，使用F中的Q，不是纯E-only。四个状态条件协方差动作头仅改善验证均值约.01–.08mm，0新增控制收尾。过去去先验、势差/空间头等负结果不重复包装。

独立动作头容量诊断已收尾：新RGB绝对1h的384参数线性头扩为50304参数、同初始函数/相同事实标签，5000更新9.55s，按验证MSE选择3000步后38.40→37.12mm，但最细20%5.46→5.88mm；未满足预定双指标10%改善门，0新增控制，不推广或归因所有非线性头。E/F从未加载或更新，仅使用冻结普通缓存。原始conditional-rgb-factual-head-capacity-20261117。

## 固定边界

固定12006普通TRAIN；独立头仅17920训练/1536验证的同转移实际a_t，标签梯度不进E/F。输入仅RGB历史/实际输入，自然未来RGB/自身码可作目标，目标图只用于控制。无外部强教师、reward/state/taskID/语言、未来/最优动作、专家路径/策略。同步相机同一个物理样本，独立样本不交换表示/统计。受保护评测/oracle/接触/闭环资料不进训练或proposal memory。

旧正交加法前向在代数上含当前输入inverse项，不能称完全无inverse信号；新一步RGB只把动作当F输入。均无未来/最优动作。不同读出/容量/教师/累计祖先与预算单列，同初值分支不是独立seed；32确认不参与方法选择。

正常续训保存model/optimizer/scheduler/各rank RNG/游标；只查真实8/16及进度，出实际错误才修，不重复CPU审计。不改已启动冻结源、检查点或vendored runtime。Git仅核心源码/必要配置/精简结论及验证较优配置；原始细节/负结果留EFS，未晋级best-config。详见[CONCLUSIONS](experiments/CONCLUSIONS.md)和[PROTOCOL](experiments/robocasa_stable_scaling/PROTOCOL.md)。
