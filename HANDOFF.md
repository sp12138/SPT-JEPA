# 当前交接

目标：从零SPT-JEPA后果学习＋独立动作头，验证实际控制可靠稳定的scaling，并超过参照。**尚未达成**；loss下降、粗回报上升或零星成功均不算达标。时间优先算法、关键诊断和闭环结果，避免重复审计、兼容检查及细碎记录。

只用dev13、dev15、dev6；dev6-2c/grid16/dev2不连接。同会话每小时检查并按证据改进，不在有限pilot后自行HOLD，不恢复旧goal/监控或启动第二套监督。R=`/efs/prescottsun/SPT-JEPA/runs/robocasa_reach3d_scaling_20261114`；实际状态查R/MASTER.json、R/SESSION_CHECKS.json，不能按旧ACTIVE恢复。

| 主机/卡 | 当前run；根后缀加robocasa_reach3d_ | 可训练参数 / 头参数 |
|---|---|---|
| dev13 0–3 | spt-conditional-rate-only；conditional_rate_only_20261117 | 22.220M / 384 |
| dev13 4–7 | spt-trajectory-flexible-likelihood；trajectory_flexible_likelihood_20261117 | 41.625M / 384 |
| dev15 0–3 | spt-action-scale-rate；action_scale_rate_20261117 | 22.220M / 384 |
| dev15 4–7 | spt-joint-profile-rate；joint_profile_rate_20261117 | 22.224M / 384 |
| dev6 0–3 | spt-joint-forward；joint_forward_20261117 | 22.224M / 384 |
| dev6 4–7 | spt-joint-action-scale-rate；joint_action_scale_rate_20261117 | 22.224M / 384 |

最近实际小时检查UTC04:59:59：训练rank分别8/8/4，dev6另4卡在冻结读出；GPU瞬时利用率dev13 86–100%、dev15 85–97%、dev6 0–97%，不代表持续满载。六组无ERROR，新profile实际保留8/恢复16且进入主训。下一同会话检查UTC05:59:25。

## 决策所需结果

下述均同64反复使用的DEV、双224RGB、最多5命令/norm .3/native2mm；格式为回报% / 平均末端mm / final成功数。各预算按活跃后果训练小时，不是墙钟或独立案例；32确认封闭。

- 无熵绝对前向1/2/3/4h：82.62/79.27/70.21/65.53%，39.02/47.52/69.11/77.59mm，final1/0/0/0，持续退化。状态相关F＋像素/后果辨认1→4→9h为83.90→91.24→91.08%、38.67→19.81→20.44mm、final均0；中间5/6h各1，后段粗平台。
- 每视角尺度前向1/2/3h：77.00/75.55/75.80%，43.84/52.84/55.03mm，final0/3/1、any2/5/4。最小20%事实输入3.06–7.68mm的一小时当前命令RMSE4.68→2.63mm，<5mm案例2→11，但粗定位及尾部变差；细分点不改写严格成功。
- 双相机联合128维绝对前向1/2h：91.43/90.23%，21.02/26.33mm，final1/0、any3/1。比每视角绝对前向首小时四厨房都好，区组95%回报收益[4.34,14.10]点、终距改善[8.69,28.30]mm；后续回退，尚非稳定精度。联合只多3200可训练参数、同总128码/384头，初始化pool/Q改变，不是独立种子或等更新比较。
- 联合尺度前向1/2h：69.52/83.03%，69.82/30.30mm，final5/4、any5/7。细动作可读性更好，粗控制仍弱于联合绝对组，尚未统一两种收益。
- 联合profile新组保留联合尺度组完整初始NN，仅将每转移Gaussian预测KL的额外等方差解析消去：原率l变为D/2 log(1+2l/D)。无新网络、权重或可调尺度，不称归一化预测密度/MI下界。4独立KL数学合同及4废弃GPU更新、物理配对/点读出/样本隔离通过，主祖先0；首小时控制未齐。源code-joint-profile-rate-v1，入口joint-profile-rate-20261117。
- 固定联合尺度1h模型/原头的有限64DEV对照正在执行：仅将W换成W Q_XYZ Q_XYZᵀ，0新增body/头拟合。普通验证中约99.8%读出权重能量在前向动作子空间外，却只贡献约.31%事实命令能量；投影验证29.81→30.00mm并无整体收益，远目标外推假说须由闭环判断。使用冻结世界模型Q，**不是纯E-only probe**。单次入口joint-projection-readout-20261117，flat根joint_projection_20261117；不启动重复任务/监控。

## Lingjing最新上传与机制

最新远端anchored-state-jepa为484f10f3、decoder-free-as-jepa为cd03863b。最新增量主要是分开表示/预测器的scaling评测协议，没有真实RGB控制胜出的证据。原AnchorReader beta .3/1码头仅15.36/15.47%、243.62/250.68mm、全0，均完整1h及部分2h保全；相对锚点修复熵原点后1/2h16.09/15.40%、239.14/252.57mm、全0，完整2h及部分3h保全。未跑作者PushT门，不称完成PartP/PartR；未来动作chunk/专家策略分支不采用。

自己的当前条件熵绝对→相对首小时56.27→85.28%、121.78→33.48mm，四厨房都改善，但相对2/3/4h79.60/78.24/75.01%、49.68/49.45/61.66mm、final全0。相对完整4h及部分5h保全后交给profile，绝对完整2h及部分3h交给联合绝对组。无证据证明熵优于匹配无熵前向，reader拟合变好不等于控制收益。

四个冻结联合1h体的状态条件MSE/Gaussian协方差头各5000更新，额外88.05GPU秒，均只监督普通事实当前输入。Gaussian NLL明显下降，但均值RMSE改善至多约.08mm，最细组未稳定改善；按验证收尾，0新增控制，不推广核心代码。入口conditional-action-uncertainty-20261117。历史动作先验去偏、势差/状态条件头、响应校准及空间读出负结果见CONCLUSIONS，不重复包装再跑。

强加法前向并非“无inverse信号”：QᵀQ=I时，||d−Qa_t||²=||Qᵀd−a_t||²+||(I−QQᵀ)d||²，含代数等价的**当前事实输入**重建；显式目标仍为自然下一码，不含未来/最优动作。此结构可对齐控制坐标，但常数加法对接触/夹限/物体交互有限，Reach结果不能代替manipulation。

## 固定边界

固定12006普通TRAIN日志；头仅17920训练/1536验证的同转移实际a_t，表示冻结/detach，标签梯度不进后果学习器。输入仅RGB历史/实际动作，自然未来RGB或自身编码可作目标，目标图只用于控制。无外部教师、reward/state/taskID/语言、未来/最优动作、专家路径/策略。同步相机属于一个物理样本，独立样本不交换表示或统计。受保护评测/oracle/接触/闭环输入不进训练、教师或proposal memory。

新学习链随机祖先0；同初值分支不作独立seed。不同读出、容量、数据、教师、累计预算单列。32确认仍封闭，不把不同公开协议当公平胜出。正常续训保存model/optimizer/scheduler/全rank RNG/游标，实际8/16恢复已做，不重复CPU审计；有实际错误才修。不改已启动冻结源码、检查点或vendored runtime。

Git仅核心代码、必要配置、精简结论及已验证较优配置。原始细节/负结果/完整与部分状态留EFS；未有新best-config晋升。稳定规则见AGENTS，机制和历史见[CONCLUSIONS](experiments/CONCLUSIONS.md)，流程见[PROTOCOL](experiments/robocasa_stable_scaling/PROTOCOL.md)。
