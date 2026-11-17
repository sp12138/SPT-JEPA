# 当前交接

目标：从零SPT-JEPA后果学习＋独立动作头，验证实际控制可靠稳定的scaling，并超过参照。**尚未达成**；loss下降、粗回报上升或零星成功均不算达标。时间优先算法、关键诊断和闭环结果，避免重复审计、兼容检查及细碎记录。

只用dev13、dev15、dev6；dev6-2c/grid16/dev2不连接。同会话每小时检查并按证据改进，不在有限pilot后自行HOLD，不恢复旧goal/监控或启动第二套监督。R=`/efs/prescottsun/SPT-JEPA/runs/robocasa_reach3d_scaling_20261114`；实际状态查R/MASTER.json、R/SESSION_CHECKS.json，不能按旧ACTIVE恢复。

| 主机/卡 | 当前run；根后缀加robocasa_reach3d_ | 可训练参数 / 头参数 |
|---|---|---|
| dev13 0–3 | spt-joint-absolute-profile-rate；joint_absolute_profile_rate_20261117 | 22.224M / 384 |
| dev13 4–7 | spt-joint-full-covariance-profile；joint_full_covariance_profile_20261117 | 22.224M / 384 |
| dev15 0–3 | spt-action-scale-rate；action_scale_rate_20261117 | 22.220M / 384 |
| dev15 4–7 | spt-joint-profile-rate；joint_profile_rate_20261117 | 22.224M / 384 |
| dev6 0–3 | spt-joint-forward；joint_forward_20261117 | 22.224M / 384 |
| dev6 4–7 | spt-joint-action-scale-rate；joint_action_scale_rate_20261117 | 22.224M / 384 |

最近实际小时检查UTC06:00:03：训练rank分别4/4/8，其他卡在冻结读出或启动，不称持续满载。完整协方差组的启动登记误写dev15，与cfg的dev13不符，在训练前退出；已保留原日志并仅修正主机参数，0更新丢失，冻结源码/训练配置不改，真实8/16及主训待确认。绝对等方profile已实际8→16并主训8600步。下一检查UTC07:00:03。

## 决策所需结果

下述均同64反复使用的DEV、双224RGB、最多5命令/norm .3/native2mm；格式为回报% / 平均末端mm / final成功数。各预算按活跃后果训练小时，不是墙钟或独立案例；32确认封闭。

- 旧每视角无熵绝对前向1/2/3/4h：82.62/79.27/70.21/65.53%，39.02/47.52/69.11/77.59mm，final1/0/0/0；保全完整4h及部分5h全状态后交给联合绝对profile组，后继继承0更新。状态相关F＋像素/后果辨认1→4→10h为83.90→91.24→91.40%、38.67→19.81→18.92mm、final均0；中间5/6h各1，完整10h及部分11h状态保全后交给完整协方差组。
- 每视角尺度前向1/2/3h：77.00/75.55/75.80%，43.84/52.84/55.03mm，final0/3/1、any2/5/4。最小20%事实输入3.06–7.68mm的一小时当前命令RMSE4.68→2.63mm，<5mm案例2→11，但粗定位及尾部变差；细分点不改写严格成功。
- 双相机联合128维绝对前向1/2/3h：91.43/90.23/90.70%，21.02/26.33/23.34mm，final1/0/0、any3/1/0。首小时比每视角绝对四厨房均好，区组95%回报收益[4.34,14.10]点/终距改善[8.69,28.30]mm；未形成稳定精度。联合仅多3200可训练参数、同128码/384头，pool/Q初始化改变，非独立seed或等更新比较。
- 联合尺度前向1/2h：69.52/83.03%，69.82/30.30mm，final5/4、any5/7。细动作可读性更好，粗控制仍弱于联合绝对组，尚未统一两种收益。
- 联合等方profile将每转移原Gaussian率l换成D/2 log(1+2l/D)，无新网络/权重/尺度。尺度归一版首小时76.75%/51.08mm/final3、any3，较未profile同尺度组粗指标改善但严格成功5→3，继续2h。绝对尺度配对只关闭尺度归一，复用不可变code-joint-profile-rate-v1，4废弃GPU更新通过、主祖先0，真实8/16和主训已过，暂无控制。入口joint-profile-rate-20261117、joint-absolute-profile-rate-20261117。
- 完整协方差profile去掉等方限制：Σ≥bI下Gaussian KL最优Σ=bI+eeᵀ，目标.5 log(1+2l)，添加零残差坐标时不变，无手选秩/新网络/系数；仍不是归一化预测密度、推理方差估计器或MI下界。99.9%以上残差在动作子空间只是结构依据，尚无性能证据。7独立KL/梯度/最优性/填充/隔离合同(.016s)和4废弃GPU更新(2.10s)通过，主祖先0、完整初始NN同09bc10e5。入口joint-full-covariance-profile-20261117、code-joint-full-covariance-profile-v1，启动参数错误已纠正，实际续训待确认。
- 固定投影W→W Q_XYZ Q_XYZᵀ已完成联合尺度1/2h同64DEV对照，0新body/头拟合：final5→7/4→5，any5→11/7→8，回报69.52→69.61/83.032→83.029%，终距69.82→69.37/30.30→29.47mm。两个预算均只有小幅精度收益，严格案例有得有失，投影自身7→5亦未scaling，不扩展新预算或晋级。子空间外99.8%权重却只贡献.31%事实命令能量的诊断不能解释整体失败。使用冻结Q，**不是纯E-only probe**；128模型—案例记录仍是64物理DEV。入口joint-projection-readout-20261117，flat根joint_projection_20261117，依据analysis/joint-projected-readout-first-two-hours.json。

## Lingjing最新上传与机制

最新远端anchored-state-jepa为484f10f3、decoder-free-as-jepa为cd03863b。最新增量主要是分开表示/预测器的scaling评测协议，没有真实RGB控制胜出的证据。原AnchorReader beta .3/1码头仅15.36/15.47%、243.62/250.68mm、全0，均完整1h及部分2h保全；相对锚点修复熵原点后1/2h16.09/15.40%、239.14/252.57mm、全0，完整2h及部分3h保全。未跑作者PushT门，不称完成PartP/PartR；未来动作chunk/专家策略分支不采用。

自己的当前条件熵绝对→相对首小时56.27→85.28%、121.78→33.48mm，四厨房都改善，但相对2/3/4h79.60/78.24/75.01%、49.68/49.45/61.66mm、final全0。相对完整4h及部分5h保全后交给profile，绝对完整2h及部分3h交给联合绝对组。无证据证明熵优于匹配无熵前向，reader拟合变好不等于控制收益。

四个冻结联合1h体的状态条件MSE/Gaussian协方差头各5000更新，额外88.05GPU秒，均只监督普通事实当前输入。Gaussian NLL明显下降，但均值RMSE改善至多约.08mm，最细组未稳定改善；按验证收尾，0新增控制，不推广核心代码。入口conditional-action-uncertainty-20261117。历史动作先验去偏、势差/状态条件头、响应校准及空间读出负结果见CONCLUSIONS，不重复包装再跑。

强加法前向并非“无inverse信号”：QᵀQ=I时，||d−Qa_t||²=||Qᵀd−a_t||²+||(I−QQᵀ)d||²，含代数等价的**当前事实输入**重建；显式目标仍为自然下一码，不含未来/最优动作。此结构可对齐控制坐标，但常数加法对接触/夹限/物体交互有限，Reach结果不能代替manipulation。

## 固定边界

固定12006普通TRAIN日志；头仅17920训练/1536验证的同转移实际a_t，表示冻结/detach，标签梯度不进后果学习器。输入仅RGB历史/实际动作，自然未来RGB或自身编码可作目标，目标图只用于控制。无外部教师、reward/state/taskID/语言、未来/最优动作、专家路径/策略。同步相机属于一个物理样本，独立样本不交换表示或统计。受保护评测/oracle/接触/闭环输入不进训练、教师或proposal memory。

新学习链随机祖先0；同初值分支不作独立seed。不同读出、容量、数据、教师、累计预算单列。32确认仍封闭，不把不同公开协议当公平胜出。正常续训保存model/optimizer/scheduler/全rank RNG/游标，实际8/16恢复已做，不重复CPU审计；有实际错误才修。不改已启动冻结源码、检查点或vendored runtime。

Git仅核心代码、必要配置、精简结论及已验证较优配置。原始细节/负结果/完整与部分状态留EFS；未有新best-config晋升。稳定规则见AGENTS，机制和历史见[CONCLUSIONS](experiments/CONCLUSIONS.md)，流程见[PROTOCOL](experiments/robocasa_stable_scaling/PROTOCOL.md)。
