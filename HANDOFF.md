# 当前交接

目标：从零SPT-JEPA后果学习＋冻结表示的独立当前动作头，验证可靠稳定的实际控制scaling并超过参照。**尚未达成**。只用dev13/dev15/dev6；禁用dev6-2c/grid16/dev2。时间优先算法与闭环结果，不恢复旧goal、monitor或有限HOLD。

状态真源R=/efs/prescottsun/SPT-JEPA/runs/robocasa_reach3d_scaling_20261114，查MASTER.json及SESSION_CHECKS.json。每小时同会话检查；最近第78次UTC13:15:59，下一次UTC14:15:59。该快照训练rank为8/8/4、利用率0–100/81–97/0–97%，新组处在启动/恢复，不能称持续24卡满训。

| 主机/卡 | 当前run；根后缀加robocasa_reach3d_ | body语料 / head参数 |
|---|---|---|
| dev13 0–3 | spt-canonical-world；canonical_world_20261117 | 12518 / 9 |
| dev13 4–7 | spt-input-scaled-pixel-world；input_scaled_pixel_world_20261117 | 12006 / 384 |
| dev15 0–3 | spt-joint-relative-full-covariance-profile；joint_relative_full_covariance_profile_20261117 | 12006 / 384 |
| dev15 4–7 | spt-joint-profile-rate；joint_profile_rate_20261117 | 12006 / 384 |
| dev6 0–3 | spt-orthogonal-input-world；orthogonal_input_world_20261117 | 12518 / 384 |
| dev6 4–7 | spt-joint-action-scale-rate；joint_action_scale_rate_20261117 | 12006 / 384 |

新canonical/强对照UTC13:12启动，实际保留8→恢复16、主训900步、完整初始NN同6246cbd2已确认；40.888348M全部从零，未继承短测或旧模型。spt-canonical-fixed-noise-world已通过短测，拟接替dev13 4–7，须先收齐旧input-scaled的2h/64例、保全状态并确认实际进程退出，不能重复启动。

## 当前成绩

均同64反复使用的DEV、双224RGB、最多5命令/norm .3、native最终距离<2mm；any另列，32确认仍封闭。预算是活跃后果训练小时，不是独占墙钟或等FLOPs。

- 相对等方profile1–6h：回报76.75/80.14/81.71/87.10/81.96/84.23%，距离51.08/41.72/40.30/25.48/42.67/34.85mm，final3/6/12/8/11/13、any3/8/13/13/12/15。13/64是当前DEV最高，仍有预算回退和案例丢失，不是可靠稳定scaling。
- 相对完整协方差1–5h：72.17/78.41/76.40/80.34/82.07%，67.87/54.56/65.96/51.53/45.90mm，final6/6/8/11/10；粗指标和严格成功尚未统一。绝对版4h84.88%/43.99mm/final1，完整4h＋部分5h已保全退役。
- 联合尺度前向1–8h final5/4/7/5/7/5/7/8；末8h87.01%/24.23mm/any9，继续。普通RGB绝对1/2/3h74.67/87.08/88.62%、54.28/25.81/23.73mm、final0/1/0；完整3h＋部分4h已保全，由新强对照接替。
- 整项风险相对加权RGB1/2h25.13/28.24%、205.44/195.82mm，严格全0，完整2h＋部分3h退役。全128维输入尺度噪声1h22.40%/247.27mm/final0，保留2h结果后决定接替；不把该负结果包装成改进。
- 新12518语料组尚无正式控制结果。原固定12006、追加数据、不同head容量及其他预训练来源分别报告；无SOTA胜出结论，best-config未晋级。

## 关键insights与新实验

Lingjing最新仍为484f10f3（评测协议）、cd03863b/d4da0d16（讨论及gap工具）。已借鉴其拆开表示、预测器及读出的诊断；原AnchorReader实际控制负结果保留，尚未跑作者PushT PartP/PartR，不称全部复现。含未来动作chunks的阶段不采用。

普通验证显示新自由RGB先验首小时误差为copy的.449，但最细输入为copy的2.062倍；16个普通大小转移的posterior/copy像素误差大动作.309、小动作.944。大幅后果学会不等于精细码可控。扩大独立头、局部Jacobian、补64旋转事实的六个头对照均未同时改善正常/精细/旋转误差，0新控制收尾。

一个结构性漏洞：若u在轨迹内不变，E'=E+Q c(u)不改变前向差分或事实inverse，当前RGB条件解码器亦可吸收该偏移；goal/current的u不同则产生沿动作方向的假位移。理想化推导不是唯一因果证明。只读证据：4h0.57mm处仍发11.46mm命令，其中11.06mm沿Q_XYZ；相同轨迹精确重放时当前/目标腕姿态差.46°。普通XYZ=0的.005rad输入被3/4/5h读出2.14/3.13/3.04mm，不能直接等同实际EEF位移。训练/在线同AREA224，无缩放错误。

新先验设P=QvQv^T，m=m_free+P(z+Qa_t−m_free)：仅3维平移码按当前平移输入推进，余125维保留非线性世界状态。候选用Sigma=sigma0²(I−P)+sigma_p²P，sigma_p=sigma0*已知当前输入长度；另跑canonical固定噪声消融和自由F固定噪声强对照。p=Qv^T E(RGB)是冻结视觉输出，9参数当前输入校准；强对照保留128维码/384参数头。mean和读出均有变化，不作单因素归因。该结构有当前输入inverse的代数信号，绝无未来/最优动作监督；命令偏移不是实测位移，伺服/限幅误差仍存在。

512条普通TRAIN旋转日志全部保留，与原12006合成12518；不按state/性能筛选，collection servo显式执行旋转，评测不改。body-manifest SHA=f3b46db167e60a7a6a09ab16e444f5cd4b9506c7017ee070b402e47ab592031c，源orientation_interventions_20261117/BODY-CORPUS.json；仅RGB/实际当前输入进入学习。一次普通重放RGB不匹配已标无效，不能用其位置统计验证原始记录或影响训练。

模型源code-canonical-state-v3，采集源code-canonical-world-v2；六独立Gaussian/梯度/协方差/混合精度及实际XYZ＋旋转短测通过。强对照/canonical-block各8废弃短测更新（含旧整项噪声4）、canonical-fixed4，均不继承；最终短测分别5.16/4.47/3.70s。主入口orthogonal-input-world-20261117、canonical-world-20261117；待启canonical-fixed-noise-world-20261117。原始诊断统一见R/analysis，不复制逐例文件进Git。

## 接下来

1. 收齐全维输入噪声2h；若仍弱，保全完整2h/部分3h后，以canonical-fixed-noise替换该4卡。已启两组继续完整小时读出并长期训练，不有限HOLD。
2. 每小时检查实际PID/GPU/ERROR及真实成功；有明确问题再改。新源、数据与head容量边界单列；32确认不开给方法选择。稳定性和独立初始化尚须验证。
3. 只使用RGB历史/实际输入与自然未来RGB或自身码；当前a_t标签仅用于冻结/detach动作读出，无未来/最优动作、专家路径/策略、state/reward/taskID/语言或外部强教师。受保护评测/物理诊断不进训练、教师或proposal memory。

恢复含model/optimizer/scheduler/各rank RNG/游标；不改旧冻结源码或检查点，不重复CPU优化器审计。Git只存核心源码、必要配置与精简结论，未验证配方不进best-config。详情见[CONCLUSIONS](experiments/CONCLUSIONS.md)及[PROTOCOL](experiments/robocasa_stable_scaling/PROTOCOL.md)。
