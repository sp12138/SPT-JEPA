# 当前交接

目标仍是改进SPT，并在**独立动作头实际控制**下验证可靠稳定的scaling，最终超过RoboJEPA；尚未达成。用户要求本会话每小时检查，发现明确问题便修复/改进并继续，不在有限pilot后HOLD。只用dev13、dev15、dev6；不连接dev6-2c/grid16/dev2，不恢复旧goal、旧监控或旧ACTIVE任务。

## 状态与口径

- EFS总入口：`/efs/prescottsun/SPT-JEPA/runs/robocasa_reach3d_scaling_20261114`（下称R），其父目录下存放各模型根。**实时状态看R的MASTER.json、SESSION_CHECKS.json及实际进程/GPU**，本文不是持续利用率证据。
- 正式小时检查由当前主会话执行，不另建LLM监控。第45次快照`session-checks/20261116T0332-{dev13,dev15,dev6}.json`记录三机各8个实际GPU训练rank、瞬时78–98%、六组无ERROR。最新检查与下次时间只从SESSION_CHECKS读取，不把瞬时值当持续利用率。小时记录/有限交接用`MASTER.primary-write.lock`保护元数据，重复快照不推进时钟。
- 同Reach blocking执行接口，固定12006条普通TRAIN交互日志、无body数据增长，全学习链随机。编码器输入仅RGB，正向预测另输入实际a_t；自然未来RGB/自身码为后果目标，没有动作target或外部教师进入body。
- 动作头始终冻结/detach表示，仅用同一事实转移输入a_t的XYZ：17920训练/1536验证标签。普通交互日志不是专家策略或目标条件动作标签。读出时将自然next RGB换为控制目标RGB，分布差异必须用真实控制检验。
- 每点同64个反复使用的DEV案例、两路224 RGB、最多5命令/norm .3、native2mm final/any。32确认例仍封闭。单点、平滑、loss下降及偶然1/64不算稳定scaling；不同读出/头容量/FLOPs/预训练来源分别报告。
- 主动训练时钟是warmup900后inverse-sqrt active-wall，**不是旧controller的常数LR**。训练恢复必须含model/optimizer/scheduler/4rank RNG/数据游标；不修改在跑冻结源码。

## 当前六组

各占4卡；根名均以`robocasa_reach3d_`开头，位于R的父目录。表中是最近完整64DEV结果，不能把训练中阶段当已完成。

| 主机/卡 | 方法 / 根后缀 | 参数/动作头参数 | 最近完整结果：回报、末端距离、final/any |
|---|---|---|---|
| dev13 0–3 | E384/D6，`encoder_allocation_20261116`；spt-encoder-heavy-shared-mean | 41.625M / 2304，E均值 | 8h：90.69%、21.06mm、0/1 |
| dev13 4–7 | E384/D6+视角地址联合状态，`encoder_view_joint_20261116`；spt-encoder-heavy-view-address | 47.575M / 384，联合码 | 5h：85.78%、30.42mm、1/2 |
| dev15 0–3 | E192/D6，`allocation_small_shallow_20261116`；spt-small-encoder-shallow-decoder | 24.986M / 1152，E均值 | 5h：86.21%、29.06mm、0/1 |
| dev15 4–7 | RGB恒等支路+条件线性响应/非线性余项，`rgb_structured_command_20261116`；spt-rgb-innovation-structured-command | 41.662M / 2304，E均值 | 1h：87.30%、32.46mm、0/1 |
| dev6 0–3 | 字面RGB恒等支路，`rgb_innovation_20261116`；spt-current-rgb-innovation | 41.625M / 2304，E均值 | 3h：90.31%、23.43mm、1/2 |
| dev6 4–7 | RGB恒等支路+root条件先验，`rgb_root_marginal_v2_20261116`；spt-rgb-innovation-root-marginal | 41.625M / 2304，E均值 | 第一小时在训，尚无控制结果 |

- 私有源码：dev13分别R/`code-encoder-allocation-v1`、`code-encoder-view-joint-v1`；dev15小E/D6用`code-encoder-allocation-factorial-v1`，新余项组用`code-structured-command-v1`；dev6的RGB组用`code-precise-consequence-v1`，新先验组用`code-root-marginal-v2`。每组祖先更新0，不继承旧训练权重；公共随机子张量配对不算独立seed。
- 实际启动证明见R/`encoder-allocation-20261116`、`encoder-view-combination-20261116`、`encoder-allocation-factorial-20261116`、`precise-consequence-20261116`、`structured-command-20261116`及`root-marginal-prior-20261116`的`ACTUAL-STARTUP-VERIFIED.json`。均实际保留8/16更新、全状态恢复、四GPU rank/配置/源码及CPU模型/优化器加载核验，不以准备门替代实际启动。
- 此前dev6结构对照分别检验：`current_RGB+D_delta(anchorE,z)`按FP32传递已知画面；或`z+MLP(z,A a_t)`允许离散后果对当前命令非线性。保留原目标、Gaussian噪声、rate/beta/优化器及E384/D6；4次功能更新废弃，无新loss或动作标签。E均值子图状态/实际函数同原零点40.49%/154.26mm才复用，不是新案例。
- dev15新余项组保留`z+f(z)+B(z)A a_t`，再加`g(z,A a_t)-g(z,0)`；仅多37120参数，旧响应保留，零输入余项精确0。与RGB恒等支路保持原子张量及初始全函数/loss/共同梯度/RNG相同，4废弃更新后梯度/非线性已验证；无新loss、权重或标签。旧大E/深D完整4h/42120步/64DEV及CPU全状态后0重放交接，实际8/16恢复/四GPU/源及CPU优化器均过，入口`structured-command-20261116`。仿射F响应读出已明确拒绝非线性模型，不能沿用其旧解析basis。
- 旧fused完整6h为83.27%/37.77mm、final0/any1；旧视角组完整4h为82.33%/43.20mm、final1/any2，单例不是稳定精度收益。各完整64DEV及CPU全训练状态核验后退役，旧SGD重放0。
- 本次有限交接在成功Popen/MASTER更新后，错误断言一个准备阶段未创建的待启动标记；只核对实际唯一controller并补记录，未重启/重复启动训练。原助手/错误保留`precise-consequence-20261116/HANDOFF-POSTSTART-MARKER-FAILURE.json`。启动核验v1又误从train config读取只在plan中的readout选项；v2只修核验器并通过，训练未改。

## 已获insights与待办

- **保留已知结构优于本轮直接替换为普通MLP。** RGB恒等支路1h86.72%/32.31mm，较同E/D基组+4.89点，四厨房回报区组95%[.86,12.47]，大部分来自厨房10；距离差区间[−25.73,.28]mm含0，strict0。普通MLP1h77.24%/54.73mm，四厨房均差，但其F从815360降到37120参数，不能将差异纯归因非线性。8相同普通TRAIN窗口中，原/RGB/MLP的一步潜预测/copy .675/.555/1.652，XYZ置零后的模型预测误差比1.52/1.85/.706；MLP此时动作响应未学好，该消融不是物理反事实。RGB一步像素MSE .01388仍略高于原.01364，不能用像素拟合排名解释控制收益；长程latent仍差，RGB7步/copy1.833。原始`analysis/precise-consequence-first-hour-{paired,factual-audit}.json`。
- **容量与吞吐都重要。** E384/D6的1/2/3/4/5/6h回报81.83/86.07/86.47/89.35/89.17/91.35%，距离42.23/30.87/30.73/24.52/24.13/17.78mm，final全0。2→3h暂平台、4h改善、5h近平台、6h再改善，第7/8h回报90.39/90.69%、距离22.45/21.06mm、final1/0，未保持第6h粗指标，严格精度尚未scaling。2×2在3h的E192/D12、E192/D6、E384/D12、E384/D6分别75.50/84.83/85.86/86.47%；宽E在浅D下优势区间含0。E384/D12-3h与D6-2h更新30416/30468，回报85.86/86.07%、距离32.98/30.87mm，数值接近但墙钟LR调度不同，是事后近更新对照而非等价证明。4h小E/浅D与大E/深D为85.29/86.09%、均30.19mm、final1/0；参数/FLOPs/头宽及dev15共卡研究开销不同。原始`analysis/encoder-decoder-factorial-{first-hour,second-hour,third-hour}-paired.json`。
- **保留视角来源确有早期粗收益，但联合压缩不是已经成功的解法。** 旧同容量视角地址1/2/3/4h为69.64/78.98/80.77/82.33%，距离62.68/43.14/45.62/43.20mm；2h比global融合+12.98点、四厨房均改善，后续精度仍不足。组合组使用384参数码头，不能直接用其分数与2304参数E均值头给body排优劣。固定组合2h父模型的2304参数E均值读出已DONE：81.53%/40.31mm、严格0；同父384参数码头84.63%/33.41mm，四厨房均较好，均值差−3.10点、95%[−4.54,−1.66]，距离+6.90mm、[6.32,7.65]。同宽的分视角E均值2h86.07%/30.87mm，但组间回报区间含0/体量及算力不同，不证明body普遍优劣。旧码默认/实际E均值/缓存—策略/隔离门均过，0body更新/同64DEV；额外dev15 GPU6/7共卡成本单列，负原型代码留EFS，未进入Git。入口R/`encoder-view-mean-probe-20261116`及`analysis/encoder-view-matched-mean-second-hour-paired.json`。
- **增大这类动作头没有解决精度。** 固定容量组3h/45998步父模型，同5000头更新，MSE/MAE均值线性头为86.47/87.95%、30.73/25.93mm、final0/0、any0/1；460035参数状态相关Response头为87.07%/28.05mm、final0/any1，较MSE小改而弱于小MAE，不扩展。另压缩码LP384与冻结F响应C9为86.94/83.64%、28.92/37.58mm、final0/2、any1/3；F严格点数略增但粗指标下降，不整体推广。均0body更新、同64物理DEV；入口`encoder-{robust-mean,code-response,nonlinear-mean-head}-probe-20261116`及对应analysis配对JSON。C9校准矩阵奇异值1.134/1.034/1.014、条件数1.118，未见数值退化或整体缩小，不能归因为简单增益不足；只读`analysis/encoder-response-calibration-conditioning-audit.json`，非物理Jacobian。MAE诊断曾15分钟分片超时，只续同权重剩余1例，63完成例SHA不变，0重训，原错误保留。 新增固定5h/77184步父模型的空间Potential头：4620可训练参数+原2304固定线性权重，同5000更新，验证选回step0，完整64例逐案例回报/距离与原89.17%/24.13mm/final0/any1精确相同；不扩大该头预算，不证明空间信息无用。v1评测入口引用错误在评测/SGD前发现，保留7完整FP32 token缓存；新不可变v2复用后补齐1片，0body/头SGD重放，实际梯度与缓存—Policy门均过。负原型仅EFS：`encoder-spatial-action-head-probe-v2-20261116`及`analysis/encoder-spatial-head-fifth-hour-paired.json`，额外共卡成本单列。
- **不能凭猜测调rate或LR。** 4h模型16普通TRAIN转移的latentleaf KL/像素梯度中位小/大命令.102/.200，未显示rate支配；当前图重构MSE/自然变化MSE中位.551/.038，静态错误能量.224/.143，未显示背景支配。这里只是一部梯度/少量事实，不是唯一归因。固定参考12288对实际150528元素归一化会缩小rate12.25倍，目前不据此改beta。原始`analysis/fine-factual-coding-budget-audit.json`。固定1216条动作头日志的全部RGB SHA及19456事实转移检查，仅2训练/1验证转移两路画面完全不变且输入非零；由此产生的零差分头验证RMSE下界仅.064mm，远小于当前约29.5mm，没有支持大量完全相同画面的标签冲突解释；仍不能排除近似视觉别名。0模型/头/控制更新、11秒CPU，`analysis/factual-current-action-observability-audit.json`。
- **短期后果学到不等于控制精度。** 8TRAIN窗口的E384/D6一步预测/copy在1/2/3h为.675/.474/.443，三步.850/.499/.456，七步仍2.121/1.343/1.141；3h消融当前XYZ令一步误差2.29倍。实际F响应条件数1.57–1.86，非真实物理Jacobian。只读诊断v1曾把动作错移到下一帧，已作废并按训练`act_emb[:,:-1]`重跑v2；正式训练、动作标签和控制一直正确。勿沿用v1。 Current rate already centers the complete clean rollout; its residual is e_t-rho*e_(t-1), not a teacher-forced one-step mean.

RGB恒等支路1/2/3h为86.72/87.51/90.31%、32.31/31.85/23.43mm、final0/0/1；第3h较同预算基组+3.84点、四厨房区组95%[.59,7.67]、距离−7.30mm、[−14.04,−.57]，3/4厨房改善，是继续训练的粗定位信号，尚非严格scaling；余项组首小时87.30%/32.46mm、final0/any1，相对RGB的回报差+.58点、四厨房区组95%[−.89,1.35]，距离差+.15mm、[−2.77,4.03]，暂无可靠增益。基组第7/8h未保持第6h粗指标，5/6/7/8h读出alpha均.01、验证RMSE仍降，不能再次归因为alpha切换。小E第5h86.21%/29.06mm、严格0；组合第5h85.78%/30.42mm、final1，均未严格scaling。配对定位`analysis/structured-command-first-hour-paired.json`、`analysis/precise-consequence-third-hour-paired.json`及`analysis/encoder-heavy-late-control-scaling.json`；同64已见DEV，不含种子不确定性。

当前新组只改多步先验依赖图：在现有完整clean rollout中心不变时，未来码条件独立于前一未来码、共同条件于初始码，保持各时刻先验边缘方差及root协方差；误差由`e_h−rho*e_(h−1)`改为`(e_h−rho^h*e0)/sqrt(S_h)`，`S_h=sum_j rho^(2j)`，加精确Gaussian KL常数。不是修复teacher forcing或统一调beta；7步原弱误差模态的惩罚变为4.00倍、强模态为.156倍，无新增参数、尺度或标签。同8个普通TRAIN窗口的RGB1h/2h及原基组7h残差中，原最弱时域模态占误差能量77.7/81.4/84.6%，却只占原均值rate的26.0/30.3/34.6%；新几何实际均值惩罚为1.34/1.50/1.66倍，提供小样本依据而非唯一归因或控制证明，见`analysis/root-marginal-observed-error-audit.json`。独立18个完整Gaussian值/梯度/同边缘检验、五类实际旧默认回归、实际新loss与leaf梯度、单步/rho0全模型门及4废弃功能更新通过。根`root-marginal-prior-20261116`，私有`code-root-marginal-v2`，已在dev6 GPU4–7从零主训；v1 FP32等价图元素梯度守卫失败及源码/日志保留，v2保留通式e0并记录范数守卫，实际两个极限梯度逐位同。有限助手只完成普通MLP完整3h/44028步/64DEV及CPU全状态收尾，结果83.53%/36.72mm、final/any0；仍在粗改善但落后原/RGB，F容量和第2h空间头、第3h先验门共卡成本单列，不证明该方法类失效。主会话复核后唯一启动新组，0旧SGD重放/祖先0。新组实际8/16保留/全恢复、四GPU rank、私有train及CPU绑定analytic loss源码SHA与模型/优化器均过，证明`ACTUAL-STARTUP-VERIFIED.json`；无新控制成绩。

继续主会话小时检查，收齐后续实际控制曲线，必要时在完整阶段保全后调整。不把准备门、loss或偶然1/64当成功。有限诊断DONE不结束目标，32确认仍封闭，尚无新best-config晋升。

## 历史边界与恢复注意

- 完整历史与负结果见[CONCLUSIONS](experiments/CONCLUSIONS.md)，当前协议见[PROTOCOL](experiments/robocasa_stable_scaling/PROTOCOL.md)，已验证配方见[best-configs](experiments/best-configs.json)，原始记录见[归档说明](docs/experiment-archives.md)。Git只保核心源码/配置/精简结论；日志、权重、逐例、临时脚本和监控快照留EFS。
- 旧冻结G分支用了外部V-JEPA；如原仿射F响应12h94.86%/9.70mm/3final、13h回落，不与当前从零链或公开RoboJEPA作公平胜出比较。最早Push日志的执行器z约束与Reach不同，已经整阶段保全；当前同Reach语料是设定修正，不算纯算法收益。
- 多个旧freeze的共享refine链接曾被写穿；独立原版/私有源码修复与实际加载证明在`source-recovery-20261114T2353`。不能追溯证明未记录的全部历史加载。新版本用`experiments.robocasa_compare.source_freeze.freeze_with_overrides`安全冻结；不写穿旧链接，不动vendored runtime。
- 旧shared小语料完整8h/128308步已保全，额外未完成第9h的1992–2092更新排除、未继承/重放；若未来明确决定恢复8h须披露可能重放范围，不能自动恢复。其他近阶段旧组均完整64DEV和CPU全状态收尾后交接。
- 学习器严禁状态/reward/成功/任务ID/语言、未来/最优动作和专家策略；保护性评测/oracle/接触/闭环诊断不得用于训练、教师或proposal memory。调试只读聚合结果或过滤错误，勿把含模拟状态的逐步日志当方法灵感来源。
