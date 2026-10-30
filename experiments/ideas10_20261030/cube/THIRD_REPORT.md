# D7 / D8：两种尚待证伪的学习规则，及其不能声称的内容

2026-10-31。独立设计记录见 `THIRD_DESIGN_FIRST.md`，先设计、后查最近工作。本轮只运行标准库 CPU 合成检查，不训练模型、不运行 GPU、不改变旧 pool/score。所有精确数值见 `third-cpu.json`，脚本为 `third_cpu.py`。**这些是规则区别与失败边界，不是 TwoRoom/Cube 性能结果。**

建议把 D8 留为“数学上有独立可检验内容、采集代价和新颖性尚未解决”的候选，把 D7 留为“目标偏好的条件性假设”。D7 的线性例确实有不同最优解，但实际 Cube 的观察步长可能直接破坏它最关键的假设。两者均未达到应启动长训练的证据标准。

## D7：按可控响应的阶次确定必须预测的观测输出

### 具体失败与选择

Cube 的冻结诊断显示：TF 相当准确地预测自己学出的几何，但真实编码目标成本主要随手的位置变化；删除一个 3 维线性手读出 span 不会删除手信息。不能由此推出“所有手的信息都应删除”，因为 TwoRoom 的 agent 本身正是目标对象，Cube 的手也仍是预测物块的必要中介。

D7 的选择是：完整状态仍供动力学使用；另从真实干预中学习“最晚才响应、但确实可控”的观测输出，要求世界模型显式保留并预测它，目标距离优先用这个输出。它把“更靠近动作链末端”设成一个公开的归纳偏好，**没有从无奖励数据中识别用户目标的普遍保证**。

### 可实现的最小版本

预先固定有界图像特征 `phi(x)`，例如固定空间网格的 RGB/Haar 系数，不使用物块/手标签。每个训练单位是一对从相同允许初始条件开始的真实短轨迹：初始 bounded pulse 不同，后续动作相同。记实际观测差

`d_k = phi(x_k_plus) - phi(x_k_minus)`。

先学习单位长度的线性输出 `w`（多个输出用正交行），而不是任意可学非线性读出：

`R_k(w) = E[(w^T d_k)^2]`。

阶次 `r` 的候选头满足所有 `k<r` 的 `R_k` 小，同时 `R_r` 明显非零；选择有独立验证证据的最深阶次。每一个随机梯度项只是同一 paired rollout 的平方差；单位范数是参数约束，不是 minibatch whitening。恒定背景所有响应均为零，所以不能因“永远不动”进入最深层。

一个明确的训练方式是先在训练交互上拟合并冻结这些头 `W_r`，然后把 `psi_r(x)=W_r phi(x)` 作为普通观察目标，加入

`L_output = ||h_r(E(x))-psi_r(x)||² + ||h_r(F(E(x),a_0:k))-psi_r(x_k)||²`。

这不预测最优动作，不用物理状态作训练标签。保持原完整预测 state，输出头共享固定的总维数预算；新增头、分支数据、额外 forward 和监督项必须计入 matched controls。部署成本用 `||h_r(z_pred)-psi_r(goal)||²`。需要分别比较“只新增预测约束”“只改目标成本”“两者同时”，否则控制器改进不能冒充世界模型改进。

此版本已经是一个可编码的学习规则，但并不天然优雅：多阶段头估计、层级检验、额外交互与噪声门槛都是代价。若只在已训练 latent 上做投影再改 CEM，则仅算诊断/控制器方案，不能算新世界模型算法。

### 精确性质与旧 Gramian 的不同解

线性系统 `s_next=A s+B u` 中，输出 `c^T s` 的阶次为最小的 `r>=1`，使 `c^T A^(r-1) B !=0`；更早项全为零。这个性质是经典相对阶/Krylov 数学，不是我们的定理。候选新规则是**优先最后一个有实际响应的输出，而非保留所有 controllable coordinates 或按控制能量缩放**。

合成系统：`r_next=u; b_next=b+r; background_next=background`。手 `r` 是 1 阶，物块 `b` 是 2 阶，背景不可控。目标 `(r,b)=(0,1)`，从零出发，两条有界动作序列分别为 `(0,0)` 与 `(1,2)`：

| 候选终点 | Euclidean 成本 | H=2 inverse-Gramian 成本 | 深层输出 b 成本 |
|---|---:|---:|---:|
| `(0,0)` | 1 | 1 | 1 |
| `(2,1)` | 4 | 4 | 0 |

两步 controllability Gramian 恰为单位阵，所以旧 I10 在这里与 Euclidean 完全同序，D7 确实选了相反动作。若任务只在乎物块，D7 把误差从 1 降至 0；若任务在乎手姿态，同一个选择反而把误差从 0 升至 4。这是两个评价函数共享同一动力学、图像、目标图像的可构造反例：**无奖励交互不可能普遍告诉算法这张目标图中究竟哪个差异应优先**。

纯导航 `x_next=x+u` 没有更深的非恒定可控层，回退到唯一 1 阶位置；不会像“对所有可逆动作取商”那样把连通房间内所有位置都合并。这只是一个条件正确的例子，不证明真实 TwoRoom 的视觉头能学出位置。

### 有限样本认证与失败界

若某个固定头在独立验证 pair 上满足 `||d_k||<=L`，则平方响应在 `[0,L²]`。Hoeffding 给出单个固定头/阶次的半径

`b_n = L² sqrt(log(2/alpha)/(2n))`。

验证早期响应 `R_hat<=eps` 只能保证真实响应 `<=eps+b_n`；验证后期 `R_hat>=gamma` 只能保证真实响应 `>=gamma-b_n`。多个头/阶次需分摊 alpha；不能在同一验证集挑头后继续套单头保证。真实图像微弱响应会使这个检验没有 power。

脚本的弱耦合例 `b_next=b+eta*r`，`eta=1,.1,.01,.001` 的精确阶次都等于 2，但后期响应能量分别为 `1,.01,.0001,.000001`。因此“更深”本身不能保证可估计性或实际可控效果。

更严重的是**时间分辨率**：把同一合成系统两个 primitive steps 合成一次观察，并在这两步保持同一输入，观测级输入向量从 `(1,0)` 变为 `(1,1)`。原物块坐标 `b` 现在立即响应；早期不变的输出变成 `b-r`。系统物理语义没变，D7 偏好的坐标却变了。Cube stride5、接触和控制器内部状态都可能造成这种问题。不能把“动作经由手传给物块”直接当成在当前图像步长可识别的阶次差。

其他明确反例：导航中有受 agent 控制的门，最深头可能优先门而非位置；Cube 的姿态/接触/保持条件不一定与最深位置输出一致；全局线性 `phi` 若让接触区的第一步响应 span 已覆盖所有物块方向，则不存在需要的深层空间。任意非线性头还能在训练样本附近人为变平、制造假阶次，所以最小版本故意限定可审计的线性观测头。

### 三个最近的主文献与额外碰撞检查

1. [Isidori & Ruberti, input-output response synthesis (1984)](https://www.sciencedirect.com/science/article/pii/S0167691184800468)：控制响应阶次与线性化是既有控制理论。D7 不声称发明它；区别只能是把观测阶次当作 reward-free 目标优先级及预测保留规则。
2. [Causal Dynamics Learning, ICML 2022](https://arxiv.org/abs/2206.13452)：学习依赖结构以改善动力学和任务无关 abstraction。它支持保留必要依赖；D7 的特殊主张则是在完整动力学之外优先最末端响应输出，这会丢弃某些任务所需的目标区分。不能把“学习因果动力学”算新。
3. [InfoPower, ICLR 2022](https://arxiv.org/abs/2204.08585)：以 action-related information/empowerment 优先功能相关因素。D7 在同一链上有意优先较晚响应，甚至不让最直接的动作响应主导目标成本。这里只证明了规则上的区别，没有证明与 InfoPower 端到端最优解的一般差异。

另一个必须纳入对照的 [Action-bisimulation, RLC 2024](https://arxiv.org/abs/2403.16369) 已研究 reward-free 多步 controllability，并递归折扣远处影响。D7 不能把多步控制相关性本身声称为创新；可能区别在于对经证实更深的输出给予优先，而不是折扣它。它与 Krylov/相对阶输出学习的重叠很大，当前只给 **conditional rule-level novelty hypothesis**，不作已证明原创的宣称。

## D8：让实际噪声通道承载被真实未来实验要求的区分率

### 具体失败

“干净 latent 上可读出”与“加训练/生成所用噪声后仍能作条件预测”是两个命题。TwoRoom 的尺度诊断提示过这种缺口，但已完成的去噪/clean-state 控制没有稳定救回任务；所以不能把它宣布为唯一病因。Cube 还存在目标几何与物块预测不足，单纯扩大 latent 并不能自动修好。

D8 要检验一个严格、更窄的命题：两个真实初始条件在**同样动作实验和相同侧信息**下产生可区分的未来时，有限噪声 latent 通道是否有能力承载这种区分？若能力不够，任何共用后续 decoder 都无法补救。

### 必要误差界及证明

设两个噪声码为 `q_i=N(mu_i,sigma² I)`，令

`tau=TV(q_1,q_2)=2 Phi(||mu_1-mu_2||/(2sigma))-1`。

真实未来分布是 `P_1,P_2`。任意固定有界测试 `v(Y) in [0,1]` 给出

`Delta=|E_P1 v - E_P2 v| <= TV(P_1,P_2)`。

若同一后续 Markov kernel `K` 从 noisy state 生成未来，且 `e_i=TV(P_i,q_i K)`，triangle inequality 与 data processing 给出

`Delta <= e_1 + TV(q_1 K,q_2 K) + e_2 <= e_1 + tau + e_2`，

所以 **`max(e_1,e_2) >= max(0,Delta-tau)/2`**。

这是标准数据处理思想的直接推论，不是新信息论。它的用途是把世界模型的经验区分需求转成实际噪声尺度下、可验证的最低容量条件；不是一个不依赖数据的 variance floor。

要求每个条件未来误差 `<=epsilon` 的必要条件是

`tau >= b=max(0,Delta-2epsilon)`，

对应均值距离

`d_min=2sigma Phi^-1((1+b)/2)`。

若两个码关于 0 对称，在标准正态先验下，**均值贡献**的平均 KL 最少为 `d_min²/8`；固定 sigma 所带来的方差 KL 是另一项，不能把上述数值误称为完整模型 rate。这个下界不保证所有 pair 的最小码能同时在有限维空间实现。

### 真正不同的学习规则

paired sample 包含两个初始观察/历史、共同的 bounded open-loop action experiment、其实际未来，以及固定的图像测试。用训练采集和独立确认得到 `Delta_LCB`，优化

`L_contract = [max(0,Delta_LCB-2epsilon) - TV(q_theta(.|x1),q_theta(.|x2))]_+²`。

这个 loss 是 paired-unit sample-level；不跨 minibatch 求均值/协方差/negative mining。保留原行动条件预测目标。`epsilon` 在概率单位上是一个全任务共用的模型容错预算，不按 Cube/TwoRoom 成功率各自调。固定噪声 sigma 必须是被约束的真实路径上的 sigma，不能任意把噪声缩至零绕开条件；若允许可学方差，需要使用实际后验可分辨率并计入完整 rate。

`theory_literature` 提供的配套采集规则不另计为一种方法：从同场景训练初态中选 noisy-state TV 小的 pair；执行同一个已记录动作 prefix bank；用实际观测测试的 `Delta_LCB-tau` 最大违反量选择 witness，再用独立执行/测试确认。**模型想象只可用于挑选查询，不可生成差异标签。**

这种不同初态/相同动作的 witness 与旧 I09 不等价：令真实状态为 `(cursor,block)`，下一步为 `(action,block)`，编码器只保留 cursor。同初态不同动作的动作识别可达 100%，但相同动作下不同 block 的未来仍被编码器合并；D8 会检出后者。额外环境查询、重复重置、图像测试覆盖与确认数据均要作为额外训练暴露供给 baseline，不能偷加数据后归功于 loss。

实际随机系统的 Delta 需要重复或有证明的估计。对每个初态独立执行 n 次、有界测试的经验均值作 Hoeffding 下界：

`Delta_LCB=max(0,|mean_1-mean_2|-2 sqrt(log(4/alpha)/(2n)))`。

脚本用 `n=2048,alpha=.05` 时单概率半径仍约 `.0327`，小差异的证据明显缩水。它揭示真实代价而非推荐采集 4096 次重置。确定性系统可以用固定有界测试的真实输出直接获得差异；若把每个不同像素都当作完美二元状态标签，Delta 变成 1，又会退化普通固定负样本 margin。自然环境无法重复同一隐状态时，还需明确近邻误差/历史输入，而不是默认为可重置仿真。

### 精确 CPU 检查：必要界可达，且最小码与固定 margin 不同

在一维 Gaussian code 与 Bernoulli 未来实验中，Gaussian 最优阈值测试的对比度正是 tau。通过共同随机化阈值解码器，可实现任意 `<=tau` 的 Bernoulli 概率差；当真实差为 Delta 时，最佳最坏误差恰为 `max(0,Delta-tau)/2`。脚本枚举 `101` 个 Delta 与 `81` 个均值距离，共 `8181` 个配置，理论下界与构造可达误差最大差 `5.6e-17`。

设 `sigma=.1,epsilon=.02`：

| 两个同时间间隔 pair 的真实未来差异 | 必需 noisy TV | 最小码距离 | 最低均值 KL |
|---|---:|---:|---:|
| Delta=.8 | .76 | 约 .2350 | 约 .00690 |
| Delta=.1 | .06 | 约 .0151 | 约 .0000283 |
| 同一真实状态的循环再访，Delta=0 | 0 | 0 | 0 |

两个不同 pair 若都采用第一行的固定强 margin，均值 KL 总和是 D8 逐 pair 必需容量的约 `1/.5021` 倍。这个受控例显示最小解确有区别；不代表世界模型整体 KL 可节约一半，也不保证实际训练能到达最小解。

另一个**仅为说明尺度的合成假设**：64 维每维 clean std `.00015`，代表距离 `sqrt(128)*.00015`，加 `.1` 噪声后 TV 只有约 `.00677`，最优二分类准确率约 `50.34%`。若真实未来测试差 `.8`，最坏条件预测 TV 误差至少约 `.3966`。这不是对真实 TwoRoom pair 的测量，不能据此声称该任务已满足假设。

### 与 TD-WorldModel 的具体区别

[TD-WorldModel](https://td-worldmodel.github.io/TD-WorldModel/) 对同一 clip 内 prediction/target cosine 拟合随时间间隔指数衰减的标签；其项目页同时报告 expert-only 模型的 off-policy 退化，以及加入扰动重跑数据后的改进。它目前是单一 Scene 环境的 preliminary 工作，不能当作我们的 Cube 受控结果。

对固定间隔的 pair，TDWM 的目标相同，D8 的目标取决于共同动作真实未来测试的区分率。更关键的是精确循环：若同一静态图像在相隔 4 步再次出现，当前 encoder 对两帧给相同码，某一个预测与它们的 cosine 必相同。时间尺度 2 的两个 TDWM 标签却分别为 `1` 与 `exp(-2)=.1353`。仅这两个项的平均平方误差就至少为 `(1-exp(-2))²/4=.1869`。D8 若两个初始条件与侧信息真的相同，共同动作实验的 Delta 为 0，不强迫把循环中的同一状态分开。

这说明时间位置并非普遍可实现的状态语义；它不否认 TDWM 在无精确循环数据上的经验效果，也不证明替换时间 target 本身就有创新。D8 可辩护的新内容必须同时保留**真实受控实验、实际有限噪声通道、可验证的误差预算**；去掉这些后若只是一个依未来图像差别变化的 contrastive margin，应降级为 metric-learning 对照。

### 三个最近的主文献与必须排除的等价类

1. [Le Cam meets LeCun (2014)](https://arxiv.org/abs/1402.4884)：feature quality 用 statistical deficiency/决策能力刻画是既有思想。D8 仅把局部必要条件变成行动条件生成通道的训练约束；不能宣称发明“表示应保留可区分性”。
2. [DrLIM, Hadsell–Chopra–LeCun (2006)](https://yann.lecun.org/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf)：pairwise margin/contrastive embedding 是旧方法。所有 deterministic negative 都得到同一个 Delta 时，D8 的零损失可行域等于固定距离 margin；这种版本明确不计新。
3. [Temporal Discriminative World Models (2026)](https://td-worldmodel.github.io/TD-WorldModel/)：同 clip sample-level temporal discrimination 是直接且必须实现的对照。D8 与其在循环及同间隔不同未来难度上的最优约束不同，但数据/计算代价也不同。

还需直接比较 [JEPA-Bisim (2026, v2 2026-10-01)](https://arxiv.org/abs/2602.18639) 与 [DBC (2020)](https://arxiv.org/abs/2006.10742)。JEPA-Bisim 已把 reward-free 行为等价用于 JEPA 规划；因此“同动作未来一致就合并”不是创新。D8 当前不是对学得 latent 转移距离做递归等式拟合，而是让**实际外部观测实验**给 noisy channel 一个一侧最低区分约束；是否构成足够本质的方法贡献尚未证实。若把完整未来测试集作为 PSR/Blackwell 充分性目标，它又落入已有理论；不能靠改名避开。

### 重要反例与任务边界

1. **额外侧信息可以完全绕过这个界。** 若 latent 两状态完全相同，但 decoder 从 bypass 得到状态标签，未来仍可预测准确。`as_jepa_lewm/research.py` 的 forecast 分支确从 noisy initial state 开始，但 renderer 另拿 observed anchor，FM 训练还提供 noised target；不能把那张训练计算图错误写成 `state -> future`。必须针对整个生成/规划条件通道固定相同侧信息，或对联合 `(state,side-info)` 通道证明。不同初态通常伴随不同确定性 anchor，联合通道 TV 可能已经等于 1，使 state-only 容量条件完全无约束。若条件固定无法做到，D8 不适用；此时附加 margin 只能叫保留偏好，不能叫修复 TF 的已证瓶颈。
2. D8 的 bound 是必要而非充分。有限测试可能看不到物块/姿态；所有 pair 的分离条件也不保证好的 Euclidean goal metric。Cube 的手会使很多测试有更大 Delta，不能凭这个 loss 宣称自然获得正确 object priority。
3. 操作仅在低噪声预测器 clean state 上进行时，扩大 encoder 的噪声分离不一定改善部署；必须先确认被约束通道确实参与必要学习/预测路径。已经完成的 clean-state 控制负证据必须一起报告。
4. 纯外观变化也可造成真实未来测试差。固定 patch tests 会保留部分背景，除非实验设计消掉它；这不是 reward-free 自动语义选择器。
5. 若许多真实区别都趋近 Delta=1 且 epsilon趋近0，Gaussian 均值距离需求发散。有限容量模型不可能满足任意精度的所有观察区别；D8 只有明确容错/查询范围后才是可实现要求。

## 本轮判断与下一步最小证伪条件

| 候选 | 已经成立 | 尚未成立 | 当前判断 |
|---|---|---|---|
| D7 | 相对阶偏好与 Euclidean/Gramian 可以给不同选择；不把纯导航连通域全合并 | 实际 stride 下存在可估计的“物块更深”观测层；这层与 strict pose/hold 一致；预测头能改善控制 | 条件性保留，先检验响应层级；不得作优越性/普适语义声称 |
| D8 | noisy-channel 预测误差必要界；精确构造可达；与固定margin/时间标签有不同解 | 必须先证明实际计算图适用：不同 observed anchor 可使约束完全无效；真实 pair 有违反且可低成本确认；收益超过同数据普通contrastive/TDWM | 保留完整规则假设，固定margin退化版本排除，尚不启动长训练 |

对 TwoRoom，D7 的最小正面要求是普通 agent 位置仍为可控输出且可从图像稳定估计；D8 要证明有真实受控未来差异的 pair 在实际必要噪声通道上确被合并。对 Cube，D7 要在当前 action/observation stride 下找到跨初态稳定的更深物块/姿态输出；D8 要显示被恢复的是物块/接触可预测性，而不只是手/背景区分。物理坐标可用于这些诊断，不能偷偷进入学习目标。

本轮没有新增 benchmark 数字，没有 GPU 作业，没有 commit，也未占用或修改旧训练状态。
