# 最后一轮：同步驱动下的初态记忆几何

2026-10-31。先写 `FOURTH_DESIGN_FIRST.md`，后检索主文献，再执行 `fourth_cpu.py`。本轮保留 **一个有具体操作和可检验性质的条件性候选 D9**；不为补数加入第二项。它不是已证实适合 Cube 的新算法。尤其其最便宜的零动作版本，已被旧冻结 bank 的直接检查否决为当前的可靠解法。

本轮仅 CPU 合成验证和旧 Cube bank 只读分析。没有 GPU、训练、环境 step、新数据采集或主代码改动；没有 commit。物理坐标只参与诊断，不是候选学习目标。

## 1. 操作：保留相同未来动作无法洗掉的初态差异

TF 在 Cube 的自有成本预测排序很准，但这个成本高度跟随机械臂。D9 提出一个新的、明确有边界的归纳偏好：**可被随后同一控制序列迅速覆盖的状态差异，不应和持续存在的配置差异占据同样的目标距离。**

它与前几条机制不同：

- D7 问“动作多久后才首次影响输出”；D9 问“相同动作执行后，输出还记得多少初态”。两个坐标都可以一阶响应，但一个迅速忘记初态、另一个长期保存。
- D4 把“能用可逆动作互相到达”的状态合并；D9 只有在相同动作同步驱动后差异消失才趋向合并。自由平移导航不会因连通而塌缩。
- D8 检验 noisy code 是否承载未来实验所需的区分率；D9 不使用噪声容量下界，而是指定一种希望世界模型保留的几何。两者可能共享采集基础设施，但不是两个数据集算两项创新。
- 不是让模型输出自己的“未来距离”再追随它。训练标签必须来自真实成对执行的后果，避免自有成本已经错、却越拟合越自洽。

设 `F_A(s)` 是从状态/允许历史 s 执行同一 open-loop 序列 A 后的实际状态，`psi(x)` 是统一固定的图像测量，例如 bounded RGB/Haar 网格，均不含物体或成功标签。在确定性环境中定义

`d_H²(s,t)=E_{A~nu} ||psi(F_A(s))-psi(F_A(t))||²`。

A 的分布 nu 对两初态完全相同，与它们的差异、测试目标、最佳动作无关；H 是预先声明的全任务公共交互长度。`A` 可以来自普通记录动作的固定 bank，但必须在两个初态上**真正执行同一序列**。从两条行为轨迹分别取它们自己的动作，不满足这个定义。

最小训练规则是 paired-sample 距离蒸馏：

`L_obs = ( ||h(E(s))-h(E(t))||² - T_A(s,t) )²`，

其中 `T_A` 是实际共同 A 后的观测差目标；`h` 是固定预算内的目标读出，全动力学 state E 仍保留。还需要把同样的真实目标直接约束 forecast，而不只提高 observed probe：

`L_forecast = ( ||h(F_B(E(s0)))-h(E(t))||² - T_A(s_B,t) )²`，

其中 `s_B` 是普通已执行 B 的实际终点；标签来自真实 `B+A` 分支与真实从 t 执行 A 的分支。s0、t、B、A 和所有图像是一个 paired interaction unit。没有把实际 B 标成最优动作，没有把某个事后状态标成奖励 1，也不训练 goal-to-action policy。

原行动条件预测目标保留；encoder geometry 与 forecast geometry 的新增项要独立消融。固定总模型容量、CEM 候选数与步数；部署用 h 的距离，不增加 rollout 次数。额外交互、重复执行、头参数和训练 FLOPs 都必须计入，并供给 matched baseline。只在冻结表示上增加 h 而无 forecast 约束，应如实称为 readout/controller extension。

这条 forecast 项是必要的实验拆分，而非新定理：最近 24 个 relative-query gates 中，Cube observed 查询和 block-height probe 改善，但 forecast 接近 zero baseline、初态交换效应约 `1e-5`，规划未改善。不能再用 observed 读出改善代理世界模型传递能力。

## 2. 可证明的性质，以及它与常见平均未来距离的不同解

### 2.1 几何可实现性

把状态映成函数 `Phi_H(s): A -> psi(F_A(s))`，则

`d_H(s,t)=||Phi_H(s)-Phi_H(t)||_{L2(nu)}`。

因此 d_H 是伪度量，对应 kernel `k_H(s,t)=E_A <psi(F_A(s)),psi(F_A(t))>` 为 PSD。固定 M 条动作和 q 维观测的精确 embedding 最多需要 Mq 维。若压进 64 维，可能有不可消除的低秩近似误差；不能因为理论存在 Hilbert embedding 就保证当前预算足够。

这些是已有 kernel/系统理论的直接性质。候选方法层面的主张是把**同步输入下的初态遗忘**作为 goal geometry，并从真实 paired 后果同时约束 observed 和 forecast；不是声称提出了新的 PSD 理论。

### 2.2 同步耦合不能替换成混合后的未来边缘分布

执行了一个 64 状态的环形导航例：`x_next=x+u mod64`，u 在全部 64 个动作上均匀。观测为圆上的 `(cos,sin)`。

任意初态的下一状态边缘分布都相同，因此比较两个边缘分布的距离为 0。但两个初态相差 7 时，同一个 u 保持它们的相对位置，故

`d_sync² = 4 sin²(pi*7/64) = 0.4539790933`。

独立抽两个 u 的平均平方距离则为 2，连同一初态对自己的“独立距离”也为 2；减去两边自距离后的 reduced independent distance 又为 0。同步目标自距离为 0，同时保留不同导航位置。

这证明“先把动作积分掉，再比较未来分布”和 D9 确实不是同一算子。它**不证明与所有 action-conditioned PSR/bisimulation 方法均不同**：如果先行方法同样保留 A 的条件索引，就不能用这个例子区分。

### 2.3 一阶响应相同，但目标排序不同

构造 `r_next=alpha*r+u; b_next=b+v`。两者都是直接一阶可控，D7 的响应阶次不能区分；相同未来控制下

`d_H² = alpha^(2H) * delta_r² + delta_b²`。

目标为 `(r,b)=(0,1)`，候选 A=`(0,0)` 与 B=`(2,1)`。Euclidean 成本 `[1,4]` 选 A；alpha=.5、H=2 时 D9 成本 `[1,.25]` 选 B。若 r 是可快速覆盖的执行器姿态、b 是持久配置，规则有合理作用。

同一规则在纯导航 `x_next=x+u` 上保留完整相对位置，不需要手工指定任务类型。然而 alpha=1 的相对位置执行器也会全部保留；这正是需要实际检验而不能靠“机械臂”名称推断的地方。

### 2.4 有耦合时保留的是未来后果，不保证是当前物块目标

改为 `b_next=b+c*r`。相同输入下初态手差异会泄漏到后续 b：

`delta_b(H)=delta_b(0)+c*(1-alpha^H)/(1-alpha)*delta_r(0)`。

alpha=.5、H=4、初始 `(delta_r,delta_b)=(2,0)` 时，终点差为 `(.125,3.75c)`。c=.4 时成本为 `2.265625`，又超过物块初始误差为 1 的候选。这里 D9 正确发现手的差异有长期物理后果，但任务可能只要求“物块现在已经到位”，不要求它在随机后续动作下继续到位。**模型后果相关性与用户目标相关性仍不是同一个概念。**

## 3. 随机环境修正：不能偷偷要求共用 simulator noise

设计原稿的 PSD 表述在确定性 F 下成立。若两个实际环境的噪声独立，直接比较两个随机终点会使同一初态的自距离大于 0。不能把 simulator 的同一随机种子当作真实世界免费可用的观测能力。

可定义 `m_A(s)=E[psi(Y)|s,do(A)]`，再令

`d_H²(s,t)=E_A ||m_A(s)-m_A(t)||²`。

每个初态在同一 A 下分别独立执行两次，形成

`T_A = <psi(Y_s1)-psi(Y_t1), psi(Y_s2)-psi(Y_t2)>`。

这个单 paired-unit 目标无偏估计上述平方距离；单次值可能负，不能 clip 成 0 后仍声称无偏。脚本枚举均值差 .3、两环境各自独立正负1噪声：直接平方差期望为 2.09，paired-repeat 内积期望为 .09，恰好等于真实均值差平方；单次目标可为负。代价是 4 条执行；真实环境重置不精确会引入额外识别误差。有限观测均值可以忽略仅体现在方差中的差异；换成 characteristic kernel 特征可减少这种别名，但又直接进入已有 conditional embedding 理论，不是另一项创新。

## 4. 已执行的 Cube 最小证伪：零动作没有选择性地忘记手

读取已有不可变 bank：

`/efs/prescottsun/SPT-JEPA/runs/mechanism_cubeaudit_recovery_20261030/pool`。

核对了恰好 50 个 case、每个 39 个 candidate、每个恰好一个 `zero` family；其 `25×5` 原始动作逐元素恰等于 0。没有重新仿真。用 physical position 只诊断共同 zero action 下哪些初态差异存留，绝不把这些字段作为训练标签。

对 50 个初态的 1225 对，计算末端/初始 pair 距离 RMS 比：

| 原始步数 | effector 距离保留率 | block 距离保留率 |
|---:|---:|---:|
| 1 | .998842 | .999575 |
| 5 | .992702 | .992959 |
| 10 | .983846 | .984219 |
| 25 | **.956548** | **.959699** |

25 步的初末 pair 距离相关分别为 `.997546`、`.994878`。个体平均位移为手 `3.72 cm`、物块 `2.16 cm`；因此 zero control 并非物理完全不动，但手与物块的跨初态差异都几乎同样存留。

**这否决了“共同零动作会显著抹掉手状态、保留物块”的廉价版本。** 没有检查其他共同动作序列，因此不能推出所有 nu 都无效；也不能在这 50 个验证 case 上反复选 nu、H，找到有利结果再当成原创算法证据。任意进一步提案必须用训练交互定义一个全任务规则，再独立检验。

这项检查只涉及位置。不能凭位置比例声称已有 strict pose/24-symmetry/hold 改善，也没有把 25-step open-loop 数据当作 50-step closed-loop 结果。

## 5. 不能删除世界模型必需的状态：一个结构性限制

设目标几何把 s,t 合并，`d_goal(s,t)=0`，但某个相同动作 a 导致 `d_goal(F(s,a),F(t,a))>0`。那么不存在仅以这个 goal code 为状态的确定性 Markov predictor，能同时正确预测两者；任何有限 L 的

`d_goal(F(s,a),F(t,a)) <= L*d_goal(s,t)`

也立即失败。

Cube 中不同手位可以在下一次接触后造成不同物块位移，所以“目标不要求同一手位”不意味着“世界状态可以不含手位”。D9 必须保留完整预测 state，只让 goal readout 有不同几何；或保留有限非零权重，但这会使所需动力学 Lipschitz 常数随压缩增大。这个限制解释了为什么删手读出或直接令所有手位等价并不是可靠方法。

另一个限制是使用有限观测 psi 时，finite-H 等价一般不封闭于再执行一步。脚本的反例是两坐标交换动力学、psi只看第一坐标：状态 `(0,0)` 与 `(1,0)` 的 H=1 末端观测距离为0，H=2却为1。隐藏区别会重新出现。若 psi 给出完整 Markov 状态且所有长度H动作都已覆盖，则有更强的闭合性质；不能把这个理想条件未经验证套在固定 RGB/Haar 查询上。因此不能把一般 finite-H 伪度量的零集直接视作可独立 rollout 的 Markov abstraction。

## 6. 最近主文献：真正区别在哪里，哪些内容已经是旧的

| 主文献 | 已有内容 | 本候选的可检验区别与边界 |
|---|---|---|
| [MICo, NeurIPS 2021](https://arxiv.org/abs/2106.08229) | 用 independent transition coupling 构造可采样状态距离并塑造表示。 | 上面的环导航给同步/独立耦合不同解。D9 不使用 reward difference 和该递归算子，但“以行为距离塑造表示”显然不是新概念。 |
| [Conditional Hilbert embeddings, ICML 2009](https://icml.cc/2009/papers/538.pdf) | 把条件分布嵌入 Hilbert space，并应用于动力系统。 | D9 的 stochastic 修正版直接属于这套数学。新颖性不能放在 conditional embedding 或 PSD 性质，只能审查“用同指令初态遗忘定义 goal geometry，并同时约束 observed/forecast”的具体学习规则。 |
| [Angeli, incremental stability, 2002](https://flore.unifi.it/handle/2158/307517) | 研究轨迹相对轨迹的稳定性；same-input 初态遗忘是经典控制问题。 | D9 不发明 contraction；它尝试把这种行为作为视觉世界模型目标几何的偏好。目标相关性没有从稳定性理论自动导出。 |

还查了 [Sequential Action-Induced Representation, 2023](https://arxiv.org/abs/2309.12628)。其具体辅助目标是动作序列分布的 characteristic function，且使用当前策略、奖励序列等；因此不能仅因标题中的 sequential action 就认定与 D9 等价。D9 不模仿动作分布，动作只是相同外部试验输入。

更近的 [Dynamics-based behavioral similarity / RDS](https://www.sciencedirect.com/science/article/pii/S0893608025013498) 已在 reward-free 表示中使用多步 latent trajectory distance。出版商摘要/正文检索能确认这一层重叠，但直接全文访问受限，本轮没有核清其全部耦合与目标公式。故不能作“排除所有近邻后的已证原创”结论；需要把它列为未解决的 priority risk。

**准入判断：** D9 与 D1–D8、MICo 的所述算子有明确的行为分离；不是可直接断言完全等价的旧公式换名。但它也明显建立在 conditional predictive geometry / incremental stability 上，现有真实 Cube 证据不支持最便宜实例。当前最多保留为 *conditional mechanism hypothesis with a falsified cheap instantiation*。不以此补足“已发现十个新算法”的数量。

## 7. 这轮给下一研究决策的具体结论

1. 从“某物是 actuator”推断应该删除它不成立。应先测真正的输入条件下初态记忆能否消失；当前 zero-action Cube 检查没有这种分离。
2. 如果将来有真实训练交互支持选择性遗忘，可测试 D9 的共同动作目标与 independent-action、相同原始图像距离、observed-only 三个直接对照；同时检查 forecast 是否传递目标差异。这个条件性测试不等于现在启动训练。
3. 如果没有这种行为分离，就不应再给这条轴加任务专用规则。现有 gate 反而强调：提高 observed 信息不保证把信息传入 forecast，更不保证正确 goal geometry；这是三件要分别建立证据的事。

精确结果与来源记录见 `fourth-cpu.json`。本轮没有第二个可辩护的新机制，没有新增 benchmark 性能主张。
