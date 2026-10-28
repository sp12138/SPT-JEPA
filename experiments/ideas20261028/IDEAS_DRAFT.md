# 从SPT-JEPA结果出发的10个候选方法（预备草案）

这份草案记录机制假设与可证伪实验，不代表方法已经有效或已证明新颖。原始全量规格的缺项见任务审计。用户已授权原实验补齐和十候选验证，三台机器无新一轮时间上限；此文保留最初机制草案，实际实现与状态见同目录README。

设计依据是50k单种子正面结果、同轨迹预算改善、双种子anchor分工、Native反馈对照、dot泄漏与精度控制。既有方法会明确标注，候选价值取决于新增机制和匹配验证。

| ID | 候选 | 筛选优先级 |
| --- | --- | ---: |
| I01 | 把生成收益和状态价格统一到 nats | 1 |
| I02 | 按解码器几何分配状态噪声 | 5 |
| I03 | 为缓慢动力学设计有时间相关性的后验噪声 | 6 |
| I04 | 让anchor定义外观坐标系，而不是让state反复学外观不变性 | 7 |
| I05 | 直接训练prior状态也能渲染的条件分布 | 2 |
| I06 | 用局部梯度约束越过坍塌势垒 | 8 |
| I07 | 按可预测性而不是像素面积分配生成梯度 | 3 |
| I08 | Native把语义记忆闭环和图像生成闭环分开 | 4 |
| I09 | 按真实动作干预压缩状态 | 9 |
| I10 | 把latent距离改成局部可控能量 | 1 |

## I01 把生成收益和状态价格统一到 nats

**结果依据：** 当前可用设置 β=.01，而 β≥1 多数失败；flow-MSE并不自动是图像NLL。原故事也把校准β列为开放问题。

**机制：** 用正确的密度/去噪积分估计状态带来的条件码长收益，再与Gaussian KL比较；先验证likelihood差分估计的残差和边界项，不能把普通CFM差值直接叫log-ratio。

**预测：** 相同物理数据在RGB单位、噪声时刻分布改变后，不必重新大范围搜索β；动态信息仍保留，随机dot RGB泄漏下降。

**关键对照：** 相同网络/数据/步数，普通FM+β网格对照；单独控制时间权重、边界修正、额外训练时间。

**否定条件：** 换像素单位仍要大幅改β，或只是数值缩放而没有改善噪声状态/生成/规划；密度差估计不能在可精确求解的Gaussian例子复原真实值。

**先做的检查：** 线性Gaussian路径上验证最优uniform-time FM风险=π√v/2，而熵=.5log(2πev)，二者不成固定比例。这个toy不是当前logit-normal时间采样器的精确损失。

**创新边界：** 校准β的动机已在原计划中出现；候选贡献是可检查的估计器与world-model码长选择流程，不把已知I-MMSE或CFM理论宣称为新定理。

相关一手来源：[When Can Conditional Flow Matching Replace Pointwise Negative Log-Likelihood?](https://arxiv.org/abs/2608.28010), [Mutual Information and Minimum Mean-square Error in Gaussian Channels](https://arxiv.org/abs/cs/0412108)

## I02 按解码器几何分配状态噪声

**结果依据：** 高BF16方差可以没有可用信息；各状态方向对图像/动作的影响很不均衡。

**机制：** 从训练数据估计全局decoder pullback度量F，令q/p共享Q∝(F+εI)^−1；用完整Gaussian KL计费，令噪声在可见图像变化上接近等强。Q只用训练段估计并保存，不从未来图像决定部署噪声。

**预测：** 任意线性重参数化不再改变有效噪声和surprise；同样状态码长下小agent信息更早变得可用。

**关键对照：** 固定等方差噪声、仅对角Q、随机旋转Q，以及同等计算的FP32对照；报告entry先验是否同时变换。

**否定条件：** 收益只来自增加实际信道容量；同码长下生成或规划没有改善，或Q退化为数值不稳定的近奇异矩阵。

**先做的检查：** 完整Gaussian KL在均值和q/p协方差一起做可逆线性变换后不变；固定N(0,I)不一起变换则不应声称全目标不变。

**创新边界：** 不是普通whitening或把MSE改成Gaussian NLL；候选是把解码可见性、共享噪声与真实KL统一。

相关一手来源：[VJEPA: Variational Joint Embedding Predictive Architectures as Probabilistic World Models](https://arxiv.org/abs/2601.14354)

## I03 为缓慢动力学设计有时间相关性的后验噪声

**结果依据：** 独立帧噪声在prior中形成额外预测负担，当前prior需要处理噪声与真实运动；长rollout仍有误差累积。

**机制：** 定义q(s_t|s_{t−1},x_t,x_{t−1})=N(μ_t+ρ(s_{t−1}−μ_{t−1}),σ²(1−ρ²)I)，保持每帧边际噪声σ不变；用真实Markov chain KL训练并匹配部署采样。

**预测：** 相同每帧噪声和总KL下，慢变化维度不再反复支付独立抖动的代价，长horizon误差下降。

**关键对照：** ρ=0、错误地仍按独立后验计费、相同总rate的基线；同时检查agent灵敏性是否因过平滑而损失。

**否定条件：** 收益完全是噪声变小；真实joint KL没有下降或动作响应受抑制。

**先做的检查：** joint Gaussian KL必须与正确的条件KL之和相等；原来的per-frame独立公式通常不适用。旧纯度证明需要重推，不能直接沿用。

**创新边界：** 视频相关噪声已有工作；候选区别在状态后验的联合码长与prior闭环，而非仅重用renderer噪声。

相关一手来源：[Go-with-the-Flow: Motion-Controllable Video Diffusion Models Using Real-Time Warped Noise](https://arxiv.org/abs/2501.08331)

## I04 让anchor定义外观坐标系，而不是让state反复学外观不变性

**结果依据：** anchor配色分工两种子复现，但state仍有残余palette R²；直接相减anchor也可能抹掉动力学几何。

**机制：** 由anchor估计受约束的点式颜色canonicalizer，再编码canonicalized当前帧；只允许共享颜色变换，不允许空间attention把anchor中的物块位置传进state。目标图像仍由anchor恢复原配色。

**预测：** 在颜色变化下state的背景泄漏下降，几何信息保持；跨palette目标规划更稳。

**关键对照：** 无canonicalizer、仅全局颜色标准化、无约束anchor cross-attention、直接减anchor的负对照；同一物理窗口改变palette。

**否定条件：** 去除颜色同时损失位置/接触信息，或只在合成RGB affine而非新的外观变化上有效。

**先做的检查：** 线性style子空间投影保留动力学需要明确的子空间条件；另造反例展示盲目减anchor会丢绝对位置。

**创新边界：** 不是把已成功的anchor swap换个名字；新增的实质是限制encoder的外观规范化路径，并验证它不泄漏几何。

## I05 直接训练prior状态也能渲染的条件分布

**结果依据：** 50k后后验物块重建约5px，但预测状态的h13物块约19px；当前renderer主要读后验state训练。

**机制：** 固定比例选择后验state或多步prior-state作为同一个FM renderer的条件，预测路径从真实过去出发，目标仍是真实未来图像；让生成损失直接给prior和过去encoder提供任务信号。

**预测：** 相同训练步数下posterior/forecast重建差缩小，噪声rollout与动作反事实改善。

**关键对照：** 后验条件基线、单步/多步prior条件、相同总FM调用数；分别报告后验重建是否退化。

**否定条件：** renderer为兼容差prior而学会忽略state，或prior视觉改善来自牺牲goal embedding和P0。

**先做的检查：** 在可微toy里确认forecast rendering对prior的梯度通路存在；真正效果只能由完整world-model对照决定。

**创新边界：** 候选重点是用同一生成条件分布把码长训练和部署状态接通；不能只叫scheduled sampling而不检查信息路径。

## I06 用局部梯度约束越过坍塌势垒

**结果依据：** 默认初始化时renderer到state的初始梯度为0；固定小批量诊断中某些配置的rate梯度远大于FM梯度，暖启动/冻结本身没有打通规划。

**机制：** 对encoder的rate更新施加局部约束：不允许它在一阶近似上破坏当前可见重建信号；冲突方向投影，renderer尚无state梯度时不收缩encoder。保留原始priced loss用于诊断。

**预测：** 高β更少落入空状态，且不需要L0预训练；有效信息和生成共同改善。

**关键对照：** 原始更新、固定warmup/冻结、PCGrad通用实现、等计算基线；记录投影比例和原始目标变化。

**否定条件：** 只是人为强留无用方差，或只有牺牲原始目标才变好且生成/规划无收益。

**先做的检查：** 投影可保证一阶FM不增，但不能保证原始FM+βKL单调下降；必须明确它是优化约束的改变。

**创新边界：** gradient surgery本身已有工作；候选创新必须体现在对码长坍塌势垒的特定约束与可预测的机制结果。

相关一手来源：[Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782), [Lagging Inference Networks and Posterior Collapse in Variational Autoencoders](https://arxiv.org/abs/1901.05534)

## I07 按可预测性而不是像素面积分配生成梯度

**结果依据：** 小agent在25k后验重建很弱，50k才改善；flow在dot控制中保留大量无关RGB。

**机制：** 让仅看过去/动作/anchor的网络估计每patch的不可预测残差方差v，用r²/(2v)+.5log v的异方差FM回归，避免随机dot消耗大量编码梯度，增加可预测的小运动信号的相对权重。

**预测：** agent信息更早出现，dot RGB leakage下降，同时动态生成和P0保住；不是简单降低所有细节质量。

**关键对照：** 普通FM、固定空间权重、motion magnitude权重、去掉log v的负对照；v不得读取目标帧或目标state。

**否定条件：** 权重只是把难区域藏起来、视觉对象消失，或随机dot反而因变化大而获得更高权重。

**先做的检查：** 正确log方差项的最优v是条件均方残差；去掉log项会通过v→∞作弊。加权FM也不自动等于端点NLL。

**创新边界：** 异方差回归是已知工具；候选是用因果可预测性处理小agent与随机细节之间的梯度竞争。

相关一手来源：[When Can Conditional Flow Matching Replace Pointwise Negative Log-Likelihood?](https://arxiv.org/abs/2608.28010)

## I08 Native把语义记忆闭环和图像生成闭环分开

**结果依据：** Native S1同检查点h7真实历史agent/block16.7/6.6px，自由rollout145.4/58.1px；裁剪生成历史反而更差。

**机制：** 传播专门的语义记忆并由它生成图像，生成图片不再作为唯一动力学历史重新编码；收到真实观测时再做状态校正。优先从已有S1表征进行匹配finetune，另设同预训练预算的对照。

**预测：** 一阶预测不变或接近，h7/h13漂移下降；改进集中在反馈路径，而不是单步能力。

**关键对照：** 原S1、裁剪history、真实history oracle、latent carry；相同预训练与finetune预算，记录是否冻结encoder。

**否定条件：** latent自身同样不稳定，或图像质量变好但动作/物块接触更差。

**先做的检查：** 在线性toy中区分A与A·E·D的反馈谱半径；当E·D=I时没有结构性优势，不能保证所有latent carry都更稳。

**创新边界：** latent world model本身不新；候选要用当前Native受控结果定位接口，并验证保留语义记忆能解决已观测的反馈放大。

相关一手来源：[VJEPA: Variational Joint Embedding Predictive Architectures as Probabilistic World Models](https://arxiv.org/abs/2601.14354)

## I09 按真实动作干预压缩状态

**结果依据：** 低latent误差和高rank不能保证动作跟随；50k反事实识别整体88.3%，物块68.8%，仍有可改进空间。

**机制：** 从训练段的真实模拟器状态收集多动作fork，训练encoded真实future在给定共同过去时识别候选动作，辅以原生成/码长目标；约束的是观测到的动作后果，不能让分类器直接读prior动作输入取巧。

**预测：** 更少编码只与外观/随机dot有关的信息，接触相关block动作识别上升，规划不下降。

**关键对照：** 相同fork数据但无干预项、打乱动作标签、只给prior输出的作弊负对照；训练/确认测试fork严格分开。

**否定条件：** 识别来自branch编号/随机种子泄漏，或只学agent而block和控制没有提高。

**先做的检查：** 共同nuisance和独立标签下不应能靠背景/ID识别动作；专门审计新数据中的branch-key泄漏。

**创新边界：** 动作相关性在原故事中已有开放问题；候选是可验证的同状态fork信息约束与抗泄漏控制，不把反事实评估本身叫新方法。

相关一手来源：[Counterfactual Action Evaluation, Observation Bottlenecks, and Representation Geometry in Joint-Embedding Predictive World Models](https://arxiv.org/abs/2610.02860), [MC-JEPA: A Joint-Embedding Predictive Architecture for Self-Supervised Learning of Motion and Content Features](https://arxiv.org/abs/2307.12698)

## I10 把latent距离改成局部可控能量

**结果依据：** 50k生成/反事实已好但P0仍有波动；GoalMSE依赖任意embedding坐标的尺度，未必衡量达到目标的动作代价。

**机制：** 冻结世界模型，用J=∂z_H/∂u估计局部可控Gramian，G=JJᵀ+αC_train；规划成本dᵀG⁻¹d。C只用训练features估计，G可每次replan估计一次，以低秩更新控制开销。

**预测：** 在相同CEM搜索预算下P0提升，对可逆线性state重参数化更稳；生成结果保持原值，因为模型冻结。

**关键对照：** 原GoalMSE、仅C_train的Mahalanobis、随机低秩G、相同墙钟时间/函数调用预算；α只在训练段或新增validation选择。

**否定条件：** 只靠增加CEM时间获益，局部J不可靠，或改变坐标后目标选择依然严重变化。

**先做的检查：** 若C和J按同一可逆坐标变换，dᵀ(JJᵀ+αC)⁻¹d严格不变；ridge I不相应变换时不成立。

**创新边界：** Gramian/控制能量是经典概念；候选贡献是由JEPA几何诊断得到的、无需重训且有坐标不变检验的规划接口。它是新cost协议，不能冒充原P0 GoalMSE。

相关一手来源：[Counterfactual Action Evaluation, Observation Bottlenecks, and Representation Geometry in Joint-Embedding Predictive World Models](https://arxiv.org/abs/2610.02860), [VJEPA: Variational Joint Embedding Predictive Architectures as Probabilistic World Models](https://arxiv.org/abs/2601.14354)

## 验证顺序与判定

先做数学、数据泄漏和梯度通路检查；通过后再运行world-model实验。数学性质成立不等于PushT有效。I10可先冻结模型做规划对照；I01先验证密度/损失单位；I05/I07分别针对已测到的prior差距与agent/dot竞争。

方法开发只读现有数据和权重，使用新的提交、冻结source/config与独立输出目录。Native工作继续复用现有Trainer/检查点实现。所有GPU调度前重新检查进程。

训练方法先保留原21.58M/64维工作点与同数据顺序的基线。8k只能筛掉有多项一致证据的完全坍塌，不能因agent较差就提前判负；原50k运行在24k时P0仍只有24%。有希望的配置应比较相同50k步和实际GPU小时，再补最有价值的训练种子。

筛选使用训练段/预先声明的validation；最终确认使用预先封存的新case。旧256窗口与P042/43/44可保留作诊断，不能反复择优后当作独立确认。与原协议不同的目标、样本量、额外数据或finetune必须单独标记。

主要结果分别报告：噪声状态probe、mean/sample生成、动作反事实、P0/新cost规划、吞吐/GPU小时与定性图。单个指标改善不算解决任务。

基线热步耗时约.325秒，50k训练本体约4.54GPU小时。10个同成本冷启动约45.4GPU小时，尚未包含新方法开销、对照和复现；这是调度粗估而非承诺。2026-10-28用户已取消新一轮时间上限；原截止只适用于历史12小时结果。
