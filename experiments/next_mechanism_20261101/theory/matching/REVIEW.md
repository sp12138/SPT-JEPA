# 同初态 fork matching 直接规划：精确条件与旧实现边界

2026-11-01。仅源码/数学与指定近邻审查；未读取当前pilot结果、未运行模型或GPU。

**判定：有条件成立，值得作为旧I09分数的部署控制；不是新loss。** 区分“从候选动作中识别结果的来源”与普通行为逆策略，确实使所需的加性自由度与同goal规划兼容。但这里只保证支持内、给定同一query的似然排序，不能推出Cube目标语义、容差成功率或未见候选外推。

## Bayes推导

令X为模型可见初始观测/历史，H为实际隐藏初态，B=(A1,…,AK)为完整open-loop动作序列bank。所有Aj在看到本轮结果前选定；每次从同一H执行对应Aj并得到Yj。独立均匀取J，再仅把(X,B,YJ)交给分类器。需要以下额外条件之一：X充分确定所需H，或 **B与H在给定X后独立**；每条分支噪声边缘必须与部署所定义的随机干预相同。此时

\[
p_i(y\mid x)=\int P(y\mid H=h,do(A_i))P(dh\mid X=x),\qquad
P(J=i\mid x,B,y)=\frac{p_i(y\mid x)}{\sum_kp_k(y\mid x)}.
\]

只给同一隐藏初态而没有上述独立性，不够：若整个bank由隐藏状态生成，则积分用的是 `P(H|X,B)`，一个不读bank的head未必能表示正确后验；“动作在结果前决定”本身不排除这类选择。

逐候选head `sθ(x,Ai,y)`，fork loss为
`−K⁻¹ Σj log softmax_i sθ(x,Ai,Yj)[j]`。
它等于先均匀抽J再计算CE的期望，是一个fork内部的proper分类log score，无跨minibatch项。模型可实现且达到population最优时，在正密度、可比较的支持上

\[
s^*(x,A_i,y)-s^*(x,A_k,y)=\log p_i(y\mid x)-\log p_k(y\mid x).
\]

于是固定(x,g)时最大化s与最大化目标密度相同。任何加性 `c(x,g)` 都精确抵消。只有在相关动作通过训练bank充分共同出现、对该query可比较时，才能将不同bank的相对尺度接通；不相连的动作组可各自加任意常数，bank-independent head本身不能消除这个不可识别性。

这**不需要减去动作proposal q(A|x)**：给定已抽好的bank，J的先验是1/K，不管bank原先是否来自非均匀q。普通行为逆策略却满足 `P(A|x,y)∝P(y|x,A)q(A|x)`，其跨动作排序通常要校正q。若分支抽样J不均匀，或按观察到的后果筛掉分支，则上述无校正结论失效；需使用实际设计概率/选择机制。

## 必须保留的数学边界

- **open-loop vs feedback。** 若把执行时根据途中观察才决定的动作序列当成预先给定Aj，仍会重现未来动作泄露随机响应。预选反馈policy可以研究，但干预对象应是policy而非事后实现的动作序列。
- **相关随机性。** 若head只看单一YJ，不看其他终点或seed，跨分支噪声可以相关；相同期望CE仍只依赖各分支边缘，主要改变估计方差。共同随机数不自动成为真实世界可获取能力。分支编号决定的背景/seed伪特征、恢复状态不全、或噪声边缘随标签改变，均可使分类学到错误线索。
- **重复终点。** 多个动作有相同结果分布时，最优后验应平分或按实际似然分配，不能要求每个终点都把自己的index分成100%。确定性M个完全碰撞分支贡献的条件熵可为log M，这不是训练坏掉。相同候选动作必须给完全相同分数。
- **有限样本。** 连续高维图像中，经验分类可能完全可分、logit不断增大；低训练CE不等于校准了人口似然。CEM会查询新候选，population支持内结论不是有限模型OOD保证。若编码器丢信息，实际只保证其可见特征条件下的分类关系，不恢复被删掉的物块变量。
- **density不是容差质量。** 同一精确g的density排序有意义且其坐标Jacobian对所有动作相同，但与 `P(Y∈G)` 不等价。未知 `c(x,y)` 又禁止直接把 `exp(s)` 在多个y上积分当概率质量。目标物块/手位语义仍未由匹配loss解决。
- **不拼接未经标定的cost。** `c(x,g)`在同次固定query的候选排序中安全；若把不同预测起态、不同goal或不同时域的分数相加，c可通过候选改变总排序。当前主张只覆盖直接 `score(x0,goal,整段A)`；不自动覆盖多段energy相加或新的可组合world model。

## 源码与公开近邻：准确的贡献说法

`as_jepa_lewm/research.py:107–118` 的旧I09已经把每个真实fork endpoint与全部候选actions组合，以 `inverse_head(concat(origin,endpoint,action))` 逐候选打分并作相同CE，随后加到原flow+rate。head为MLP且不读bank；`as_jepa_tools/fork_evaluate.py` 已用此分数直接做held-out候选匹配，并检查候选置换与相同action零差异。训练源码仍使用80维flattened action输入，不能未经接口审计就当作当前Cube horizon可用的现成权重。现有搜索未发现该head接入原任务CEM的代码。

因此待检验改变是：**让已训练匹配准则直接决定规划候选，而不是只作表示辅助目标再交给Euclidean cost。** 不应把它包装成新监督公式，也不能用旧fork分类准确率代替新规划实测。

- [CFM，Learning Predictive Representations for Deformable Objects Using Contrastive Estimation，§III](https://sim2real.github.io/assets/papers/2020/yan.pdf)：已有行动条件前向预测、未来观测对比与MPC；其log-similarity就是负latent L2，部署同样按L2排序。因此“对比训练的评分用于规划”也不是首次。它对比未来观测，而此处在同初态实执行fork内对比候选动作；不能宣称完整算法完全等价。
- [CRL，NeurIPS 2022](https://arxiv.org/abs/2206.07568)：已有从未来分布/密度比得到goal-conditioned value及动作选择的联系。当前是固定时域、预先随机执行的候选bank分类，不是其完整actor/critic与未来占据学习流程；Bayes解释不是新定理。
- [OnlineWM，§3.2–3.3](https://arxiv.org/html/2609.23753v1)：已有同初态真实动作分叉和反事实后果训练；其具体目标是负分支扰动的flow训练与EMA混合正负重建，并非这里的候选CE或直接匹配energy规划。

建议用语：**有Bayes排序依据的I09直接energy规划对照**。这是可行、可检验的接口改变；是否明显胜过Euclidean、是否可泛化到新bank和新goal，留给预先固定的实验，不由以上推导承诺。
