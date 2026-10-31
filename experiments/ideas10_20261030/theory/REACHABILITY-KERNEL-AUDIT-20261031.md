该方向属于已有功能距离/未来状态密度家族；所提比值具有精确坐标不变性，但不能作为一般控制距离，当前不实施、不计新机制。最新冻结Cube候选池中，TF预测与真实latent成本排序约.95，真实latent oracle却只有14/18%而physical oracle约40%，支持“预测准确与目标几何是两层问题”；这是该候选池内的诊断，不是全局控制上限。

固定观测表征下，记录轨迹的逐样本目标可为
\[
L=-\log p_{\theta,H}(u_{t+H}\mid u_t),\quad
K_H(s,g)=\frac{p_H(g\mid s)p_H(s\mid g)}{p_H(s\mid s)p_H(g\mid g)}.
\]
若密度存在、分母为正，且T为同一光滑双射，则\(p'_H(Tg\mid Ts)=p_H(g\mid s)/|\det J_T(g)|\)，四项Jacobian精确抵消。故同一个真实核不能靠坐标收缩改变K。但**密度学习本身**的latent NLL没有此不变性：若同时学习T而不计Jacobian，收缩仍能降低NLL；比值不防编码塌缩，也不保证有限模型拟合的变换一致性。

可理解“快/慢”偏好的特例是\(p_H(g\mid s)=\mathcal N(g;As+b,\Sigma)\)：直接展开得到
\[
-\log K_H(s,g)=(g-s)^\top\operatorname{sym}(\Sigma^{-1}A)(g-s).
\]
A=I时为行为扩散协方差的Mahalanobis距离：充分探索的快机械臂可能被下调；它不需要物体标签。但A=0时K恒等于1；A=-I时非同点成本为负。更直观地，二状态转移矩阵\(P=[[.1,.9],[.9,.1]]\)给K(0,1)=81。即使正确学到密度，该表达也不保证非负性、可分辨性或三角不等式。

可逆、PSD半群核的限制下，归一化后的K是归一化热核的平方；适当半时间展开使其对应未来分布特征的余弦平方。这与[diffusion maps/diffusion distance](https://www.math.ucdavis.edu/~strohmer/courses/270/diffusion_maps.pdf)密切相关；一般diffusion distance比较完整未来分布，而非这里四个点密度，不能直接等同。

限制是实质性的：该核描述**记录行为策略**的H步分布，不是最短路径、最优控制或H内首次到达；策略窄支持可把易移动机械臂误判为难移动。不可逆成功若\(p_H(s\mid g)=0\)，对称化会给易达目标无穷成本。H太短看不到物体变化，太长可能混合并丢区分；静态/奇异轨迹未必存在全维密度，自转移密度可能缺数据。单帧部分观测及隐藏的专家目标又可使条件核混合不同历史；没有机制自动识别benchmark所要求的物体语义。

最接近直接系统先例是[MBOLD](https://arxiv.org/abs/2012.15373)：离线无标签交互学习功能距离，给视觉world-model规划评分。[Eysenbach等C-Learning](https://arxiv.org/abs/2011.08909)明确未来状态密度和on/off-policy区别；[Naderian等C-Learning](https://arxiv.org/abs/2011.12363)学习指定horizon内累计可达性。[TDM](https://arxiv.org/abs/1802.09081)、[dynamical distance learning](https://arxiv.org/abs/1907.08225)、[ChronoSRL](https://arxiv.org/abs/2609.36238)分别覆盖时间条件价值、到达步数和到达时间分布/停留时间几何。我们的旧TEMPORAL-TARGET-FEASIBILITY已指出时间目标的全局不一致与转为图几何的先例；S3已覆盖路径事件可达性。四密度归一式未在旧实现中出现，但这不足以建立新的方法贡献。

引用核验记录：本轮曾误以为2011.12363是错误C-Learning引用；直接打开primary后确认两篇同名论文均真实，该误判已撤回。旧记录只补全标题并区分两篇理论；实验事实不改。本审计仅推导/读文献，无实现、拟合或GPU工作。
