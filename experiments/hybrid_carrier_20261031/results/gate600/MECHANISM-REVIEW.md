# Hybrid600：固定信息、可学习坐标与预测利用

只读审阅；未执行模型、GPU、训练或新探针。模型源为 `f2ae93d4f0da2fb32021bfade7426b11ea06602b`，见 [source record](model-source-record.json)。本次核对的本地 `hybrid_carrier.py / robust.py / train.py` 与该不可变快照逐字节相同。数值来自已全部完成的600门结果，权威半支统计为 `/efs/prescottsun/SPT-JEPA/reports/hybrid-carrier600-20261031/core-half-stats.json`；不是额外拟合或新增实验。

**最直接的结论：F观测信息得到保护，F的预测和规划排序没有因此得到保护。** TwoRoom adaptive Q仍有明显中心化变化，不能称为常量塌缩；但F预测明显退化、Q预测接近carry且对既有动作交换几乎不敏感。Cube Q幅度收缩，F预测略好，任务表现没有改善。这些证据不支持把adaptive的低信息代价或TwoRoom短门P0增长直接解释为更好的动作条件预测。

| 已有64窗口量；seed1 / seed2 | TwoRoom fixed → adaptive | Cube fixed → adaptive |
|---|---|---|
| Q中心化RMS | 1.038 / 1.027 → .734 / .726 | 1.040 / 1.064 → .376 / .358 |
| Q均值向量RMS | .331 / .338 → 3.977 / 3.814 | 本表不补推未引用量 |
| F clean MSE / F carry MSE | .675 / .662 → .953 / .950 | .559 / .564 → .536 / .535 |
| adaptive Q clean / carry | .981 / .971 | .816 / .837 |

TwoRoom adaptive的动作交换/原clean误差比在F中约 `1+3e-6`、Q中约 `1+3e-7`。这只描述该缓存干预下的误差变化，不证明数学上的动作独立。原控制器P0 fixed→adaptive：TwoRoom `14/22→26/36%`，Cube `6/6→2/4%`；共同相关动作控制器为 `14/32→30/32%` 与 `0/2→0/0%`。全两seed保留；600仍在1249步warmup内，P0和预测诊断的方向不能互相替代。

**当前损失究竟约束什么。** [Hybrid encode](../../../../as_jepa_lewm/hybrid_carrier.py#L99) 使用固定初始统计

\[
z_Q(x)=(q_\theta(x)-m_{Q,0})/s_{Q,0}.
\]

初始化之后没有重新居中、单位方差、白化或F/Q互补约束。[Robust loss](../../../../as_jepa_lewm/robust.py#L208) 在本门使用one-step rate，Q对应的可变部分为

\[
L_{Q,\mathrm{rate}}=\frac{\beta}{7\cdot12288}
\left[\frac12\mathbb E\|z_{Q,0}\|^2+
\frac{1}{2\sigma^2}\sum_{t=1}^{7}\mathbb E\|z_{Q,t}+\rho n_{Q,t-1}-\widehat z_{Q,t}\|^2\right].
\]

其中 `n` 已含σ，预测器输入是有时间相关噪声的状态历史；β=.25、σ=.1、ρ=.8。若用每坐标二阶矩和七步平均每坐标残差MSE表示，两个系数分别为 **0.00009300595238** 与 **0.06510416667**，相差700倍。后者是训练中的带噪one-step残差，不是表中的clean rollout MSE；二者系数比较也不是实际梯度大小比较。首帧项惩罚“均值平方+中心方差”，不只是均值，但它不强制每帧居中。rate的固定KL项不会因Q丢信息而消失。

把QueryPool末层weight置零即可产生观测无关Q；shared16维bias和固定64维校准均值会留下固定slot偏移，不保证恰好零向量。这说明架构允许Q删去信息。若DiT已能依靠F、anchor及带噪目标解释图像，可变rate下降未必需要保留Q信号；反过来，FM仍可能奖励Q，因此“rate必然导致当前结果”没有得到证明。损失也不要求Q含有F之外的信息：重复F可能帮助有限解码器，或提供另一组独立坐标噪声下的冗余观测，相关性本身不能证明无用。

**大梯度是坐标与优化事实，不是信息结论。** TwoRoom `s_Q≈.0003` 使raw Q参数的链式梯度带约3333倍归一化因子。seed2 step200的pool范数8025.62与renderer12.02来自 [startup](STARTUP.md)。[训练hook](../../../../as_jepa_lewm/train.py#L67) 在裁剪前记录；实际安装的Lightning `precision.py::_after_closure` 先调用该hook，再以 `clip_grad_norm_` 对同一个AdamW的所有参数组一起裁剪至1，bf16无GradScaler。因此该步共同梯度乘数至多约 `1.246e-4`；renderer等其他模块也被缩放。这不是“裁剪失效”，也不能等同于实际AdamW步长同比减小：历史一、二阶矩与独立weight decay会改变更新结果。

实际native Predictor先做 `Linear(128,192)` 混合F/Q，再进入LayerNorm及条件块。因此Q均值漂移可以改变F预测所依赖的隐藏表示；但缓存未测隐藏层方差/灵敏度，也没有隔离裁剪处理，**不能把LayerNorm或全局裁剪宣称为已证原因**。此处核的是实际优先导入的 `sources/stable-worldmodel/.../lewm/module.py`（SHA256 `d276fb896712bed23f70f25ae9774e5ce7b3ab44fbb6bf4dea61065ba1388ef9`），不是环境中的另一份副本。

目标成本是 `C128=(C_F+C_Q)/2`。Q的中心化尺度、预测误差和目标几何都可能改变候选排序；共享预测器也能让F预测变差，即使F观测逐字节相同。**共同常量平移本身会在终点距离中抵消**，不能仅由Q均值大推出它主导规划距离。若Q预测与目标都变成相同常量、F预测完全保持，删Q反而仅将F成本乘常数，不改变排序；当前不能假设这些条件成立。

已有缓存可检验的量应保持分支/时间分解：均值能量、中心能量、原始second moment；clean/zero/carry MSE以及 `MSE=E||y||²+E||p||²−2E<p,y>`；动作/初态交换的绝对误差变化与比值；原始预测与目标的Q/F终点成本分量（仅限实际已保存goal的缓存）。梯度trace若覆盖全部互不重叠参数组，可重建共同裁剪系数及Q占比；不能据此反推参数实际更新。跨支相关性和线性可读性也不能单独证明条件互信息为零。无需新增拟合即可完成这些描述性区分。

**relative-Q仅有条件成立的距离恒等式，不能当干净修复。** 若 `r_t=q_t-c`，且预测终点、目标都用同一c，则 `||r_H-r_g||²=||q_H-q_g||²`。要连预测也等价，必须使用共轭规则 `f_c(r,a)=f(r+c,a)-c`，并保留c；直接将现有framewise输入改成 `q_t-q_0` 不满足这个条件。只留下F绝对初态不能保证可恢复Q初态：F是64维压缩，没有证明它对Q或动力学充分。rolling3历史、目标和每次replan还必须共享同一原点或一致重定基准。

此外，若先减带噪初态，差分噪声方差变为 `2σ²(1−ρ^t)`；若先减clean Q再沿用原噪声，便是另一个训练通道。clean `r_0=0` 还移除了Q初态entry惩罚，但不约束Q的尺度、重复信息或未来偏移。relative表示可消掉公共bias，不能仅凭终点距离恒等式解决这些信息、噪声与上下文问题。本审阅不将其认定为新方法或已证修复，不授权实施或追加训练。
