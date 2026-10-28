# I01：包含边界项的有限步密度候选

此实现验证一个已知的概率建模工具能否解决当前beta单位问题，不宣称新的扩散定理。依据是[Variational Diffusion Models](https://arxiv.org/abs/2107.00630)第4节式13；代码为 `as_jepa_lewm/density.py`，方法名 `vdm_nats`。

定义64个log-SNR区间，范围[-10,10]。每个Gaussian反向转移的KL为半个SNR差乘denoising平方误差；训练均匀抽样区间并乘64。条件renderer仍用原FM的bridge坐标，数据估计为bridge+t*velocity。额外加入干净端的256级离散Gaussian重建NLL，以及噪声端对单位Gaussian的KL。三项均按渲染值平均，状态KL使用同一normalizer，beta=1具有一致的nats单位。

这是指定有限反向Gaussian链的负ELBO，即NLL上界，绝不是精确图像NLL。采样使用同一64步反向链，加离散Gaussian端点；不能把它的采样成本和原50步Euler假定相同。以前的 `snr_weighted` 仍保留作没有边界项的surrogate控制。

检查包括：转移权重等于精确Normal KL、离散端点概率归一化及梯度有限、已知Gaussian例子完整bound接近真实entropy而删边界失败、观测单位变换下的中间项/离散概率一致、反向采样shape/数值及8bit支持。完整world-model收益需相同数据和预算的GPU实验；CPU数学性质不是任务成功。

I04另增加 `anchor_canonical_color`：由anchor通道极值决定共享的点式颜色变换，当前帧仅经此颜色变换再编码；不能读取anchor空间排列。CPU检查完整空间置换anchor不会改变编码输入。旧framewise分位数方法 `canonical_color` 保留作为不同的颜色标准化对照；两者都不等于原AS-JEPA的anchor-isolated encoder。

NativeS2增加可选 `state_width=32` 做128维容量检查，默认仍4×16；训练记录microbatch可变rate、rate_share及模块梯度范数。正式active_dims和逐维rate由固定heldout的collapse audit给出，不能把train microbatch计数冒充正式容量结论。

原规格全部价格坍塌的固定fallback在NativeS2上通过公开beta_scale参数实现，由Native任务的step给定前5%进度；常规运行默认1，不改变任何旧snapshot。没有新增Trainer或checkpoint实现。
