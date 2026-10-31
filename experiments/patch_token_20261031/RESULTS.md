# Shared patch-token predictor：已核算的机制对照，尚未训练

独立CPU原型可行，有一个明确的判别问题：**保留同一local1024观测、raw目标度量和sample-level TF损失，取消所有样本共享的191维输出仿射空间，是否改善动作条件多步预测与任务表现？** 它不是新idea，也未接入production。CPU2线程一次审计0.92秒、CUDA未初始化，没有读取数据/物理标签、编码器调用、优化器更新或GPU任务。结果见 [cpu-contracts.json](cpu-contracts.json)，执行前源码记录见 [source-record.json](source-record.json)，独立原型为 [prototype.py](prototype.py)。完整小报告同存EFS `reports/patch-token-design-20261031`。

**结构与可证明的改变。** 单一配置为64个16D空间格、hidden64、6层、4heads×16、MLP256，共享16D输出。保留原 `[1,7,1024]` 位置参数；原1024D action encoder完整输出经每帧一次 `Linear1024→64`，广播到所有空间格，AdaLN也只计算一次/帧。原raw carry不变；head和条件gate按native惯例零初始化，不要求首步所有深层梯度非零。两token臂所有参数初始化逐元素相同，外部Torch/NumPy RNG不变。

原模型增量属于 `c+range(W diag(γ)H192)`，秩最多191；所有k步终点属于 `u0+k c+range(V)`。此前真实Cube local6000的空间外误差占clean MSE **51.03/50.14%**，但只占真实变化 **11.59/11.09%**；这既不是81%信息丢失，也不是规划失败的因果证明。见 [原空间审计](../predictor_span_20261031/results/RESULTS.md)。

新输出为 `δ=concat_s W16×64 LN(h_s)+b`。每个token的输出线性空间可以满16维；独立hidden输入处的局部Jacobian实测rank16，因此64块输出阶段的潜在秩为1024。**这只取消固定全局191维天花板，不证明训练网络实际达到1024秩或固定初态下动作可控制1024个方向**；真实动作、共享网络及观测流形仍有限制。与STATE readout不同，STATE每个固定状态仍只旋转/变形原191维增量空间，依靠后续状态变化改变方向；token输出没有这个共同全局子空间限制。既有STATE2000没有一致任务收益，不能据此推断更大表达能力必有益。

**唯一token对照：信息与raw metric不变，预测器的局部结构改变。** 固定正交Q使预测增量为 `T(Q u,E(a)) Qᵀ`（行向量记法），输出回raw再加carry；两臂action embedding相同。Q采用旧signed/permuted Hadamard规则，QᵀQ在FP64精确为I；合成目标距离差≤2.22e−16。它只在predictor内变基，不改变encode、renderer、rate接口或raw goal L2，也不重复宣称旧whole-model basis控制必优。固定混合会打散空间/通道结构；它保信息，但不保有限网络的优化和泛化。位置参数仍可学习slot特异性，所谓共享不是严格平移等变。

**参数少，计算更贵；没有compute-matched声明。** 实际构造并逐参数计数，Linear MAC通过真实CPU forward hook核对。1 MAC表示乘加一次，约为2 FLOPs；以下不含encoder、保留的action encoder、DiT、反向、softmax/归一化/激活及访存。

| 配置 | predictor参数 | TF一步前向GMAC，batch128 | CEM一次300候选rollout GMAC，H1 / H3 |
|---|---:|---:|---:|
| 原native flat192 | 11,389,184 | 51.164 | 51.296 / 85.654 |
| token hidden64 | 523,536 | 162.200 | 138.387 / 271.540 |
| 同token + 正交变基 | 523,536 | 171.595 | 147.825 / 287.269 |

保留的action encoder另有TwoRoom **4,240,494**、Cube **4,302,474**参数，三种配置相同。正交矩阵为1,048,576个固定元素；**每次predict前后各一次1024² FP64乘法已计入**，TF多9.395GMAC，H1/H3多9.437/15.729GMAC。其MAC仅多5.79–6.82%，不表示FP64 wall time也只增加这个比例。

精确公式（B=batch、T=帧数）：native `B(11,354,112T+12,288T²)`；token `B(19,218,432T+3,145,728T²)`；正交再加 `2BT·1024²`。训练TF为one-step长度7加forecast长度1..7，共8次predict，所以token总前向MAC为原3.170倍。未执行宽度/结构网格；旧未归档的“约3.32倍”估算由这里的明确结构及计数取代。

**真实mask/action/rollout合同。** SDPA mask允许 `key_frame≤query_frame`，同帧所有64格互见；不能用flatten之后的普通三角causal mask。测试特意打开合成head和gate，避免零初始化造成虚假的无泄漏结果：未来帧/动作扰动对过去输出差严格0、未来梯度严格0；末slot扰动可影响同帧首slot，当前动作确有非零效应；独立prefix差1.19e−7；多余future action padding被显式拒绝。

原生 `LeWM.rollout` 的缓存合同实跑H1/H3（无encoder调用），逐项核对执行过的H−1动作及candidate前缀；更改最后候选动作不会改变前四个预测。实际规划是 **300候选×30次CEM、horizon=receding=5、block5、最多50 raw steps**，不是1280候选。policy保留3个观测帧，但predictor默认history_size=7，所以H1五次预测长度1..5，H3为3..7，而非始终截成3。无提前成功时通常两次replan；两轮/50case的predictor MAC还需乘30和50，不能把表中单次候选rollout当整个评测成本。

**是否值得600门。** 它能区分“当前全局输出空间/空间共享确实妨碍学习”与“取消限制仍不能改善条件预测和控制”，因此有信息价值。identity-token versus orthogonal-token是参数匹配的空间先验对照；两者 versus old flat同时改变参数量、隐藏结构与计算，不能把差异全归于rank。相同encoder缓存可以复用诊断，正式图像训练的原始encoder/DiT路径与数据暴露应保持。

CPU单次B1,T7 forward+backward为native .027秒、token .100秒；不是GPU时间预测。batch128的TF图若显式保留attention score，一份矩阵/层/调用合计约 **4.43GiB BF16**，还不含softmax、反向和其他激活；SDPA是否使用合适的融合kernel尚未在CUDA验证，可能决定显存和速度。原训练已有较高显存压力，因此下一步若另获授权，必须先做原batch128的实际短预检、动作梯度/完整状态及native CEM300候选显存检查，失败则停，不静默减batch/候选数。CPU证据不足以直接批准600训练，更不支持自动延长。

相关机制已有公开先例：[DINO-WM](https://proceedings.mlr.press/v267/zhou25t.html) 使用空间patch特征预测和规划；其[论文](https://openreview.net/pdf?id=D5RNACOZEI)明确采用frame-level causal attention。本原型使用既有随机ViT/local投影及当前TF目标，而非DINOv2表征；这是受既有机制启发的受控研究，不作创新或性能承诺。
