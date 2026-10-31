# Predictor输出子空间：CPU机制审计

限制真实存在，并构成Cube当前1024维预测误差的重要部分；它不能独立解释规划成败。全部20个旧checkpoint、18份现有缓存核验通过；两任务各使用相同64个validation窗口×8帧，保留全部两seed。没有新编码、网络forward、物理标签拟合、GPU或训练。

冻结32fad/da/8e的模型使用carry，native pred_proj为Identity，最终输出是LayerNorm(192)→Linear(192,D)，之后无非线性；selected Robust/spatial/carrier类无predict覆盖。设W、b是输出参数，γ、β是LayerNorm参数，H=I−11ᵀ/192，则V=W diag(γ)H、c=Wβ+b。每次增量属于c+range(V)，k步预测属于u₀+k c+range(V)，有效秩≤191，比宽松span(W,b)≤193更紧。实测全部12个local/basis的秩W、[W,b]、V分别192、193、191。FP32缓存离精确仿射空间max残差仅2.19e−6–6.25e−6；误差正交分解相对偏差<2.9e−9。

当前输出层固定时，真实终点的不可达oracle为 ‖(I−P_V)(u_k−u₀−k c)‖²/D；one-step替换为当前真实u_(k−1)和c。下表均为seed1/seed2；比例由总能量相除，不平均不稳定的逐窗口比率。

| 任务 / checkpoint | Clean MSE | Oracle MSE | Oracle / clean误差 | Oracle / 真实变化 | Oracle / one-step误差 |
|---|---:|---:|---:|---:|---:|
| tworoom local2000 | 0.38500 / 0.39942 | 0.05732 / 0.06757 | 14.89% / 16.92% | 3.56% / 4.17% | 22.17% / 23.63% |
| tworoom local6000 | 0.13781 / 0.15378 | 0.01953 / 0.01761 | 14.17% / 11.45% | 1.21% / 1.09% | 23.37% / 22.88% |
| tworoom basis2000 | 0.36340 / 0.32679 | 0.10163 / 0.09496 | 27.97% / 29.06% | 6.31% / 5.86% | 40.07% / 38.40% |
| cube local2000 | 0.46818 / 0.45579 | 0.21498 / 0.20593 | 45.92% / 45.18% | 16.57% / 15.99% | 49.53% / 49.49% |
| cube local6000 | 0.29457 / 0.28465 | 0.15033 / 0.14273 | 51.03% / 50.14% | 11.59% / 11.09% | 54.21% / 53.51% |
| cube basis2000 | 0.42080 / 0.41893 | 0.23447 / 0.22922 | 55.72% / 54.72% | 18.08% / 17.80% | 59.78% / 60.72% |

Cube local6000约一半clean误差处在当前子空间外，但占真实变化仅11.59/11.09%；不能以1024−191维声称丢失81%的任务信息。TwoRoom6000对应真实变化仅1.21/1.09%，第7步oracle只占预测误差8.81/6.79%；其主要剩余误差仍在可输出子空间内。

实证反例：local从2000→6000秩完全不变，TwoRoom oracle由.0573/.0676降到.0195/.0176，Cube由.2150/.2059降到.1503/.1427，说明子空间会学习对齐。basis2000降低总MSE，却提高oracle：更好的子空间内拟合可能掩盖更差的输出方向覆盖；但这仍不证明basis的任务下降由此造成。

| 64维参考6000 | 有效秩seed1/2 | Oracle / clean误差seed1/2 |
|---|---:|---:|
| tworoom F64 | 64 / 64 | 1.68e-29 / 2.36e-29 |
| tworoom TF64 | 64 / 64 | 无既存cache / 无既存cache |
| cube F64 | 64 / 64 | 1.60e-29 / 2.02e-29 |
| cube TF64 | 64 / 64 | 3.48e-28 / 1.45e-28 |

64维参考没有这一固定线性输出秩障碍，但表征、训练自由度和原始MSE尺度不同，不能跨家族直接比较MSE。TwoRoom TF本身规划弱，满秩显然不是充分条件。

固定初态、目标g和horizon时，native欧氏goal cost可分为可输出方向的代价，加上 ‖(I−P_V)(u₀+k c−g)‖²。后者与候选动作无关，这些目标方向无法经模型cost指导动作选择；但它不直接证明排序错误，也可能主要对应任务无关变化。没有可复用的local1024固定物理probe系数，本次未新增拟合或声称姿态/接触分量丢失。

边界：这是**当前checkpoint**的必要可达下界，允许任意隐藏向量，忽略真实动作、attention与LayerNorm半径的额外限制；不是充分动力学模型，也不是跨训练所有W的架构不可解定理。local2000/6000观测逐元素相同；basis与固定Q旋转max误差≤2.39e−7。独立CPU toy验证低秩但oracle为零、正交目标有正下界及Pythagorean分解。没有实施新预测器或作创新/规划因果声明。

源码定位：冻结local32fad和basis da的as_jepa_lewm/model.py:81、84、120；native stable_worldmodel/wm/lewm/lewm.py:21、43；module.py:139、154、173、285。Dropout只在输出前，原audit eval关闭。

Helper `d1ab309c629ed5828092f6a563a4adb7c5a66ee8`；计划B/runs/predictor_span_20261031/plan.json；完整结果B/reports/predictor-span-20261031/results.json（SHA `ec813331000470ef3dd2099d5856abd59f3eb65f531d55b3173d1dcc06e03ddc`）。exit0，wall11.86秒，矩阵分析5.50秒，CUDA未初始化。summary.json保留全部seed、horizon与结果hash，完整输入/源hash留在EFS。
