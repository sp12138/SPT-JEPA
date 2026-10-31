# 动态输出基原型：CPU审查完成，尚未提交或启动GPU

`spatial_readout_mode=none/static/state`，默认none。仅local、identity basis、chartnone允许；原loss/noise/calibration、原始状态与目标、planner和Native Trainer不变。STATIC与STATE均增加5376个trainable参数；STATIC有1024个state-input权重元素不参与数据梯度。矩阵head前缀`dynamic_readout_module`，每个预测状态额外401408 MACs，不代表实测吞吐。

v2全部5项CPU契约通过（76.72秒wall）：两任务seed1的553个原始tensor、原生loss/metrics、全部原参数梯度及RNG严格相同；seed2原tensor/RNG相同；旧0c非零rawchart默认loss/grad/RNG相同。现有factory严格加载、新head roundtrip、样本与时间局部性、有限32步构造rollout均通过。v1的double-head位置buffer dtype错误已保留并修正，生产FP32行为未因此改变。

机制反例已执行：同一初态的第二步预测离开原始rank1 span，正交残差norm=.63738。更强的共享head构造中，旧16维增量空间经STATE得到跨当前状态的1024维增量族，STATIC仍为16维。后者使用人为构造状态，不证明真实图像流形或同一物理初态有1024维可达空间；固定当前状态的瞬时action rank仍受旧输出限制。

零初始化有真实启动延迟：fresh旧输出层也为0，因此新head初始梯度全0。一次原生梯度SGD更新后，STATIC/STATE输出weight最大梯度为7.36e−6/1.18e−4，输入层仍0；第二次更新后输入weight为1.35e−12/5.99e−10。这是有限启动证据，梯度很小，不能据此断言真实训练已健康。未手动打开门或增加loss。

旧6000缓存的δRMS为.760–.904；从FP32 p,u重算p−u的相对舍入误差仅1.44–1.66e−8。该检测不恢复原carry加法已舍入掉的sub-ULP δ；exact carry初始化时新分支也为0。实际CUDA/BF16、长轨迹、600步学习和规划表现均未测试。

最近先例是[Dynamic Filter Networks](https://arxiv.org/abs/1605.09673)与[HyperNetworks](https://arxiv.org/abs/1609.09106)，本原型是针对已测固定输出span的机制干预，不作新算法优先权声明。详细运算/证明见DESIGN.md，全部CPU sourcehash与结果保存在summary.json。只改SpatialCarrier可选分支，model.py/carrier.py/robust.py和所有冻结源未改。
