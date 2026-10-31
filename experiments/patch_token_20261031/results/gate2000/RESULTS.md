# Patch-token2000：完整成熟门结果

72/72实际exit0；8组从原600原生恢复，1200/2000共16份完整归档、全部最终fullstate/22分层梯度组/health通过。精确owned dispatcher/monitor关闭并HOLD；无自动>2000。

|任务|臂|primary600→2000，seed1/2|fresh2000，seed1/2|
|---|---|---:|---:|
|tworoom|flat_reference|16/24 → 60/56|60/56|
|tworoom|identity|28/26 → 46/34|36/32|
|tworoom|orthogonal|26/22 → 58/56|54/52|
|cube|flat_reference|0/4 → 24/20|16/10|
|cube|identity|8/10 → 8/8|12/8|
|cube|orthogonal|6/2 → 2/12|4/8|

|病例/任务|对比|均值差pp [条件95%CI]|
|---|---|---:|
|primary/tworoom|identity−orthogonal|-17.0 [-29.0,-5.0]|
|primary/tworoom|identity−flat_reference|-18.0 [-28.0,-9.0]|
|primary/tworoom|orthogonal−flat_reference|-1.0 [-12.0,+11.0]|
|primary/cube|identity−orthogonal|+1.0 [-6.0,+7.0]|
|primary/cube|identity−flat_reference|-14.0 [-24.0,-5.0]|
|primary/cube|orthogonal−flat_reference|-15.0 [-24.0,-6.0]|
|fresh/tworoom|identity−orthogonal|-19.0 [-31.0,-7.0]|
|fresh/tworoom|identity−flat_reference|-24.0 [-36.0,-12.0]|
|fresh/tworoom|orthogonal−flat_reference|-5.0 [-18.0,+7.0]|
|fresh/cube|identity−orthogonal|+4.0 [-2.0,+11.0]|
|fresh/cube|identity−flat_reference|-3.0 [-10.0,+4.0]|
|fresh/cube|orthogonal−flat_reference|-7.0 [-14.0,+0.0]|

|任务/seed|臂|clean/copy600→2000|clean/zero600→2000|别段动作/clean600→2000|teacher/邻帧copy2000|
|---|---|---:|---:|---:|---:|
|tworoom/1|flat_reference|0.806→0.239|1.381→0.409|1.0414→3.7753|0.183|
|tworoom/1|identity|0.859→0.568|1.470→0.972|1.0160→1.3480|0.425|
|tworoom/1|orthogonal|0.773→0.612|1.323→1.048|1.0001→1.0690|0.793|
|tworoom/2|flat_reference|0.828→0.246|1.397→0.416|1.0447→3.7545|0.198|
|tworoom/2|identity|0.869→0.590|1.466→0.996|1.0147→1.2985|0.449|
|tworoom/2|orthogonal|0.772→0.618|1.302→1.044|1.0003→1.0635|0.786|
|cube/1|flat_reference|0.774→0.361|1.090→0.509|1.1563→2.6531|0.435|
|cube/1|identity|0.812→0.622|1.144→0.876|1.0327→1.1806|0.658|
|cube/1|orthogonal|0.796→0.703|1.121→0.990|1.0019→1.0478|0.822|
|cube/2|flat_reference|0.765→0.354|1.088→0.503|1.1509→2.7479|0.423|
|cube/2|identity|0.804→0.619|1.143→0.880|1.0329→1.1679|0.643|
|cube/2|orthogonal|0.794→0.696|1.129→0.989|1.0010→1.0525|0.814|

全部raw observed/actions跨两token、旧flat以及600/2000缓存逐元素一致；病例/solver/归一化/hash匹配。两token共享初值542参数及共同buffers，466非predictor初值与flat相同，冻结210/211张量不变。准备16次实际模型/AdamW/scheduler恢复及归档审计保留，RNG仅保存/复制值核验，sampler next32与start0序列参考相同；不声称不中断GPU轨迹逐值复现。

所有7horizon的raw zero/copy/teacher/clean、预测能量与对齐项保留在summary，核MSE=zero+energy−alignment。zero是固定训练中心，无验证拟合；平均误差接近zero并不证明常数或无信息，teacher额外接收真实前缀。动作干预与任务收益分别报告。

两token参数/初始化/损失一致，orth仅在predictor内部固定变基并逆回raw goal坐标。旧flat predictor11,389,184参数，token523,536；token训练predictor forward约3.17倍MAC，orth另计两次FP64变基，不能作匹配计算或单独归因于rank的对照。固定普通动作作为输入，没有专家最优动作标签，损失保持sample-level。

CI对共同50病例进行10000次配对重采样并在两固定seed间共享病例索引；它不覆盖训练seed总体不确定性、多重比较或Cube原协议重复波动。fresh同属验证集、复用既有生成规则，不能称独立测试集。报告不授权继续训练或宣称原创。

[逐步预测误差图](cold-horizon.png) · [PDF](cold-horizon.pdf) · [机制解释](INTERPRETATION.md)。图由完整保存数组生成；不等同原生history3规划曲线。
