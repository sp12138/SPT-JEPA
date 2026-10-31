# Patch-token 600：完整短门结果

40/40实际exit0；8组均fresh600，全部原生完整状态、分层梯度、初始/冻结张量及原病例协议核验通过。8槽空闲，三个dispatcher已关闭并HOLD；monitor末条记录全部完成后已退出。未续训。

|任务|旧flat1024 seed1/2|token identity seed1/2|token orthogonal seed1/2|identity−orthogonal pp [条件95%CI]|
|---|---:|---:|---:|---:|
|tworoom|16/24|28/26|26/22|+3.0 [-12.0,+18.0]|
|cube|0/4|8/10|6/2|+5.0 [+0.0,+11.0]|

|对比|TwoRoom均值差pp [条件95%CI]|Cube均值差pp [条件95%CI]|
|---|---:|---:|
|identity−旧flat|+7.0 [-2.0,+16.0]|+7.0 [+2.0,+13.0]|
|orthogonal−旧flat|+4.0 [-10.0,+18.0]|+2.0 [-2.0,+6.0]|

|任务/seed|臂|clean/copy|clean/zero|teacher/相邻copy|错初态/clean|别段动作/clean|
|---|---|---:|---:|---:|---:|---:|
|tworoom/1|flat_reference|0.806|1.381|0.841|1.296|1.04141|
|tworoom/1|identity|0.859|1.470|0.936|1.227|1.01595|
|tworoom/1|orthogonal|0.773|1.323|0.941|1.216|1.00006|
|tworoom/2|flat_reference|0.828|1.397|0.837|1.278|1.04473|
|tworoom/2|identity|0.869|1.466|0.942|1.227|1.01465|
|tworoom/2|orthogonal|0.772|1.302|0.940|1.222|1.00033|
|cube/1|flat_reference|0.774|1.090|0.831|1.626|1.15630|
|cube/1|identity|0.812|1.144|0.923|1.395|1.03267|
|cube/1|orthogonal|0.796|1.121|0.932|1.362|1.00192|
|cube/2|flat_reference|0.765|1.088|0.828|1.602|1.15090|
|cube/2|identity|0.804|1.143|0.922|1.366|1.03286|
|cube/2|orthogonal|0.794|1.129|0.932|1.332|1.00102|

空间identity在两任务两seed的动作交换效应均强于orthogonal，但均弱于旧flat（TR1.041/1.045、Cube1.156/1.151）。identity的clean/copy也没有超过旧flat。identity规划点估计相对两种对照均更高，但共享50病例的配对区间仍需完整保留，不能据此宣称稳定优势。orthogonal平均clean MSE更低；这与任务排序相反。现有结果支持继续区分动作条件信号与总体重构误差，尚未证实移除global输出秩限制带来大幅控制收益。

所有token的clean误差仍高于固定训练中心zero；这不等于输出为常数或没有信息。summary保留7horizon的预测能量、对齐项和恒等式MSE=zero+energy−alignment，并记录真实初态/动作干预。teacher接收真实前缀，不能与纯因果rollout混同。

两token臂共享全部542初始参数张量及共同buffers；466非predictor初值与旧flat一致。24份200/400/600 portable均有限，8份600完整checkpoint逐hash绑定，模型/AdamW/scheduler原生恢复通过，RNG保存字段齐备，sampler cursor76800及后续索引核验通过。22个predictor分层梯度组/每run全部有限非零；单参数零值另列，不能由模块非零推断每个参数均非零。

原raw局部观测和动作缓存跨三臂逐元素一致；数据/校准/噪声/rate/renderer/goal L2/原CEM300×30,h5,history3、50rawsteps不变。token predictor523,536参数 vs flat11,389,184；TF predictor forward约3.17倍MAC，orth另有双FP64变基，故旧flat只作容量/计算不同的历史上下文。峰值显存112.86–112.97GiB。

600仍在1249步warmup内。建议有限2000成熟检验有研究价值：identity相对旧flat四个task/seed的任务点估计均提高，Cube条件区间为正；但动作效应和平均MSE不优于flat，尚不足以把收益归因为更好的条件动力学。此为预算建议，但本报告不授权2000、不声称方法创新，也不把短门当成熟效果。置信区间仅条件于这两个已训练seed，未覆盖seed不确定性、多重比较或Cube渲染重复差异。

Sampler证据的精确范围：保存cursor=76800；用相同cfg与该cursor两次构造原生WindowSampler，next16索引一致。未将这项检查表述为与未中断连续训练的逐样本轨迹复现；本次也未执行RNG状态restore来证明下一训练噪声逐值相同。
