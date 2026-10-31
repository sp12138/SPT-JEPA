# Cube TF：6000→12000

10/10端点exit0；primary平均变化+0pp [-8,+8]，fresh变化-1pp [-9,+7]，未见规划收益。区间为50病例配对bootstrap95%区间（10000次，同一索引同时重采样两个固定训练seed），不包含训练seed总体不确定性，未作多重校正。原primary/fresh有序病例、CEM、strict判据及train-episode动作归一化均与6000精确一致。原8e0f821训练源、0934d3c收集/归档helper及runtime全部manifest哈希通过；6000完整状态恢复、12000完整状态归档/last哈希、8000/10000/12000保留权重有限性与相同键/形状均通过，优化器/scheduler/RNG/sampler审计通过。

| 病例 | seed | 6000成功率 | 12000成功率 | Δpp |
|---|---:|---:|---:|---:|
| primary | 1 | 24% | 18% | -6 |
| primary | 2 | 14% | 20% | +6 |
| fresh | 1 | 14% | 14% | +0 |
| fresh | 2 | 18% | 16% | -2 |

同16 validation窗口的健康检查保持有限且encoder/predictor/renderer均有非零训练梯度；t=.95 forecast条件去噪MSE由0.00566/0.00719降至0.00394/0.00474（-30.3%/-34.1%），feature std从0.594/0.587变为0.800/0.783。这是带噪初态和renderer的匹配噪声诊断，不是独立clean-latent预测误差；改善没有转化为这两组规划收益。原max_steps=124910/warmup=1249保持不变，训练在12000停止，保留全部指定checkpoint；不据此自动续跑。逐case比较和全部健康指标保留EFS，summary.json含完整结果SHA。
