# 活跃维度不能单独充当早期非坍塌证明

一个精确反例：令encoder对任何图像都输出mu=0，状态只有sigma噪声，因此不含输入信息。若prior仍为初始identity carry(m=s_prev)，独立后验每个转移、每维的期望均值失配KL为1/2nat。W=8时，日志使用的variable-rate每维期望为7/2=3.5nat，高于0.1阈值，64维仍可全部被记作active。

即使采用rho=.8的Markov后验，扣除协方差KL常数后，期望仍是7*(1-rho)^2/2=.14nat，也高于0.1阈值。正码长可以来自prior对噪声的失配，并不等于编码了图像信息。

[MonteCarlo数值核验](ACTIVE-DIMENSION-COUNTEREXAMPLE.json)与解析式一致。此反例不是对实际600步模型权重的追溯判定。原五种子模型的非坍塌结论还有独立含噪后验重建证据，未被该反例替代。

实际早期门必须联合有限数值/梯度通路、posterior对anchor重建优势、置零/打乱状态响应和学习趋势。日志active_dims=64本身不足以放行长实验；早期P0低也不足以判死一个尚未学会任务的模型。
