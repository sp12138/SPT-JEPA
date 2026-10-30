# 2000步fresh确认：Cube local优势尚未稳定复现

全部12项fresh评测exit0，使用原先固定的50 fresh病例及未更新的2000步checkpoint。Cube local从primary24/20%变为fresh16/10%；fresh与dense、训练集BN校准native均无胜出证据。此前primary上的候选优势应保留为探索结果，不能外推为稳定提升。

| 任务 | 方法 | primary seed1/2 | fresh seed1/2 |
|---|---|---:|---:|
| tworoom | local | 60/56 | 60/56 |
| tworoom | dense | 10/12 | 20/12 |
| tworoom | native-BN | 82/76 | 74/90 |
| tworoom | F-identity | 72/70 | 66/56 |
| tworoom | F-unconstrained | 74/82 | 70/78 |
| tworoom | F-bounded | 76/80 | 64/90 |
| tworoom | TF | 2/6 | 4/8 |
| cube | local | 24/20 | 16/10 |
| cube | dense | 16/12 | 12/14 |
| cube | native-BN | 14/16 | 12/16 |
| cube | F-identity | 8/16 | 8/6 |
| cube | F-unconstrained | 10/10 | 12/6 |
| cube | F-bounded | 8/14 | 10/6 |
| cube | TF | 12/6 | 6/6 |

数值为成功率%。所有模型均2000更新；旧F/TF行复用已完成同fresh评测，不是重新运行。

| 任务 | fresh差值 | seed1/2 Δpp | 均值 [配对case95%区间] |
|---|---|---:|---:|
| tworoom | local−dense | +40/+44 | +42 [+29,+55] |
| tworoom | local−native-BN | -14/-34 | -24 [-36,-12] |
| tworoom | local−F-identity | -6/+0 | -3 [-16,+10] |
| tworoom | local−TF | +56/+48 | +52 [+40,+65] |
| cube | local−dense | +4/-4 | +0 [-7,+6] |
| cube | local−native-BN | +4/-6 | -1 [-7,+6] |
| cube | local−F-identity | +8/+4 | +6 [-2,+15] |
| cube | local−TF | +10/+4 | +7 [+1,+14] |

区间按50配对cases重采样10000次，同一case索引同时用于两个固定训练seed；未涵盖训练seed总体不确定性、多重比较。primary/fresh是不同病例，不能把两集合的成功率差当作配对模型改进。fresh来自相同validation pool，与primary精确pair重合0、episode重合TwoRoom34/Cube1，不是独立test。

完整核验通过：原fresh manifest哈希/有序病例、CEM配置、horizon/budget、strict判据和train-episode动作归一化，与父primary及旧F/TF各自病例集一致；checkpoint路径与父primary相同，权重SHA对local完整归档或native校准导出记录匹配，冻结源全部manifest文件通过哈希核验。CEM seed仍42，fresh manifest的policy/selection seed为2026103101，均未变化。

native-BN保留预先固定的train-only precise-BN处理：只改4个running_mean/variance buffers，其他状态bitexact；它不是literal native推理。native约18.04M参数/192维状态，local/dense约55.9M/1024维，旧F/TF约51.1M/64维，容量、冻结参数、校准暴露和计算不匹配。不能把跨家族差值单独归因于局部结构或某个loss。

本次只确认现成checkpoint，不新增训练、校准、GPU assay或自动续跑。完整逐case结果、来源与所有对比保留EFS，summary.json含其SHA；本报告不改写历史primary报告。
