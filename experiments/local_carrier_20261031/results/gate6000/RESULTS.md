# Local/dense6000：Cube小幅候选收益，未超过强基线

全部64端点exit0；Cube local在fresh较dense有小幅改善，但primary差距较小，对native/TF没有稳定胜出证据。TwoRoom local明显好于dense，仍低于64维F与native。两任务、两seed和raw/calibrated native均完整保留，不按任务选择较好native变体。

| 任务 | 方法 | primary seed1/2 | fresh seed1/2 |
|---|---|---:|---:|
| tworoom | local | 60/60% | 72/60% |
| tworoom | dense | 12/14% | 20/18% |
| tworoom | F-identity | 80/86% | 74/80% |
| tworoom | F-unconstrained | 90/94% | 82/86% |
| tworoom | F-bounded | 92/88% | 80/94% |
| tworoom | TF | 2/4% | 4/2% |
| tworoom | native-raw | 80/72% | 76/70% |
| tworoom | native-cal | 82/78% | 84/88% |
| cube | local | 16/22% | 24/20% |
| cube | dense | 12/18% | 10/14% |
| cube | F-identity | 10/12% | 12/10% |
| cube | TF | 24/14% | 14/18% |
| cube | native-raw | 24/22% | 12/14% |
| cube | native-cal | 16/18% | 20/16% |

| 任务/病例 | local减对照 | seed1/2 Δpp | 平均 [配对case95%CI] |
|---|---|---:|---:|
| tworoom/primary | dense | +48/+46 | +47 [+35,+59] |
| tworoom/primary | F-identity | -20/-26 | -23 [-34,-13] |
| tworoom/primary | TF | +58/+56 | +57 [+45,+69] |
| tworoom/primary | native-raw | -20/-12 | -16 [-28,-4] |
| tworoom/primary | native-cal | -22/-18 | -20 [-31,-9] |
| tworoom/primary | same2000 | +0/+4 | +2 [-3,+7] |
| tworoom/fresh | dense | +52/+42 | +47 [+34,+60] |
| tworoom/fresh | F-identity | -2/-20 | -11 [-23,+0] |
| tworoom/fresh | TF | +68/+58 | +63 [+50,+75] |
| tworoom/fresh | native-raw | -4/-10 | -7 [-20,+6] |
| tworoom/fresh | native-cal | -12/-28 | -20 [-31,-10] |
| tworoom/fresh | same2000 | +12/+4 | +8 [+0,+16] |
| cube/primary | dense | +4/+4 | +4 [-3,+12] |
| cube/primary | F-identity | +6/+10 | +8 [-1,+18] |
| cube/primary | TF | -8/+8 | +0 [-7,+7] |
| cube/primary | native-raw | -8/+0 | -4 [-11,+2] |
| cube/primary | native-cal | +0/+4 | +2 [-5.025,+10] |
| cube/primary | same2000 | -8/+2 | -3 [-11,+5] |
| cube/fresh | dense | +14/+6 | +10 [+2,+19] |
| cube/fresh | F-identity | +12/+10 | +11 [+4,+20] |
| cube/fresh | TF | +10/+2 | +6 [-1,+14] |
| cube/fresh | native-raw | +12/+6 | +9 [+0,+19] |
| cube/fresh | native-cal | +4/+4 | +4 [-7,+15] |
| cube/fresh | same2000 | +8/+10 | +9 [+0,+18.025] |

区间按50相同病例联合重采样两个固定训练seed（10000次，seed1042），不是训练seed总体区间，未多重校正。primary与fresh分别比较；fresh仍来自同validation pool，不能视为独立test。Cube free/bounded未训练6000，不以2000替代。

| 任务/方法 | seed | clean/initial-copy 2000→6000 | wrong-order/clean 6000 | renderer t=.95 forecast MSE 2000→6000 |
|---|---:|---:|---:|---:|
| tworoom/local | 1 | 0.239→0.086 | 7.51 | 0.00696→0.00103 |
| tworoom/dense | 1 | 0.370→0.161 | 5.04 | 0.00604→0.00171 |
| tworoom/local | 2 | 0.246→0.095 | 6.80 | 0.00749→0.00208 |
| tworoom/dense | 2 | 0.367→0.190 | 4.35 | 0.00733→0.00228 |
| cube/local | 1 | 0.361→0.227 | 3.20 | 0.02120→0.00595 |
| cube/dense | 1 | 0.499→0.362 | 2.24 | 0.02052→0.00711 |
| cube/local | 2 | 0.354→0.221 | 3.27 | 0.02061→0.00605 |
| cube/dense | 2 | 0.482→0.341 | 2.36 | 0.02059→0.00656 |

潜变量误差、动作干预和renderer去噪衡量不同机制：latent readout是64固定窗口的clean rollout；health是16窗口带噪初态的条件renderer诊断。它们的改善不等于更准确的物理状态或更高任务成功率；不跨不同表示直接比较MSE。冻结观察编码未改变，未重复训练/执行observed probe。

4000/6000各8个完整状态审计通过，原32fad0c训练源、da26748 helper及runtime全文件哈希通过；本收集重新核验所有完整归档/portable SHA。每次审计确认210冻结参数/测量buffer不变、原global-RMS校准不变、optimizer/scheduler/RNG/sampler恢复完整。原有50病例顺序、manifest、CEM/budget、strict判据及train动作归一化与2000和全部6000对照一致。两阶段边界2000→4000→6000、总scheduler124910不变。

local/dense约55.9M参数、1024维状态；F/TF约51.1M、64维；native约18.04M、192维。更新/数据/规划预算匹配不代表容量和算力匹配，local/dense固定投影与计算也有区别。native-cal额外固定128训练clips做无梯度BN统计，是单独报告的统一变体。当前结果不支持大幅超过基线或自动续训；保留所有checkpoint/原报告。完整逐case数据在EFS，summary.json含其SHA。
