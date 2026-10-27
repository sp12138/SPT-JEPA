# SPT-JEPA：统一评估结果
核查时间：2026-10-27T20:51:30.780688+00:00

这是单次探索性 campaign 的结果。不同训练种子分别列出；规划为 3 个评估种子的均值与标准差，生成比值的 95% 区间按测试 episode 重采样。未完成项目不会算作通过。
生成主表使用均值状态推进；每 case 4 次状态采样的独立结果见 summary.json 中的 generation_sample，以及 summary.csv 的 sample_* 字段。

| 模型 | 步数 | P0 均值±SD (%) | h7 agent / copy | h7 moving block / copy | 反事实动作识别 |
| --- | ---: | ---: | ---: | ---: | ---: |
| L0-standard-cursorfix-seed0 | 25000 | 74.7±4.2 | — | — | — |
| L0-standard-seed0 | 25000 | 82.7±5.8 | — | — | — |
| L0-standard-seed1 | 25000 | 63.3±6.1 | — | — | — |
| L1-color-bn-seed0 | 25000 | — | 1.272 | 1.842 | — |
| L1-dot-bn-seed0 | 15000 | — | 1.130 | 0.563 | — |
| L1-standard-bn-dev2 | 25000 | 16.0±2.0 | 0.976 | 0.418 | 0.359 |
| L1-standard-bn-seed1 | 25000 | 24.7±4.6 | 0.989 | 0.434 | 0.371 |
| L1-standard-cursorfix-seed0 | 25000 | 11.3±8.3 | 1.053 | 0.536 | 0.336 |
| L1-standard-mlpbn-seed0 | 25000 | 33.3±8.1 | 0.972 | 0.404 | 0.359 |
| L1-standard-seed0 | 25000 | 2.0±3.5 | 1.289 | 1.652 | 0.281 |
| L2-color-calibrated1-b025-seed0 | 25000 | — | 0.986 | 0.645 | — |
| L2-color-calibrated1-b025-seed1 | 25000 | — | 0.978 | 0.701 | — |
| L2-color-noanchor-calibrated1-b025-seed0 | 25000 | — | 0.951 | 0.772 | — |
| L2-color-noanchor-calibrated1-b025-seed1 | 25000 | — | 0.983 | 1.094 | — |
| L2-dot-flow-calibrated-b001-seed0 | 15000 | — | 1.180 | 0.779 | — |
| L2-dot-mse-calibrated-b001-seed0 | 15000 | — | 0.908 | 0.472 | — |
| L2-standard-b001-seed0 | 25000 | 11.3±6.1 | 1.019 | 0.476 | 0.352 |
| L2-standard-b1-extended-dev13 | 25000 | 5.3±1.2 | 1.078 | 1.041 | 0.312 |
| L2-standard-b1-seed1-continuous | 25000 | 6.7±1.2 | 0.978 | 0.653 | 0.375 |
| L2-standard-b4-cursorfix-seed0 | 25000 | 4.7±1.2 | 1.282 | 1.755 | 0.250 |
| L2-standard-calibrated-b001-50k-seed0 | 50000 | 70.0±8.7 | 0.234 | 0.234 | 0.883 |
| L2-standard-calibrated-b001-seed0 | 25000 | 12.7±4.6 | 1.010 | 0.486 | 0.348 |
| L2-standard-calibrated-b001-seed1 | 25000 | 7.3±4.2 | 1.003 | 0.495 | 0.371 |
| L2-standard-fullfp32-b025-seed0 | 25000 | 4.7±1.2 | 1.114 | 1.227 | 0.309 |
| L2-standard-original-renderer-b001-budget | 17859 | 12.7±5.8 | 0.879 | 0.422 | 0.371 |
| L2-standard-poolfp32-b025-seed0 | 25000 | 4.7±3.1 | 1.104 | 1.229 | 0.293 |
| L2-standard-staged-b025-seed0 | 25000 | 12.0±5.3 | 1.043 | 0.504 | 0.352 |
| L2-standard-staged-b4-seed0 | 25000 | 5.3±5.8 | 0.978 | 0.814 | 0.367 |

背景颜色交换与噪声状态探针见 color-mechanism.png。完整逐配置协议与区间在 summary.json，简表在 summary.csv。

旧 L0 seed0 与旧 L1 无 BN 使用修复前的迭代器游标。主比较优先使用 cursorfix seed0 与修复后的 seed1；源版本和恢复来源在 summary.json 中明确记录。

原方案要求三训练种子、h7 agent 不超过复制基线的 0.5 倍、移动 block 优于复制，以及反事实和人工画面检查。本轮只用于关键 insight 筛选，不能把单项表征、规划、方差或低 loss 当作通过这个门槛。

LPIPS、SSIM 和 FVD 未测。权威检查点、视频、逐 case 数据与日志留在 EFS，未进入 Git。

本批28组已登记终点评估尚未完成：无
