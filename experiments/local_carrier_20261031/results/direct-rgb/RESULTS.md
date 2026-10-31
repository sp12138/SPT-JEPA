直接RGB局部投影没有显示Cube信息读出优势，因此本结果不支持以“保留更多信息”为理由启动4×600训练。原始局部性仍是可检验的参数化假说，但这里未测试其可学习性或规划性能。此审计只运行一次CPU，exit0，约23秒、峰值0.863GiB；没有encoder/model/GPU调用。

| 表征 | 通道 | 读出 | train/val位置cm | train/val姿态R² | val角度 | ≤3cm / ≤15° / joint |
|---|---|---|---:|---:|---:|---:|
| rawRGB | clean | linear | 1.66/15.62 | 0.905/-6.514 | 23.42° | 4.46/31.70/1.34% |
| rawRGB | clean | rbf | 5.32/7.92 | 0.423/-0.063 | 21.99° | 12.95/34.82/4.02% |
| rawRGB | sigma0.1 | linear | 0.91/17.02 | 0.970/-6.882 | 23.37° | 1.12/32.37/0.22% |
| rawRGB | sigma0.1 | rbf | 5.56/7.97 | 0.442/-0.060 | 22.40° | 12.50/34.15/4.46% |
| oldViT_seed1 | clean | linear | 1.51/12.11 | 0.904/-4.863 | 22.32° | 7.59/35.04/2.68% |
| oldViT_seed1 | clean | rbf | 4.83/6.65 | 0.383/-0.075 | 22.15° | 21.21/35.04/7.37% |
| oldViT_seed1 | sigma0.1 | linear | 0.90/15.08 | 0.974/-6.097 | 21.37° | 2.46/37.05/0.45% |
| oldViT_seed1 | sigma0.1 | rbf | 5.02/6.93 | 0.411/-0.076 | 22.18° | 17.41/36.83/4.69% |
| oldViT_seed2 | clean | linear | 1.39/13.36 | 0.919/-5.730 | 22.62° | 3.57/33.26/1.12% |
| oldViT_seed2 | clean | rbf | 4.97/6.74 | 0.377/-0.061 | 21.59° | 22.54/39.29/8.04% |
| oldViT_seed2 | sigma0.1 | linear | 0.91/14.66 | 0.972/-5.777 | 21.94° | 0.67/35.27/0.22% |
| oldViT_seed2 | sigma0.1 | rbf | 5.15/6.98 | 0.410/-0.063 | 21.41° | 20.76/40.40/6.70% |

| 表征 | train/val RMS | train/val平均坐标std | train/val top16能量% | 解析初始carry rate train/val |
|---|---:|---:|---:|---:|
| rawRGB | 0.957/0.953 | 0.697/0.680 | 16.21/16.96 | 4.241/4.216 |
| oldViT_seed1 | 0.964/0.957 | 0.723/0.707 | 16.32/17.29 | 3.611/3.580 |
| oldViT_seed2 | 0.960/0.949 | 0.718/0.698 | 17.48/18.25 | 3.679/3.593 |

rawRGB只报告一次：两旧ViT seed共享原128训练clips与64验证clips，分别比较同一raw特征。共享192×16投影与旧checkpoint buffer逐元素一致；1536帧的normalized224→恢复字节→area64/round逐字节匹配native render，块内HWC顺序通过单点合同。原128校准初帧给出globalRMS=0.6202318，无floor；个别坐标std=0没有造成除零，因为统一使用全局标量。旧clean读出复现，最大标量误差5.4e−12。

RBF的raw位置7.92cm比旧6.65/6.74cm差，姿态仍约22°；σ=.1后为7.97cm/22.40°，并非噪声把整个通道抹掉。linear训练拟合好而validation差，不能把训练分数或其加噪后改善当成信息增益，也不能据读出失败证明信息不存在。rate是原目标在初始carry下的解析期望，不是训练实测。固定ridge=.001、train-median RBF与原Cube24码本，不调validation；物理标签仅拟合诊断读出。TwoRoom缺少对应128/64缓存，未另选cohort、未调用encoder、未外推结论。未获准或启动后续训练。

冻结helper：531a69e6ae36b26111f2964daa1d542f92e6b7fc；EFS完整证据：/efs/prescottsun/SPT-JEPA/reports/direct-rgb-information-20261031。
