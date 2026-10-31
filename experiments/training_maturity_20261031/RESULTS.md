# Cube 训练成熟度复核：当前不是充分收敛证据

只读已完成配置、原始 `metrics.jsonl` 和已有评测；没有模型调用、新拟合、GPU任务或续训。完整数值与输入SHA见 `reports/training-maturity-20261031/summary.json`；本目录 `collect.py` 可复现。历史“maturity reference”应理解为较长预算对照，不是收敛认证。

9000个训练episode共1,809,000行，window8、frameskip5有1,494,000个合法窗口起点。batch128的原sampler每轮取1,493,888个窗口，**一个loader epoch=11,671更新**。全局cursor继续推进，permutation cycle按natural_length划分；112个尾部起点会接到下一loader，不能称为每轮永久丢弃。23342更新是两个loader长度，比两次完整natural排列cycle少224个窗口。窗口高度重叠；窗口曝光量不是独立物理轨迹数，也不意味着半epoch只见过半数图像。

| 截止步数 | loader epoch | 完整124910日程进度 | 当前LR/峰值 |
|---:|---:|---:|---:|
| 2000 | 0.1714 | 1.601% | 99.991% |
| 6000 | 0.5141 | 4.803% | 99.636% |
| 12000 | 1.0282 | 9.607% | 98.147% |
| 124910 | 10.7026 | 100% | 0% |

warmup为1249更新；6000/12000仍接近峰值LR。数据已曝光多少、优化是否停止、是否过拟合是不同问题。短曝光和高LR不能单独证明延长必有效，却足以阻止把当前结果称为完整日程收敛结果。

## 原始训练末1/4仍在变化

预先固定区间local4500–6000、native/TF9000–12000，各按时间中点分两半，报告后半相对前半中位数变化；阶段结束重复记录按step仅保留最后一条。分别16/31个独立step日志，仍是相关的随机minibatch，**不报告IID显著性或把梯度非零等同未收敛**。下表每格seed1/2，两seed全部保留，不跨方法比较loss绝对值。

| 方法 | total loss变化 | 分量变化 | predictor梯度范数中位数（前→后） |
|---|---:|---|---|
| local6000 | −5.38% / −4.39% | flow −10.15%/−10.55%；rate −5.10%/−4.13% | .1248→.1225 / .1274→.1262 |
| native12000 | −6.00% / −4.45% | prediction −6.81%/−6.74%；SIGReg −6.22%/−4.19% | .0436→.0428 / .0426→.0436 |
| TF12000 | −4.37% / −4.76% | flow −4.63%/−5.39%；rate **+0.62%/+1.57%** | .0370→.0359 / .0364→.0404 |

全部日志有限，活动模块梯度仍可测；local encoder/pool为预先冻结，0梯度符合合同。TF的rate与feature scale上升而flow下降，不能把总loss下降统称为动力学改善。模块梯度起伏、尤其renderer的中位数与OLS方向不一致，未构成梯度消失或系统性爆炸证据。

## 验证与规划：预算内的改善和反例都保留

| 方法/预算 | clean rollout / initial-copy（同64窗口） | primary成功率 | fresh成功率 |
|---|---|---|---|
| local2000 | .361 / .354 | 24 / 20% | 16 / 10% |
| local6000 | .227 / .221 | 16 / 22% | 24 / 20% |
| native2000 raw | .387 / .517 | 12 / 6% | 未作raw同组评测 |
| native6000 raw | .177 / .298 | 24 / 22% | 12 / 14% |
| native12000 raw | .0505 / .0543 | 26 / 26% | 28 / 26% |
| TF2000 | 无这套clean-transition测点 | 12 / 6% | 6 / 6% |
| TF6000 | 主阶段未做同套transition | 24 / 14% | 14 / 18% |
| TF12000 | 未做同套transition | 18 / 20% | 14 / 16% |

local固定观测坐标跨预算完全相同，raw clean MSE .468/.456→.295/.285；其fresh提升+9pp，既有配对条件95%区间[0,18.025]，primary −3pp[−11,5]。这支持继续学习，但不支持稳定任务提升。结构性输出span限制也不会因更多训练自动消失。

native6000→12000的raw clean MSE .221/.388→.0682/.0684；copy分母1.246/1.304→1.350/1.261、observed std .998/1.021→1.005/.955，改善不是仅分母暴涨。teacher/shifted-copy .240/.345→.0398/.0393。raw fresh +14pp[2,26]，primary +3pp[−5,11]；这是反对“6000已成熟”的直接证据。预声明的train-only preciseBN变体也全部保留：primary6000 16/18→12000 30/28%，fresh20/16→30/26%；不事后取raw/cal较大值。native的可学习坐标仍会变化，误差不是物理单位，不与local1024或TF64绝对误差横比。

TF同16窗口t=.95 forecast条件去噪MSE：2000 .02086/.02152，6000 .00566/.00719，12000 .00394/.00474。后段减少30.3%/34.1%，却primary +0pp[−8,8]、fresh −1pp[−9,7]；这保留了“学得更好未必规划更好”的反例。此health含带噪初态和renderer，不能替代clean latent dynamics。既有其他probe也不是统一三预算的纵向transition曲线。

**没有9000–12000的同口径heldout密集测点**，所以无法宣称最后1/4验证误差仍“显著下降”。现有端点没有典型的“训练改善、同目标验证系统恶化”证据；不能据P0不涨诊断过拟合。任务区间只条件于两个固定seed与已实现的50病例，不包含seed总体、环境重复波动或多重比较不确定性。

## 公开预算边界与下一步

本地已固定的官方主源 `stable-worldmodel@21446f1ede6d5284e981bd7b47f432b994e6d812`，`scripts/train/config/lewm.yaml:22,29,43–44` 默认100 epochs、batch128、history3/window4；collector验证文件与该Git blob逐字节一致。`scripts/train/lewm.py:128,150–160` 使用random clip split、按loader长度建日程，并有原optimizer-factory的interval声明。我们的episode split、history7/window8、124910更新且按update调度是明确适配。**仓库默认100 epochs不是已发布checkpoint实际跑满100 epochs的证明**，没有把未核实的公开实际训练时长编入对照。

结论是三者均缺少充分成熟证据，native已有继续受益的最强正证；local需要补齐与12k基线的曝光差，TF值得作为“loss改善但P0不涨”的同预算反例保留。不能因此自动延长所有失败候选，patch2000的同预算落后仍是有效阶段结果，原HOLD不变。

Root据此只授权**准备、暂不激活**六条Cube成熟度控制：local/native/TF原seed1/2、原fullstate与日程统一封顶23342（两loader epochs）；local补11671完整归档，全部18000/23342完整归档；最终原primary/fresh、原CEM及固定公共correlated CEM并列。下一阶段仍是有界成熟度对照，不是新算法、收敛保证或自动跑完整124910日程。
