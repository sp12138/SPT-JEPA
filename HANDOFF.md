# 当前交接

目标是在**独立动作头实际控制**下验证可靠、稳定的scaling，最终超过RoboJEPA；尚未达成。用户要求本会话每小时检查，有明确问题便改进重跑，不在有限pilot后HOLD。仅dev13/dev15/dev6，不连接dev6-2c/grid16/dev2，不恢复旧goal、旧监控或文档中的旧ACTIVE。

## 状态与评估口径

- EFS总入口：`/efs/prescottsun/SPT-JEPA/runs/robocasa_reach3d_scaling_20261114`，简称R；模型根在其父目录。实时状态查R/MASTER.json、SESSION_CHECKS.json及实际进程/GPU。
- 正式检查仅由主会话执行；第48次快照`session-checks/20261116T0641-{dev13,dev15,dev6}.json`，实际训练rank为8/4/8，dev15另四卡进入冻结读出，瞬时利用率0–100%，六组无ERROR。最新/下次时间只查SESSION_CHECKS，快照不代表持续满载；元数据用`MASTER.primary-write.lock`保护、重复快照不推进小时钟。
- 同Reach blocking接口，固定12006条普通TRAIN日志，全部新主链随机初始化、祖先0。body只输入RGB/实际a_t、学习自然未来RGB/自身码，无外部教师或动作target。
- 冻结/detach动作头只监督同一事实转移当前输入a_t的XYZ，17920训练/1536验证标签。推理用目标RGB替换自然next RGB，必须以实际控制验证该分布差异。
- 每点同64个反复使用的DEV案例、双224 RGB、最多5命令/norm .3/native2mm final与any。32确认例封闭；单点、loss下降、平滑或偶然成功不算稳定scaling。
- warmup900后inverse-sqrt active-wall调度，不是常数LR；完整恢复必须含model/optimizer/scheduler/4rank RNG/数据游标。不同参数、读出容量、更新数/共卡开销与预训练来源分别报告。

## 当前六组

各占四卡，根名均以`robocasa_reach3d_`开头。表为第48次检查时最新完整64DEV结果，非训练中阶段。

| 主机/卡 | 方法 / 根后缀 | 参数 / 头参数 | 训练小时：回报、距离mm、final/any计数 |
|---|---|---|---|
| dev13 0–3 | E384/D6，`encoder_allocation_20261116`；spt-encoder-heavy-shared-mean | 41.625M / 2304 E均值 | 10h：90.65%、20.96、2/4 |
| dev13 4–7 | E384/D6+视角地址联合状态，`encoder_view_joint_20261116`；spt-encoder-heavy-view-address | 47.575M / 384联合码 | 7h：85.51%、32.86、3/4 |
| dev15 0–3 | E192/D6，`allocation_small_shallow_20261116`；spt-small-encoder-shallow-decoder | 24.986M / 1152 E均值 | 7h：86.77%、27.31、0/2 |
| dev15 4–7 | 同步视角随机未来码目标，`synchronized_view_target_20261116`；spt-synchronized-view-target | 41.625M / 2304 E均值 | 第1h在训，尚无控制成绩 |
| dev6 0–3 | RGB恒等支路，`rgb_innovation_20261116`；spt-current-rgb-innovation | 41.625M / 2304 E均值 | 5h：88.85%、28.83、0/1 |
| dev6 4–7 | RGB恒等支路+root条件先验，`rgb_root_marginal_v2_20261116`；spt-rgb-innovation-root-marginal | 41.625M / 2304 E均值 | 2h：88.07%、29.55、1/2 |

对应私有源为R/`code-encoder-allocation-v1`、`code-encoder-view-joint-v1`、`code-encoder-allocation-factorial-v1`、`code-synchronized-view-target-v1`、`code-precise-consequence-v1`及`code-root-marginal-v2`。初始化子张量配对不算独立seed。

实际启动证据在`encoder-allocation-20261116`、`encoder-view-combination-20261116`、`encoder-allocation-factorial-20261116`、`synchronized-view-target-20261116`、`precise-consequence-20261116`及`root-marginal-prior-20261116`的`ACTUAL-STARTUP-VERIFIED.json`。验证必须涵盖实际保留8/16更新、全状态恢复、四GPU rank、配置/私有源及CPU模型/优化器；准备门不能替代实际启动。同步视角组v2已核验实际8/16更新、10600步进度及7536步CPU完整状态；v1误查CUDA的CPU核验错误保留，训练未改。

## 当前关键insights与下一步

1. **粗定位随训练改善，精度仍不稳定。** 基组1/3/6/8/9/10h回报81.83/86.47/91.35/90.69/91.43/90.65%，距离42.23/30.73/17.78/21.06/19.81/20.96mm，final0/0/0/0/0/2。5–8h读出alpha均.01、验证RMSE仍降，不能归因为alpha跳变。视角联合组与小E组也未保持严格成功；当前仍以完整曲线判定，确认集不开。
2. **RGB恒等支路提供早期粗收益，尚未稳定。** 1/2/3/4/5h为86.72/87.51/90.31/83.13/88.85%，32.31/31.85/23.43/41.33/28.83mm，final0/0/1/2/0。第3h对同预算基组+3.84点、四厨房区组95%[.59,7.67]，距离−7.30mm、[−14.04,−.57]；第4h回落主要在厨房10（−29.14点/+71.78mm），另三厨房近持平。3/4h的alpha均.01、事实验证RMSE仍降、无数值重试，实际评测已使用最高FP32精度，不伪造精度/alpha故障解释。详见`analysis/precise-consequence-third-hour-paired.json`及`analysis/independent-head-objectives-and-priors-review.json`。
3. **更大/更复杂动作头与按小动作重权尚未解决。** 固定父模型的MAE、状态非线性、空间Potential、F响应等有限探针保留全部结果。最新相对误差头在预先固定3/5/8h、同2304参数/标签下仅77.93/81.79/83.81%、31.99/28.85/26.23mm、final0/3/2，四厨房粗回报都弱于匹配MSE（−8.53/−7.39/−6.88点），距离未可靠改善、严格计数不随预算增长；不扩展。两次新解析拟合+一次原拟合SHA复用、0body更新、192次模型—案例仍是同64DEV，dev15共卡成本单列。`relative-mean-scale-probe-v2-20261116/DONE.json`及上述配对review。
4. **保留有效作用结构优于本轮直接替换。** 普通MLP转移1/2/3h77.24/79.61/83.53%，3h36.72mm、final/any0；F由815360减为37120参数，不能纯归因非线性。保留条件线性作用再加37120参数非线性余项，1/2/3h87.30/86.59/90.28%，3h23.42mm、final0/any2，基本持平RGB3h90.31%/23.43mm/final1，没有展示额外收益。完整3h/43046步、64DEV及CPU全状态保全后退役，0旧SGD重放；新同步视角组唯一启动，证据见`synchronized-view-target-20261116/OLD-READY-FOR-PRIMARY.json`，旧源与检查点不改。
5. **新同步目标实验检验相机表示的坐标自由度。** 每一物理样本/未来时刻内均匀循环置换同步相机的posterior均值；保留初始码、原clean rollout源、实际a_t及每相机自然未来RGB/当前anchor。原Gaussian噪声/rate/beta/解码混合/参数/头均不变，无新loss系数。条件KL仍精确；对置换取期望会在原公共均值误差外产生由sigma/rho决定的视角方差惩罚，不能消除所有场景相关坐标自由度或预告控制收益。同8TRAIN窗口中RGB3/4h及root1h的跨视角码差为单步时变的14.41/13.33/25.28倍，把自然future码换为另一同步相机码会使原相机像素误差增6.81–15.88倍；这是可测视角差异，不是厨房10失败的因果证明。`analysis/view-randomized-target-theory.json`及`synchronized-view-code-audit.json`。六类独立真实旧默认回归、完整Gaussian值/梯度、样本隔离/未来因果/数据SHA/实际读出门通过，4功能更新废弃。v1回归器旧模块绑定错误保留，v2只修核验脚本，模型冻结源未改。新链祖先0。
6. **root条件先验只是另一可检验机制。** 保留原完整clean rollout中心，仅把AR未来—未来依赖改为共同条件初始码，保持边缘方差/root协方差、补精确Gaussian常数。7步误差弱/强模态惩罚比4.00/.156，不是统一调beta或修teacher forcing。1/2h86.93/88.07%、31.55/29.55mm、final0/1，尚无可靠早期优势，更非稳定scaling。独立Gaussian、真实旧默认、实际8/16恢复/CPU绑定loss源与优化器均过；v1等价FP32图守卫失败和v2修正留档。

TRAIN审计未支持简单rate支配或大量画面完全相同却动作冲突的解释；最小事实动作组预测误差高于零输出、但范数未统一缩小。此前联合状态、容量2×2及各种冻结头的精简结论/负结果查[CONCLUSIONS](experiments/CONCLUSIONS.md)，不重复启动失败探针。后续继续主会话每小时核查和完整控制收集，优先检验明确机制；有限DONE不结束主目标，尚无新best-config晋升。

## 归档与恢复边界

- Git仅核心源码/必要配置/精简结论；原始结果、模型、监控和临时审计留EFS。协议见[PROTOCOL](experiments/robocasa_stable_scaling/PROTOCOL.md)，已验证配方见[best-configs](experiments/best-configs.json)，[归档说明](docs/experiment-archives.md)。
- 相对头v1嵌套根发生EGL库定位错误，episode前失败。核心环境改为向祖先查找真实bundle；独立旧平层完整环境字典不变、布局/缺库及真实离屏GL门通过，私有v2修环境、不重放拟合，旧源/错误/首头保留。`relative-mean-scale-probe-v2-20261116`。
- 旧冻结V-JEPA G的12h94.86%/9.70mm/3final随后回落，不与从零主链或公开RoboJEPA混作公平胜出。旧Push日志z约束与Reach不同，已保全，不当纯算法收益。
- 旧共享链接曾写穿，独立原版/私有修复见`source-recovery-20261114T2353`；不能追溯证明全部历史加载。新版本用`source_freeze.freeze_with_overrides`，不改已启动源码、共享链接或vendored runtime。
- 原小语料完整8h/128308步保全，额外未完成9h的1992–2092步排除、未继承/重放；未来恢复须披露可能重放范围，不自动恢复。交接均先完整64DEV及CPU全状态。
- 保护性评测/oracle/接触/闭环诊断不进入训练、教师或proposal memory；只读聚合指标/过滤错误。严禁未来/最优动作标签、专家路径或目标条件策略监督。
