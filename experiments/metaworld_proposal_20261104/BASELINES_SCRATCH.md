# 从随机初始化训练的基线合同

最新用户明确要求不能使用已有强模型先验，必须原生train from scratch。此前DINO-WM、官方JEPA-WMs、SmolVLA和Evo-World的预训练版本仅是文献背景，不列入本研究的主成绩对照。编码器、动力学、生成器和任何学习器/规划器所依赖的教师都不得加载外部预训练权重；预训练视觉、语言、分割、深度、光流教师也在排除范围内。同一研究数据上的自身续训可保留，但必须计入完整起点与累计预算，不把继承模型称为新随机初始化。

同时保留已有约束：RGB+已执行动作；无reward、success、物体真值、任务语言/ID或下一最优/示范策略动作目标。物理状态仅用于独立诊断和共享评测重置。候选方法保持sample-level loss；公开基线的batch regularizer须明确披露。

当前代码检查：as_jepa_lewm/model.py直接调用ViTModel(ViTConfig(...))；native_lewm.py调用vit_hf(...,pretrained=False)。现有MetaWorld原始训练从随机权重开始，新控制pilot仅冻结自身在同一数据训练至16480步的模型，没有新增外部教师/权重。20轮TF22.1875%/native15.625%等本地成绩不构成公开SOTA；AS与native容量/优化器不同仍需披露。

优先基线：

| 方法 | 原生机制 | 适用性与边界 |
|---|---|---|
| LeWM | 未来latent预测＋SIGReg，端到端学习视觉/动作动力学，MPC | 首要主基线，SIGReg跨batch；原论文控制套件不是MetaWorld，当前MetaWorld成绩来自本地适配 |
| LpWM | 未来latent预测＋RDMReg，非负稀疏latent，MPC | 近期强结构基线；论文Table5明确from-scratch ViT；论文未给MetaWorld成绩，需要同协议重训。RDMReg为分布正则，不是sample-level候选 |
| PLDM | 预测＋VICReg/时序正则＋可选/原配方IDM，MPC | 重要历史基线，不能省掉原配方项后仍叫完整PLDM；IDM从两个已观察状态回归实际执行动作，不是从当前状态预测下一最优动作。各项/输入需单列披露 |

还应有同encoder/predictor配置的SIGReg/VICReg控制，以区分损失机制、参数容量和训练预算；这些是适配消融，不能冒充原论文实现。原版LeWM单独保留。

若另设允许奖励的scratch赛道，TD-MPC2、DreamerV3是强参考；其原生reward/value/actor训练结果不可搬入当前reward-free主表。去奖励后用统一图像目标MPC属于明确命名的适配版本，需要重新评测，不能沿用原RL成绩。

当前没有核验到完全匹配“本数据版本＋8任务图像目标＋RGB-only＋无奖励/最优动作监督＋全链路随机初始化”的公开冠军表。不得给未运行的LpWM/PLDM编造MetaWorld成绩，不能把TwoRoom/PushT/Cube或DMC-Reacher分数算作MetaWorld。

补充审计：F-JEPA在AppendixA明确PushT选用训练好的LeWM encoder/projector并冻结，Reacher也使用LeWM warmstart；其公开整表不能直接作为同预算单阶段scratch成绩。若加入，应从随机起点重训完整流水线、统一累计预算。其余任务的joint训练和MetaWorld尚未验证，不自动启动新训练。

来源（2026-11-04核验）：

- [LeWM方法与基线](https://arxiv.org/html/2603.19312v1#S3)
- [PLDM原论文](https://arxiv.org/html/2502.14819v4)
- [LpWM，尤其AppendixD/Table5](https://arxiv.org/html/2608.22764v1#A4)
- [F-JEPA，AppendixA训练初始化](https://arxiv.org/html/2608.29029v1#A1)
- [公开reward-free Dreamer/IRIS/TD-MPC2适配说明](https://arxiv.org/html/2610.27851v1#S4)

本次只更新约束与基线审计；既有有限16路proposal pilot继续，未新启动上述基线训练、长训或周期监控。
