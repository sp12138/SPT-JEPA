# 当前交接

目标是匹配条件下超过 RoboJEPA，尚未达成。用户授权在本会话每小时检查，性能差则复现参考、逐组件融合／改进。仅用dev13/dev15/dev6/dev6-2c；旧goal／旧监控不恢复。没有创建托管定时任务。以下成绩均为进度，严格成功另报；源验证只含Reach-v2/v3、Push-v2/v3四个物理案例（seed7401），重复评测不增加独立案例数。

## 当前推进

新有限根 `runs/robocasa_target_refinement_20261111` 已在三台主机实际启动，六个10002步完整恢复／梯度／Push-H5入口全部通过，正在继续到15000步。三个配对方向：原目标续训、冻结读出表示、把像素流匹配目标换成自然未来帧的冻结V-JEPA特征。六个模型均从原10000步V-JEPA-SPT的两个完整父状态开始，每个新增5000步；同1024普通随机日志、batch32、window8、四rank RNG／游标，统一5000步cosine重启／50步warmup。冻结组仅冻结pool／feature_adapter，动力学和像素解码器继续训练。特征组保留解码器384×12主干，重建1664通道输入／输出和14×14位置缓存；兼容参数／Adam动量按名称保留，新I/O无旧动量。各组保留原beta=.25和rate分母12288；像素／特征失真单位不同，不冒称等单位ELBO。先做私有梯度／因果性检查和真实10002恢复／Push入口，最终15000步24源回合，六小时cap，无自动长训。核心 `target_refinement.py`；冻结源 `sources/target-refinement-20261111-v1`。启动 `launch-dev13/dev15/dev6.json`，有限collector汇总 `results/summary.json`；私有检查六项均通过，像素重构前后loss／RNG／所有参数梯度完全一致。新目标只是有证据的假设，尚无最终性能结果。

本会话检查记录在 `runs/robocasa_encoder_budget_20261111/SESSION_CHECKS.json`。上次21:35UTC之后，用户恢复会话时已到Nov11 03:59UTC；中间没有实际检查，不能补写。最近实际检查04:00:56UTC，下一次约05:00:56UTC；统一helper在 `runs/robocasa_robojepa_fusion_20261111/session-check.py`。20:34漏读旧参考gpu目录的问题已单独correction，旧0计数不是实验结果。04:00–04:18实际快照：三台主机原队列全部结束、24卡空闲；正在转入本轮，不能把分配卡数称为持续满载。dev6-2c原参考队列仍在跑，截止约07:05UTC不延长。

## 已收齐的关键结果

所有下表源回合严格final／any成功均0；Reach为平均归一化进度，Push为最终归一化进度。同案例旧零动作Push已有9.09%，约9.2%不是操纵突破。

| 对照／有限根 | Reach | Push | 结论 |
| --- | ---: | ---: | --- |
| own-SPT 10000步 |15.73%|9.29%|未解决操作 |
| V-JEPA-SPT 10000步 |64.77%|9.25%|Reach随预算明显提高，Push不动 |
| own-dense／V-JEPA-dense 10000步 |20.11%／38.85%|9.31%／9.14%|换dense头也没有操作收益 |
| 官方2B RGB全视场 |77.19%|11.25%|非论文完整输入，非同训练预算 |
| 同2B square224 |70.67%|9.29%|全视场有差异，未解决操作 |
| 官方8B RGB全视场 |43.51%|9.18%|该源队列未显示更大模型收益 |
| 官方2B released proprio |92.02%|28.33%|Push-v2为55.39%，v3为1.26%；仍0严格成功 |
| 官方2B native proprio |93.14%|4.54%|训练式本体输入并未更好 |
| 官方8B released proprio |95.44%|9.15%|Reach高不代表Push完成 |
| SPT续训plain／self／cross 11000步 |75.67%／77.75%／80.05%|9.20%／9.19%／9.17%|跨相机消息未改善操纵，不长训 |
| 官方2B动力学＋SPT读出 |86.69%|10.89%|低于原92.02／28.33；读出额外适配预算和矩形分布偏移须披露 |

预算根 `runs/robocasa_encoder_budget_20261111` 已完成八模型／64新增回合，保留32继承回合。原1000→5000→10000完整状态续训，统一9000步cosine重启／90warmup；不是固定LR的纯scaling law。V-JEPA-SPT Reach为11.03→47.58→64.77%，说明短训会遮蔽骨干收益。所有头1000步前重新初始化；own骨干继承7000步，外部视觉骨干预算不同。

公开RGB／预处理诊断 `runs/robocasa_robojepa_fusion_20261111`、公开主输入 `runs/robocasa_public_inputs_20261111`、跨视角 `runs/robocasa_crossview_20261111`、读出融合 `runs/robocasa_readout_20261111` 均已完成，HOLD，无有效执行错误。各正式回合前实际入口、NN不变和分布式物理／动作一致检查通过。模式2到3同时改变proprio、embodiment和动作编码，不是纯proprio消融。公开预训练／状态参考与RGB候选分轨，不作同预算SOTA排名。

## 采样器假设已被物理结果否定

`runs/robocasa_endpoint_20261111` 已完成24回合、0SGD、严格成功全0。`endpoint_cem.py`在相同1024候选／20elite／原refit数下比较发布式径向分布、初始矩匹配Gaussian和终点／路径分解协方差。CPU小实验确实发现分布／高斯方差refit不一致会在无信息目标下冷却，分解协方差可保留路径探索；这没有转成控制收益。公开2B released轨的Push从原28.33%降为Gaussian0%、endpoint1.72%。SPT只有一个父模型的单Push-v2病例有Gaussian收益，另一父模型不成立；不能叫稳健改进。无长训晋级。实际源、CPU证明和分family结果留在该根及同级 `robocasa_endpoint_{gaussian,endpoint}_20261111`。

## 表示／数据诊断的边界

固定盐抽8个TRAIN初态组×8分支，64条旧日志／1024事实转移全部严格重放，初始物理、RGB、实测位移误差均0。16/64条有机器人接触，36/1024动作步有接触；8/8组跨动作物体XY位移差>1cm。这否定完全没有对象干预数据，不能排除接触稀疏、目标失配或泛化问题。对象／接触字段仅诊断，不进训练、教师或proposal。

固定距离统计（无拟合probe、无SGD／模拟）3584同时间跨分支pair：dense V-JEPA的物体XY／机械臂EEF相关为.122／.740，SPT两读出为.090/.910、.092/.921；控制EEF的部分秩相关分别.100、.054、.060。没有符合阈值的固定EEF匹配pair，所以只能说该度量明显跟随机械臂运动，不能证明物体信息不存在或找到了唯一原因。新增同pair的RGB224／64固定距离统计中，物体XY相关为.117／.121，EEF为.579／.580，未显示明显分辨率退化；这削弱了单独归因64像素的解释，不是生成器或控制能力证明。入口 `robocasa_robojepa_fusion_20261111/coverage/{protected,distance-statistics}`。

## 第四台公开目标厨房参考

`runs/robocasa_reference_adopt_20261110` 接管原24小时队列；结果实际在 `runs/robocasa_intervention_20261110/public-evaluation/*/gpu*/*/summary.json`。保留原chunk，无重复实验，原截止不延长。04:00UTC released轨的20个Reach案例均齐，50M／2B／8B平均进度91.00／93.40／96.12%，严格final为0／3／1次。50M Push19例均齐，平均0.49%；2B6例、8B4例未齐，不拿部分均值排名。50M native已开始。该39有效目标厨房案例、外部预训练和状态输入与上表源验证不同，不能直接拼表成公平差距。参考BF16预测按论文表格，发布客户端FP16差异仍披露。

## 历史轮次

旧 `robocasa_compare/refine/intervention/local/forks_20261110` 均已结束；没有可靠严格操纵成功，均未获长训晋级。局部目标support和共享根direction曾提高个别源Push进度，但牺牲Reach或跨父模型不稳定。冻结几何／动作辨识诊断在 `runs/robocasa_geometry_review_20261110`；分支预测77%–82%可辨识动作，不证明对象接触已学好。全部原始错误、修复、祖先与结果仍在EFS。旧R3参考已STOP，不恢复。旧MetaWorld困难六任务均未可靠突破；历史分任务正结果及边界见结论。

- [关键结论](experiments/CONCLUSIONS.md)
- [较优配置](experiments/best-configs.json)：本轮无新配方晋级
- [原始记录归档](docs/experiment-archives.md)
- [稳定约定](AGENTS.md)
