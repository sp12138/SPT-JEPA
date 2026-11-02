# 转向 Meta-World：现有实验收尾与新基准准备

用户2026-11-03指定后续主攻Meta-World。本次做覆盖审计、公开协议核对和独立接口预检，不启动新世界模型长训，不恢复旧goal或LLM周期监控。

## 旧实验是否已足够结束

**最后核验：03:56北京时间，135主要配方87个已完成处置、48个未结束（21个已有worker、27个待启动），D组六条仍未开始，无执行错误终点。** 以下03:23表格与37个完整性能导出保留为原始快照，不冒充同一时刻；最终覆盖见 [coverage-after-preparation.json](results/coverage-after-preparation.json)。

还不能宣称提案已全部做完。19:23:06UTC / 北京时间11月3日03:23的快照中，135个主要配方已有85个完成处置，另50个未结束：

| 部分 | 已完成处置 | 尚未结束 |
|---|---:|---:|
| TwoRoom主要配方 | 26个门槛失败，停止 | 0 |
| Cube K组 | 12个12000步完成，停止延长 | 0 |
| 标准L0 P1 | 3个10epoch完成 | 0 |
| PushT其余主要配方 | 37个名义完成、7个首epoch早停 | 50 |

50个中21个已有worker记录，29个尚未开始；这不是当前SGD/GPU占用数。剩余为P2十五、P3十、P4一、DF B二/C四/D六/E二/G八/H二。**特别是D组sample-level window/pooled同域对照六条仍未开始，不能把此前pooled版本的失败代替这个问题。** 原有限队列继续完成这些已授权问题；不再增加PushT/Cube扩展网格，不触发P2c新种子、硬任务Stage2或DF Phase2。执行错误终点为空；两个旧保存故障已有独立补充结果。

P2已有一条重要的正向分离：同为24982步，F7白噪声的双seed均值为8.00/7.33%，F8仅改时间相关系数ρ=.8后为87.33/83.00%（primary/fresh）。这说明当前forecast-scored并非全线无效，但不是超过旧TF的结论；旧TF三epoch94.00/92.33%的预算不同，必须另取同24982参照后比较。F8−F7的条件episode聚类95%区间为primary+79.33pp[73.70,84.33]、fresh+75.67pp[69.87,81.25]，条件于这两个固定模型、不覆盖训练seed总体或simulator-repeat方差。F7/F8及全部37个已完成模型均保留，不能只报F8。

完整覆盖证据：[old-campaign-inventory.json](results/old-campaign-inventory.json)、[37个完整模型全部终点](results/completed-performance.json)、[F7/F8预设配对](results/F7-F8-paired.json)。其余原始检查点、各阶段评测和CORRECTIONS继续保留原EFS根目录。

## Meta-World选哪套数据

Meta-World本身是50任务的模拟环境基准，并非只有一个固定离线数据集。[Farama官方文档](https://metaworld.farama.org/)提供环境和标准评估说明。

后续优先采用LeCun署名的[Meta官方JEPA-WMs](https://github.com/facebookresearch/jepa-wms)发布的[离线数据](https://huggingface.co/datasets/facebook/jepa-wms)。原论文说明它含42任务×3个采集种子×100条轨迹，合计12600条，每条100步；行为来自TD-MPC2训练早期的混合质量数据。不是12600条最优示范。README的“每任务100条”是简写，不能漏掉三个采集种子。[论文数据说明](https://arxiv.org/html/2512.24497v4#A5)

实际公开文件元数据已核验：126个原始Parquet分片，共737175770字节（约703.0MiB），另有一个预览分片。视频与轨迹内容尚未下载，实际行数、图像和动作时序仍待授权后核验。代码固定`13cf1d9c7e476f53c17714d2e0f1dc239a883ce0`，数据版本固定`6116f042ae7ae4c8e3f1fd2f194f432615664182`。见 [source-audit.json](results/source-audit.json)、[访问与文件清单](results/access-and-inventory-v1.json)。

公开主成绩的MW-R/MW-RW是Reach/Reach-Wall，不代表完成42种物体操作。CEM L2下论文报告DINO-WM44.8/35.1%、JEPA-WM58.2/41.6%。论文另指出物体操作会出现想象抓取、实际未抓住的失败。这使接触/抓取成为有依据的重点，而不是继续围绕手臂位置优化。[原论文Table2与抓取失败分析](https://arxiv.org/html/2512.24497v4)

## 接下来的固定方向

1. 先复现公开Reach/Reach-Wall，保留原数据、相机、帧率、目标构建、controller和成功定义。这是检查实现的参照，不把它当成全部研究任务。
2. 固定push、pick-place、drawer-open、door-open、peg-insert-side、bin-picking六类接触任务，逐任务报告，成功规则在看新方法结果前冻结。后续再扩展全部42任务。
3. 主线保持图像+已执行动作的世界模型学习。RGB-only和RGB+4D机器人本体状态分开比较，每条赛道各方法输入一致；39维仿真真值、奖励、成功和最优动作不进入主线训练。已有公开最佳checkpoint使用本体信息，不能在推理时把它置零就冒充RGB-only训练结果。
4. 先区分预测接触错误、目标度量错误与短视规划错误，再选择一个sample-level机制。详见 [PROTOCOL.md](PROTOCOL.md)。不会把噪声/anchor扫参继续搬到新任务，也不提前声称原创方法成立。

## 实际准备与当前依赖

官方299个文本源码/配置已冻结到EFS独立目录。Meta-World3.1.1/MuJoCo3.3.0已在独立native环境完成3条100步接口检查；图像渲染只使用原基线已释放且实际检查为空的dev2 GPU1，不占21个旧实验槽。首次检查早于依赖就绪的失败日志保留；旧Gym在Python3.12需要setuptools提供distutils，并需在导入前配置已有EGL路径。另外补齐了Gymnasium渲染所需的packaging依赖。这些修复均未改变官方wrapper、NN或现有LeWM环境。

这里的native检查使用Python3.12，不等于完整JEPA-WMs训练栈已就绪；官方项目要求Python3.10，后续完整栈另建环境。接口检查不产生学习算法成绩，也不作为官方训练数据。

数据访问是当前唯一需要用户账号配合的前置：官方仓库要求先在网页接受访问条件；三台机器均未发现现有HF登录，实际文件请求返回GatedRepoError。需要在[官方数据页](https://huggingface.co/datasets/facebook/jepa-wms)获得访问权限后，在dev2执行`/efs/prescottsun/project_jepa/envs/lewm/bin/hf auth login`。不要在聊天发送token。

授权后先运行一分片检查，确认后才下载该版本的全部Meta-World分片，不下载其他环境：

```bash
/efs/prescottsun/project_jepa/envs/lewm/bin/python -B \
  /efs/prescottsun/SPT-JEPA/reports/metaworld_transition_20261103/dataset_preflight.py \
  --cache /efs/prescottsun/SPT-JEPA/data/metaworld-jepawms-6116f04 \
  --output /efs/prescottsun/SPT-JEPA/reports/metaworld_transition_20261103/data-preflight-v1
```

脚本[dataset_preflight.py](dataset_preflight.py)检查真实Parquet、39D状态/4D动作结构、视频解码和声明的100观测/99动作合同；它不把shape检查当成精确时序或物理动力学正确性的证明。

实际接口预检已全部通过：[三任务证明](results/environment-preflight-v5.json)。Reach、Push、BinPicking各执行100步，224RGB/4D动作/39D诊断状态和完全相同初始状态重建均通过，重建max-abs差为0。仅seed重建失败的v3及依赖错误原日志均保留在EFS。Reach/Push脚本策略此次成功，BinPicking未成功；这些是接口检查，不是学习模型成绩，也不能据此推出BinPicking在100步内不可解。目标构建成功要独立记录，不能默认为所有expert参考都成功。
