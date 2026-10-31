# NGC 冻结共同控制器可行性审计

2026-10-31。结论：**值得作为已公开方法的共同控制器基线；可只换终端成本，不重训。当前未实现、建图、编码、拟合或运行评测。** 官方代码钉到 `1312d35b62d7f643af494ae3954c10725cfab1ca`；逐文件SHA见 `source-audit.json`。只读源码临时checkout在 `/tmp/ngc-readonly-git`，未下载权重/数据/大缓存。

## 是否已经覆盖

本地全部Markdown对题名、arXiv号、NGC、Neighborhood Graph Cost、Isomap/geodesic的检索未发现此前审阅该方法的记录。旧I10是动作Jacobian可控Gramian的逆二次型；decoder geometry是固定平均JᵀJ二次型；两者都没有训练样本邻接图与最短路。因此那些控制的负结果没有排除NGC。NGC属于已知公开方案，不计我们的原创想法；也不能把它的收益称为sample-level训练方法收益。

## 方法与信息条件（已核代码）

每个冻结编码器将相同训练RGB行编码为节点v。以欧氏边长构建mutual-15NN图；若不连通，反复加最短跨分量边，再计算全对Dijkstra距离D。默认N=5000，每个不同训练episode选一个节点。候选终点q和目标g各接最近4节点；若直接距离落在双方第15近邻半径内，取直接欧氏距离；否则取16条锚点路径中最短的 `||q−vi||+Dij+||vj−g||`。它不是时间距离、可达性图、动作能量或平方Mahalanobis距离。[官方graph.py](https://github.com/youzunzhi/ngc-latent-manifold/blob/1312d35b62d7f643af494ae3954c10725cfab1ca/ngc/graph.py#L57)、[cost.py](https://github.com/youzunzhi/ngc-latent-manifold/blob/1312d35b62d7f643af494ae3954c10725cfab1ca/ngc/cost.py#L71)

图不读取动作、物理状态、reward或最优动作标签，也不训练额外网络。原训练仍是LeWM观测动作条件预测与SIGReg；原数据来自expert并不等于goal→最优动作监督。每个测试goal只计算近邻及半径，既不把goal加入离线图，也没有test-time参数优化。物理状态用于论文的pair筛选、成功判定、诊断；simulator oracle是另一个诊断实验。[nodes.py](https://github.com/youzunzhi/ngc-latent-manifold/blob/1312d35b62d7f643af494ae3954c10725cfab1ca/ngc/nodes.py#L21)、[pairs.py](https://github.com/youzunzhi/ngc-latent-manifold/blob/1312d35b62d7f643af494ae3954c10725cfab1ca/evaluation/pairs.py#L40)、[训练loss](https://github.com/youzunzhi/ngc-latent-manifold/blob/1312d35b62d7f643af494ae3954c10725cfab1ca/training/lewm.py#L88)

原采样细节：从seed控制的90%训练clip划分中，排除被评估episode后，均匀选不同episode，再均匀选该episode的训练clip和clip中的帧；不是整列均匀采样。默认frameskip5、4帧clip，graph seed与pair seed关联。我们必须改为权威episode级train split，保持同一节点行清单跨checkpoint，明确这是匹配本项目的适配，不能重新调用其random_split混入本项目validation。[nodes.py:21–46](https://github.com/youzunzhi/ngc-latent-manifold/blob/1312d35b62d7f643af494ae3954c10725cfab1ca/ngc/nodes.py#L21)、[实验常数](https://github.com/youzunzhi/ngc-latent-manifold/blob/1312d35b62d7f643af494ae3954c10725cfab1ca/experiments/plan.py#L53)

## 论文条件不能直接迁移

| 项目 | NGC公开主实验 | 本项目固定对照应保持 |
|---|---|---|
| 数据 | 同官方Cube/TwoRoom文件revision；训练episode内start/goal，graph排除评估episode | 9000/500/500 episode划分，primary/fresh来自既有validation清单 |
| Cube成功 | 位置≤4cm，任一步满足即可 | ≤3cm、24对称姿态≤15°、连续3步 |
| TwoRoom成功 | 位置16px，跨房间pair | 原strict协议与topology不变 |
| pair/重复 | 每offset300 pairs，3 planner seeds/对；Cube offsets25/50/100，TR25/50/75 | primary/fresh各原50，原policy seed，不新选简单pair |
| 起点 | Cube等任务置零速度；首轮1真实帧，后续自行维护3真实帧/已执行动作 | 原录制状态与原World/policy上下文，不能偷偷采用其新history |
| planner | H=5、stride5，执行25raw步再规划；300候选×30轮、30elite；预算2×offset | 同数值预算，但严格保留原控制路径/命令动作 |
| action scaler | full-data StandardScaler | 现所有own/native适配器的train-only scaler |

来源：[论文§5及附录A](https://arxiv.org/html/2610.25013v1#A1)、[官方评测入口](https://github.com/youzunzhi/ngc-latent-manifold/blob/1312d35b62d7f643af494ae3954c10725cfab1ca/scripts/evaluate.py#L63)、[真实history与执行端clip](https://github.com/youzunzhi/ngc-latent-manifold/blob/1312d35b62d7f643af494ae3954c10725cfab1ca/evaluation/history.py#L55)。官方main在返回动作端clip并记入history；这不等于我们此前“预测候选也投影”的控制。其comparability分支又用默认单帧history；两套入口都不宜直接代替本项目evaluator。

训练配置是ViT-tiny/patch14/224/192维、6层predictor、history3、SIGReg=.09，10epoch，batch128、bf16、AdamW5e−5、wd1e−3、1%warmup后cosine至10epoch结束。我们native的8帧/history7、原有step schedule和6k/12k曝光不是该recipe；本次若仅在已有模型上换cost，既无需复制这些训练设置，也不能引用其结果为同预算成绩。[training/config/lewm.yaml](https://github.com/youzunzhi/ngc-latent-manifold/blob/1312d35b62d7f643af494ae3954c10725cfab1ca/training/config/lewm.yaml)、[scheduler](https://github.com/youzunzhi/ngc-latent-manifold/blob/1312d35b62d7f643af494ae3954c10725cfab1ca/training/lewm.py#L143)

文献数值仅供定位：offset25时TwoRoom L2→NGC为91.4→99.8%，Cube为47.6→57.9%；这些都不是我们的严格heldout结果。其报告NGC相对L2在线CEM耗时最多+7%（RTX4090），不能当本机吞吐保证。[论文表3/5](https://arxiv.org/html/2610.25013v1)

## 实现与预算可行性

1. **接口吻合。** `NGCCost`接受原`predicted_emb[B,S,T,D]`/`goal_emb[B,T,D]`并只取末态，返回[B,S]；可在既有metric_evaluate的objective替换位置接入，原模型、scaler、solver、World和严格成功函数均不改。不得复制官方整套评测。L2控制直接返回原GoalMSE；NGC直接使用官方未平方路径长度，不随意加identity混合、whitening或投影。[cost接口](https://github.com/youzunzhi/ngc-latent-manifold/blob/1312d35b62d7f643af494ae3954c10725cfab1ca/ngc/cost.py#L31)
2. **训练集足够。** 本次只读两张dataset_card及episode数组，均9000训练episode；与论文数据revision一致。5000不同训练episode可取，无需缩N或读取validation。现128clip缓存只有128episode，不满足该默认规则。未来如授权，每个唯一真实编码器需5000张单帧前向；batch32是157次，12个不同编码器上限60000张/1884次，只有全编码器/归一化/state hash相同才可复用。没有DiT/predict/拟合需求。
3. **内存不构成障碍。** 由矩阵shape直接计算，FP32 D为100MB；nodes另加1.28MB(64D)、3.84MB(192D)、20.48MB(1024D)。离线float64全距离/D各200MB，排序/连通桥还有临时数组；实际CPU时间应单报，不声称免费。官方在线全排序300×5000节点，再查16route。[graph](https://github.com/youzunzhi/ngc-latent-manifold/blob/1312d35b62d7f643af494ae3954c10725cfab1ca/ngc/graph.py#L57)
4. **数值条件明确。** 官方离线FP64、在线FP32、稳定排序tie按节点顺序；近邻距离用cdist矩阵乘法，直接goal距离用差分。官方assert第15邻域radius>0，重复/坍塌节点可能使它失败；不能无声加floor/fallback。记录桥数量/跨度、finite、节点/source/checkpoint/状态hash和原identity代数合同；图只是观测局部连接，不保证物理可达或接触方向性。[cost数值路径](https://github.com/youzunzhi/ngc-latent-manifold/blob/1312d35b62d7f643af494ae3954c10725cfab1ca/ngc/cost.py#L20)

## 最小公平对照建议（未授权执行）

最小基线验证：本项目raw native两任务×两seed的既有held checkpoint，均比较原L2和NGC，primary/fresh各50，即16评测端点；这仅验证公共cost在现严格条件的效果，不能建立ours优势。若要比较当前ours/TF/native的排序，三类冻结模型必须都跑同两种cost，共48端点，且保留TR F64free/Cube local为不同candidate、训练曝光和18M/51M/56M容量差异。不能只给ours套NGC，或选各任务较优raw/BNcal。

锁定单一train-only5000节点行清单、k15/anchors4/mutual/连通规则，不以validation调图；各模型使用自己的原生完整状态坐标。先按原command CEM做两臂，不混相关CEM或action projection；若将来另比较更强controller，则所有模型两臂都统一。报告固定solver调用预算及离线编码/建图、在线时间。保存同一实际候选bank的L2/NGC分数；后续bank自然分叉，不能当共同bank。病例配对CI保留每seed，并说明环境复测波动未被单次case bootstrap覆盖。

若目标是验证“图拓扑”而非仅报告公开方法性能，已有公开NGC_Euc控制将D换成节点间直线距离、保留相同attach/radius，可区分图最短路与贴近训练支持的惩罚；这属于额外机制控制，不能从L2/NGC两臂单独推导。当前不自动加该臂。[官方ablation](https://github.com/youzunzhi/ngc-latent-manifold/blob/1312d35b62d7f643af494ae3954c10725cfab1ca/ngc/ablations.py)

判断边界：此前固定池“预测编码成本排序好、物理argmin差”与这个假说相容，但不证明NGC会纠正Cube姿态/hold。它也可能沿机械臂主导的流形走得更准确，或通过桥产生跨墙/不可能接触捷径。若统一控制器让TF/native同样获益，应更新共同强baseline，不能记作我们的学习贡献。
