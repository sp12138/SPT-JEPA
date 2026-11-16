# 当前交接

目标：从零 SPT-JEPA 的独立动作头控制出现可靠稳定的 scaling，并超过已有参照。**尚未达成**；两点上升、loss 下降或零星成功都不算达标。用户要求时间优先算法机制、关键实验和实际控制，检查与记录最小化，避免重复兼容回归或 CPU 审计。

只用 dev13、dev15、dev6；dev6-2c/grid16/dev2 不连接。当前主会话每小时检查并按证据改进，不在有限 pilot 后自行 HOLD；不恢复旧 goal、监控或第二套 LLM 监督。实际分配和检查入口为 R/MASTER.json、R/SESSION_CHECKS.json，不能根据旧 ACTIVE 文字恢复任务。

R=`/efs/prescottsun/SPT-JEPA/runs/robocasa_reach3d_scaling_20261114`；模型根在其父目录，下表后缀均加 `robocasa_reach3d_`。

| 主机/卡 | 根后缀 / run name | 当前方法；可训练参数 / 动作头参数 |
|---|---|---|
| dev13 0–3 | `global_natural_query_20261117` / spt-global-natural-query | 全局动作自然参数＋学习上下文；22.245M / 384 |
| dev13 4–7 | `trajectory_flexible_likelihood_20261117` / spt-trajectory-flexible-likelihood | 状态相关F＋样本内后果辨认；41.625M / 384 |
| dev15 0–3 | `likelihood_pair_20261117` / spt-likelihood-pair | 当前/下一帧二元后果似然＋正交动作输入；22.220M / 384 |
| dev15 4–7 | `anchor_reader_exact_nats_20261117` / spt-anchor-reader-b1 | 同初值AnchorReader、单位nats价格；25.165M / 384 |
| dev6 0–3 | `anchor_reader_20261117` / spt-anchor-reader-b03 | Lingjing条件去噪熵梯度＋Gaussian后果率；25.165M / 384 |
| dev6 4–7 | `likelihood_factual_q_20261117` / spt-likelihood-factual-q | 单步Q＋8候选受控对照；22.220M / 384 |

## 决策所需的关键结果

- **纯动作平移已完整8h收尾。** 1/2/4/6/7/8h回报71.70/86.30/87.95/85.56/85.94/87.28%，距离67.51/29.59/29.87/37.58/37.48/31.75mm，final0/0/0/2/0/1；未稳定保持后段收益。完整8h及部分9h全状态保留，后继继承0更新，dev6 4–7已给单步8候选Q对照。源`code-pure-action-transport-v1`，完整曲线保留EFS。
- EMA未来码1/2h49.27/49.96%、124.00/123.78mm、全0，已退出。EMA当前码完整6h88.04%/27.28mm、严格0，仍有粗改善但未精确scaling；保全完整6h及部分7h（含自身EMA）后将卡交给单步后果辨认；当前码＋平移组合1/2h57.24/71.11%、101.00/64.30mm、全0，保全完整2h及部分3h状态后退出，让卡给新相对编码。均为自身同数据随机初始化模型EMA，decay=exp(-32/120060)，无外部教师。源 `code-ema-initial-target-v1`；读出容量分别2304/384，不能混报。
- 同8普通TRAIN窗口，未来码换成当前码，原RGB1h解码误差增至2.21–3.54倍，纯平移1h仅1.02–1.30倍；后者实际动作生成画面比自然未来编码解码更准。支持自然编码弱于动作生成路径，不能只改F。原始 `analysis/motion-code-necessity-first-hour.json`。EMA组合的在线/教师码差异在减去各自当前码后从.02033降至.000774，存在显著原点漂移；直接套原教师动作头，在线表示并未更好，不能宣称换读出就解决。原始 `analysis/ema-current-coordinate-gauge-first-hour.json`。
- 观察映射试验解码确定状态均值，Gaussian rate作正则。纯F恒等/绝对版本2h为79.56/84.60%、48.26/37.28mm、final均0；恒等组多一次D前向。两组保全完整2h及部分3h状态后退出，让卡给相对编码对照与完整RGB预测。状态相关F恒等1/2/3h78.00/85.88/88.63%、54.76/34.68/27.46mm、全0，粗指标在改善但仍未突破；保全完整3h及部分4h状态后退出，将卡给后果辨认。源 `code-state-observation-v1`；入口 `state-observation-pair-20261116` 和 `flexible-state-observation-pair-20261116`。**flexible-absolute仅准备，未启动**，需重新选空出的卡位。
- **相对状态编码已有早期对照。** 观察映射只读`zh-z0`、零变化基准0，已知当前RGB的entry rate为0。dev13版本用QR学习Q、满足QᵀQ=I，固定动作潜空间单位；dev15版本A自由学习，作同384参数头的对照。无EMA或新增loss系数，a_t仅为正向条件。各4废弃更新梯度正常、原点平移下loss/fm/rate误差0；正交版本更新后Q误差2.4e-7。正交版本1/2h77.15/77.62%、52.57/51.65mm、全0，平台后保全完整2h及部分3h状态退出；自由版本1/2/3h74.78/79.31/80.69%、57.18/42.05/38.31mm、全0；保全完整3h及部分4h后将卡交给二元单步后果似然。尚无性能突破。源`code-isometric-relative-v1`/`code-relative-state-v1`，入口`isometric-relative-state-20261116`/`relative-state-coding-20261116`；自由版本未启动前重分配到dev15 0–3，原功能检查主机仍是dev13，未改旧证明。
- **完整RGB预测＋正交动作基已收尾。** 已核实历史最佳长训参照没有RGB直通，而RGB直通分支早期较快、6h反而86.57%/33.97mm，弱于无直通参照91.35%/17.78mm。新分支恢复从学习的当前E特征预测完整未来RGB、保留原Gaussian码噪声/rate/其余loss，仅改为纯平移与学习正交动作基；冻结E均值头2304参数，和384码头分开报告。1/2h75.91/77.90%、56.06/50.19mm、全0，尚无突破；保全完整2h及部分3h状态后让卡给后果似然唯一目标。源`code-isometric-relative-v1`，入口`dense-isometric-action-20261116`。
- **两组后果辨认已启动。** 分别沿正交相对模型及原状态相关F恒等观察模型，只替换Gaussian未来码率：预测码从同一相机流的8个自然观测码中辨认实际后果，距离温度沿用sigma=.1、beta=.25及原归一化，不新增权重、网络或跨样本候选。正交相对1h的预测码误差/copy为1.93，自然编码在预测动作方向的投影幅度比仅.143，直接已知动作基读出也弱于原头，故不扩展直接几何头。新似然的手工公式、原点不变、样本隔离、塌缩状态非零编码梯度及4废弃更新通过；这不是性能证据。原状态相关F版本保留原Gaussian初始码率，未约束动作正交。正交组完整3h81.11%/39.85mm、final0，保全完整3h及部分4h后退出，让卡给全局自然参数；状态相关组1/2/3h83.90/89.00/89.72%、38.67/24.11/25.42mm、final全0，第三点终距回退，继续长训，尚非稳定精细scaling。相应Gaussian78.00/85.88/88.63%、54.76/34.68/27.46mm，前两预算四厨房均改善。后果检索更准而Euclidean码预测更差，不能称世界状态预测校准；详见CONCLUSIONS。两组4功能更新均废弃，全链随机、祖先0，同384参数detach头和固定标签/64DEV/32封闭确认。源`code-trajectory-likelihood-v1`/`code-trajectory-flexible-v1`，入口`trajectory-consequence-likelihood-20261117`/`trajectory-flexible-likelihood-20261117`；正常入口保留8/16恢复，不重复CPU审计。
- **后果似然唯一目标已完整3h收尾。** 同正交相对模型的E/pool/Q随机初值、同8帧样本内候选、sigma=.1，但以categorical NLL均值作为唯一loss，不用像素重构或辅助码率/权重。随机D保留在检查点中但冻结且从不执行；22.220M可训练/40.834M总参数，384参数detach头。4废弃更新确认E/pool/Q梯度、D零调用/零梯度，完整1/2/3h82.36/80.54/79.70%、37.94/44.51/46.38mm、严格均0/64，连续退化，已保全完整3h及部分4h后退出，将卡交给AnchorReader；同初始NN的Gaussian相对对照2h77.62%/51.65mm，更新24691对21983、后者更少，仍非稳定精细scaling。源`code-likelihood-only-v1`及核心`consequence_likelihood.py`，入口`consequence-likelihood-only-20261117`。三条初期似然分支首小时均已收齐，按查询/状态语义区别启动两个单步版本。
- **两个单步后果似然版本已在dev15启动。** 4–7原状态相关F只从本次已观测当前E(x_t)及实际a_t查询下一观测，8候选，23.036M可训练；0–3用Q a_t固定单位、只比较当前/下一帧，22.220M，二元式见CONCLUSIONS。无像素/辅助码率/新增权重或动作target，随机D冻结不执行。各4更新验证预测输入均1当前状态/1当前动作且不反馈查询、E/pool/动作/F有效梯度、D零调用；二元式另验解析最优与样本隔离。状态相关1/2h82.25/82.13%、41.21/48.80mm、final均0；保全完整2h及部分3h后退出，让卡给单位nats AnchorReader。二元1/2h85.00/86.14%、27.89/26.46mm、final1/2（各64DEV），继续第三小时；已实际保留8/恢复16，不重复CPU审计。源`code-likelihood-factual-v1`/`code-likelihood-pair-v1`，入口`likelihood-factual-20261117`/`likelihood-pair-20261117`。
- **新增两项结构对照，均从零且无像素项/动作target。** dev6 4–7保留同Q/E及8候选，仅把滚动查询改为逐次已观测当前状态，区分单步与二元候选的作用；复用不可变`code-likelihood-pair-v1`，入口`likelihood-factual-q-20261117`。dev13 0–3用q=g(E_current)+Q a_t，动作线性进入条件分布自然参数，学习上下文g而非假定物理均值恒定；仅多启用已有24832个上下文参数。样本内8候选NLL、同sigma和384参数detach头，源`code-global-natural-query-v1`，入口`global-natural-query-20261117`。各4废弃更新通过、D零调用、输入仅当前1状态/1动作；新全局版本验证初始函数同纯Q及混合log-odds恒等式。两控制器已启动，正常8/16入口待实际检查，不据功能通过宣称有效。
- 宽E/浅D参照完整16h89.46%/27.26mm/2final，已保留额外部分17h全状态后退出；历史6h91.35%/17.78mm，后段未稳定保留收益。全局A＋漂移完整3h79.73%/52.43mm、绝对视角对齐5h86.08%/32.81mm、变化量对齐4h85.68%/35.38mm等已收尾。全部旧结果及完整/部分状态保留，后继均继承0更新；原始退休记录在MASTER，详情见 [CONCLUSIONS](experiments/CONCLUSIONS.md)。

- **Lingjing新方案已接入当前Reach3D机制检验。** 来自`6f9cadfa`/`cd03863b`的AnchorReader及SPT适配器按原样引入；自己的随机小CNN读取初始RGB，去噪器在detach码上学噪声，后果学习器只接收detach score给出的条件熵梯度及原Gaussian码率，不使用像素目标或动作target。beta_nats=.3和1是两个预设原始nats权重，不能等同原pixel beta=.25；reader另组lr5e-4、weight=.01，不用跨样本统计初始化。22项合同测试和4废弃GPU更新通过；E/pool/F/action/reader梯度有效、D零调用。25.165M可训练/43.754M总参数（含未执行冻结D）、reader2.129M、384参数当前a_t detach头；祖先0、固定12006日志和原标签/64DEV/32封闭确认。dev6 0–3的.3组已实际8/16恢复并主训；dev15 4–7的1组也已实际8/16恢复并主训。两者初始完整NN SHA相同，不是独立种子，均暂无控制成绩；另准备同检查点2304参数均值头作有限归因对照，未启动。此为直接迁移实验，未完成作者PushT门或宣称完整PartP/PartR复现；评测handoff的未来动作chunk监督不符合当前约束，不采用。两组复用不可变`code-anchor-reader-v1`，入口`anchor-reader-20261117`和`anchor-reader-exact-nats-20261117`。

## 固定边界

固定12006普通TRAIN日志，动作头仅用17920训练/1536验证的同一事实转移当前输入a_t；表示冻结/detach，动作标签梯度不进入后果学习器。输入仅RGB历史和实际执行动作，自然未来RGB/自身编码可作目标，目标图只用于控制；无未来/最优动作、专家路径、reward/state/taskID/语言或外部强教师。同步双相机属于一个物理样本，独立样本不交换表示/统计。

同64反复使用的DEV、双224 RGB、最多5命令/norm .3/native2mm final与any；32确认仍封闭。不同读出、容量、数据、教师来源、完整祖先与累计预算分别报告；同随机初值分支不是独立种子。保护性评测/oracle/接触/闭环输入不进入学习或proposal memory；不能把预测/验证误差下降当控制提升。

正常检查点保存model/optimizer/scheduler/各rank RNG/数据游标及EMA状态，实际8/16续训检查保留；不重复独立CPU恢复或广泛兼容核验，有实际错误才修。新源码用freeze_with_overrides，不修改已启动冻结源、旧检查点或vendored runtime，不把额外部分小时报作完整小时。

Git仅核心代码、必要配置、精简结论及已验证较优配置；原始实验和监控留EFS，不新增重复状态文件、不绕过ignore。未有新best-config晋升。流程边界见 [PROTOCOL](experiments/robocasa_stable_scaling/PROTOCOL.md)，原始归档见 [说明](docs/experiment-archives.md)。
