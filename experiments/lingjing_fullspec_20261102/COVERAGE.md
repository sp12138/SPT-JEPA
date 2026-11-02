# 逐问题覆盖，而不是把失败配方跑满

2026-11-02。精确作者版本：

- `lingjing/anchored-state-jepa`，`43962c526f825a8bbba28f424feddfada330910b`，round2 rev3。
- `lingjing/decoder-free-as-jepa`，`fbb3efa6a44fce8ca4eaa53b9c36eb2b962ffcf2`；它引用的 round2 文档是 rev2，未悄悄用 rev3 替换。

作者原文及逐文件 SHA 见 [upstream-provenance.json](upstream-provenance.json)、[upstream/](upstream/)。四个上传模型文件保持作者字节；本轮只补注册、评测和执行工具，不调 loss/LR/batch/架构。标准训练源固定 `0196dc17bf1f6d3823a2edd5509e1508d19bb8e8`；主队列评测/恢复 helper 固定 `e9281dde6d198215808921ac912234605805fa43`。

## 135 个主要配方的覆盖清单

| 提案块 | 配方数（含种子） | 要回答的问题 | 当前处置及预算 |
|---|---:|---|---|
| R2 D0 | 非新配方 | 成熟 prior 是否收缩、校准是否失效 | 已完成40个既存 PushT 导出的80个 BF16/FP32冻结诊断；另完成6个既存 Cube 导出的12个诊断 |
| R2 P1 | 3 | 对齐曝光的原生 L0 baseline | 三种子按30530→45795→152650；3/10epoch 都保留 raw 与固定 train-only BN 两套结果 |
| R2 P2 F1–F10 | 20 | horizon、forecast noise、λ、ρ 的配对因素 | 逐12491→24982，未校准的原配方；首epoch只在联合强失效时早停 |
| R2 P3 a/sham/b/c | 10 | 初始尺度、均值平移、噪声分辨率 | 逐12491→24982；sham 等同种子a的 `original_std`，不按结果挑尺度 |
| R2 P4 I2 | 1 | InfoNCE reconstruction-free 对照 | 逐12491→24982→37473；批耦合，非 sample-level 候选 |
| R2 T1–T9 | 18 | TwoRoom 初始化、anchor、价格、σ、forecast | **全部完成2000步门槛并失败，结束 Part T**；T3/T4复用严格匹配的既存检查点 |
| R2 K1–K6 | 12 | Cube anchor、价格、σ、目标变量 | **12/12完成2000步门槛并通过**；完整状态续到12000，所有配方/种子/控制器均保留 |
| DF A | 18 | β×λ 价格/预测平面 | 逐12491→24982→37473，首epoch联合强失效可提前结束 |
| DF B | 12 | sample/mean 与有效权重、open-loop | 同上；区分通道噪声和隐含预测权重 |
| DF C | 4 | prior input 读 mean 或 sample | 同上；不将 mean-input 对照悄悄选为 canonical |
| DF D | 6 | pooled/window/batch-conditioning、估计器 gaming | 同上；显式区分 sample-level window 与 batch-coupled controls |
| DF E | 8 | transient dot 的保留阈值 | 同上，seed1；读取 noisy nuisance probe，不靠标准化 clean probe 判定 |
| DF F | 5 | colour nuisance、keyed representation、L0正对照 | 4个DF至37473，colour L0至45795；固定512帧 invariance/recoloured-goal retrieval |
| DF G | 8 | 初始陷阱、尺度0.3、gap0 | 同DF主要预算；不在未校准2000步低spread时误杀 |
| DF H | 2 | σ不同但有效参数 π、w 对齐 | 同DF主要预算，比较 A 的 β.3/λ.5，参数不扫 |
| DF T | 8 | decoder-free能否救 TwoRoom | **全部完成2000步门槛并失败，不续6000**；canonical pooled 两种子复用既存匹配检查点 |

主要配方计数：Round2 `3+20+10+1+18+12=64`，DF `18+12+4+6+8+5+8+2+8=71`。合计135：**26个 TwoRoom 已按门槛结束，12个 Cube 进入成熟评测，3个匹配 L0 基线独立执行，其余94个 PushT 进入有限阶段队列。** 不是135个已跑满或已成功。

额外4条预先列出的条件诊断（不充当主要配方计数）：TwoRoom T4 β=0 ×2，DF初始化0.3 ×2，均完成2000步但未通过。14条200更新的原生技术 smoke 不当作名义任务完成；原冻结 FP32-pool repair 对照也单列。

## 评测合同与已知偏离

- 推理使用同一 heldout phase0 数据/原 CEM300×30、top30、history3、H5、block5/budget50；Cube 同一 primary/fresh50 与固定 train-only correlated-CEM，全部保留。σ/β改变不改变评测条件；不使用最优动作标签。
- PushT所有DF（含D/E/F/G/H）在2epoch执行42/43及short latent包，3epoch执行原42/43/44和fresh1042/1043/1044及full latent包。L2所有主要P2/P3在2epoch执行六个固定planning seeds、质量64×1×20Euler、CF16、decomposition128×50Euler、posterior/forecast BF16/FP32、rollout gain与probes。首epoch增加的联合失效判断是用户“insight优先”的实现，不是原文宣称的收敛定理。
- Cube12000包包括原生CPU完整恢复/最近梯度、状态使用、内容探针、固定128train/64validation的 xyz/对称姿态读出、white/correlated × primary/fresh。匹配2000步全状态先做实际复制前后 CPU model/AdamW/scheduler 恢复，随后继续原schedule，无二次校准。
- TwoRoom 已明确不使用状态的分支，按用户最新优先级省去其后6000步训练和重复的大规模2k规划；故这一部分完成的是 **提案门槛问题**，不是原文每一张规划/视频表。
- 旧3epoch导出缺失就报告缺失，不用新训练替换旧证据。D0来自既存导出，不能当成新同预算/同模型容量随机对照。
- NN默认 BF16 和 `fp32_pool=False` 保持作者 canonical；FP32科学判定优先。旧 `fp32_pool=True` 训练修复结果独立呈现。P1、colour L0 使用真正原模板且**不移植 encoder**；其他新配方只加载原始 RANDOM encoder，不加载已训练/预训练表征。
- 相同更新数不等于相同参数量、算力、收敛程度；colour/dot 训练后的P0仍按作者指定的原规划入口执行，与离线 nuisance 读出/变色检索的含义分开。
- L0保留作者原模板的window4，AS/DF为window8；这里的epoch是各自loader的epoch，不是相同frame-forward数或等FLOPs。P1不应被表述成已严格排除模型容量/训练算力差异的公开榜单复现。

## 条件路线，不按空想扩张

| 条件项 | 状态/触发器 |
|---|---|
| T4 β0、DF init.3 | 已完成，均失败，不再重复 |
| TwoRoom6000 | 当前没有任何配方满足双种子 gate，故不触发 |
| P2c：选中配方续3epoch，新增seed3/4至2epoch | 必须等20个cell与参考的全包结果，按作者排除/配对规则选择；当前未执行 |
| P2 prior-gradient fallback/低β新增seed | 仅在作者明确条件满足时审查；当前未执行 |
| K-mature23342 | 等12k matched控制结果和既存native成熟收益；不自动延长 |
| R2 Stage2 selected hard-task transfer | 等P2与K/T机制报告再选择；当前未执行 |
| DF Phase2 stability/final/post-hoc decoder | 等Phase1可行性、gaming和规划判断；若无合格候选即结束，不启动 |
| 任意新的architecture/feature/K/controller sweep | 未选择；不会靠填满GPU添加 |

上传文档中的“等我们同意”不是本次再向用户索取许可的理由；用户已授权补全有意义的提案。这里暂不触发后续条件路线，是因为其所需实验证据尚未产生。旧 goal 与LLM轮询保持暂停；阶段内由有限本地队列存档、评测、早停和结束汇总。
