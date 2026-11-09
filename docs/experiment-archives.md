# 实验记录与 Git 边界

Git 只维护核心源码、必要配置、精简结论与 [较优配置](../experiments/best-configs.json)。逐回合结果、运行日志、监控快照、临时审计、调度状态和重复源码不纳入版本管理；`.gitignore` 已统一约束。

2026-11-09清理只移除索引记录，原始文件仍保留在工作目录；远端数据、冻结源码、权重、检查点和错误记录没有修改。清理前23,366个索引文件已完整备份并逐文件SHA256校验，其中包含尚未提交的暂存内容。

- 本地备份：仓库旁的 `../SPT-JEPA-local-archives/git-cleanup-20261109T081109Z/`。
- 完整包：`tracked-files.tar.gz`，132,134,583 bytes，SHA256 `64ebd94feaedef6100001996c83d4f0544e5ebb4fb5596f7509219892629afc4`。
- 同目录的 `manifest.json`、`index.before` 和两个patch保留文件摘要及清理前工作区／索引状态。
- 清理前提交：`5d4405bbaca1f6964413b01843cf9672097a33aa`。既有Git历史未改写；已提交的历史文件也可按该版本恢复。

需要历史细节时，从归档解出对应路径供本地分析，不把整批文件重新加入Git。历史实验脚本可能需要这些外部配置／数据才能重放；不因恢复文件而启动训练。

共享实验存储为 `/efs/prescottsun/SPT-JEPA`，通过已授权机器读取。主要索引：

| 内容 | 存储中的相对位置 |
| --- | --- |
| 较优PushT模型与原配方 | `runs/promising_20261029/temporal-forecast-seed{1,2}-selected-epoch3/` |
| TwoRoom local1024模型 | `runs/local_isotropic6000_20261031/tworoom-spatial-local-seed{1,2}/` |
| Cube成熟模型 | `runs/cube_maturity23342_20261031/cube-temporal-forecast-seed{1,2}/` |
| MetaWorld Reach／ReachWall历史评测 | `runs/metaworld_proposal_20261104_v3/` |
| 原生MetaWorld基线及续训结果 | `runs/metaworld_baselines_native_20261108/`、`runs/metaworld_baseline_scaling_20261108/` |
| 最新已结束的causal-inverse研究 | `runs/metaworld_causal_inverse_20261109/` |
| 当前RoboCasa三路线及官方冻结参考对照 | `runs/robocasa_compare_20261110/`；`pipeline/`为队列状态，`results/summary.json`为有限收集结果 |
| 最近完整元数据快照 | `reports/metaworld_persistent_world_20261109/hour11-evidence.tar.gz` |

最后一个元数据包SHA256为 `2de5f54b26381404b5b4c7b615b4f0d298099426781df73a7e3893293df9ca60`，其中5,420个内容文件已在本地核验。它是逐文件快照，并非全局原子状态；最新终态以对应run的 `CLOSED.json` 和 `results/summary.json` 为准。

清理未重写历史，因此现有 `.git` 对象的磁盘体积不会按移出的文件数同步缩小。目标是让当前版本和后续提交保持精简。
