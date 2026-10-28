# SPT-JEPA十个候选：验证进行中

- [机制假设、可证伪实验与一手来源](IDEAS_DRAFT.md)
- [结构化候选](ideas-draft.json)
- [数学前提检查](PRECHECK_RESULTS.md)
- [追加实验状态](../completion20261028/README.md)

用户已明确授权补完原实验并验证十候选，dev2/dev13/dev15无新一轮时间上限。权威文件和运行在EFS；旧草案里尚待预算回复的叙述已失效。

基线和候选使用独立、已提交且不可变的source/config快照。已进行200步GPU执行检查并启动相同步数长跑；I08语义记忆修复后的GPU重试已通过，I09配有相同fork数据但零辅助权重的对照，I10冻结模型的三评估种子对比已完成；原Gramian缓存矩签名存在平移碰撞，正在用完整context相等检查重跑，旧数值不用于科学判定。

I01保留log-SNR加权surrogate控制，并增加含离散重建/噪声先验边界的64步Gaussian负ELBO候选，不能把bound叫精确NLL。I02当前先用训练参考decoder估计对角Q。I04保留framewise分位数控制，并增加只看anchor颜色极值的点式canonicalizer。新实现及CPU检查见[概率目标与实际差异](../../docs/lingjing/research-density-objective.md)，GPUworld-model收益仍待验证。

world-model收益尚在验证，方法有效性需看相同数据、预算、case和训练种子下的生成、动作跟随、规划与成本。数学检查和训练不崩都不足以宣布方法有效。
