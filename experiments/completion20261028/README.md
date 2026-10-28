# 追加实验：原始方案补齐与十候选验证

用户已撤销新一轮12小时限制，授权dev2/dev13/dev15持续完成未结束任务和十候选验证。权威运行状态在EFS的 `runs/completion_20261028/plan.json`、`workers/`、`exits/`；恢复入口见[恢复说明](../../RECOVERY_RESEARCH_20261028.md)。本文是执行进度，不能当成全部任务完成声明。

- 正在补齐：轻量L2成功配置seed1/2、L0/L1匹配三训练种子、g0和128维状态对照、NativeS2低beta筛选及S0/S1额外种子、原大renderer余下步数。
- 全数据：Lance固定revision的2,336,736行、18,685episode、934testepisode已保存；像素/episode拆分核验通过。W8每epoch12,491步，10epoch124,910步。原尺寸L2/L0已通过200步GPU训练smoke；全beta网格先跑37,473步，之后按真实含噪后验/anchor和active-dim规则筛选。
- 终点评估：50k训练帧/5k测试帧probe、逐维rate、256例mean/sample生成、LPIPS/SSIM/PSNR/FVD、角度、64例动作反事实、标准GoalMSE规划和32例gallery。内部子任务错误必须显式判失败，外层退出0不算完成证据。
- 新方法与原方案分开报告。修改后的噪声、目标、优化器梯度、canonicalizer、fork训练数据和语义记忆路径均不是原AS-JEPA同一协议。
- 仍待完成：完整预算选择、最终配置额外种子、条件继续训练/消融、正式人工gallery签收。不得用smoke、旧单种子或AI视觉审阅代替。

已有额外结果将保存于本目录；完整原始媒体、权重和数组留在EFS。原12小时结果目录保持不变。

已发布追加材料：[完整数值、科学图和固定32例总览](results/README.md)，[25k基线的三训练种子对比](results/BASELINES_25K.md)，[冻结配置和规划队列快照](configs/planned-jobs.json)。这些材料不代表全部训练已完成。

自动后续使用原规格的选择次序：排除坍塌；generation score在最佳10%以内；L2在tie中用P0的10百分点窗口；再选最低heldout rate。网格边界按4倍扩展；全部价格坍塌才启用前5%更新线性beta的原规格fallback。最终种子、g0、128维及20epoch任务使用独立、冻结目录，只有对应的科学条件满足才解锁。Dormant条目只是条件配置，不会自动消耗全部网格的多种子预算。人工gallery签收仍须另行完成，自动判据不代替它。

媒体补齐工具使用原始冻结RGB银行生成fixed32、best8、worst8，附全部动作/误差/seed及review.csv；规划直接索引已保存的五个成功/失败视频。反事实与anchor-swap分别有8例的完整视频和元数据。新增GPU媒体入口须先通过独立smoke；所有视频留在EFS，人工review状态保持pending。

[三训练种子50k完整结果](results/L2-50K-STABILITY.md)已齐：mean h7 agent/copy=.411，但动作识别平均74.22%低于75%阈值，P0=46.67±29.87%。不能把单种子正面结果当作稳定解决；原始完整预算和方法验证继续。
