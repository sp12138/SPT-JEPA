# 一epoch短线筛选已完成

[完整结果与关键图](results/SHORT-SCREEN.md)、[筛选决定](results/DECISION.md)。没有明确规划赢家；暂停单独I03/I05，只保留降低beta与组合方向做有上限的第二阶段。

以下为执行前协议。

---

# Short-budget method screen

The user corrected the long16run matrix: inspect short-run evidence before deciding whether to continue. All16 old methods/seeds were gracefully paused around9.6-10.3k updates with finite full optimizer/scheduler/RNG/data cursor checkpoints. No long continuation is automatic.

Only diagnostic training seeds1(high original performer) and2(weaker original performer) of the four methods resume to exactly12491updates, the first epoch, while keeping the124910-step LR schedule. Compare each to the matching original beta=.25 checkpoint already saved at12491updates. Other seeds stay paused.

P0 uses two paired evaluation seeds(42/43),50episodes each; generation64cases,one sample,50Eulersteps; counterfactual16cases,50steps. These are diagnostic screens, not final benchmark confirmation. See [frozen protocol](protocol.json). Continue only evidence-supported directions; publish all informative outcomes.

[早期健康检查](EARLY-HEALTH.md)：最新短线权重均未坍塌，但该结果不是规划收益。
