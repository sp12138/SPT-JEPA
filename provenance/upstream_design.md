# RGB16 架构 scaling 实验

## 已确定的基线与本轮训练

- 单条 RGB diffusion 路线，内部为 patch DiT + pixel refiner。
- Split ViT：E 3 层、C 18 层，width 1024；输出 condition 长度沿用原实现。
- Patch DiT：8 层，width 768，12 heads；pixel refiner 2 层，width 16。
- 沿用既有 +70M 的 CA 设置：第 0、4 层执行 CA。
- 不使用 REPA；mask 位于 E 前。
- 精确可训练参数量 423,579,328；E 42,143,308，C 233,962,952，DiT 147,473,068。
- 从零连续 pretrain 到 50k，mask ratio 0.9，distant target probability 0.5。
- Global batch 64，8×H200，BF16，FA4，关闭 activation checkpointing。
- 继承原优化器、学习率和数据设置，constant LR，不在 10k 边界重新 warmup。

## Training budget 实验

```text
随机初始化 → pretrain 10k → 20k → 30k → 40k → 50k
                      │      │      │      │      │
                      └2k    └2k    └2k    └2k    └2k tuning
                       │      │      │      │      │
                      FVD    FVD    FVD    FVD    FVD
```

每个 tuning 分支独立输出，完整恢复其源 checkpoint 的权重、optimizer、EMA、训练状态与数据游标。只切换 mask ratio=0 和 distant target probability=0，不把 tuning 后权重接回主线。

主线每 2500 step 保存，tuning 每 500 step 保存。主指标固定为每条分支 tuning 满 2000 step 后的 EMA FVD-2048，不以不同分支的不同 tuning step 择优作为主比较。完整保存中间 checkpoint，方便之后进行相同规则的附加评估。

评估沿用既有固定 2048 case 和输入 bank：16 帧历史、128 帧生成，50 Euler steps/state，CFG=1，seed=20260920，I3D400 FP32。协议的公共采样与检测设置保持一致；模型、源码和 checkpoint 身份单独记录。

报告包括：FVD 对 pretraining steps、实际训练 GPU-hours 的曲线；相邻 10k 的 FVD 改善与每 GPU-hour 收益；pretrain 和 tuning FM loss 分开；训练吞吐与 NaN/gradient 事件。20k、30k 等分支的全局 checkpoint step 是 22000、32000 等，展示时明确区分 pretraining steps 和 tuning steps。

## Predictor / renderer 假设的检验

当前 c_i 是历史表示，没有直接的未来预测监督。Patch DiT 接收 noisy future RGB、t、c_i，仍然参与未来内容和动态建模。Pixel refiner 更接近局部渲染器；整个 DiT 不能直接视为 LLM 的线性输出 head。

已有容量消融（原 REPA + E 后 mask 设置，按各自最佳 EMA FVD）：baseline 840.40，ViT C +70M 为 851.27，DiT +70M 为 630.36。C 扩容尚无明确收益；不据此把当前瓶颈归于 C，也不重复优先扩大 C。该实验只扩大 C，E 仍为 3 层。

历史 +70M 同时改变了 patch depth 4→8、pixel depth 1→2、CA 数量 1→2，因此其收益不能全部归于渲染能力。

容量实验立即使用 dev2，先跑 D-mid（patch 16 层）和 D-large（patch 24 层）；两组都从零 pretrain 10k，独立 tuning 2k，并测试固定 2k 端点的 EMA FVD-2048。与 dev1 基线的 10k + 2k 分支对比，不等待 50k 主线完成。其他容量组暂未启动。

按用户最新要求，所有组均为每卡 batch 8、gradient accumulation=1、global batch 64。D16 保留已正常运行的设置；D24 开启 Native activation checkpointing 解决显存需求，不使用梯度累积。训练计算与 GPU-hours 的比较会计入重计算开销。早期的 D24 microbatch 4 / accumulation 2 配置未启动，已由 v3 计划取代。

| 组 | E/C 层数 | Patch DiT 层数 | Pixel 层数 | 目的 |
|---|---|---|---|---|
| B | 3/18 | 8 | 2 | 本轮基线 |
| D-small | 3/18 | 4 | 2 | DiT 容量下降的影响 |
| D-mid（dev2 先跑） | 3/18 | 16 | 2 | DiT 扩展收益；503,723,232 参数 |
| D-large（dev2 后跑） | 3/18 | 24 | 2 | DiT 边际收益是否降低；583,867,136 参数 |
| Pixel-large | 3/18 | 8 | 4 | 单独扩大 pixel 渲染部分 |
| C-large（暂缓） | 3/30 | 8 | 2 | 只在 DiT 平台明确后检查条件容量 |
| C+D-large（暂缓） | 3/30 | 16 | 2 | 更大 C 是否重新提升 DiT 扩展收益 |

上述组保持宽度、head dim、condition token 数、数据和采样协议相同；CA 数量固定为 2，位于第 0 层和中间层（cross_attention_every=patch_depth/2），避免深度增加时自动增加 CA 数量。精确参数量与 FLOPs 在实施前按实际模型统计，不以层数代替计算量。

容量比较同时报告两种视角：同样训练样本量（同 batch/steps）与实际训练计算量。不能用同 steps 的结果宣称计算效率更高。若要严格等计算比较，应按实测训练 FLOPs/token 确定各组训练 steps，再给所有组相同 2k tuning，并把 tuning 计算量计入总账。

## 支持或反驳假设的结果

- 固定 E/C，扩大 DiT 后 FVD 改善逐渐变小，只支持该配置范围出现平台；不能证明理论下限。
- 在 DiT 平台附近，扩大 C 再带来明显改善，才更支持历史条件成为瓶颈。
- 若 pixel-only 扩展主要改善局部细节，而 C 扩展主要改善运动和 rollout drift，更支持功能分工。
- 若 patch DiT 扩展继续明显改善运动与长期一致性，则它仍承担较多动态预测任务。

辅助诊断使用相同 case 的 GT history 单步预测和正常 generated-history rollout，按未来时间段统计误差、运动一致性和 drift。GT history 只替换历史，不给未来 GT condition；未来 GT condition 的 privileged reconstruction 不能用于证明 forecasting 能力。

进一步的功能诊断可固定同一 checkpoint 的 E/C，训练不同容量 DiT，以保证所有组读取相同 condition 表示。该实验能检查表示是否足够供更大 DiT使用；即使更大 DiT 的结果改善，也不能单凭冻结结果认定它只做渲染，因为它仍能从固定历史表示中学习未来动态。

如要检查 c_i 是否容易读出未来信息，可在冻结 E/C 后训练轻量线性/小 MLP probe，预测一个固定参考 encoder 的未来 feature 或运动目标；所有模型共享同一参考 encoder、训练/验证划分和 probe 容量，并比较当前状态 persistence。Probe 本身不能很大，否则可能由 probe 承担预测。它只说明未来信息可读性，不能单独证明模块承担了全部预测。

有限 case 的 FVD 可做配对 bootstrap 估计评估样本不确定性；它不包含训练 seed 方差。接近平台且差异较小时，再对关键组复跑训练 seed。

## 执行方式

现有 H200-2D dev1 的 8 卡负责连续 pretrain 50k；dev2 的 8 卡立即运行 D16、D24 容量实验，各自 pretrain 10k + tuning 2k + EMA FVD。

预算实验的五条 tuning/FVD 分支由 dev1 上的有限 dispatcher 在对应 checkpoint 完整保存、2D 配额满足 8 卡时提交到集群。若集群持续无空余配额，在 dev1 的 50k 主线完成且 GPU 空闲后使用 dev1 补齐尚未提交的分支，不中断主线。所有五条分支完成后 dispatcher 自动退出。没有创建 Codex heartbeat、cron 或定时监控。

远端报告：`/efs/stormyxie/reports/rgb16_scale_budget_20261010/`

远端输出：`/efs/stormyxie/outputs/vforge/rgb16_scale_budget_20261010/`

容量报告：`/efs/stormyxie/reports/rgb16_scale_capacity_20261010_v3/`

容量输出：`/efs/stormyxie/outputs/vforge/rgb16_scale_capacity_20261010_v3/`。D16 的预训练正在原输出目录 `/efs/stormyxie/outputs/vforge/rgb16_scale_capacity_20261010/d16_pretrain/` 连续运行，后续 tuning 和 FVD 使用 v3 目录。调度切换保留同一 D16 torchrun 进程，没有中断或重启训练。
