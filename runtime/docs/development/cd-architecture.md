# VForge CD 架构设计

## 设计目标

CD（Causal Consistency Distillation）是多模型训练 recipe，不是模型 variant。
`training.recipe=cd` 声明 teacher、student、ema_model 三个独立角色；teacher 和
EMA target 冻结，只有 student 接收梯度。CD 与 DMD 复用公开训练设施，不依赖
DMD 的私有 bundle、adapter 或 rollout 实现。

## 职责边界

- **算法层 `algorithms/cd/`**：定义配置校验、训练步骤和一致性 loss，装配唯一 student
  phase；loss 与 CD 裁剪适配直接位于 `algorithm.py`。
- **模型组合 `RoleModelBundle`**：注册角色、恢复冻结状态、提供独立 FSDP execution
  roots 和统一 checkpoint 根节点，不拥有算法数学或 optimizer 循环。
- **模型家族 `tasks/wan_taichu/cd/`**：`integration.py` 校验并构建角色；`data.py`
  复用 Wan 缓存和数据流，产生 `PreparedCdBatch`；`adapter.py` 将统一输入映射到
  模型公开的 `forward_teacher_forcing`，不复制 Transformer forward。`schedule.py`
  提供 Wan 时间点采样、加噪、Euler 推进和 x0 转换，保留其时间单位及计算精度。
- **公共训练层**：PhasedTrainer 执行 backward、裁剪、optimizer、scheduler、更新后
  hook 和 checkpoint；FSDP2、USP 与数据恢复继续使用现有实现。

入口由 `tasks/cd/train.py` 解析算法与模型家族，先完成 preflight，再延迟构建模型和数据。
配置入口是 `wan22_ti2v_5b_cd` 与 `wan22_a14b_cd` 两个 experiment。

## Schedule 接口

参考 DMD 的 family 回调模式，`CdFamilySpec.create_schedule` 构建 `CdSchedule`，
由 `CdFamilyComponents` 传给算法。接口包含 `sample_pair`、`add_noise`、
`teacher_step`、`velocity_to_x0`；`CdNoisePair` 只携带相邻时间点及 sigma。
算法提供持续随机流和高低专家选择，task 实现不持有或重置 generator。算法层不直接
导入 Wan schedule，采样后再生成噪声的顺序保持不变。

## 单步流程

1. 准备同一批 clean latent、文本和条件，采样相邻时间点与噪声。
2. 冻结 teacher 预测 velocity，以一步 Euler 得到下一时间点的 noisy latent。
3. student 在当前时间点预测 x0，EMA target 在下一时间点预测 x0。
4. 计算一致性 MSE；只有 student 分支反向传播。
5. 公共 trainer 完成梯度归约、裁剪和 optimizer 更新，随后更新 FP32 EMA shadow。

teacher 和 EMA forward 共同构造 student loss，不拆成额外的训练 phase。
EMA shadow 是训练状态，不是第四个执行 forward 的模型；后续前向前同步到 target。

## 随机流、精度与状态

- pair/noise 使用公共 `RandomStreams` 的持续 `cd.student` 流，数据随机流独立管理；
  不按 optimizer step 重新设种子，恢复沿用公共 RNG 状态协议。
- 模型精度边界由模型与公共并行配置管理；CD 不在 trainer 外复制混合精度逻辑。
  Wan CD experiment 通过 `parallel.mixed_precision.preserve_forward_input_paths`
  显式保留 `blocks.*` 的输入 dtype，追加到公共模型契约；该配置不改变 DMD 等任务的默认边界。
  x0 MSE 按 prediction dtype 归约，避免外层 autocast 改变目标函数的计算精度。
- CD 通过 `gradient_clipper_factory` 为当前 student 注册 FP32 逻辑范数裁剪，
  沿用 trainer 按参数身份建立的贡献范围，在 student 子模型上计算并裁剪梯度。
- EMA 使用公共 `ShardedEmaState`；分片 student 与复制 target 之间通过 DTensor
  布局转换同步。模型、EMA、optimizer、scheduler 和数据状态交由公共 full DCP 管理。

## Wan 接入与扩展边界

5B 的首帧锚点、A14B 的专家选择及条件布局见
[Wan CD 接入说明](wan22-5b-cd-extension-plan.md)。当前要求 offline latent cache 与
FSDP2，不包含 student self-rollout、KV-cache 链或 CD/DMD 联合训练。

新增模型家族时实现 `CdFamilySpec` 的模型构建、数据支持和 teacher-forcing adapter，
保留算法层与模型布局的隔离。新增算法时通过 CD 算法注册入口扩展，不复制 trainer。
通用 checkpoint 入口不代表 CD 的真实 GPU 中途恢复已通过验收。

使用配置见 [CD 训练](../features/cd-training.md)。实验补丁与数值证据在仓库外维护：
`/efs/weilunfeng/vforge/validation/taichu-cd/protocol-0916/`。
