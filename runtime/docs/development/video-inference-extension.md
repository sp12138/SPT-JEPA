# 视频推理扩展边界

新增 eval 的唯一最小、可复制示例见
[新增 Native 模型](adding-native-model.md#evalcomponents)。本页不再
维护第二份接入步骤，只解释 prepared sample、flow-match 和 incremental KV 三组可选
helper。模型不满足对应假设时，只实现 `EvalComponents` 与
`infer_sample(sample, *, seed)` 即可。

## 可选 prepared-sample helper

`vforge.tasks.prepared_eval` 适合“单 latent、单样本 batch、leader 负责读取和编码”的
视频模型：

- `PreparedEvalShape`：输出像素高度、宽度和帧数；
- `PreparedEvalBatch`：encoded bundle、显式 shape 和 sample context；
- `build_prepared_eval_loader()`：调用模型提供的
  `prepare_batch(samples, source_kind, device)`，规范化后广播；
- `move_prepared_eval_tensors()`：按需迁移顶层 tensor。

它不创建 scheduler，不编码 negative prompt，也不拥有 pipeline。模型不使用 prepared
sample 时无需接入。

官方 Wan 与 WanTaichu 可以复用这组中立 helper，但两者的 task assembler 不共享：
`vforge.tasks.wan.*` 和 `vforge.tasks.wan_taichu.*` 分别构造自己的 model、codec、
condition encoder、dataloader、inference 与 output writer，互不 import 对方 task。

## 可选 flow-match helper

`vforge.models.video_diffusion.flow_match` 为单 latent、单 scheduler 的视频 flow-match
模型提供 scheduler traversal、CFG 顺序、guidance 合成和可选 prefix latent 重投影。
两个独立 Wan 实现还可通过 `vforge.models.video_diffusion.wan` 复用 Wan asset/latent
约定：

```python
run_flow_match_denoising(
    *,
    scheduler,
    latents,
    num_inference_steps,
    conditioning,
    predict,
    guidance_scale=1.0,
    negative_conditioning=None,
    frozen_prefix_latents=None,
) -> torch.Tensor
```

唯一模型扩展点是：

```python
predict(current_latents, scheduler_timestep, conditioning, cache_name) -> prediction
```

公共循环不拥有以下模型语义：

- condition channel、prompt/ref/audio 参数；
- scalar、batch 或 per-token timestep；
- `cache_context` 以及 cond/uncond cache 生命周期；
- `engine_forward` 调用和模型输出解包；
- dual-expert 路由、first-frame latent、decode 和 output leader。

官方 Wan 2.2 teacher-forcing causal eval 不使用该双向 sampler；它保留 dedicated
generated-history block rollout。WanTaichu causal pipeline 同样在自己的模型/task 中
装配 block loop。

MiniMax H3 的 packed video/audio rows、双 scheduler 和无 CFG 语义不满足单 latent
contract，因此不接该 helper。

## 可选 incremental KV helper

`vforge.models.causal` 提供：

- `CausalKVCache`；
- `IncrementalKVAttentionContext`；
- `IncrementalKVExecutor`；
- `allocate_incremental_kv_guidance_caches()`。

`IncrementalKVExecutor.transformer` 是 evaluation engine 已持有的 transformer 引用，
executor 本身不是 `nn.Module`。模型注入 `block_forward`，公共 executor 负责能力校验、
cache geometry 和完整 clip RoPE；CFG 的 conditional/unconditional cache 必须隔离。

patch embedding、condition/timestep embedding、block layout、scheduler 重置、clean
history re-pass 和 output projection 仍属于模型。bidirectional 或不支持 incremental
KV 的模型无需实现这些接口。

## 输出 ownership

所有模型都只能在 `get_parallel_state().is_batch_leader` 上 decode 并返回可写 artifact。
其它 TP/CP/USP rank 仍参与 `engine_forward`，但不写输出。不能只用 `usp_rank == 0`
判断 ownership，因为组合并行时可能产生多个解码者。
