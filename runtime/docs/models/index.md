# 支持模型

本页汇总当前文档覆盖的模型、experiment 和成熟度。代码注册不等于公开支持；范围外
模型不在这里作状态声明。

!!! tip "第一次使用"
    尚未跑通任何任务时，先完成 [Quick Start](../start/index.md)；确认环境可用后再从
    本页选择正式模型流程。

## 支持矩阵

| 模型 / extra | `experiment` | 状态 | 文档 |
| --- | --- | --- | --- |
| RepWorld / VideoRAE、V-RAE | `rep_world`、`rep_world_1d`、`rep_world_eupe`、`rep_world_vjepa` | **内部验证**；在线 train/eval、已完成的 VideoRAE3D latent shard 训练、Native DCP，H200 单机与 B300 两节点指定并行组合已验证；未做生产验收 | [RepWorld](rep-world.md) |
| Gen Engine / base | `gen_engine` | **内部验证** | [Gen Engine](gen-engine.md) |
| Gen Engine MMDiT MoE 32B / 4096 维文本、裸 Tensor | `gen_engine_mmdit_moe32b` | **内部验证**；受限 synthetic GPU Replay | [Gen Engine](gen-engine.md#mmdit-moe32b-pt)；真实数据训练需提供已确认的 encoder codec |
| Wan 2.1 I2V 14B / `wan` | `wan21_i2v_14b` | **稳定** | [Wan 2.1](wan21.md) |
| MiniMax H3 FL2VA / `h3` | `minimax_h3_fl2va` | **预览** | [MiniMax H3](minimax-h3.md)；先读[数据与 Cache](minimax-h3-data.md) |
| LTX-2.3 Stage 1 TF | `ltx23_causal_ar_tf` | **已支持** | [LTX-2.3 Stage 1 Teacher Forcing](ltx23.md) |
| LTX-2.3 DMD 推理 | `ltx23_dmd_eval` | **已支持** | [LTX-2.3](ltx23.md) |
| LTX-2.3 DMD | `ltx23_dmd` | **已支持** | [LTX-2.3 DMD](ltx23.md)；支持 Ulysses，使用方法与数值对齐记录见说明 |
| Wan 2.2 A14B / `wan` | `wan22_a14b` | **内部验证** | [Wan 2.2 内部验证说明](wan22.md) |
| Wan 2.2 TI2V 5B / `wan` | `wan22_ti2v_5b` | **内部验证** | [Wan 2.2 内部验证说明](wan22.md) |
| Wan 2.2 TI2V 5B TF / `wan` | `wan22_ti2v_5b_tf` | **内部验证** | [Wan 2.2 内部验证说明](wan22.md)；适配步骤见[Teacher Forcing](../development/adapting-teacher-forcing.md) |
| Wan Taichu variants / `wan_taichu` | `wan_taichu_wan21_i2v_14b`、`wan_taichu_wan21_i2v_14b_audio`、`wan_taichu_wan22_a14b`、`wan_taichu_wan22_ti2v_5b`、`wan_taichu_wan22_ti2v_5b_ref`、`wan_taichu_wan22_ti2v_5b_causal_teacher` | **内部验证** | [Wan Taichu 模型指南](wan-taichu.md) |

## 状态含义

- **稳定**：有完整用户流程、持续 CPU 回归和真实 GPU 闭环验收；
- **预览**：有可运行指南和模型专属 gate，但范围或接口仍可能调整；
- **内部验证**：代码和测试存在，尚不作为普通用户默认工作流。

## 选择建议

- 第一次使用选择 `wan21_i2v_14b`；
- H3 FL2VA 用户按专用指南准备已验收的模型版本和 offline cache；
- LTX-2.3 常规训练入口支持 Stage 1 causal teacher forcing；TI2AV SFT 待支持；
- LTX DMD 支持 Native FSDP2 训练和 Ulysses；启动命令、CI 与数值对齐记录见模型指南；
- Wan22 目前面向框架开发和内部验证，不承诺完整用户支持；
- Wan Taichu 当前开放普通双向、Audio、ref 和 TI2V causal-teacher train preset；配置
  `grid` stream 的在线训练可从 `c2ws/Ks` 现场生成 Gameplay grid；CLIP、DINO 和
  added-KV 只保持 Wan21 范围；
- 新模型统一实现为 Native task/model 插件，不新增 compat 接入。

注册表的代码真源是
[`vforge/tasks/registry.py`](https://git.woa.com/ieg-infra/wmtrain/blob/master/vforge/tasks/registry.py)，
配置真源是 `vforge/conf/experiment/*.yaml`。
