# Wan 2.2 CD 接入说明

本文保留 Wan CD 的接入背景与模型约束；公共职责和训练流程见
[CD 架构设计](cd-architecture.md)，不再重复实施阶段、代码草案或逐次实验记录。

## 接入方式

Wan CD 通过 `CdFamilySpec` 注册到 Native CD recipe，复用现有 Wan 模型、离线缓存、
条件处理和 teacher-forcing 接口。三个角色独立装配，student 为唯一 optimizer target。
外部 Taichu 仅作为数值参考，不成为 VForge 的运行时依赖。

配置分别为 `wan22_ti2v_5b_cd` 和 `wan22_a14b_cd`。模型权重、condition streams、
block 几何与并行约束在 family preflight 中校验；算法只接收统一的 batch 和预测接口。

## 5B：首帧锚定

- 使用 `grid/normal` 条件，首帧独立成块。
- 加噪及 teacher 推进后恢复首帧 clean latent，首帧 noisy timestep 为零。
- teacher、student、EMA 使用一致的 clean/noisy teacher-forcing 布局；loss 排除首帧。
- 三个角色从同一组初始化权重开始，student 的 FP32 EMA shadow 独立保存。

## A14B：双专家

- 使用 `RGB/grid/normal` 条件，不采用 5B 的首帧 inpaint 规则，首帧参与 loss。
- 偶数步训练高噪声专家，奇数步训练低噪声专家；相邻时间点必须落在同一噪声区间。
- 单个 optimizer 持有两个 student 专家，未参与计算的专家不推进 Adam 状态。
- 高低专家分别维护 EMA，仅在对应专家完成 optimizer 更新后推进其 EMA。

## 与公共实现的衔接

数据侧负责缓存规范化、latent 截取与 USP 组内广播；adapter 负责条件拼接和时间点
布局，调用现有 `forward_teacher_forcing`。按帧 timestep projection 和 attention
分片由模型及公共 USP 实现承担，不在 CD 内复制。`schedule.py` 通过 family 的
`CdSchedule` 接口提供 Wan 噪声规约；算法保留 loss 与梯度裁剪适配。

训练默认值和允许的并行规模以 experiment 与 `validate_wan_cd` 为准。验收使用的
短帧、空间裁剪或两层专家是实验设置，不改变完整模型的生产配置。

## 验收边界

首步 reduce 前梯度一致、短程 loss 一致、完整训练状态一致是不同结论。
归约顺序和随机轨迹可能影响多步误差，不能将单个 seed 的 bitwise 结果推广到任意训练。
真实 GPU 的 full DCP 中途恢复、A14B 完整深度和长程收敛仍需独立验收。

参考补丁、精度协议及历史数值证据保留在
`/efs/weilunfeng/vforge/validation/taichu-cd/protocol-0916/`；本文不维护逐次实验日志。
