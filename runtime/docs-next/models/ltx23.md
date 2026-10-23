# LTX2.3

LTX-2.3 在 vForge 中支持 SFT、TF、DMD 和 Long Tuning 训练，以及 DMD 独立推理。
训练通过 `vforge.tasks.train` 启动，推理通过 `vforge.tasks.eval` 启动：

- **[SFT / TF](ltx23/sft-tf.md)**：使用视频、音频和文本缓存进行监督训练。SFT 训练
  bidirectional 模型，TF 训练 causal 自回归模型，分别使用 `ltx23_ti2av_sft` 和
  `ltx23_causal_ar_tf`。
- **[DMD](ltx23/dmd.md)**：从 causal TF 权重初始化 generator，通过 teacher 和 critic
  蒸馏训练，使用 `ltx23_dmd`。
- **[Long Tuning](ltx23/long-tuning.md)**：从已训练的 DMD student 继续训练多段生成，
  段间保留前缀，并通过视频 VAE 重建下一段的首帧条件，使用 `ltx23_long_tuning`。

- **[DMD 推理](ltx23/eval.md)**：加载训练后的 generator，用缓存参考首帧和缓存文本或
  自定义 prompt 生成带音轨的视频，使用 `ltx23_dmd_eval`。

按对应教程准备权重、数据和启动命令。各类训练均支持 checkpoint 保存与恢复，
通用配置见[保存、恢复与自动容错](../guide/training-recovery.md)。
