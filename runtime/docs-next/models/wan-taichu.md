# Wan Taichu

Wan Taichu 基于 Wan 2.1 / 2.2，针对世界模型训练增加 RGB、相机几何 grid 和 normal
场景条件，并支持首帧条件和按时间块组织的因果训练。不同模型变体使用不同条件布局，
训练时需准备匹配的权重与缓存。

训练通过 `vforge.tasks.train` 启动，按任务选择对应配置：

- **[CD](wan-taichu/cd.md)**：从 causal teacher 权重初始化，支持 Wan 2.2 TI2V-5B
  单模型和 A14B 高低噪声双专家，提供离线缓存准备要求与单机 SP8 启动命令。

按对应教程准备权重、数据和启动命令。保存与恢复的通用配置见
[保存、恢复与自动容错](../guide/training-recovery.md)。
