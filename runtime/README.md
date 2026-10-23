# vforge

`vforge`（Video Forge）是面向视频大模型训练的原生 PyTorch 分布式框架。它统一训练生命周期、分布式执行、checkpoint 与可观测性，同时允许原生模型任务和既有第三方训练项目复用同一套运行时。

使用指南、支持模型和开发文档请访问[在线文档](https://ieg-infra-vforge.devcloud.woa.com/)。

## 开发环境

首次开发时安装开发依赖并启用 pre-commit：

```bash
pip install -e '.[dev]'
pre-commit install
```
