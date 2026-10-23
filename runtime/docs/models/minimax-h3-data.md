# MiniMax-H3 数据与 Cache

本页定义 H3 FL2VA 的媒体、manifest 和 offline cache contract。运行命令见
[MiniMax H3 训练流程](minimax-h3.md)。

## 媒体要求

每条样本建议包含：

- 可由 PyAV 解码的视频，推荐 MP4/H.264、24 fps；
- 视频容器内嵌 stereo 32 kHz 音轨；没有音轨时 audio loss 被 mask；
- 描述视频和声音的 prompt；
- 可选首帧、尾帧条件图，推荐 PNG/JPEG，宽高比与视频一致。

当前不接受独立 `audio` 路径。所有文件必须能被 prepare 节点访问。

## 帧数和时长

默认读取 124 帧，约 5.17 秒：

- 少于 124 帧时重复最后一帧；
- 多于 124 帧时保留前 124 帧；
- 音频转成 stereo 32 kHz，并补零或截断到相同时长；
- encoder 不按源 fps 重采样画面，非 24 fps 素材应先离线转换。

H3 video VAE 要求：

```text
num_frames = 17 × n + 5
```

发布模型常用约 5–15 秒，即 124、141、158、…、345 帧。修改时在 prepare、train 和
eval 中使用同一个值：

```bash
model.extras.num_frames=175
```

Native schema 不提前验证该 VAE 约束；非法值会在 prepare/encode 阶段失败。

修改帧数后必须重新生成 cache。旧 video/audio latent、position ids 和 packed layout
不能复用。当前同一任务只能有一个全局 `num_frames`。

## 画布

默认画布由首帧条件图决定：

- 短边 768；
- 像素数不超过 `768 × 1344`；
- 高宽对齐到 32。

也可同时设置 `model.extras.height` 和 `model.extras.width`；两者必须同时出现且都是
32 的倍数。视频与条件图应保持一致宽高比，避免拉伸。

## Manifest

支持 JSON 数组或 JSONL，推荐 JSONL：

```json
{"sample_id":"sample-0001","video":"/data/0001.mp4","image":"/data/0001.png","prompt":"A singer performs on stage."}
```

| 字段 | 要求 | 说明 |
| --- | --- | --- |
| `video` | 必填 | `video_path`、`rgb` 是兼容别名 |
| `sample_id` | 强烈建议 | 必须唯一稳定；用于 cache 和 resume contract |
| `prompt` | 建议 | `text` 是兼容别名；缺省为空 |
| `image` | 可选 | 首帧条件；缺省时使用视频第一帧 |
| `last_image` | 可选 | 填写后使用首尾帧条件 |

首尾帧样本：

```json
{"sample_id":"fl2va-0002","video":"/data/videos/0002.mp4","image":"/data/images/0002-first.png","last_image":"/data/images/0002-last.png","prompt":"A train crosses the frame with rhythmic wheel sounds."}
```

以上是 prepare/train 以及 `evaluation.data_source=validation` 的 target-backed
schema。只生成、不提供目标视频时，使用 `evaluation.data_source=inference`：

```json
{"sample_id":"infer-0001","image":"/data/images/0001.png","prompt":"A singer performs on stage.","num_frames":124}
```

Inference manifest 不接受目标 `video` 作为 geometry 载体；画布由图片或显式
`height`/`width` 解析。两种 raw source 都通过 `evaluation.manifest` 指向样本文件；
`cache` source 则直接指向 cache 根目录下的 `manifest.json`。

## 生成 Cache

先验证媒体：

```bash
ffprobe /data/videos/0001.mp4
```

再用少量样本跑 prepare：

```bash
vforge-prepare experiment=minimax_h3_fl2va \
  model.pretrained_path=/models/MiniMax-H3 \
  data.train_manifest=/data/h3/train.jsonl \
  data.cache_dir=/data/h3/cache
```

检查 cache 后再扩大 manifest。Prepare、train 和 eval 必须共享：

- 模型版本；
- `num_frames` 和画布；
- VAE/text/audio encoder 配置；
- sample id 与输入媒体身份。

Cache fingerprint 不一致时，严格 DCP 恢复应失败，不应通过复制目录绕过。

数据和 cache 检查通过后，返回
[MiniMax H3 One Pager：USP4 首次训练](minimax-h3.md#usp4-first-train)。
