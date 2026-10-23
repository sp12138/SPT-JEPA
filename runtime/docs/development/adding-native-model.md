# 新增 Native 模型

模型插件只注册已经实现的 task builder。已有 train、后续补 eval 时，继续使用原来的
`model.name`，复用训练侧的模型构造和 published weight source，并在 registry 的同一项
增加 `eval=`。不要增加模型专属 CLI、Trainer 或 runner。

## 最小目录

下面只维护一份“已有 train，新增 eval”的最小示例：

```text
vforge/models/acme/v1/
  model.py
  inference.py                 # 新增
vforge/tasks/acme/v1/
  shared.py                    # train 已使用的模型与权重构造
  train.py
  eval.py                      # 新增
vforge/conf/experiment/acme_v1.yaml
tests/tasks/acme/v1/test_task_contracts.py
```

`vforge/models/` 只放模型和 inference；配置翻译、dataloader、权重来源和输出装配属于
`vforge/tasks/`。插件不得依赖外部业务框架。

## EvalComponents 最小示例

假设 train 已经从 `shared.py` 使用下面两个函数。eval 必须直接复用它们，不能复制一套
模型配置读取或 published 权重初始化规则：

```python
# vforge/tasks/acme/v1/shared.py
from vforge.models.factory import ModelSpec
from vforge.models.loading import PretrainedWeightSource
from vforge.models.acme.v1.model import AcmeModel


def build_model(*, extras, init_device):
    if init_device != "meta":
        raise ValueError("acme_v1 requires model.init_device=meta")
    return ModelSpec(
        build=lambda: AcmeModel(
            width=int(extras.get("width", 64)),
            init_device="meta",
        )
    )


def build_published_weight_source(pretrained_path):
    return PretrainedWeightSource(
        path=pretrained_path,
        subfolder="transformer",
        model_prefix="",
        loading_mode="auto",
    )
```

模型 inference 对象只需实现 `infer_sample(sample, *, seed)`。每次模型调用必须经过注入的
`engine_forward`，不能直接调用 prepared model：

```python
# vforge/models/acme/v1/inference.py
import torch


class AcmeInference:
    def __init__(self, model, engine_forward):
        self.model = model
        self.engine_forward = engine_forward

    @torch.no_grad()
    def infer_sample(self, sample, *, seed):
        features = sample["features"]
        generator = torch.Generator(device=features.device).manual_seed(seed)
        noise = torch.randn(
            features.shape,
            generator=generator,
            device=features.device,
            dtype=features.dtype,
        )
        prediction = self.engine_forward(features, noise=noise)
        return {
            "sample_id": sample["sample_id"],
            "prediction": prediction.detach().cpu(),
        }
```

最小 eval 先支持 condition-only inference manifest。下面文件可直接复制后替换
`acme_v1` 的 sample/model 细节：

```python
# vforge/tasks/acme/v1/eval.py
import os

import torch

from vforge.data.dataset import InferenceRequestDataset
from vforge.data.loader import build_eval_loader
from vforge.distributed.parallel_state import get_parallel_state
from vforge.evaluation.runner import (
    EvalComponents,
    EvalOutputWriter,
    sample_artifact_name,
)
from vforge.tasks.acme.v1.shared import (
    build_model,
    build_published_weight_source,
)
from vforge.tasks.requests import EvalRequest, validate_eval_source


def build_eval_components(request: EvalRequest) -> EvalComponents:
    validate_eval_source(request)
    if request.data_source != "inference":
        raise ValueError("minimal acme_v1 eval supports data_source=inference")

    model = build_model(extras=request.extras, init_device=request.init_device)
    dataset = InferenceRequestDataset.from_manifest(request.manifest)

    def collate(samples):
        if len(samples) != 1:
            raise ValueError("acme_v1 eval requires batch_size=1")
        sample = samples[0]
        return {
            "sample_id": str(sample["sample_id"]),
            "features": torch.tensor(sample["features"], dtype=torch.float32).unsqueeze(0),
            "__vforge_eval_padding__": bool(sample["__vforge_eval_padding__"]),
        }

    dataloader = build_eval_loader(
        dataset,
        parallel_state=get_parallel_state(),
        collate_fn=collate,
        num_samples=request.num_samples,
        sample_selection=request.sample_selection,
    )

    def build_inference(prepared_model, engine_forward):
        from vforge.models.acme.v1.inference import AcmeInference

        return AcmeInference(prepared_model, engine_forward)

    def write_output(result, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(
            output_dir,
            sample_artifact_name(result["sample_id"], suffix=".pt"),
        )
        temporary = path + ".tmp"
        torch.save(result["prediction"], temporary)
        os.replace(temporary, path)
        return path

    return EvalComponents(
        model=model,
        dataloader=dataloader,
        initial_weights=build_published_weight_source(request.pretrained_path),
        build_inference=build_inference,
        output_writer=EvalOutputWriter(
            artifact_name=lambda sample_id: sample_artifact_name(sample_id, suffix=".pt"),
            write_output=write_output,
        ),
    )
```

`EvalComponents` 的五项缺一不可：

- `model`：普通 `nn.Module` 或延迟构造的 `ModelSpec`；
- `dataloader`：使用 eval sampler，先应用全局样本限制再做 DP 分片；
- `initial_weights`：`PretrainedWeightSource` 或 `TrainingCheckpointWeightSource`；
- `build_inference(prepared_model, engine_forward)`：返回带 `infer_sample()` 的对象；
- `output_writer`：声明预期路径并原子写入产物。

registry 在已有 train 项上增加 eval：

```python
# vforge/tasks/registry.py
"acme_v1": ModelTasks(
    train="vforge.tasks.acme.v1.train:build_train_components",
    eval="vforge.tasks.acme.v1.eval:build_eval_components",
),
```

最小 manifest：

```json
{"sample_id":"demo-0","features":[0.1,0.2,0.3]}
```

从仓库根目录运行：

```bash
NPROC_PER_NODE=1 bash scripts/eval.sh \
  experiment=acme_v1 \
  model.pretrained_path=/models/acme-v1 \
  latent_cache.mode=online \
  evaluation.data_source=inference \
  evaluation.manifest=/data/acme-eval.jsonl \
  evaluation.output_dir=/runs/acme-eval
```

## 按需扩展，不进入最小示例

- **raw validation**：需要读取原始媒体并在 batch leader 编码时，再接
  `open_eval_dataset()` 和模型自己的 encoder/collate。
- **cache**：需要 `evaluation.data_source=cache` 时，再定义模型专属
  `EncodedBatchCodec`；cache manifest 必须与 codec 匹配。
- **multi-variant**：同一样本运行多种策略时，复用
  `MultiVariantInference` 和 `MultiVariantOutputWriter`，模型 task 只定义 variant
  集合及 sample 转换。
- **causal**：只有模型声明 causal inference 能力时，才实现 generated-history rollout
  或 KV cache；不要把 causal block loop 塞进通用 runner。
- **training checkpoint**：需要评估训练产物时，按请求返回
  `TrainingCheckpointWeightSource`；published 权重仍使用 train 已拥有的 source。

这些扩展都不改变最外层的 `EvalComponents` / `infer_sample` contract。prepared sample、
flow-match 和 incremental KV 的可选 helper 见
[视频推理扩展边界](video-inference-extension.md)。

## Train preflight

已有 train 只有在模型与 engine 存在组合限制时才注册 `train_preflight`：

```python
def validate_train_request(request: TrainRequest, engine: str) -> None:
    ...
```

它在 process group、CUDA、模型、dataloader 和权重初始化之前运行，只能基于
`request` 与 `engine` 做无分配校验；支持时返回 `None`，不支持时抛出带可操作替代方案的
异常。没有组合限制的模型不要注册空回调。

## 验证

至少增加 CPU contract，覆盖 registry 可解析、builder 五项完整、固定 seed 可重放、
writer 的声明路径与真实路径一致。文档和定向测试按
[本地开发与验证](testing.md)运行；真实 GPU、FSDP、TP/USP 或长视频另做对应 gate。
