# Vendored Megatron-Core

这棵树是上游 Megatron-LM 的 `megatron/core`，导入命名空间被重写成了
`vforge.mcore`。它不是依赖，也不是我们写的代码。改这里任何东西之前先读完本文。

| | |
|---|---|
| 上游 | https://github.com/NVIDIA/Megatron-LM |
| Tag | `core_v0.18.2` |
| Commit | `571370c829ca768fe37244f4e2e7f28d8accc4ab` |
| Vendor 日期 | 2026-08-11 |
| 内容 | 上游 `megatron/core` 去掉一批用不到的子树（见下），外加上游 `LICENSE` |
| 体积 | 251 个 Python 文件（上游 437） |
| 本地改动 | `megatron.core` → `vforge.mcore`，机械替换（1522 处，291 个文件） |

## 为什么要重写命名空间

上游是绝对导入自己的 —— 约 1500 条 `from megatron.core.x import y`。保留这个路径就
意味着要发布一个顶层 `megatron` 包，那会占用全局导入名，使得装了 vforge 的环境无法
再装 pip 的 `megatron-core`，两者互斥。所以我们选择重写，让 mcore 成为 vforge 的一个
普通子包，二者可以共存。

重写只是一个 token 替换：`\bmegatron\.core\b` → `vforge.mcore`，由
`scripts/vendor_mcore.sh` 施加。它同时覆盖导入语句和 **mcore 在运行时才解析的字符串
模块路径** —— `transformer/spec_utils.py` 里的 `ModuleSpec.module`，以及
`safe_globals.py` 里的 pickle 允许名单。这类字符串漏改不会在导入时报错，而是等到某个
特性首次被用到才炸，所以它们才是重写的真正风险点。

裸 `megatron`（不跟 `.core`）是**故意不改**的：在 `core_v0.18.2` 上它只出现在
docstring、tokenizer 库标识（`'megatron'`）、缓存目录名和 HuggingFace 模型 id 里，
没有一处是模块路径。**升级换 tag 时要重新验证这个结论。**

因为重写是机械且确定性的，升级依然便宜：`--check` 会对一份全新的上游 checkout 重放
同样的替换再和本棵树比对，所以任何手工改动都会以"漂移"的形式暴露出来。

### 重写的代价

FP8 extra-state checkpoint 会把模块路径写进 pickle，而 `safe_globals.py` 是按字符串
匹配允许名单的。重写之后我们产出的 checkpoint 里带的是 `vforge.mcore.*` 路径，因此
**与原版 megatron-core 写出的 checkpoint 双向都不通用**。这是自洽的 —— 我们写的我们
读得回来 —— 并且是明确接受的取舍：vforge 不依赖 mcore 格式的 checkpoint 互操作。
如果哪天这个前提变了，这条要回头重新评估。

## 裁剪掉的子树

`scripts/vendor_mcore.sh` 顶部的 `PRUNE` 数组声明了不 vendor 的子树，同步时作用在暂存
目录上，所以 `--check` 比对的是"上游减去这份清单"，而不是把缺失报成漂移。

| 组 | 内容 |
|---|---|
| 导出 | `export/`（TensorRT-LLM） |
| RL refit | `resharding/`（训练与推理布局间搬权重） |
| 数据集 | `datasets/`（GPT/BERT/T5 的预训练数据管线） |
| 模型库 | `models/{T5,bert,huggingface,mamba,mimo,multimodal,vision}` |
| tokenizer 实现 | `tokenizers/{text,utils,vision}` |
| 量化后训练 | `post_training/modelopt/{gpt,hybrid,mamba}` 的 model specs（只有 `layers.py` 被用到） |
| 推理服务 | `inference/{apis,engines,model_inference_wrappers,sampling,text_generation_controllers,text_generation_server}` 及 `async_stream.py`、`common_inference_params.py`、`communication_utils.py`、`data_parallel_inference_coordinator.py`、`headers.py`、`inference_client.py`、`scheduler.py` |

入选门槛只有一条:**`megatron/core` 里没有任何其它地方引用它**。
`tests/scripts/test_mcore_prune.py` 强制这一条 —— 它解析 `PRUNE` 数组，再扫描保留下来的
每个文件，只要有人 import 到被裁掉的模块就失败。那个测试自己也有一条自测，因为把
`models/T5` 拼成模块名时漏掉斜杠会让整条扫描静默失效。

有几个看上去同样没用、实际删不掉的，值得记下来省得下次再查一遍:

- `models/common/` 和 `models/backends.py` 根本不是模型 —— 前者装着 rope 工具，
  `transformer/attention.py` 在模块级就要；后者是 spec provider 协议。
- `models/gpt/` 被 `transformer/module.py`、`pipeline_parallel/combined_1f1b.py`、
  `models/common/model_chunk_schedule_plan.py` 和 `post_training/modelopt/` 引用。
- `models/hybrid/` 在关键路径上:`attention.py` → `inference.contexts` →
  `contexts/__init__.py` 加载 `dynamic_context` → 模块级 import
  `models.hybrid.hybrid_layer_allocation`。上游在那个 `__init__` 里留了 warning，说这组
  导入造成了循环依赖、计划在 0.14 移除。
- `inference/contexts`、`moe`、`quantization`、`symmetric_memory`、`utils`、
  `communication` 同理被 `transformer/`、`parallel_state.py`、`tensor_parallel/` 模块级引用。
  这个包的名字有欺骗性:它不只服务于推理。
- `ssm/` 只被 `transformer/cuda_graphs.py` 惰性引用，但它和 `models/hybrid/` 互相 import。

还有两个**盲区**，静态扫描看不见，扩大清单前要留意:

- **条件导入。**`post_training/` 被 `transformer/multi_latent_attention.py` 和
  `transformer/experimental_attention_variant/absorbed_mla.py` 在 `if HAVE_TE:` 分支里
  引用。笔记本上没有 Transformer Engine，那段不执行；B300 上装了，就会执行。也就是说
  **在没有 TE 的机器上做静态判断会低估 GPU 机器上真正跑到的代码**。
- **按字符串动态加载。**`tokenizers/megatron_tokenizer.py:206` 用
  `importlib.import_module(f"vforge.mcore.tokenizers.{tokenizer_type}.models")` 拼模块名，
  所以裁掉 `tokenizers/text` 之后，一旦真去构造 tokenizer 就会 `ModuleNotFoundError`。
  我们只用到 `MegatronTokenizer` 这个类型标注（`inference_request.py` 里两个标了
  deprecated 的参数），走不到那个函数，所以这一项是**基于判断**而非工具证明。
  `ModuleSpec.module` 和 `safe_globals` 的 pickle 允许名单是同类风险。

要扩大裁剪范围，就往 `PRUNE` 里加一行再跑一次同步；上面那个测试会告诉你 import 语句层面
是否成立，但它看不见上面这两类。请在装了 Transformer Engine 的机器上验证。

## 规则：这棵树只可引用，不可修改

每一处本地改动都是下次升级时要还的合并冲突，而下面的校验是唯一便宜的发现手段。
CI 的 lint 阶段会强制这条规则。如果你确实必须改 mcore：

1. 优先在 `vforge/` 里用继承或包装解决。
2. 实在不行，把改动写成 `scripts/vendor_mcore.sh` 里的一段 patch（和现有的
   `rewrite_namespace` 并列），让它在每次同步时自动重放，然后跑一次同步脚本
   重新生成清单，并登记到下面这一节。

**不要直接编辑文件**：那样的改动会在下次升级时被无声覆盖掉，而且 CI 会拦住你。

### 本地补丁

除命名空间重写外，无。

## 校验与升级

有两种校验，用途不同。

```bash
# 离线：对照 CHECKSUMS.sha256 检查这棵树有没有被改过（约 60ms，不联网）
scripts/vendor_mcore.sh --verify

# 联网：重新从上游推导一遍标准答案再比对（升级前后、或怀疑清单被重新生成时用）
scripts/vendor_mcore.sh --check

# 升级到新 tag（重写整棵树、重新生成清单，然后手动更新本文的表格）
scripts/vendor_mcore.sh core_v0.19.0
```

`--verify` 已经接进 CI 的 lint 阶段（`scripts/ci/lint.sh` 的 `vendored mcore`
步骤）。之所以用离线的哈希比对而不是 `--check`，是因为 lint 阶段是刻意做轻的无 torch
镜像，往里塞一个 GitHub clone 会让它变慢、变不稳定，内网还未必连得通。

`CHECKSUMS.sha256` 是同步时自动生成并提交进仓库的，记录这棵树里每个文件的 sha256
（不含 `VENDOR.md` 和清单自身）。有人手工改了代码，哈希就对不上，CI 变红。
当然可以通过重新生成清单来"洗白"一次改动 —— 但那会在 MR 里表现为
`CHECKSUMS.sha256` 的 diff，而这正是我们想要的、瞒不过 review 的信号。

生成和校验的逻辑在 `scripts/vendor_manifest.py`，它的回归测试在
`tests/scripts/test_vendor_manifest.py`。这个守卫一旦悄悄失效，保护就没了却不会有任何
东西变红，所以它自己也需要被测。

这棵树已从 ruff（`pyproject.toml`）、`compileall`（`scripts/ci/lint.sh`）和 pytest
收集范围（`testpaths = ["tests"]`）中排除。上游的代码风格不是我们的风格，去 lint 它
只会逼我们产生本文极力避免的偏离。

## 导入它：triton 不是可选项

`vforge/mcore/__init__.py` 会急切地拉进 `triton`，链路是
`distributed` → `finalize_model_grads` → `pipeline_parallel.schedules` →
`transformer/moe/paged_stash.py`。Python 在导入任何子模块前必然先初始化父包，
所以**在没有 triton 的机器上，`vforge.mcore` 底下什么都导不进来** —— 包括所有
macOS，triton 在 mac 上没有 wheel。

好消息是 Linux 上这是免费的。torch 2.13.0 自己就声明了
`triton==3.7.1; platform_system == "Linux" and python_version < "3.15"`，
所以 CI 和 GPU 镜像装完 torch 就已经有 triton 了。**不要**把 triton 写进我们的
`dependencies`：torch 是精确锁版本的，我们再声明一个只会和它打架。


## mcore 期望的其他依赖

`core` 相对 Megatron-LM 的其余部分是自包含的 —— 不引用 `megatron.training` 或
`megatron.legacy`。它会去够一些可选的第三方加速库，绝大多数藏在 `extensions/` 的
惰性导入后面：

| 包 | 导入点 | 用于 |
|---|---|---|
| `transformer_engine` | 98 | FP8、融合 attention、TE 实现的层 |
| `triton` | 58 | 融合 kernel（见上，实际上不可选） |
| `apex` | 17 | 融合 optimizer / layernorm |
| `tensorrt_llm` | 15 | export 路径 |
| `transformers` | 10 | tokenizer |
| `einops` | 9 | Mamba/SSM 层 |
| `nvidia_resiliency_ext` | 9 | 容错 |
| `flash_attn` | 4 | attention 后端 |

其中 Transformer Engine 因 MCore CP 已成为 `pyproject.toml` 的 `megatron` optional
extra；其余没有写入项目依赖。在有 triton、其余都缺的情况下，mcore 能正常导入，只会
警告说正从 Transformer Engine 和 Apex 回退到 Torch 实现；具体特性会在首次使用时才抛错。
