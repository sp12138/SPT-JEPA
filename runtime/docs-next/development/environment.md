# 配置开发环境

开发 vforge 可以使用预装依赖的 Docker 镜像，也可以在本地用 uv 创建 Python 环境。本文说明如何选择镜像、安装当前源码、修改依赖，以及构建自己的镜像。以下命令均从仓库根目录执行。

## 镜像查询与选择

进入 [KAP 镜像管理](https://kap.deltaverse-intl.com/asset/image-manage)，在有访问权限的项目中找到 **vforge**，查看可用镜像。选择时：

- **复现 CI 环境**：使用仓库 [CPU 机器配置](https://git.woa.com/ieg-infra/vforge/blob/master/scripts/ci/kap/machines/cpu.yaml)或 [GPU 机器配置](https://git.woa.com/ieg-infra/vforge/blob/master/scripts/ci/kap/machines/b300.yaml)中的完整 `image` 值。
- **试用自定义构建**：选择自己构建的镜像，确认其依赖与开发分支匹配。

标签便于辨认版本；`@sha256:…` 是镜像内容的固定标识。KAP 任务配置的 `image` 字段填写完整镜像地址，CI 使用 digest 固定环境，不会自动跟随最新标签。

### 容器内源码安装

镜像预装第三方依赖。进入容器后，将需要开发的仓库放到可写目录，切换到仓库根目录，执行：

```bash
bash scripts/ci/kap/install_vforge.sh
python -c 'import vforge; print(vforge.__file__)'
```

安装脚本校验源码的 `uv.lock` 与镜像中保存的锁文件一致，再以 editable 模式安装当前源码。editable 模式会让 Python 直接使用该源码目录，修改 Python 文件后无需重新安装。

如果锁文件不匹配，先选择或构建匹配的镜像。该脚本采用离线安装，不会在启动时下载新依赖。CI 如何把源码送入容器，见[源码分发与运行环境准备](maintenance/ci.md#source-environment)。

## 本地 uv 环境配置 {#local-environment}

环境版本由仓库配置管理：

- [`pyproject.toml`](https://git.woa.com/ieg-infra/vforge/blob/master/pyproject.toml)：`tool.uv.required-version` 指定 uv 版本要求，`project.requires-python` 声明支持的 Python 范围，依赖条目声明各包的版本约束。
- [`.python-version`](https://git.woa.com/ieg-infra/vforge/blob/master/.python-version)：指定项目默认使用的 Python 解释器版本。
- [`uv.lock`](https://git.woa.com/ieg-infra/vforge/blob/master/uv.lock)：保存解析后的具体依赖版本。

按 [uv 安装说明](https://docs.astral.sh/uv/getting-started/installation/)安装满足仓库要求的 uv，再准备 CPU 开发环境：

```bash
uv --version
uv sync --locked --group cpu --group build --group docs
uv run --no-sync python -V
uv run --no-sync python -c 'import vforge; print(vforge.__file__)'
```

`uv sync` 创建项目环境并安装当前源码。后续命令使用 `uv run --no-sync` 复用已准备的环境；依赖发生变化时，显式重新同步。

Linux GPU 环境使用另一组依赖：

```bash
uv sync --locked --group cu130 --group build --group docs
uv run --no-sync python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())'
```

依赖组及其包含的包以 `pyproject.toml` 为准。按运行环境选择 CPU 或 GPU 依赖组，两者互斥，不使用 `--all-groups`。

## 依赖添加、更新与移除

先确定依赖属于运行时还是开发工具。下面用 `httpx` 和 `pytest-xdist` 演示命令形式，并非要求安装这两个包：

```bash
# 运行时依赖：写入 project.dependencies
uv add --no-sync httpx

# 仅供开发使用：写入 dev 组
uv add --no-sync --group dev pytest-xdist
```

`uv add` 同时更新 `pyproject.toml` 和 `uv.lock`。这里使用 `--no-sync`，将依赖修改与环境安装分开；完成所需修改后，按上面的 CPU 或 GPU 命令同步环境。

更新已有包的锁定版本，或移除不再需要的依赖：

```bash
# 在项目声明的版本约束内更新
uv lock --upgrade-package numpy

# 移除依赖，同步更新声明与锁文件
uv remove --no-sync httpx
```

提交前审查 `git diff -- pyproject.toml uv.lock`，运行受影响的测试，并将两个文件一起提交。更多参数见 [uv 依赖管理](https://docs.astral.sh/uv/concepts/projects/dependencies/)。

**锁文件变化后，CI 镜像也必须更新。** 镜像中的依赖与源码锁文件需要一致；只更新本地环境，不能让远端 worker 获得新依赖。

## 自定义 Docker 镜像构建 {#custom-image}

1. 修改仓库的 [Dockerfile](https://git.woa.com/ieg-infra/vforge/blob/master/Dockerfile)或依赖配置，将改动提交并推送到分支。
2. 在 KAP 创建基于代码仓库的镜像构建，选择确定的源码提交，使用仓库中的 Dockerfile。
3. 按项目的镜像配置选择目标仓库，使用独立标签区分自定义构建。
4. 如需调整可选组件，查看 Dockerfile 中声明的构建参数，仅覆盖需要改变的选项。

系统组件、基础镜像和构建参数的定义以 Dockerfile 为准，Python 依赖以锁文件为准。构建应能追溯到源码提交，以便复现环境。

### CI 运行镜像更新

1. 等待镜像构建成功，取得完整镜像地址和 digest。
2. 用新镜像运行本次修改涉及的 CPU/GPU 测试，命令见[运行测试](testing/running.md)。
3. 更新需要切换的 CPU/GPU 机器配置中的 `image` 值，固定到新 digest。
4. 提交配置改动，通过 MR 审查。

仅修改 Python 源码且锁文件不变时，可以复用已有镜像。接下来可阅读[提交代码](contributing.md)，完成提交前检查和 MR。
