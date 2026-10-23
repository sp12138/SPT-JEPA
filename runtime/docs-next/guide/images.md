# 镜像资源

以下两个 base 镜像已发布到 [KAP 镜像管理](https://kap.deltaverse-intl.com/asset/image-manage?projectId=ieg-infra)，预装 Python、PyTorch、CUDA 和 vforge 所需的第三方依赖，**不包含 vforge 源码或 gen_engine 子模块**。下载镜像并启动容器后，需要自行安装 vforge。

## 已发布镜像

两个镜像均面向 Linux amd64 / AWS B300，环境版本如下：

| 镜像 | Python | PyTorch | CUDA |
| --- | --- | --- | --- |
| PyTorch 2.13 基础镜像 | 3.12.3 | 2.13.0+cu130 | 13.0 |
| PyTorch 2.9.1 基础镜像 | 3.12.3 | 2.9.1+cu130 | 13.0 |

**PyTorch 2.13 镜像地址：**

```text
600627331169.dkr.ecr.us-west-2.amazonaws.com/ieg-infra/vforge:base-torch2.13.0-py3.12.3-cu130-c65bc86
```

**PyTorch 2.9.1 镜像地址：**

```text
600627331169.dkr.ecr.us-west-2.amazonaws.com/ieg-infra/vforge:base-torch2.9.1-py3.12.3-cu130-c65bc86
```

在 KAP 任务配置的 `image` 字段填写对应的完整镜像地址；本地使用 Docker 时，先登录有访问权限的镜像仓库，再使用 `docker pull <镜像地址>` 下载。

## 在容器中安装 vforge

先在能访问工蜂的本机下载源码。以下命令选择与这两个镜像配套的源码版本，不拉取子模块：

```bash
git clone --no-recurse-submodules --branch image-config/c65bc86 \
  git@git.woa.com:ieg-infra/vforge.git
```

将源码目录上传或挂载到容器，在容器中的 vforge 仓库根目录执行：

```bash
bash scripts/ci/kap/install_vforge.sh
python -c 'import torch, vforge; print(torch.__version__, vforge.__file__)'
vforge-train --help
```

安装脚本复用镜像中已有的依赖，离线安装当前 vforge 源码。安装后 Python 直接使用这份源码目录，修改代码无需重新安装。也可以使用自己的源码版本，但其 `uv.lock` 必须与镜像一致；脚本会检查，不一致时需要选择配套源码或镜像。

## gen_engine 按需下载

没有 gen_engine 权限的用户可以跳过子模块，安装 vforge 并使用其他业务。只有需要 gen_engine 且具备其仓库权限时，才在能访问工蜂的本机进入源码目录执行：

```bash
git submodule update --init --checkout -- vforge/models/gen_engine/dit
```

随后将包含子模块的源码目录上传或挂载到容器即可。
