# 构建与发布文档

文档使用 MkDocs 将 `docs-next/` 构建为静态 HTML，生成 `site/` 产物并发布到站点根路径。

## 文档内容与导航维护

内容目录、导航和主题设置统一由[站点配置](https://git.woa.com/ieg-infra/vforge/blob/master/mkdocs-next.yml)维护。

新增或移动页面时同步修改配置的 `nav`。导航分组只负责组织页面，无需为每组创建介绍页。站内链接使用相对 Markdown 路径，源码链接指向仓库网页。

每篇文章围绕一个读者目标：操作指南给出前提、步骤和预期结果；设计说明解释流程、职责与约束。相同命令或规则只维护一份，其他页面链接引用。

版本、镜像地址、资源规格、参数默认值和允许值清单以代码或配置为准，文档链接其来源，不复制当前取值。命令示例只包含完成操作所需的内容。

## 本地构建与预览

环境已准备好时，从仓库根目录执行：

```bash
uv run --no-sync bash scripts/ci/docs_build.sh
uv run --no-sync python -m http.server 8000 --bind 127.0.0.1 --directory site
```

若缺少 MkDocs，先按[环境准备](../environment.md#local-environment)安装 `docs` 依赖组。构建脚本不会临时安装依赖。

浏览器打开 `http://127.0.0.1:8000/` 查看文档。修改页面后重新构建并刷新；按 Ctrl+C 停止服务。

构建依次执行：

1. 校验 `docs-next/` 的导航、相对链接和标题锚点。
2. 严格构建文档到 `site/`。
3. 为已有的 `/next/` 链接生成指向根路径对应页面的跳转，保留查询参数和标题锚点。

命令以 0 退出，且 `site/index.html` 和 `site/next/index.html` 均存在，表示构建完成。预览时检查侧栏、目录、搜索和页面跳转。

## MR 文档预览 {#mr-preview}

MR 的 CPU 检查通过后，流水线的「文档 debug」步骤将该阶段构建的站点部署到 [预览站点](https://ieg-infra-vforge.devcloud.woa.com/debug/)，供合入前审查。

所有 MR 共用这一入口，每次部署覆盖上一份内容并删除已移除的文件，不保留历史版本。查看前先确认对应流水线的「文档 debug」步骤已成功；期间其他 MR 的部署也会更新该入口。

## master 合入后的文档发布

合入 master 后，[发布流水线](https://git.woa.com/ieg-infra/vforge/blob/master/.ci/publish.yml)从该提交重新生成网站：

```text
master push
  → 核对 master 最新 SHA
  → 签发本次产物的短期上传地址
  → KAP CPU worker 构建并上传压缩后的 site
  → CI 下载、核对产物源码 SHA、删除临时对象
  → 再次核对 master 最新 SHA
  → CI 通过 SSH + rsync 部署
  → 检查 HTTP 响应并清理资源
```

MR 预览和 master 发布都由 KAP 构建、CI 主机部署。CI 为当前构建签发短期上传地址；worker 只接收该对象的上传地址，不接收 AWS 长期凭据或部署私钥。CI 下载后核对产物的源码 SHA 和部署路径，再删除临时对象。

master 发布串行执行；构建前或部署前发现提交已被更新版本替代时跳过该版本。正式站点部署保留 `/debug/`，预览部署只更新该目录。临时对象在正常流程和 Finally 清理中删除。

## 发布环境与凭据配置

凭据通过蓝盾 settings 注入。维护者需要配置 KAP 访问权限、临时对象存储访问权限，以及部署机的 SSH 身份和可信主机公钥。

具体的凭据绑定和部署目标以[发布配置](https://git.woa.com/ieg-infra/vforge/blob/master/.ci/publish.yml)及[部署入口](https://git.woa.com/ieg-infra/vforge/blob/master/scripts/ci/docs_deploy.sh)为准。

目标主机需要预先提供部署入口依赖的工具和静态 HTTP 服务。部署会同步完整站点，并清理已从产物中移除的旧文件，因此站点目录应专用于生成的文档。部署不负责启动或管理 HTTP 服务。

发布失败时按阶段检查：构建错误看 MkDocs 日志，传输错误看上传和源码版本校验，部署错误看 SSH、目标目录权限及 HTTP 检查。不要将包含签名地址或私钥的环境输出贴入日志。
