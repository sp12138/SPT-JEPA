# MetaWorld公开协议比较：数据与复现审计

用户要求在公开设定下测量与SOTA baselines的差距，随后选择继续寻找他人独立公开、无需账号验证的数据版本。仅使用dev13/dev15；scratch、RGB/已执行动作、无reward/最优动作标签及sample-level候选损失约束继续有效。

## 实际完成的检查

官方数据固定`facebook/jepa-wms@6116f042ae7ae4c8e3f1fd2f194f432615664182`。公开元数据有126个原始MetaWorld parquet、737,175,770 bytes；不等于已下载或核验轨迹内容。dev13下载第一片实际返回401/GatedRepoError，HF未登录。

检索HF的273个MetaWorld、86个JEPA、95个worldmodel条目。对59个尚未由描述排除的MetaWorld候选，加85个其他JEPA仓库，实际核对144个仓库的公开文件元数据：与官方126个分片SHA256匹配的文件为0，读取错误为0。见`results/mirror-audit-v1/summary.json`。只读取公共元数据，没有请求受限文件的替代端点、代理或viewer缓存。

字节指纹阴性结果不能排除其他仓库名字、重编码或重打包副本。任何候选都仍需实际无token下载、SHA及轨迹身份/时序检查。没有启动正式SGD、模型评测、新后台launcher或周期监控；公平性能差距尚未测得。

## 免登录候选的范围

| 公开数据 | 已核验信息 | 与目标语料的关系 |
|---|---|---|
| [lerobot/metaworld_mt50](https://huggingface.co/datasets/lerobot/metaworld_mt50)及其重发布 | 已有2500条专家轨迹 | 是此前独立实验的数据，不是JEPA-WMs语料 |
| [ml-jku/meta-world](https://huggingface.co/datasets/ml-jku/meta-world) | 数据卡说明为L2M所用v2数据、每任务2M transitions | 未建立逐轨迹同源性 |
| [nicklashansen/tdmpc2](https://huggingface.co/datasets/nicklashansen/tdmpc2) | MT80/MT30回放，545M/345M transitions | 采集算法相同不等于同一批数据 |
| [jepa-world-models/h-jepa](https://huggingface.co/datasets/jepa-world-models/h-jepa) | FourRoom、AntMaze、PushT、DROID等 | 未含目标MetaWorld副本 |
| `fenfenda/jepa-wm-delta` | 实际README是NAVSIM/ReCogDrive未来图像增量 | 与MetaWorld无关 |

## 公开代码发现的额外差异

固定[LaLa-World 3356787](https://github.com/junha1125/LaLa-World/tree/33567871362aa9d5ed7832d3667b1ae122975c14)，文件SHA见`results/public-source-audit.json`。这是静态代码检查，未执行模拟器或恢复公开训练模型。

- JEPA-WMs原论文取专家100步末帧；LaLa当前`set_episode`允许最多200步、首次成功立即停止并取goal。此前把两者当作相同goal定义，需纠正。
- LaLa公开入口48案例/任务、100原生交互步、任意原生子步成功；案例seed映射为`(seed*seed+episode*seed) % (2**32-2)`。
- 当前发布代码直接渲染224、仅垂直翻转；旧LeRobot流程480缩放、双轴翻转不能算同一接口。
- CEM每次从零均值重新开始，末段预测长度随剩余预算缩短；与旧控制器warm tail等行为不同。名义CEM300×15/H6相同不足以证明相同planner。
- 此版本未找到论文DreamerV3、TD-MPC2、IRIS、LeWM的适配训练代码/配置/权重；README中LaLa世界模型checkpoint为coming soon。上游原算法代码不能自动代表论文的无奖励适配版本。

## 正式实验的准入条件

1. 同一官方语料版本和原始轨迹划分；如为重打包副本，先证明RGB/action及task/collector-seed/episode身份与官方loader相符。此前LeRobot模型不能续训后冒充该语料上的随机初始化。
2. TF/innovation与可复现scratch基线使用同一公开评测入口、同一48物理案例和预算；至少3个训练种子，明确同一案例被重复评估的相关性。冻结累计训练预算，并披露编码图像量、容量及FLOPs差异。
3. JEPA-WMs原100步专家末帧/成功规则与LaLa发布代码的200步内首次成功goal/100步any-success分别命名，预先确定主次，不择优混报。
4. 公开预训练SOTA仅作明确外部参考；主方法和scratch基线保持无外部强模型/教师。缺失适配实现的基线若重新实现，必须明确命名、审计完整损失/初始化/预算，不冒充原表复现。

目前缺少可访问的同源数据，以及部分论文基线的适配实现/完整训练预算证据。上述准入条件尚未通过；正式公平比较未启动。
