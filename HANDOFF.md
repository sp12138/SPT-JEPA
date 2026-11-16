# 当前交接

目标：从零 SPT-JEPA 的独立动作头控制出现可靠稳定的 scaling，并超过已有参照。**尚未达成**；两点上升、loss 下降或零星成功都不算达标。用户要求时间优先算法机制、关键实验和实际控制，检查与记录最小化，避免重复兼容回归或 CPU 审计。

只用 dev13、dev15、dev6；dev6-2c/grid16/dev2 不连接。当前主会话每小时检查并按证据改进，不在有限 pilot 后自行 HOLD；不恢复旧 goal、监控或第二套 LLM 监督。实际分配和检查入口为 R/MASTER.json、R/SESSION_CHECKS.json，不能根据旧 ACTIVE 文字恢复任务。

R=`/efs/prescottsun/SPT-JEPA/runs/robocasa_reach3d_scaling_20261114`；模型根在其父目录，下表后缀均加 `robocasa_reach3d_`。

| 主机/卡 | 根后缀 / run name | 当前方法；可训练参数 / 动作头参数 |
|---|---|---|
| dev13 0–3 | `trajectory_likelihood_20261117` / spt-trajectory-consequence-likelihood | 样本内后果辨认替换Gaussian码率；40.810M / 384 |
| dev13 4–7 | `trajectory_flexible_likelihood_20261117` / spt-trajectory-flexible-likelihood | 状态相关F＋样本内后果辨认；41.625M / 384 |
| dev15 0–3 | `relative_state_20261116` / spt-relative-state | 相对状态编码，动作基自由学习；40.810M / 384 |
| dev15 4–7 | `ema_initial_target_20261116` / spt-ema-initial-mean-readout | 自身 EMA 当前码＋状态相关 F；41.625M / 2304 |
| dev6 0–3 | `dense_isometric_action_20261116` / spt-dense-isometric-action | 完整未来RGB预测，无RGB直通，学习正交动作基；40.810M / 2304 |
| dev6 4–7 | `pure_action_transport_20261116` / spt-pure-action-transport | 无自主漂移的共享动作平移；40.810M / 384 |

## 决策所需的关键结果

- **保留纯动作平移继续8h。** `z_next=z+A*a_t` 的1/2/3/4/5h为71.70/86.30/86.26/87.95/85.82%、67.51/29.59/31.97/29.87/35.85mm，final0/0/1/0/1、any0/0/2/1/3；后段回落，尚非稳定精细scaling。同384参数头的漂移对照2h79.79%/49.56mm，纯平移有匹配改善。源 `code-pure-action-transport-v1`。**不要为了启动已准备的flexible-absolute对照而停这个分支。**
- EMA未来码1/2h49.27/49.96%、124.00/123.78mm、全0，已退出。EMA当前码1/2/3/4h73.60/83.54/85.81/86.86%、63.18/38.30/34.07/30.60mm、final0/0/1/0、any均1，继续第五小时；当前码＋平移组合1/2h57.24/71.11%、101.00/64.30mm、全0，保全完整2h及部分3h状态后退出，让卡给新相对编码。均为自身同数据随机初始化模型EMA，decay=exp(-32/120060)，无外部教师。源 `code-ema-initial-target-v1`；读出容量分别2304/384，不能混报。
- 同8普通TRAIN窗口，未来码换成当前码，原RGB1h解码误差增至2.21–3.54倍，纯平移1h仅1.02–1.30倍；后者实际动作生成画面比自然未来编码解码更准。支持自然编码弱于动作生成路径，不能只改F。原始 `analysis/motion-code-necessity-first-hour.json`。EMA组合的在线/教师码差异在减去各自当前码后从.02033降至.000774，存在显著原点漂移；直接套原教师动作头，在线表示并未更好，不能宣称换读出就解决。原始 `analysis/ema-current-coordinate-gauge-first-hour.json`。
- 观察映射试验解码确定状态均值，Gaussian rate作正则。纯F恒等/绝对版本2h为79.56/84.60%、48.26/37.28mm、final均0；恒等组多一次D前向。两组保全完整2h及部分3h状态后退出，让卡给相对编码对照与完整RGB预测。状态相关F恒等1/2/3h78.00/85.88/88.63%、54.76/34.68/27.46mm、全0，粗指标在改善但仍未突破；保全完整3h及部分4h状态后退出，将卡给后果辨认。源 `code-state-observation-v1`；入口 `state-observation-pair-20261116` 和 `flexible-state-observation-pair-20261116`。**flexible-absolute仅准备，未启动**，需重新选空出的卡位。
- **相对状态编码已有早期对照。** 观察映射只读`zh-z0`、零变化基准0，已知当前RGB的entry rate为0。dev13版本用QR学习Q、满足QᵀQ=I，固定动作潜空间单位；dev15版本A自由学习，作同384参数头的对照。无EMA或新增loss系数，a_t仅为正向条件。各4废弃更新梯度正常、原点平移下loss/fm/rate误差0；正交版本更新后Q误差2.4e-7。正交版本1/2h77.15/77.62%、52.57/51.65mm、全0，平台后保全完整2h及部分3h状态退出；自由版本1h74.78%/57.18mm、全0，继续第二小时。尚无性能突破。源`code-isometric-relative-v1`/`code-relative-state-v1`，入口`isometric-relative-state-20261116`/`relative-state-coding-20261116`；自由版本未启动前重分配到dev15 0–3，原功能检查主机仍是dev13，未改旧证明。
- **完整RGB预测＋正交动作基已在dev6 0–3主训。** 已核实历史最佳长训参照没有RGB直通，而RGB直通分支早期较快、6h反而86.57%/33.97mm，弱于无直通参照91.35%/17.78mm。新分支恢复从学习的当前E特征预测完整未来RGB、保留原Gaussian码噪声/rate/其余loss，仅改为纯平移与学习正交动作基；冻结E均值头2304参数，和384码头分开报告。1h75.91%/56.06mm、final/any0，尚无突破。源`code-isometric-relative-v1`，入口`dense-isometric-action-20261116`。
- **两组后果辨认已启动。** 分别沿正交相对模型及原状态相关F恒等观察模型，只替换Gaussian未来码率：预测码从同一相机流的8个自然观测码中辨认实际后果，距离温度沿用sigma=.1、beta=.25及原归一化，不新增权重、网络或跨样本候选。正交相对1h的预测码误差/copy为1.93，自然编码在预测动作方向的投影幅度比仅.143，直接已知动作基读出也弱于原头，故不扩展直接几何头。新似然的手工公式、原点不变、样本隔离、塌缩状态非零编码梯度及4废弃更新通过；这不是性能证据。原状态相关F版本保留原Gaussian初始码率，未约束动作正交。两组4次功能更新均废弃，新主训全链随机、祖先0，同384参数detach头和固定标签/64DEV/32封闭确认。源`code-trajectory-likelihood-v1`/`code-trajectory-flexible-v1`，入口`trajectory-consequence-likelihood-20261117`/`trajectory-flexible-likelihood-20261117`；正常入口保留8/16恢复，不重复CPU审计。
- 宽E/浅D参照完整16h89.46%/27.26mm/2final，已保留额外部分17h全状态后退出；历史6h91.35%/17.78mm，后段未稳定保留收益。全局A＋漂移完整3h79.73%/52.43mm、绝对视角对齐5h86.08%/32.81mm、变化量对齐4h85.68%/35.38mm等已收尾。全部旧结果及完整/部分状态保留，后继均继承0更新；原始退休记录在MASTER，详情见 [CONCLUSIONS](experiments/CONCLUSIONS.md)。

## 固定边界

固定12006普通TRAIN日志，动作头仅用17920训练/1536验证的同一事实转移当前输入a_t；表示冻结/detach，动作标签梯度不进入后果学习器。输入仅RGB历史和实际执行动作，自然未来RGB/自身编码可作目标，目标图只用于控制；无未来/最优动作、专家路径、reward/state/taskID/语言或外部强教师。同步双相机属于一个物理样本，独立样本不交换表示/统计。

同64反复使用的DEV、双224 RGB、最多5命令/norm .3/native2mm final与any；32确认仍封闭。不同读出、容量、数据、教师来源、完整祖先与累计预算分别报告；同随机初值分支不是独立种子。保护性评测/oracle/接触/闭环输入不进入学习或proposal memory；不能把预测/验证误差下降当控制提升。

正常检查点保存model/optimizer/scheduler/各rank RNG/数据游标及EMA状态，实际8/16续训检查保留；不重复独立CPU恢复或广泛兼容核验，有实际错误才修。新源码用freeze_with_overrides，不修改已启动冻结源、旧检查点或vendored runtime，不把额外部分小时报作完整小时。

Git仅核心代码、必要配置、精简结论及已验证较优配置；原始实验和监控留EFS，不新增重复状态文件、不绕过ignore。未有新best-config晋升。流程边界见 [PROTOCOL](experiments/robocasa_stable_scaling/PROTOCOL.md)，原始归档见 [说明](docs/experiment-archives.md)。
