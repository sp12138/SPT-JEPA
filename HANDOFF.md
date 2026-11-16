# 当前交接

目标：从零 SPT-JEPA 的独立动作头控制出现可靠稳定的 scaling，并超过已有参照。**尚未达成**；两点上升、loss 下降或零星成功都不算达标。用户要求时间优先算法机制、关键实验和实际控制，检查与记录最小化，避免重复兼容回归或 CPU 审计。

只用 dev13、dev15、dev6；dev6-2c/grid16/dev2 不连接。当前主会话每小时检查并按证据改进，不在有限 pilot 后自行 HOLD；不恢复旧 goal、监控或第二套 LLM 监督。实际分配和检查入口为 R/MASTER.json、R/SESSION_CHECKS.json，不能根据旧 ACTIVE 文字恢复任务。

R=`/efs/prescottsun/SPT-JEPA/runs/robocasa_reach3d_scaling_20261114`；模型根在其父目录，下表后缀均加 `robocasa_reach3d_`。

| 主机/卡 | 根后缀 / run name | 当前方法；可训练参数 / 动作头参数 |
|---|---|---|
| dev13 0–3 | `flexible_mean_identity_20261116` / spt-flexible-mean-identity | 状态相关 F＋恒等观察映射；41.625M / 384 |
| dev13 4–7 | `ema_action_transport_20261116` / spt-ema-action-transport | 自身 EMA 当前码＋共享动作平移；40.810M / 384 |
| dev15 0–3 | `mean_state_identity_20261116` / spt-mean-state-identity | 动作平移＋恒等观察映射；40.810M / 384 |
| dev15 4–7 | `ema_initial_target_20261116` / spt-ema-initial-mean-readout | 自身 EMA 当前码＋状态相关 F；41.625M / 2304 |
| dev6 0–3 | `mean_state_absolute_20261116` / spt-mean-state-absolute | 动作平移＋绝对观察映射；40.810M / 384 |
| dev6 4–7 | `pure_action_transport_20261116` / spt-pure-action-transport | 无自主漂移的共享动作平移；40.810M / 384 |

## 决策所需的关键结果

- **保留纯动作平移继续 3/4/8h。** `z_next=z+A*a_t` 的 1→2h 从71.70%/67.51mm升至86.30%/29.59mm，native final/any仍全0。同384参数码头、同预算带漂移对照2h为79.79%/49.56mm；有粗控制改善，尚非稳定精细 scaling。源 `code-pure-action-transport-v1`。**不要为了启动已准备的 flexible-absolute 对照而停这个分支。**
- EMA未来码1/2h仅49.27/49.96%、124.00/123.78mm、全0，已保全完整2h及部分3h状态后退出。相反方向的EMA当前码首小时73.60%/63.18mm/0final/1any；EMA当前码＋平移组合首小时57.24%/101.00mm/全0；两者继续第二小时判断 scaling。均为自身同数据随机初始化模型的EMA，decay=exp(-32/120060)，没有外部教师。源 `code-ema-initial-target-v1`，读出容量分别2304/384，不能混报。
- 同8普通TRAIN窗口，未来码换成当前码，原RGB1h解码误差增至2.21–3.54倍，纯平移1h仅1.02–1.30倍；后者实际动作生成画面比自然未来编码解码更准。支持自然编码弱于动作生成路径，不能只改F。原始 `analysis/motion-code-necessity-first-hour.json`。EMA组合的在线/教师码差异在减去各自当前码后从.02033降至.000774，存在显著原点漂移；直接套原教师动作头，在线表示并未更好，不能宣称换读出就解决。原始 `analysis/ema-current-coordinate-gauge-first-hour.json`。
- 观察映射试验使用确定状态均值，恒等版本为 `D(x0,z0,zh)=x0+g(x0,zh)-g(x0,z0)`，绝对版本为对照；原Gaussian rate保留作正则，不宣称完整像素ELBO。各4废弃更新梯度正常，同状态解码误差0；表中三个观察映射分支已主训，尚无完整控制成绩。恒等版本多一次D前向，额外计算单列。源 `code-state-observation-v1`；入口 `state-observation-pair-20261116` 和 `flexible-state-observation-pair-20261116`。**flexible-absolute仅准备，未启动**，需重新选择合适空出的卡位。
- 进一步准备了**相对状态编码**：观察映射只读 `zh-z0`、零变化基准为0，已知当前RGB的entry rate为0；纯平移下整个loss不依赖每条物理相机流的任意码原点，无EMA或额外loss系数。4废弃更新、非零梯度、恒等解码及随机平移下loss/fm/rate误差0均通过；**仅准备，未主训，无性能结论**。源 `code-relative-state-v1`，入口 `relative-state-coding-20261116`，等证据和卡位，不停掉正在改善的纯平移。
- 宽E/浅D参照完整16h89.46%/27.26mm/2final，已保留额外部分17h全状态后退出；历史6h91.35%/17.78mm，后段未稳定保留收益。全局A＋漂移完整3h79.73%/52.43mm、绝对视角对齐5h86.08%/32.81mm、变化量对齐4h85.68%/35.38mm等已收尾。全部旧结果及完整/部分状态保留，后继均继承0更新；原始退休记录在MASTER，详情见 [CONCLUSIONS](experiments/CONCLUSIONS.md)。

## 固定边界

固定12006普通TRAIN日志，动作头仅用17920训练/1536验证的同一事实转移当前输入a_t；表示冻结/detach，动作标签梯度不进入后果学习器。输入仅RGB历史和实际执行动作，自然未来RGB/自身编码可作目标，目标图只用于控制；无未来/最优动作、专家路径、reward/state/taskID/语言或外部强教师。同步双相机属于一个物理样本，独立样本不交换表示/统计。

同64反复使用的DEV、双224 RGB、最多5命令/norm .3/native2mm final与any；32确认仍封闭。不同读出、容量、数据、教师来源、完整祖先与累计预算分别报告；同随机初值分支不是独立种子。保护性评测/oracle/接触/闭环输入不进入学习或proposal memory；不能把预测/验证误差下降当控制提升。

正常检查点保存model/optimizer/scheduler/各rank RNG/数据游标及EMA状态，实际8/16续训检查保留；不重复独立CPU恢复或广泛兼容核验，有实际错误才修。新源码用freeze_with_overrides，不修改已启动冻结源、旧检查点或vendored runtime，不把额外部分小时报作完整小时。

Git仅核心代码、必要配置、精简结论及已验证较优配置；原始实验和监控留EFS，不新增重复状态文件、不绕过ignore。未有新best-config晋升。流程边界见 [PROTOCOL](experiments/robocasa_stable_scaling/PROTOCOL.md)，原始归档见 [说明](docs/experiment-archives.md)。
