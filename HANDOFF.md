# 当前交接

目标仍是让从零SPT-JEPA在独立动作头控制下出现可靠稳定的scaling，最终超过RoboJEPA；尚未达成。用户最新两次强调：**时间优先投入算法机制、关键实验和实际控制性能；常规检查最小化，不重复广泛兼容回归或独立CPU审计，记录只留核心结论与较优配置。** 有实际错误再修。

只用dev13/dev15/dev6。旧goal、旧监控及dev6-2c/grid16/dev2不恢复。当前主会话每小时检查并改进，不在有限pilot后HOLD。小时入口为R/SESSION_CHECKS.json，实际在跑分配以R/MASTER.json及进程为准，不根据旧文档ACTIVE恢复。

R=`/efs/prescottsun/SPT-JEPA/runs/robocasa_reach3d_scaling_20261114`；模型根在R父目录。下表根后缀均加`robocasa_reach3d_`。

| 主机/卡 | 根后缀 / run name | 当前算法 |
|---|---|---|
| dev13 0–3 | `encoder_allocation_20261116` / spt-encoder-heavy-shared-mean | E384/D6参照，41.625M，2304参数E均值头 |
| dev13 4–7 | `global_response_strong_encoder_20261116` / spt-encoder-heavy-global-action | 共享动作方向+非线性自主变化，40.834M，384参数状态码头 |
| dev15 0–3 | `synchronized_increment_target_20261116` / spt-synchronized-increment-target | 同步相机变化量对齐，41.625M，2304参数E均值头 |
| dev15 4–7 | `ema_initial_target_20261116` / spt-ema-initial-mean-readout | 自身EMA已知当前码、在线未来码，41.625M可训练，2304参数EMA骨干均值头 |
| dev6 0–3 | `ema_future_target_20261116` / spt-ema-future-mean-readout | 新启动自身EMA未来目标；41.625M可训练，2304参数EMA骨干均值头 |
| dev6 4–7 | `pure_action_transport_20261116` / spt-pure-action-transport | 去除自主漂移的共享动作平移，40.810M可训练，384参数状态码头 |

## 关键进展与下一步

- 宽E/浅D参照从1h81.83%/42.23mm改善到6h91.35%/17.78mm，14h90.77%/22.60mm/1final，后段未稳定保留收益，native2mm末端成功稀少。不能把粗回报、验证loss或零星成功当达标。RGB恒等支路早期改善，但后段平台/回落；已保全完整8h控制及额外部分9h的完整训练状态，后者不报完整9h、不继承给新模型。
- 绝对视角对齐1/2/3/4/5h为70.21/80.58/82.74/84.45/86.08%，5h32.81mm、final/any全0。对齐伴随运动码变化变弱；1h直接读实际码也仅59.83%，低于RGB同384参数码头86.40%，不是均值头漏掉好表示。完整5h状态及结果保全后退出，将卡交给EMA当前码对照。
- 变化量对齐只交换`mu_h(other)+mu_0(source)-mu_0(other)`，保留相机原点。1/2/3h80.08/86.03/83.73%、48.43/31.77/39.41mm、final全0；1h码头79.11%/50.61mm/0final，仍弱于RGB。8普通TRAIN窗口显示变化量转运改善、相对绝对对齐保留更多运动信号，但不足以带来控制优势。继续看完整后续预算，不扩大读出搜索。原始定位R/analysis/`root-increment-first-hour-motion-transfer.json`及`root-increment-code-probe-20261116`。
- 共享动作方向用`z_next=z+f(z)+A*a_t`；旧弱骨干联合状态上的失败保留。当前强骨干1/2h72.59/79.79%、61.60/49.56mm、final0/1。A的XYZ列已有非退化秩，但同一动作头应用到其前向预测时与自然观测的读出严重失配：1h验证观测39.32mm、预测码1514.76mm；2h为34.94/673.74mm，响应矩阵距单位阵仍13.35。均为内部诊断，不是环境真实位移。下一单独对照将`f(z)`去掉，强制`z_next=z+A*a_t`；其余损失/数据/随机初值/384参数码头与共享A组一致，无新增动作标签或损失权重。4废弃更新通过梯度与方程检查，主训已在dev6 4–7启动，尚无控制成绩。源`code-pure-action-transport-v1`，入口`pure-action-transport-20261116`。
- 固定一阶空间矩线性读出已完整结束：同父1/4/8h为79.34/87.56/89.50%、48.54/28.49/23.89mm，final全0；均弱于原均值头，参数6912对2304。0body更新、同17920/1536事实标签、192次模型—案例仍是同64物理DEV。R/`encoder-spatial-moments-probe-20261116/DONE.json`。此前相对误差、MAE、非线性及空间Potential等负结果保留，不继续堆头/调参。
- **EMA未来目标首小时失败：49.27%/124.00mm/0final/0any，事实动作验证读出47.71mm。** 原改动停止了全部未来编码梯度，学生只有当前帧可训练，可能切断了表示学习的重要信号；该解释仍待对照。新分支将EMA放到已知当前码，学生编码全部帧，未来码保留RGB/预测学习梯度，其他损失/数据/decay不变。4废弃更新确认未来码梯度非零、EMA当前码停止梯度；主训已在dev15 4–7启动，尚无控制成绩。两个分支均为自身随机初始化encoder/pool的EMA、decay=exp(-32/120060)、冻结EMA骨干均值读出，无外部教师/动作标签进入body。源码`ema_future.py`；入口`ema-future-target-20261116`与`ema-initial-target-20261116`，各自冻结源同名`code-…-v1`。未来目标旧分支继续看第二小时，不将单点失败当最终scaling结论。

## 固定口径与执行边界

固定12006普通TRAIN日志，head只用17920训练/1536验证的同一事实转移当前输入a_t；表示冻结/detach，动作标签梯度不进body。学习器仅RGB/实际输入动作，自然未来RGB/自身编码可为目标；无未来/最优动作标签、专家路径、reward/state/taskID/语言或外部强教师。EMA只来自自身同数据训练。

同64反复使用的DEV、双224 RGB、最多5命令/norm .3/native2mm final与any；32确认仍封闭。不同头容量、代码/均值读出、数据/预训练/祖先预算分开报告。所有新主链祖先0，同随机初值的分支不算独立种子。

模型、优化器、调度器、各rank RNG、数据游标及EMA权重/计数随正常检查点保存；实际8/16续训验证保留。小时检查只记录必要状态和性能，不再堆兼容/审计文件。发现真实错误才修，不改已启动冻结源；源码新版本用`freeze_with_overrides`，不写穿共享链接或修改vendored runtime。

小E完整9h205932步后退役；旧视角联合组完整9h124117步后退役。root条件先验完整6h为91.35%/21.57mm/1final/2any，保留后期回升事实，但尚无稳定精细控制优势；完整状态保留后将卡交给无漂移对照，后者继承0更新。当前其他旧实验不自动恢复。历史与负结果见[CONCLUSIONS](experiments/CONCLUSIONS.md)和[归档说明](docs/experiment-archives.md)，口径见[PROTOCOL](experiments/robocasa_stable_scaling/PROTOCOL.md)。Git仅核心代码、必要配置、精简结论；未有新best-config晋升。
