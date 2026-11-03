# Meta-World：无需授权的独立数据上的方法评测

2026-11-03用户授权评测我们的方法，并明确要求寻找无需HF授权的数据。选用LeRobot独立发布的`lerobot/metaworld_mt50`，固定revision `b46a1c185a2368660650a82ac5a5083a56722f48`。原`facebook/jepa-wms`数据继续保留访问失败证据，不绕过gate、不把本数据称为它的镜像或等价复现。

实际无token下载已经通过：README、完整episode/task元数据和首个99.7MB原始Parquet；首帧480×480 RGB、4D动作均可读取。元数据声明2500episode、204806frames、50个任务注册项；全语料内容和因果时序还须单独核验。全部公开文件约45.86GB。本数据来自expert policies，与JEPA-WMs的TD-MPC2早期混合质量数据分开报告。

## 固定的比较边界

- 主方法是已有temporal-forecast，保留原sample-level目标、ρ=.8、forecast_fraction=.5，不以新数据分数调noise/anchor。记录与原PushT/Cube版本相比必需的接口适配。
- 主线RGB-only，已执行4D动作作为条件；不把动作作为最优动作预测标签。不提供39D环境真值、reward/success/任务语言给学习器。若另跑RGB+4D proprio，所有方法同轨道输入一致。
- 轨迹级、按task_id分层划分训练/验证，固定同一份清单；相邻窗口不能跨划分。任务名和task_id只用于采样/分组和环境选择。先检查历史push-back/push文本合并问题；当前50项注册不能替代全帧核验。
- 核对数据camera、翻转、原生版本、动作裁剪语义和state/action时序。使用上游公开相机与环境处理，不能靠逐任务视觉调参匹配。图像、动作scaler只用训练数据拟合并对所有方法固定。
- 基线必须在相同独立数据、输入、图像预处理、目标和评测预算下比较。现成JEPA-WM/DINO-WM权重只作明确标注的外部参考，不冒充同数据重训。预训练资源/参数/FLOPs差异单列。
- 先2-update原生训练/完整model–optimizer–scheduler恢复与GPU反传，再短程训练和完整规划；技术通过不等于任务成功。预先保存数据/模型/控制器合同后才启动正式训练。
- 第一组评估预先保留Reach/Reach-Wall及push、pick-place、drawer-open、door-open、peg-insert-side、bin-picking；逐任务报告，不只挑成功任务。之后扩展全部可验证task_id。
- 原JEPA-WMs的100原始步final-success及Farama/LeRobot的500步any-success分开，若扩展两套规则，所有方法同样评估。原published Reach分数不直接与本数据/相机结果做优越性结论。

旧PushT/TwoRoom/Cube所有HOLD/STOP和paused goal保留；不重启旧队列或周期LLM监控。当前仅数据/接口准备，尚无Meta-World训练成绩。

公开来源：[数据](https://huggingface.co/datasets/lerobot/metaworld_mt50)、[作者环境与评测说明](https://github.com/huggingface/lerobot/blob/main/docs/source/metaworld.mdx)、[历史任务文本映射问题](https://github.com/huggingface/lerobot/issues/4409)。
