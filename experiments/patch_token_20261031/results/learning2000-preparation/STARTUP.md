# Patch-token2000：实际启动与恢复核验

09:55:35–38 UTC在原8槽启动；10:00:26一次检查：8组均已从600恢复到logged800，loss/fm/rate和模块梯度有限，predictor/renderer/action_encoder均有正梯度，冻结encoder/pool梯度为0。所有原生日志明确记录恢复完整checkpoint；尚无阶段退出、无失败，未读取任务评测。

模型source1018不变；helper3a3b249。旧46 helper/report expected_step元数据缺陷保留，新helper仅修该标注和对应合同；训练和CPU恢复审计字节不变。16次真实CPU模型/AdamW/scheduler恢复通过，8组checkpoint与RNG组件字节复制一致，next32 sampler与start0参考序列匹配。CPU审计未执行RNG restore，也不宣称分段GPU轨迹逐值等同连续训练。

进程cwd/PYTHONPATH/command和dispatcher source/config均逐项匹配；准备阶段实际模块import来自1018。未对活跃进程注入sys.modules检查。仅查询原patch8个GPU：显存116897–117017MiB，采样利用率100%。

每组先1200完整归档，再2000完整归档及既有primary/fresh评测；总72端点，无自动>2000。下次统一检查10:30:26 UTC。
