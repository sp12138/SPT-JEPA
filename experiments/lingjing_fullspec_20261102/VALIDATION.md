# 执行与测量验证

## 已通过

- 固定NN源0196dc17的完整 `tests/as_jepa`：145 tests，144通过、1个依赖可选历史的skip。第一次源导出缺历史Git引用/运行时provenance而失败的日志保留；第二次在独立验证目录补checksum绑定的历史参考输入，测试与NN代码未改。见 [CPU证明](results/cpu-tests-v2.json)。
- 作者模型类、forecast/anchor/noise/readout等合同已覆盖；新增3项诊断合同（包含forecast动作对齐错误的反例）通过，固定helper3再次执行通过。见 [helper3测试](results/full-tests-v3.json)。
- 14个原生200步技术smoke：完整CPU model/AdamW/scheduler恢复、finite权重/日志/梯度；真实DiT/DF/NCE接口。它们不当成名义预算性能结果。见 [全部smoke与冻结诊断](results/gates-summary-v1.json)。
- 全部36条新增2000步链 exit0，并保留完整全状态；所有主要活跃模块有正有限梯度，冻结诊断不改模型状态。算法门槛结果失败与程序exit失败分开记录。见 [完整短门槛](results/hard2000-summary.json)。
- 实际GPU接口：posterior/forecast×BF16/FP32 collapse、latent_eval、colour invariance；其后新增ForecastScored/DF/NCE一episode PushT规划、两case Cube correlated controller、视频质量两case两Euler步均可执行。所有小规模接口只验证可运行，不能据此声称任务性能。见 [第一组](results/stage-interfaces-v3.json)、[第二组原始结果](results/full-interfaces-v3.json)、[修复后的最终证明](results/full-repair-interfaces-v4.json)。
- 修复后的Cube姿态读出使用固定128训练/64验证clips，并校验原数据/行索引/动作和纯分析函数SHA、前后NN state hash；无模型更新。这里是完整固定cohort，已产生描述性早期物理读出，见 [pose原始结果](results/cube-pose-interface.json)。
- 主队列12个Cube实际CPU复制前/后各恢复一次，共24次模型/优化器/scheduler恢复通过；12个原生训练子进程均记录 `Restored all states`，第一次真实恢复步是2001，未重置到0。首批另9个PushT真实从步1开始。21个native的实际cmd/cwd/PYTHONPATH/CUDA及模型/工厂/数据源SHA全部核对，见 `results/main-startup-*-v1.json`。
- **2026-11-02 10:02:00–04 UTC**：21个主队列模型均已新增至少100步，最近最多5条日志损失与必须活跃的梯度全有限/正；12个Cube在2200、9个DF PushT在2000–2300。三个原始模板L0基线实际训练在62000/43900/43800，全部finite/正活跃梯度。共24个实际训练进程。见 `results/main-health100-*-v2.json`。这是一次启动跟进，非连续监控；不能推断后来的GPU利用率或训练结局。

## 保留并修复的实现错误

1. 全套CPU测试最初缺源导出未携带的历史参考；独立输入修复如上，不作为算法失败。
2. 新forecast collapse工具把已经截断的动作再次交给会内部切片的forecast helper，造成长度差1。改为传完整W动作，加入明确长度guard和CPU反例，四个真实精度/条件接口复测通过。老错误日志保留。NN的forecast实现没改。
3. stage-helper-v3初次登记误指向v2描述文件；保留错误写入后，从早已冻结的protocol恢复精确v2描述，另建v3描述，所有运行中的协议内嵌值与模型快照未变。EFS `descriptor-repair-v3.json`。
4. full-interface-v2误写Cube目录名，在启动GPU任务之前读config即失败；修正名字的新目录重做，未改模型参数。
5. full-interface-v3的Cube姿态探针在dataset仍打开普通HDF5读句柄时另开SWMR读句柄，HDF5拒绝。关闭已完成的dataset句柄后，在独立helper3与输出目录重做通过，原失败保留。其余五个已通过接口不重复做无用计算。
6. full-helper-v3的初次bootstrap误用旧描述文件名；exclusive open拒绝覆盖，旧v2记录未被改。逐字节验证已经解压的v3文件后，用显式v3描述名继续。见 [登记修复](results/full-setup-v3-registration-repair.json)。

未启动的helper1/2候选和main-v2协议均保留，main-v2已HOLD。实际训练源始终0196dc17；上述修复不改变模型、loss、学习率、batch、输入信息或已运行检查点。主队列helper为e9281dd；baseline仍用自身独立53010bb2执行包。实际运行中的快照禁止原地修改。

## 持续成本与失败处理

新任务使用有终点的本地队列与结束汇总：主队列72小时上限，baseline协调器48小时上限；每阶段训练/评测也有明确timeout。不是周期性唤醒LLM的monitor，不恢复旧goal。正常结束写summary/HOLD；技术错误写terminal/log，不偷改为算法负结果或自动换参数。超时、失败或主机异常时先读已有PID/日志/退出记录，不盲目重启或清HOLD。
