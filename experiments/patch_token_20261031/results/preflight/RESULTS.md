# Patch token：两步技术预检

16/16实际exit0；8组各执行原生两步，随后严格fullstate/FP32梯度审计全部通过。调度器空闲后关闭并HOLD。未启动600训练。

|task / seed|arm|native loss at2|peak GiB|predictor grad|action encoder grad|
|---|---|---:|---:|---:|---:|
|tworoom / 1|identity|2.816|112.86|0.2156|7.49e-17|
|tworoom / 1|orthogonal|2.816|112.97|0.3529|2.206e-17|
|tworoom / 2|identity|2.837|112.86|0.2442|1.11e-16|
|tworoom / 2|orthogonal|2.837|112.97|0.3437|3.226e-17|
|cube / 1|identity|2.258|112.86|0.2887|4.284e-17|
|cube / 1|orthogonal|2.258|112.97|0.2626|1.337e-17|
|cube / 2|identity|2.266|112.86|0.2977|4.539e-17|
|cube / 2|orthogonal|2.266|112.97|0.2843|1.996e-17|

466个原nonpredictor初始张量、542个成对parameter张量（含冻结参数）、原校准/source/config及portable/fullstate严格核验；原生模型/AdamW/scheduler恢复和sampler cursor256通过。predictor全部分层条件/注意力/MLP梯度组实际非零且有限，未添加额外backward。

每训练seed的predictor初值与CPU证据逐字节一致，seed1/2不同。head零初始化的早期梯度延迟不是失败；两步通过也不是功能性预测/规划成功。保留原日志的NFS finalizer警告与真实exit，不以日志stage_complete代替OS exit。

唯一批准范围到此结束。完整逐seed/分层梯度、参数状态、hash和关闭记录见summary.json；原始checkpoint与报告留在EFS。
