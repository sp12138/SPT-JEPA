# MetaWorld baseline study

当前只比较和分析强基线，使用 dev6、dev13、dev15，不启动我们的新算法探索。六种方法是 LeWM、LeWM+IDM（H-JEPA一层配方）、H-JEPA、SMWM、AC-MTM、LpWM，各三个独立种子、从零训练100个原生epoch。六种均为当前约束内的强方法代表，尚无同条件全50任务SOTA冠军结论。

本轮先纠正了一处会影响比较的数据问题：原适配器让短窗口方法取8帧窗口的前缀，漏掉了轨迹末段和短轨迹。SMWM只能看到约63%的可用训练帧，4帧方法约75%。这会把数据覆盖差异误认为算法差异，尤其接触、抓取往往发生在演示后段。这个问题说明原排名不可靠；它尚不能证明覆盖不足就是失败的主要原因。

正式根目录改为 `runs/metaworld_baselines_native_20261108`。十二份短窗口模型按各方法原生窗口、全部合法起点从零重跑；六份8帧 LeWM+IDM／H-JEPA 不受影响，保留有效训练。旧训练、负结果和错误均归档，不把它们当正式基线结果。修正后的60次真实终端窗口RGB/动作检查通过；当前已启动的六份新模型均通过3步训练、完整恢复、反向传播、20步规划和100步恢复检查。后续排队种子也必须通过相同检查。

原生训练窗口数不同，所以每100epoch的更新数为：SMWM67400，LeWM／AC-MTM／LpWM58600，LeWM+IDM／H-JEPA41200。共同数据、切分和评测相同，图像数量、更新次数、容量和FLOPs并不相同，最终会分别报告。完整协议见 [PROTOCOL.md](PROTOCOL.md)。原五天截止时间没有延长。

数据为独立公开 LeRobot MetaWorld MT50，按任务45/5条轨迹切分，时间相位按轨迹归组。模型只接收RGB和实际执行的forward动作；inverse重构同一已执行输入，没有目标图像到下一条示范指令的监督。没有外部预训练模型、奖励、状态、语言或task ID输入。主评测统一50任务、同一案例库、500步any-success和CEM；中期100案例、最终1000案例/模型。H-JEPA完整层次规划另设同权重对照。当前流程不是公开JEPA-WMs分数的原样复现。

已有完整有效的参考：两份历史20epoch scratch NativeLeWM，各1000个物理案例；另有零动作、均匀随机、固定训练集动作均值各1000回合。5000条真实轨迹的动作、成功数组、案例、RGB和哈希已核验；模型评测前后权重未变。

| 同一案例、500步any-success | 历史LeWM20epoch，两种子平均 | 固定动作均值 | 均匀随机 |
|---|---:|---:|---:|
| 全50任务平均 | 10.5% | 5.5% | 1.4% |
| Drawer Close | 82.5% | 100% | 30% |
| Handle Press | 72.5% | 65% | 15% |
| Button Press | 65% | 30% | 0% |
| Reach | 55% | 5% | 0% |
| ReachWall | 50% | 0% | 0% |
| Bin Picking／Door Open／Drawer Open／Peg Insert Side／Pick Place／Push，各项 | 0% | 0% | 0% |

零动作全50平均为0%。[参考原始汇总](results/reference/summary.json)、[无目标控制汇总](results/floors/summary.json)。这些历史模型的新评测与当前100epoch训练分开报告；物理案例在模型之间复用，32个评测分片也不是32个训练种子。

最可靠的第一条insight：**成功率中可能有很强的行为先验贡献。** Drawer Close固定动作均值已100%，LeWM的82.5%不能证明它理解了复杂目标；Reach／ReachWall相对固定动作的增益则明显，确有目标控制价值。因此需要同时看“绝对成功率”和“超过无目标行为的提升”，不能把容易被动作偏置完成的任务当成通用操作能力。

第二条insight：**公平比较需要对齐可获得的轨迹信息，而不只是对齐batch数。** 同一8帧起点集合对2帧和4帧模型造成了系统性末段缺失。已经纠正；旧 LeWM8%、SMWM5%、AC-MTM13%、LpWM7%的种子2/20epoch小样本数字仅为有混杂的pilot，不能用于正式原生方法排名。

第三条是待进一步检验的机制判断：**预测“记录中的未来”与判断“到目标应怎样控制”是两种能力。** 留出诊断表明多份模型的factual预测优于自身潜空间的copy，且对动作有响应；困难操作仍可能失败。这只说明单一预测误差不是充分指标，不能据此认定某一种表征、接触或规划问题就是唯一原因。旧短窗口模型还受数据覆盖混杂影响。

正式训练同时收集三个证据链：各时域forecast/copy与动作敏感性，防塌缩方式及特征分布，和同权重共同CEM／完整层次规划的真实控制差异。SMWM的动作回归、AC-MTM的动作辨别、LpWM的稀疏非负分布、H-JEPA的多尺度动作抽象各有不同机制；最终将用完整训练和相同物理案例区分作用。LeWM+IDM约54M、H-JEPA约87M、其他约18M，不能把大模型与小模型的差异全部解释为inverse。

代码固定版本：[LeWM](https://github.com/lucas-maes/le-wm/tree/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac)、[H-JEPA](https://github.com/kevinghst/H-JEPA/tree/97054ecaff83e13839195ec4b92d74c08a0127c5)、[SMWM](https://github.com/petr-ivashkov/sensorimotor-world-model/tree/46e4d3556410bf70dd9e76e1d773c52cc128704c)、[AC-MTM](https://github.com/jackboyla/action-contrastive-jepa/tree/d7b703cb515e26b39526f972355c616fe543f1c7)、[LpWM](https://github.com/YilunKuang/lpworldmodel/tree/bdd812d9432cccda8c350086006401b436f91982)。[恢复入口](RECOVERY.json)、[105份SHA核验启动证据](results/native-startup-v7/MANIFEST.json)。目前没有完成100epoch的正式基线冠军结果。
