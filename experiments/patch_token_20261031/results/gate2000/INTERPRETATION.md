# 2000步结论与机制边界

**保持HOLD，不继续扩展当前h64 token方案。** 两任务的primary优势均未保持：空间token相对旧flat2000为TwoRoom −18pp［−28,−9］、Cube −14pp［−24,−5］；fresh分别−24［−36,−12］、−3［−10,4］。全部seed/cases保留，区间条件于这两个固定训练seed，不能升级为普遍性结论。原flat在600→2000同时明显改善预测与规划；当前token的收益不能与其混同。

空间token相对自己的600步确实学到了条件预测。它比匹配的orthogonal token具有更低的平均factual误差和更强动作响应，但TwoRoom的primary/fresh均更差。这个反差需要进一步限定：**平均误差隐藏了短步与终点的排序变化**。TwoRoom identity的h1/zero为.353/.352，orth为.670/.663；到h5，identity为1.216/1.272，已差于orth的1.193/1.188；h7为1.346/1.378 versus1.160/1.181。旧flat的h5仅.489/.509。原生规划器的目标看末步，而非所有预测步的平均；这些曲线提供方向一致的证据，但不能认定它解释了P0：这里是initial-only cold rollout，实际planner使用history3且动作来自优化后的候选分布，二者并不相同。

不能把当前失败表述为“整体更好的动力学输给了旧flat”。旧flat2000的clean/zero在TwoRoom为.409/.416、Cube为.509/.503，均远优于token；动作交换误差比也更大。identity/orth的区别只说明局部结构有助于这项短预算factual学习，未证明更强控制。TwoRoom token接近zero级平均误差仍不是零输出：identity预测能量为.575/.607，2E〈p,y〉为.601/.610；这两个量几乎抵消，输出仍非恒零。所有horizon能量分解均在summary，无验证均值拟合。

取消固定global输出子空间的数学上限，并不保证获得更好的可学习动力学。本候选还把predictor从11,389,184参数改为523,536，并增加约3.17倍forward MAC；无法把失败单独归因于空间token、秩、容量或优化中的任何一项。已有全局span诊断保持有效，当前结果不支持继续用同一变体堆预算，也不否定公开patch predictor的一般可行性。没有新颖性或成熟优胜声明。
