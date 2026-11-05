# 操作失败的可视化

使用已经封存的MetaWorld 20 epoch标准CEM评测动作，按原始物理初始状态逐步重放，导出实际仿真画面。没有调用世界模型、重新规划或训练，也没有把重放当作新增独立评测案例。

选择规则为事先案例列表中的第一个场景seed42、训练seed1：bin-picking与pick-place各展示TF和同轮Native LeWM；另有TF的Reach成功对照。目标参考来自同一缓存案例的专家goal RGB，两个失败案例的专家目标构造均成功。这里的Native LeWM是本项目的MetaWorld比较模型，不是Meta官方JEPA-WM。

| 方法／场景 | 原记录结果 | 保存动作数 | 物体最终距任务目标 |
| --- | --- | --- | --- |
| TF／bin-picking | 失败 | 500 | 19.7 cm |
| Native LeWM／bin-picking | 失败 | 500 | 19.7 cm |
| TF／pick-place | 失败 | 500 | 31.3 cm |
| Native LeWM／pick-place | 失败 | 500 | 31.3 cm |
| TF／Reach | 成功 | 40 | 此任务不要求搬动物体 |

所有五条轨迹：源动作文件与案例SHA核对通过；初始完整物理数组精确一致；初始RGB通过原评测严格误差门槛；所有步骤的success布尔值与封存记录逐项一致。每次额外渲染前后核对物理数组、模拟时间与环境RNG不变。物体位置仅用于事后诊断，没有进入模型或规划器。

[取盒截图](results/bin-picking.png)显示绿色方块仍在红盒，夹爪移到了蓝盒。[抓放截图](results/pick-place.png)显示红色物体仍在桌上，夹爪已移动至目标附近。两例都没有发生有效抬起。[Reach截图](results/reach-success.png)作为能完成手臂位置控制的对照。截图能确认具体执行失败，不能单独区分编码、动力学、目标距离或规划中的唯一原因。

每张图的上排是完整相机画面，下排是三个栏目使用相同像素区域的放大图；橙圈为人工核对的操作物体位置。只增加标题、裁剪框和标注，没有改变场景内容。左列初始，中列执行结束，右列缓存的专家目标参考。

MP4位于本地 `artifacts/manipulation_visuals_20261106/replay-v2/<arm>-<task>/replay.mp4`，远端完整素材位于 `/efs/prescottsun/SPT-JEPA/reports/manipulation_visuals_20261106/replay-v2/`。每5个原始动作采一帧，视频以10fps展示；GIF末帧为便于阅读有额外停留。媒体核对索引见[MEDIA_INDEX.json](results/MEDIA_INDEX.json)，完整重放证明见[replay-summary.json](results/replay-summary.json)。视频与完整帧素材不进入Git，三张轻量展示图随报告保存。

重放脚本在dev13使用原封存的环境helper及MetaWorld/MuJoCo版本，EGL只用于渲染。首个展示尝试因系统字体路径缺失而停止，原错误保留；v2使用已安装的绘图字体完成全部五条重放。没有改动旧实验记录或当前DROID训练。
