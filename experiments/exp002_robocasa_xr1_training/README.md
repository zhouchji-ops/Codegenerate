# exp002: XR1 真实数据短训与开发评测

## 范围

目标是跑通“已发布专用权重 → 多样真实示范 → 实际参数更新 → 保存重载 → 独立开发评测”，不是宣称模型性能提升。模型仍为 Xiaomi-Robotics-1-RoboCasa365；本轮没有把尚未完成物理审核的新几何 rollout 混入成功监督，也不进行 RL 参数训练。

代码：
- `scripts/xr1_real_step_diagnostic.py`：严格权重覆盖、官方固定输入对照、真实一步更新及新进程重载检查。
- `scripts/xr1_short_train.py`：官方精确图像预处理、按 episode 隔离的多样示范短训和固定噪声开发集 loss。
- `scripts/serve_xr1_expert.py`：仅监听 loopback 的官方推理服务包装器，可加载经过键名/形状/有限性检查的 expert 权重，并固定推理 RNG。

## 固定来源

- OpenTau：TensorAuto/OpenTau，commit `64544b98711534f14c1ba714672e0c490e589063`，Apache-2.0。使用其中 XR1 模型和状态适配器，不复制源码进本仓库。
- 官方 XR1 代码：commit `0dd7aef8dc87296246aae812a1f59ccb708e5546`。
- 权重：`XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365`，本地三分片，Apache-2.0。HF 权重 revision 尚未独立核实，不能用代码 commit 冒充权重 revision。
- human 数据：`pepijn223/robocasa_pretrain_human300_v4`，HF revision `3f72c9307b4a132dd55e1433e2dda0f54bb8c1d0`。保留 Hugging Face 数据卡及上游 RoboCasa 来源；数据卡标 Apache-2.0，上游资产/数据另有 CC BY 4.0 要求，不将二者混为一种许可。
- 数据仅取 CounterToSink 的 108 个回合：episode 12127–12234，22410 步。原始四个下载文件已由数据准备流程校验 SHA；相机视频实际为 AV1，使用 PyAV/libdav1d 解码，不使用该旧环境无法解码 AV1 的 OpenCV。
- RoboCasa：commit `100f29b8efd239e807e62f2678a1a8489c524297`；MuJoCo 3.3.1；robosuite 1.5.2。不升级既有 benchmark 环境。

## 接口与数据隔离

human state16 顺序为 EE xyz、EE xyzw quaternion、base xyz、base xyzw quaternion、gripper2。用 `robocasa365_dataset` 转为 EE-first state14，再零填到60；不能与 simulator 的另一种 state16 顺序混用。独立使用官方四元数函数重算，已核样本差为0。action 为原始12维，零填60并仅对前12维置有效 mask。官方 checkpoint 动作 mean=0、有效维 std=1，状态不做统计归一化。

每样本使用同一回合的历史 t−6/t−4/t−2/t 和未来动作 t:t+16。三相机依次 left/right/wrist，RGB256。真实短训走官方 evaluator 的 PIL 0.95裁剪和 HF processor 路径；不是 OpenTau 近似 tensor resize。跨相机按元数据时间戳寻址，PTS误差检查小于1ms，禁止把文件全局帧索引当回合帧索引。

先90回合作训练池、后18回合作开发池，再均匀选择32个训练回合、8个开发回合，每回合一个不同时相样本。episode 严格互斥，开发集不是最终测试集。样本 manifest、NPZ和原始视频只在外部数据目录。

## 一步加载/更新/重载验证

- 三分片实际载入1120个键；仅缺共享权重的 lm_head 一端，随后恢复 tie；没有意外键或未加载的实际参数。
- expert-only 可训练参数615333888；辅助 choice heads关闭，匹配公开专用权重。
- 初始单样本官方对照：tokens、attention mask、video grid一致；OpenTau tensor resize的像素最大差为0.0078433，固定种子动作最大差0.0078125，平均差0.00089775。因此不声称 bit-exact 移植。
- 用真实 human episode12127/t6 做一次 AdamW 更新：loss 0.049046736，clip前梯度范数19.033285；输出投影1048576个元素变化，最大变化3.03015e-5。不是随机输入或仅 loss 函数测试。
- 新进程重新加载407个 expert tensors，FP32逐项完全一致；转BF16后推理有限。相对原参数固定输入动作平均变化0.021827，仅证明更新生效，不证明策略变好。
- 随后的32步短训改用官方完整预处理，消除上述像素路径差异。仍需闭环评测，不能从一次输入证明所有数值等价。

## 32步真实短训结果（2026-11-12）

- 从原始专用 checkpoint 重新开始，不继承前面的一步实验。
- 32个不同训练回合各1样本，32个 optimizer steps；batch=1、expert-only、training_repeat=1、async prefix关闭；AdamW lr=3e-6、betas=(0.9,0.95)、weight_decay=0.1、grad clip=1。不是 OpenTau 官方完整训练配方的复现。
- 参数/optimizer为FP32，前向使用BF16 autocast；VLM冻结。训练随机种子按 step固定，开发集固定 noise 和 time=0.5。
- 8个开发样本 flow loss 均值：0.355360476 → 0.351693956，约下降1.03%；5个样本下降、3个上升。样本太小，不作显著性或任务成功率结论。
- 32步训练23.46秒，包含训练后开发集检查共26.13秒；峰值显存19.08GiB。权重和 optimizer 状态均已保存到外部目录。
- 所有step原始输出保留。首版JSONL写入器误把换行编码为字面字符，已修正代码；恢复的逐行文件必须从完整原始记录解析、逐项校验，不能改写loss。

## 闭环开发评测

冻结 native PickPlaceCounterToSink，pretrain split，seed10/11，外层horizon600；模型RNG分别1010/1011，原模型与短训模型使用同一官方推理服务与参数转换。原始轨迹、初始XML和初始sim state保留以核查配对。600步只是本开发检查预算，不叫官方榜单预算。内部done/truncated可能提前结束，必须记录实际步数。

首批四回合已完成，初态审计与结果见下节。不得用开发loss下降替代任务结果。

## 运行与环境

本轮在独立 venv 进行，继承既有 torch2.8.0 / transformers4.57.1 依赖，补充的数据包仅安装于该 venv；没有改动既有模型/仿真环境。OpenTau项目元数据声明Python3.10，本轮最小源码接口实际在Python3.12验证，不能声称完整官方安装矩阵兼容。已固定 datasets3.6.0、huggingface-hub0.36.2；直接安装最新datasets会与现有Transformers冲突。

显式选择已授权GPU。所有数据、checkpoint、NPZ、日志和视频必须在仓库外。示例变量均指向使用者自己的环境；未经验证不要盲目升级依赖。

```bash
export CUDA_VISIBLE_DEVICES=1
export HF_HUB_OFFLINE=1 WANDB_MODE=disabled PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$OPENTAU_SOURCE/src"
python -B scripts/xr1_real_step_diagnostic.py --checkpoint "$XR1_MODEL" --batch "$ONE_BATCH" --xr1-repo "$XR1_REPO" --output "$NEW_DIAGNOSTIC"
python -B scripts/xr1_real_step_diagnostic.py --checkpoint "$XR1_MODEL" --batch "$ONE_BATCH" --xr1-repo "$XR1_REPO" --output "$NEW_RELOAD_CHECK" --verify-expert "$EXPERT_CHECKPOINT"
python -B scripts/xr1_short_train.py --checkpoint "$XR1_MODEL" --batch "$ONE_BATCH" --manifest "$BATCH_MANIFEST" --xr1-repo "$XR1_REPO" --output "$NEW_SHORT_TRAIN" --steps 32 --lr 0.000003
```

脚本的首批训练/重载已经实跑；公开仓库不附数据或权重。继续规模化前需补齐更多初态上的配对结果，确认保留能力与新几何数据的审核准入。RL方向可复用冻结任务、状态轨迹和独立success/物理诊断，但奖励与复位协议另行设计。

## 首批闭环结果及初态审计

原模型 seed10/11 均成功，分别136/178步；短训后也均成功，分别172/183步。四条都是官方 success，仍需独立物理审核；没有成功率提升证据，不能因 loss 小降声称策略改善。

初态读回审计发现：
- seed10：完整 sim state 最大差1.234102，state14及三相机首帧不一致，因此不是严格配对，步数差也不能归因于训练。
- seed11：完整sim state、state14、三相机首帧逐项完全一致；XML只在碰撞几何rgba上有差异。它支持这一初态的观测/物理状态对照，不代表大样本效果。
- 同seed并不足以冻结环境；后续用保存XML+sim state建立规范重建初态，并逐帧核验。
- 首个快照恢复探针state精确恢复，但与旧首帧平均相差约0.006–0.009灰度级，且旧动作前5步复演与旧轨迹不完全一致，说明仅XML和flattened state不足以声称旧过程严格重放。随后已验证两次独立新重建的首观察完全一致，且同一前5动作的完整sim state序列逐项相等；这只证明新规范协议内的重复性，不等于旧轨迹可严格回放。

存储提速只改变HDF5图像chunk，不改变输入/动作/成功判定。最后一回合使用逐帧chunk，记录耗时降至1.28秒；不同回合的总耗时不能直接当同轨迹速度提升。32步日志恢复副本已与stdout的32条记录逐项完全核对，原始文件保留。

## 规范同初态复测：已完成

新增 `scripts/restore_rollout_snapshot.py`。先恢复保存XML和完整sim state，forward后重置各机器人composite controller，再强制刷新观测。两次独立重建均校验三首图和完整前5步同动作轨迹完全相等。

首次短训模型复测在0次推理时因腕相机少量1灰度级渲染差异被硬检查拦下，失败记录保留。之后显式采用冻结规范首观察选项：物理state仍须精确相等，实时渲染与规范图最多允许每相机8个数值、1LSB的差异，然后将保存规范图作为首观察；后续实时渲染不替换。**最终成功执行的这一回合，替换前的三首图本身也已完全相等，实际像素替换差为0。** 不把该选项称为消除一般渲染随机性的方法。

最终读回原模型与短训模型的两条新轨迹：
- model XML 字节完全相同。
- 完整初始sim state、state14以及三首图逐项完全相同。
- 首16动作最大差0.021484375，说明训练权重确实生效，而非误加载原模型。
- 原模型成功137步，短训模型也成功137步；**没有改善证据**。
- 两回合约23.77秒、25.16秒；其中记录约0.96秒、1.00秒。计时不是独占硬件吞吐基准。

这些新规范结果不覆盖旧不匹配seed10结果。全部为开发验证；未把新几何未完成物理审核的轨迹用于训练，也未扩大或偷看最终测试集。

示例：
```bash
python -B scripts/restore_rollout_snapshot.py --snapshot "$BASELINE_ROLLOUT" --collector scripts/collect_rigid_rollout.py --canonical-reference "$CANONICAL_IMAGES" --probe-only
# 先独立重复probe，核对完整同动作轨迹；模型服务须另行启动
python -B scripts/restore_rollout_snapshot.py --snapshot "$BASELINE_ROLLOUT" --collector scripts/collect_rigid_rollout.py --canonical-reference "$CANONICAL_IMAGES" --freeze-initial-observation --xr1-repo "$XR1_REPO" --model "$XR1_MODEL" --output "$NEW_ROLLOUT" --port 10217 --seed 10 --horizon 600
```

采集脚本和适配器来自同仓库 exp001 工作分支；依赖该分支的文件，不能仅取本PR后声称所有示例独立可运行。

