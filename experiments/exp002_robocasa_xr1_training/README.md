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


## Matched-budget generated-geometry replay pilot

A subsequent frozen pilot replaces the odd sample slots of the original 32-step
human-only run with 16 reviewed generated-geometry policy rollouts; even slots
retain exactly the same 16 human samples. The original 32-step checkpoint is the
human-only control. Both start from the original released checkpoint and use the
same optimizer settings, step seeds and official image preprocessing. This is a
matched optimization-budget comparison, not matched human-data exposure.

The new preparation entry point is `scripts/prepare_xr1_mixed_pilot.py`.
It verifies every accepted trajectory and review SHA-256, rejects reserved
splits, selects 16 trajectories uniformly from the ordered 19-entry v2 manifest,
and excludes anchors with incomplete 16-action futures. State/RGB are pre-action;
the four history indices are t-6, t-4, t-2, t. Generated state14 is already
EE-first axis-angle and takes the explicit identity adapter plus zero-padding
to 60. Human state16 retains its dataset quaternion/reordering adapter. Both
paths check exact shape, finite values, independent state values and zero padding.
Generated trajectories are **policy pseudo-demonstrations, not human experts**.
The v2 offline stability audit freezes other joints and does not certify live
continued-control stability. Only training boxes/T/bridge enter optimization;
cross is development and elbow remains untouched test.

Measured mixed run: 32 actual optimizer steps (16 human, 16 generated), 22.88 s
for optimizer steps, 25.50 s including final development loss, peak 19.08 GiB.
Weights and optimizer state were saved outside the repository. The eight original
human development losses before training are element-for-element equal between
control and mixed runs (mean 0.3553604763). Final means are 0.3516939562 for
human-only and 0.3555818964 for mixed. This small pilot therefore does **not**
show a human-development-loss benefit from mixing. Loss is not task success.
Independent fixed-initial-state cross-family rollout comparisons remain pending.
No test-set selection or hyperparameter adjustment has occurred.

Run the preparation step with explicit external paths:
```bash
python scripts/prepare_xr1_mixed_pilot.py \
  --review-manifest "$REVIEWED_TRAIN_MANIFEST" \
  --human-manifest "$HUMAN_TRAIN_DEV_MANIFEST" \
  --output "$EXTERNAL_MIXED_SAMPLES"
```
Then supply the produced manifest to the existing short-training entry point,
keeping the original checkpoint, 32 steps and learning rate 0.000003.

## Frozen larger pilot (planned before observing new rollout results)

The next bounded run uses 256 expert-only optimizer steps per arm from the
original checkpoint, batch 1, learning rate 3e-6 and unchanged numerical recipe.
Human training expands to the original 90 training episodes, with two anchors
at 25% and 75% of each valid interval; the existing eight human-development
samples remain unchanged. The human-only schedule permutes 180 anchors with
seed 20261112, then cycles to fill 256 slots. The mixed arm keeps every even
human slot unchanged and replaces 128 odd slots with a fixed shuffled selection
from eight phase anchors in each of the 19 reviewed generated trajectories.
This compares equal optimizer/sample budgets, not equal human exposure.

The accepted-19 fixture/history audit passed for all candidates; maximum
reported sink-cache discrepancy is 3.662 micrometers, below the 50-micrometer
gate. Each trajectory also has sustained dual-finger lifting evidence.
These checks support data usability, not expert optimality.

The preregistered plan was to record fixed train-probe and human-development
losses at steps 0/64/128/192/256. In this execution a missed code insertion
left only the step-0 probe: intermediate probe curves were not collected.
Every optimizer-step loss is preserved. Final fixed-probe evaluation is run
in a fresh process after loading all 407 expert tensors. The monitor call
has now been fixed and guarded by a regression test; training was not repeated
merely to reconstruct missing intermediate metrics.
The final step is evaluated regardless of those values; no best-dev selection.
Six further cross-family development initial states (cross00 through cross05,
seeds 2001 through 2006) are frozen as a complete set before policy outcomes.
Evaluate the original model, human-only and mixed final checkpoints on all six;
do not select only failed cases or touch the elbow final-test family.
Their difficulty is unknown until baseline evaluation; calling them harder
before measurement would be unwarranted.


### 256-step run results (rollout comparison pending)

Both arms completed 256 actual optimizer steps from the same original weights.
All gradients remained finite. Initial fixed human train-probe and development
loss arrays match exactly. Each arm peaked at 21.05 GiB; elapsed training plus
final development evaluation was 184.17 s for human-only and 183.12 s for mixed.
The fixed eight human-development means were:

- Original: 0.3553604763.
- Human-only 256: 0.3427186720.
- Mixed 256: 0.3464587405.

Both reduced this small development loss, but mixed was worse than human-only.
This is not a task-success or generated-geometry generalization claim.
Final weights and optimizer state are retained externally; the original
checkpoint is unchanged. Six-case three-arm physical rollout evaluation is
handled independently using the preregistered final checkpoints.

Fresh-process reload verified all 407 expert tensors per arm; development loss
arrays exactly match the in-training final evaluation. On the identical eight
human training probes, mean loss was 0.1363142561 initially, 0.1153661637 after
human-only, and 0.1318463283 after mixed. These endpoints do not replace the
missing intermediate overfitting curves. All 512 step records have contiguous
indices and finite loss/gradient norms.

### Six-case rollout result and one-factor replay-ratio follow-up

The frozen six cross-family cases completed for all three arms. Model XML,
initial full simulation state, state14 and all three first images were exact
across arms; all 18 trajectories passed format integrity checks.

- Original: 4/6 official successes.
- Human-only 256: 5/6.
- 50% generated mixed 256: 5/6.

The shared fine-tuning gain cannot be attributed to generated geometry.
Cross05 failed for every arm. On cross04 the human-only and mixed arms needed
356 and 561 steps; success rate alone hides this slowdown. The complete human
versus mixed step pairs were 218/224, 177/176, 195/308, 373/198, 356/561, 600/600.
These are six development cases, not a definitive population estimate.

A bounded follow-up changes only the generated replay proportion to 25%:
192 original human slots and 64 generated slots, all 19 generated trajectories
represented, 256 optimizer steps from the same original checkpoint.
Slots with zero-based index i%4==3 keep the prior 50%-mixed sample; every other
slot restores the human-only control sample. Optimizer, learning rate, step
noise seeds and final-checkpoint selection remain unchanged. The existing
human-only/original/50%-mixed rollout results are reused, not redundantly rerun.
Only the new checkpoint is evaluated on all six cases.

This ratio was chosen after observing development outcomes and original-domain
losses: it is an exploratory development adjustment, not a preregistered
confirmatory result or proof that replay dilution caused failure. No cross
development trajectory enters training; the elbow final-test family stays
untouched. Failure-phase diagnosis is conducted independently from videos and
target-contact history. The repaired interval-monitor hook is used in this
new run and will be checked from actual output.

### Failure-stage evidence

Independent target-contact history confirms that cross05 had zero active
robot-target contact frames, zero dual-finger contact and no target lifting
for all three original/human/50%-mixed arms; the maximum target-bottom change
was approximately 1.9e-8 m. This is a target-reaching/selection failure, not
evidence of a failed release. Videos suggest distractor/occlusion involvement,
but that causal explanation remains a hypothesis.

For cross04, first robot contacts were at steps 99/101/102 and first dual-finger
contacts at 106/108/108 (original/human/mixed). Sustained lifting began at
118/266/484; task success followed at 192/356/561. The mixed-vs-human delay is
primarily contact-to-lift (218 steps later), not transport after lifting
(77 versus 90 steps). This supports investigating approach/grasp/lift training
coverage in independent training scenes rather than adding release-only data.
Development trajectories are used for diagnosis only, never as training labels.

### 25% replay training result

The quarter-generated arm completed all 256 updates (192 human / 64 generated)
with finite losses and gradients, saved final expert and optimizer states, and
was independently reloaded with all 407 expert keys. Reloaded development and
shared-human-probe arrays exactly match the training-end arrays.

Actual interval-monitor means (step: human training probe / human development):

- 0: 0.136314 / 0.355360
- 64: 0.122493 / 0.345137
- 128: 0.119687 / 0.348458
- 192: 0.112440 / 0.342221
- 256: 0.125747 / 0.344625

The 256-step final model remains the evaluation checkpoint; the lower monitored
192-step loss was not used to select a different model. Final human-development
loss is between human-only (0.342719) and 50% mixed (0.346459), still worse than
human-only. The monitor is non-monotonic on eight examples, not sufficient to
establish a general overfitting law. Elapsed training including interval/final
monitoring was 208.14 s, peak 21.05 GiB. The new-model six-case rollout is pending;
existing control rollouts will be reused.

### Quarter-replay rollout result and new-scene phase ablation

Quarter replay also achieved 5/6 official successes on the same frozen cases,
with steps 221/178/217/203/197/600. All six initial-state/image/XML pair checks
passed. Success rate is unchanged from human-only and 50% mixed. On the same
five successful cases its mean step count was 203.2, compared with 263.8 for
human-only and 293.4 for 50% mixed. These are descriptive development outcomes
from a single noise realization per case, not a robust efficiency claim.
Cross05 still failed. Existing controls were reused rather than rerun.

The next bounded ablation uses only new, independently accepted training scenes.
Two new arms retain the exact same 192 human slots from quarter replay and
replace only the 64 generated slots:

1. New-scene uniform: evenly spaced valid phases within each accepted trajectory.
2. New-scene phase: 32 approach anchors and 32 grasp/lift anchors, using the
   same trajectories and the same per-trajectory sample counts as uniform.

Approach anchors range from frame 6 to the metadata-derived grasp-phase start.
Grasp/lift anchors range from grasp-start minus 8 to first sustained lift plus
16, clipped so every anchor has four history frames and a complete future
16-action target. The grasp start is the recent sustained contact segment
leading to the first sustained lift; explicit fallback labels are retained.
Physics, fixture, sustained-lift history, source hashes and train/development
separation must pass before data enters either arm. Needs-review and pending
episodes are excluded even when official success is true.

Both arms use 256 updates from the original checkpoint, unchanged optimizer,
noise seeds and final-step rule. Comparing them isolates phase selection on the
same new scenes; comparing either with old quarter replay also changes scene
distribution and cannot isolate the phase mechanism. Each new final checkpoint
will use the existing six-case development protocol, without final-test access.
