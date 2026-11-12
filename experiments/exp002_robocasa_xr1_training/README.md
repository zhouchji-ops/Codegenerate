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

The new-scene admission completed with 9 accepted and 3 excluded trajectories:
4 T-family and 5 bridge-family training scenes. All nine have metadata-aware
physics/history, fixture consistency, sustained dual-finger pickup and exact
initial-state restoration checks; none needed the grasp-start fallback.
The accepted-manifest SHA-256 is
`af35ff258882b80d196325a2afec23c71142fef67cb8eae52ff4ce7a552b8e88`.
Both 256-step arms were launched only after these checks. A separate schedule
audit confirms identical human slots, identical development samples and
identical per-episode counts between arms. The preparation entry point now
also rejects any family other than T/bridge, even if a manifest is mislabeled.

### New-scene uniform versus phase: completed training

Both arms completed 256 updates, saved expert and optimizer states, and passed
fresh-process reload checks of all 407 expert keys. Reloaded eight-development
and shared-human-probe arrays exactly match training-end arrays. All 512
optimizer records are finite; 0/64/128/192/256 monitoring is present for both.

- New-scene uniform: human dev 0.3541095592; human train-probe 0.1271307115.
- New-scene phase: human dev 0.3444956187; human train-probe 0.1211182429.

Both started with dev 0.3553604763 and probe 0.1363142561. Phase selection has
lower human-development loss than uniform on these same new scenes, but neither
loss alone demonstrates improved task success. Training with monitoring took
207.21 / 207.60 s, each peaking at 21.05 GiB. Final step 256 remains the checkpoint
for both, regardless of intermediate fluctuations.

Independent review identified two future-input guard gaps: an accepted manifest
could contain duplicate run IDs, and a base schedule could contain 64 generated
slots in positions other than every fourth slot. Direct audits of the actual
run manifests confirmed 9 unique runs and exact 192-human/64-generated source
patterns in the base and both new arms, so this run is unaffected. Explicit
duplicate-run and shifted-slot rejection tests/guards have been added without
changing the running experiments. Twelve new fixed-development rollouts are
delegated, reusing old controls rather than retraining or rerunning them.

### New-scene rollout result: phase-only emphasis regressed

All twelve new rollouts passed trajectory integrity and exact initial XML,
simulation-state, state14 and three-camera checks against the frozen controls.

- New-scene uniform: 5/6, steps 223/184/224/198/194/600.
- Early-phase emphasis: 4/6, steps 600/175/227/270/186/600.

The phase arm lost cross00; cross05 remained unsolved. Uniform's mean steps on
the five successful cases was 204.6, close to prior quarter replay's 203.2.
Thus neither new-scene data nor early-phase emphasis has demonstrated a success
gain, and the phase arm has a concrete regression.

Cross00 metadata-aware replay shows phase contacted/lifted at 99/104, lost
finger contact at 134 under a roughly +1 close-gripper command, and landed on
high sink geometry at 137. It stayed there until roughly 519, regrasped, and was
still holding at step 600. Uniform maintained grip until intentional -1 opening
at 205, reached the basin at 207 and succeeded at 223. This points to grasp
stability during transport, not a demonstrated premature-open command.

Read-only expert audits found 407 valid keys in all three checkpoints.
Relative update L2 norms versus the original expert were about 0.2293% (human),
0.2297% (uniform) and 0.2289% (phase); uniform/phase update cosine was 0.7441.
These magnitudes/directions are not proof for or against catastrophic forgetting.
On fixed uniform cross00 observations at t0/120/180, three shared noise seeds
and four model versions gave 36 offline forwards with identical RNG hashes.
At t120 all predicted gripper values were near closed; phase was not earlier
to open at t180 either. Fixed uniform observations are not phase's own t134
state or its original online RNG/chunk sequence, so no counterfactual-success
claim follows. Evidence supports testing retained transport/pose skills rather
than asserting global forgetting.

### One bounded full-stage retention check

The next arm keeps the same nine accepted scenes, 192 human slots, 64 generated
slots, original checkpoint, 256 updates and fixed final-step rule. The 64
generated anchors allocate exactly 16 each to approach, grasp/lift, holding
during transport, and the successful episode tail. Transport anchors must be
actual pre-action history entries marked pickup-candidate, rather than merely
assuming all post-lift time is held transport. Terminal-tail anchors retain
complete future actions through the successful ending. This tests whether
restoring later-stage coverage recovers the lost skill; existing uniform and
early-phase arms are the controls. No additional steps are added.

Separately, cross05's initial target is outside all three 256x256 views:
segmentation yields zero target pixels, consistent with RGB inspection and
projected center x coordinates around 305/273/510. This is not solely occlusion.
The retention check is not expected to establish a solution to initial
unobservability. Further work on that failure requires independent training
scenes with legitimate observation-acquisition actions or controlled visibility,
not feeding the held-out failure trajectory into imitation training.

The sampling audit of the retention arm independently confirmed exactly 16
anchors in each of the four categories, identical 192 human slots and per-scene
counts versus the prior new-uniform arm, and true pickup-candidate evidence
for all 16 selected holding anchors.

Visibility audits of all nine accepted training initial states found a target
in at least one external camera; left/right/hand target-pixel counts were
T00 0/125/0, T02 200/0/0, T03 0/132/0, T05 165/6/0, bridge00 200/0/0,
bridge01 0/150/0, bridge02 153/0/0, bridge03 0/134/0, bridge04 0/203/0.
No accepted scene initially covered the all-three-views-empty condition seen
in cross05. Keeping the right camera is essential for several training scenes.
This distribution gap motivates a separate training-only observation-acquisition
experiment, not more repetition of the same visible-target supervision.

The full-stage retention arm completed its sole 256-step run and saved final
expert/optimizer states. Fresh-process reload of all 407 expert keys reproduced
the development and training-probe arrays exactly; all step losses/gradients
were finite. Human dev means at 0/64/128/192/256 were
0.355360/0.347093/0.348587/0.344066/0.352623. Final shared-human-probe mean was
0.121925. Training plus monitoring took 208.13 s with 21.05 GiB peak allocation.
The prescribed final step remains selected, not the lower-loss step 192.
Six new fixed-development rollouts are handed off; no further optimizer run
is implied by this check.

### Full-stage rollout outcome

The six frozen rollouts subsequently completed: five successes at
214/176/213/384/415 steps, and cross05 failure at600. Initial XML, simulator
state, state14 and three RGB views matched the reference exactly. The common
five-success mean was280.4 steps, versus204.6 for new-uniform and203.2 for the
older quarter-replay arm. Recovering phase-arm cross00 did not establish an
overall benefit. These are reused development scenes, not untouched tests.

### Frozen natural-scene refresh pilot (training complete; evaluation running)

A separate bounded data-refresh experiment uses five newly audited, naturally
initialized T-shape scene_groups: seeds6001/6002/6004/6005/6006. The candidate
manifest SHA-256 is
`5d9bfd8c112a1beee1050bfbd7bafa0accaaa7b3880cae404f36c992aba07f05`.
Seed6003 failed; naturally hidden bridge6024 reported official success but
remains needs_review under the conservative physical gate. Both are excluded.
Thus this is NOT all-hidden observation augmentation or an expert dataset.

Before preparing or training, the initial proposal to replace all64 generated
slots was superseded by the following retention-preserving schedule:
- Reuse the completed quarter256 final model as the control, without retraining.
- Train one new arm from the original checkpoint for256 optimizer steps with
  the same lr3e-6, optimizer, seed/noise schedule and final-step selection.
- Preserve192 human and48 older generated rows byte-for-byte at manifest level.
- Replace only16 rows at zero-based indices i%16==15 with new natural examples.
- Five unique groups receive4/3/3/3/3 anchors, uniformly spread over each full
  episode's valid range6..N-16, keeping complete future16 targets and three
  camera histories. No early-observation-only weighting or visibility oracle.
- Verify all source/review/safety/metadata hashes and admission gates before
  extraction. Leave the existing eight human-development rows unchanged.
- Evaluate the control and new arm on all six newly fixed development cases
  6101..6106, with the independently running original-model baseline reused.
  Do not rerun oldcross/5106 for selection; final6201..6206 remain sealed.

This contrasts two data mixtures with equal optimizer budgets, not a single
causal visibility variable: scene identity, demonstrations and sampled action
phases differ. Only five new T groups are represented. No new optimizer run
or rollout outcome is claimed at this preparation stage.

The sole new256-step run completed after actual-artifact preparation checks:
only the sixteen prescribed rows differ; the other240 training rows and8dev
rows remain exact. All candidate/review/safety/snapshot hashes passed. The
new training-manifest SHA-256 is
`1a7bc124f786bbc1c5d15d747cc842e82243b322c6f1d783cafe4f0b523ab967`.

All256 contiguous step records are finite, monitors0/64/128/192/256 are present,
and final expert/optimizer states were saved. A separate process reloaded all
407 expert tensors and reproduced development/probe arrays exactly:
- Human development flow loss:0.3403223567 (quarter control:0.3446245082).
- Shared-human training probe:0.1188280778.
- Training plus monitoring:207.84s; peak allocated GPU memory:21.05GiB.
These are demonstration-loss diagnostics, not success-rate results. The final
checkpoint remains fixed. New-six-development evaluation is now running for
the oldquarter control and newnatural arm; original-model baseline is reused.

### Natural-six development result and frozen final evaluation (2026-11-12)

All six natural development scenes6101..6106 completed. Steps in seed order:
- Original:223/208/213/245/F600/304,5/6.
- Human-only256:224/227/203/253/F600/288,5/6.
- Older quarter256:225/229/211/246/F600/F600,4/6.
- Natural16 refresh256:222/227/212/246/359/344,6/6.

The18 original/quarter/natural trajectories and6 human trajectories passed
exact initial XML, full simulator state, state14, three RGB, canonical,
actual query-log and frozen full-episode-metadata checks. On6105 the natural
arm showed sustained dual-finger lift at305; original and quarter showed none.
Natural6106 lifted at156. Both natural successes were horizontally inside the
sink and stable for a1-second offline passive check, without robot support.
Their existing conservative label remains stable_but_bbox_region_mismatch,
not strict-bbox pass. This initial local gain is one paired development case
beyond the human-only baseline, not a statistically established broad gain.

Before any final policy evaluation, freeze the three final checkpoints:
original XR1, human256 final and natural16_256 final. Evaluate every one of
six previously reserved scene snapshots6201..6206, horizon600, matching noise
seeds, actual model observations/actions and unchanged official success.
Run independent canonical/state/dynamics probes before rollouts and retain
all18 outcomes. No training, checkpoint selection, data-mixture change or
success-threshold change may use these final outcomes. Physics audit is a
separate report, not a replacement for official success. This is same-task
scene generalization; generated shape categories overlap training.

Selection history: multiple human/mixed/quarter/phase/full-stage pilots used
repeated cross development scenes; the two-group observation pilot did not
establish a gain. The final natural16 candidate was chosen after the newly
frozen six-scene development comparison and subsequent human-only check.
It is not the only candidate tried, and these development results are not an
independent test. Final results remain pending at this protocol freeze.

### Frozen final-six outcome and continuation

All18 final rollouts and exact paired-integrity checks completed. In6201..6206
order, original:220/330/170/F600/177/150 (5/6); human256:
221/429/179/F600/185/173 (5/6); natural16:
222/F600/170/F600/194/169 (4/6). The natural arm adds no final success and
regresses on6202; the development improvement did not reproduce. Keep all
these results. Do not describe generated-geometry effectiveness from this run:
scene identity, demonstration content and sampling changed together.

All18 physical and history reviews completed. Every official success had
sustained dual-finger lift and stable released sink support, with horizontal
containment. Their conservative label is stable_but_bbox_region_mismatch,
not strict full-bbox containment. All failures also had some sustained lift;
natural6202 ended outside the basin near floor height. This distinguishes
transport/placement failure from failure to perceive or initially grasp.
The external result collection natural_final_three_arm_v1 contains the18-case
paired verification, physical/history summary and locked_result_manifest.
This final set is now sealed for reporting, not subsequent recipe selection.

Continuation is data coverage on the next six previously preregistered TRAIN
scenes6007..6012 (bridge00..05), in fixed seed order, original XR1 teacher,
600-step horizon. Initial support/finite/no-penetration checks pass; independent
restore probes precede rollouts. Automatic physical/history review follows.
These are policy pseudo-demonstrations, not experts. All failures are retained;
no new SFT is authorized by a success flag alone. This extends training coverage
without sampling, tuning against, or training on the sealed final set.

### Eleven-group balanced refresh pilot

The next fixed TRAIN batch6007..6012 completed6/6 original-policy successes
at294/181/157/530/164/130 steps. All six pass sustained lift and released
stability;6011 also passes strict bbox. Independent saved-model CPU contact
checks confirm finite states, counter support and no positive target/robot
contact or significant robot/obstacle penetration. This provides11 accepted
policy demonstration groups together with the five earlier T groups.

`prepare_xr1_natural11_pilot.py` replaces the same16 of256 slots as before:
T and bridge families receive8 slots each, evenly assigned in fixed group
order, with full-episode quantile anchors. The remaining192 human+48 older
policy slots and eight human-dev samples are unchanged. Independent checks
match every new NPZ array against its original HDF5 observation/action data.
The accepted-manifest hash is
`f535694178635a1883b39368dc3c594337ed51b7169d7e094c84392bfd1686e2`;
training-manifest hash is
`206c33b8a9058b9915441144b7c09dd4e27db28de992069a9ea156ec9760d9e0`.

The first extraction stopped because the older phase helper rejected count1.
Its incomplete output is retained; the corrected extractor uses exactly the
same uniform quantile formula and supports a single midpoint anchor. Seven
CPU tests check quotas, group/holdout rejection, preservation of240 slots and
equivalence with the old formula for counts2..4. No sample/optimizer budget
was changed to bypass the error.

One fresh256-step SFT uses the original checkpoint, lr3e-6 and unchanged
seed/noise/optimizer recipe. Only the final checkpoint is eligible. A separate
407-key reload must reproduce frozen human-dev and training-probe arrays
exactly before six DEVELOP6101..6106 rollouts start. Existing comparison-arm
rollouts are reused. Sealed6201..6206 remain excluded from recipe selection.
This section records preparation and execution protocol, not a new gain.

Execution correction: the initial11-group run completed256 finite optimizer
steps, but its legacy runtime entrypoint omitted periodic monitor calls and
final training-probe output. The post-training gate correctly stopped before
rollouts. This record is retained; missing intermediate metrics are not
reconstructed or invented. A deterministic replay from the same original
checkpoint uses the corrected monitored entrypoint with unchanged256 updates,
samples, seeds and optimizer settings. Before any rollout, it must reproduce
all407 final expert tensors exactly, contain monitors0/64/128/192/256, and pass
fresh-process development/probe reload checks. A mismatch blocks evaluation
and must be reported, not labeled equivalent. This replay consumes additional
compute but is not256 extra continuation steps or an independent replicate.

The equality gate FAILED:407/407 final tensors differ, maximum absolute
parameter difference0.0002266386291. The first discrepancy is already in the
step1 gradient norm (22.00736046 vs22.00736427), followed by different step2
losses, before the first periodic monitor. Sample identities/order match.
Floating-point nondeterminism is a hypothesis, not an established cause.
Therefore the runs are not an equivalent replay. Both are retained as
same-recipe independent runs; neither is selected by lower loss.

The legacy run completed256 steps in183.27s, dev loss0.3409862630; the monitored
run completed256 steps plus all five monitors in208.12s, dev loss0.3396214824.
Both peaked at21.05GiB. Each independently reloads407 keys and reproduces its
own eight-dev loss array exactly. The monitored run also reproduces its final
human training-probe array; the legacy run did not record that array, so its
corresponding equality cannot be established. No missing monitor is invented.

A separate explicit protocol now compares BOTH runs on all six6101..6106
development snapshots, preserving failed gates and recording the provenance
limit. Both weight hashes were frozen before rollout; no6201..6206 evaluation
or tuning is permitted. Twelve rollouts plus paired and physical audits are
running. This assesses training variability, not an independent final test.

### Eleven-group16-slot development outcomes

Both independent runs completed and all12 paired/physics/history audits ran.
Legacy6101..6106:221/217/228/239/F600/329; monitored:
220/296/347/239/F600/330. Both are5/6, not an improvement over human-only5/6.
The monitored6102 official success fails the separate strict stability gate:
angular speed0.1156->0.1082rad/s exceeds the unchanged0.1rad/s limit, while
horizontal containment remains true and1-second displacement is only0.068mm.
This is not evidence of falling out of the basin. Keep official and strict
outcomes separate. Both6105 failures eventually lift, only at586/551 steps.

### Fixed three-arm dose/step comparison

The next bounded experiment actually trains THREE fresh512-step arms:
human-only; quarter replay with192human+64old policy slots; and natural64
with the identical192human slots plus64new policy anchors from11train groups.
The last arm allocates32 anchors per shape family,5..7 per group, across the
full valid trajectory interval. All64(scene,step) anchors are distinct, but
nearby temporal windows and same-trajectory dependence still exist.

The256-slot schedule repeats twice, not512 independent demonstrations.
Human/quarter/natural have90/103/95 distinct episode IDs and180/199/199 distinct
episode-step pairs. The two mixed arms preserve the same192human slots; all
three share the same eight human-dev samples, original checkpoint, lr3e-6,
seed/noise recipe, final-step selection and5monitor points0/128/256/384/512.
Existing256-step baselines are not substituted for newly trained controls.

Source arrays, manifest hashes and episode separation pass before training.
Protocol hash:
`ae70e2eebbbe19928671241a794a23dec091a764be45fbb8da5c485e91421493`.
An inherited descriptive comparison string is corrected in a separate file;
actual frozen arms, budgets, selection and execution settings are unchanged.
Primary comparison is all six natural DEVELOPMENT cases6101..6106; old
cross00..05 are prespecified secondary retention checks with their historical
full-metadata-hash limitation. Sealed6201..6206 are never rerun for selection;
any subsequent claim needs a new previously unseen final evaluation.

All three512-step trainings and fresh-process407-key/dev/probe reload gates
have now passed. Human/quarter/natural dev flow loss is0.342037/0.356238/
0.352047; training plus monitors390.71/383.48/383.90s, peak21.05GiB each.
These losses do not imply task improvements. Primary rollouts are underway;
the oldquarter512 already succeeds on6/6, so any new-arm success must be
compared with this same-budget control rather than only old256-step results.

### Dose512 primary-development outcome

All18 natural-development rollouts completed and passed exact paired initial
XML/state/state14/three-RGB/canonical/query/full-metadata checks. Steps6101..6106:
- Human512:221/279/228/236/F600/F600, official4/6.
- Old-quarter512:224/227/210/227/517/289, official6/6.
- Natural64_512:220/182/328/249/F600/296, official5/6.

Separate offline stability counts are3/6,6/6,4/6 respectively. Human6101 and
natural6102 satisfy official success but fail the unchanged strict stability
gate. They are not silently removed or relabeled as official failures.
Old-quarter512 exceeds human512 here by two cases; new11-group data does not
outperform the existing generated-data control. This is adaptive development
evidence, not an independently confirmed generalization improvement.

The prespecified cross retention check is now running after each GPU's primary
pipeline releases it. New snapshot copies retain identical model/state/RGB/
canonical bytes; a new full-metadata hash seal and old-source hashes are saved.
Old snapshots are unmodified. This proves current three-arm input identity,
not retrospectively the old256-step runs' full episode-metadata identity.
All three arms and all six cross cases are retained. No sealed6201..6206 case
is revisited; any subsequent final claim requires new unseen evaluation.

Old-quarter provenance audit: its64 policy slots comprise21 box,21T-grip and
22bridge samples, taken from19 original-policy trajectories over18 distinct
assets (six per family; rigid00 has two seeds). The new natural64 set comprises
11trajectories/11assets (fiveT, sixbridge), with32 slots per family and no boxes.
The old teacher trajectories mostly use seed0, with rigid00/seed1 and
bridge04/seed2 exceptions. Both sources are successful policy rollouts, not
expert demonstrations. Differences include geometry-family coverage, scene
initialization/layout, trajectory content and sampling weights. Success does
not isolate one causal3D-generation factor. Keep these confounds with the
result rather than crediting the latest scene generator automatically.
