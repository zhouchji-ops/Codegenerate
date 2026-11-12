# exp001: RoboCasa365 新刚体工程闭环

## 目的与范围

依用户最新优先级，先小规模生成、实际采集与微调，争取改善单任务成功率，不先铺完整 2×2。首任务 PickPlaceCounterToSink，属于 pretrain，不等于旧 target50 评测。当前六个 box 是工程样例，不是 GPT 自动生成结果，也不是完整强参数化基线。

固定原厨房、PandaOmron、原成功判定。只通过独立进程内任务子类替换目标物体，原注册表和资产划分不变。宽度 4/5/6 cm × 长度 6/8 cm，高 3 cm；单位米，视觉与碰撞同尺寸，外观相同。尺寸变化也改变体积与质量，不应称单一接触几何因果干预。

## 实现

- `scripts/generate_rigid_pilot.py`: 生成六资产，外部 manifest 和 XML 哈希；拒绝仓库内产物和覆盖既有目录。可执行 MJCFObject 加载与 MuJoCo 独立平面落稳检查。
- `scripts/smoke_rigid_kitchen.py`: 独立 RigidPilotCounterToSink 子类，目标物体绕开原 registry，保留其他任务逻辑；真实 reset、三相机截图和初始化诊断。
- `scripts/collect_rigid_rollout.py`: 调用已有专用 XR1 推理服务，默认最多 600 步，4 帧历史、间隔 2、重规划 16 步、crop .95。保存 pre-action 图像/state14/action12/完整 sim_states、初始场景 XML、终态、录像与结果。提供 `--xml` 为新物体，省略为原任务。
- 这是旧策略采样，不是公共专家或 MimicGen；成功还需视频和物理诊断复核，失败不能当成功动作监督。

## 已验证（2026-11-12）

- Windows 标准库单测 11/11 通过，其中新生成器 3 项；修改前既有8项也通过。
- 六资产全部通过真实 MJCFObject 加载与 2000 步平面落稳。
- 六资产全部完成真实厨房初始化与三相机渲染：seed=0、layout=53、style=37、action_dim=12；有限状态且初始官方成功均为 false。
- 已查看第一个物体左相机截图：蓝块可见且放在水槽旁台面。其余截图生成不等于均人工审核。
- 第一次厨房测试曾因 NumPy int64 无法 JSON 序列化中断，已改显式 int 并完成六资产重测。
- 新几何策略 rollout、成功示范、真实参数更新和重载评测仍须单独验收，以上检查不代表已完成。

## 可比性与待验收

同 seed 不保证不同尺寸的精确相同初态：placement sampler 可能因碰撞/尺寸重新采样，已观察到目标转到水槽另一侧。当前是工程可用性检查，不能据此作严格几何因果对照。正式前后策略比较必须固定同资产、同初态清单、预算、horizon，单次尝试配对评测；开发数据与最终评测隔离。旧混合回放 60.76% 不作为此任务基线。

保留官方 success，同时独立核查充分进入水槽、稳定支撑、释放与夹爪远离。不能改成功函数以让新物体更易成功。官方命中但物理失败单独记，不作合格示范。

模型/仿真固定：RoboCasa commit `100f29b8efd239e807e62f2678a1a8489c524297`，MuJoCo 3.3.1，robosuite 1.5.2，Xiaomi-Robotics-1-RoboCasa365 专用权重。训练加载/归一化/观察动作时序一致性另行验证。

## 运行约定

所有输出目录必须在仓库外，环境变量和路径由运行环境注入。生成入口不安装包、不升级仿真器、不修改已有注册或资产，不调用外部模型API。

```bash
python -B scripts/generate_rigid_pilot.py --output "$NEW_ASSET_ROOT"
MUJOCO_GL=egl python -B scripts/generate_rigid_pilot.py --output "$NEW_ASSET_ROOT" --smoke-only
MUJOCO_GL=egl python -B scripts/smoke_rigid_kitchen.py --assets "$NEW_ASSET_ROOT" --output "$NEW_RUN_DIR" --index 0 --seed 0
# 需先有已核验的 XR1 localhost 推理服务，并使用对应 client 环境
MUJOCO_GL=egl python -B scripts/collect_rigid_rollout.py --xr1-repo "$XR1_REPO" --model "$XR1_MODEL" --xml "$NEW_ASSET_ROOT/rigid_00/model.xml" --output "$NEW_ROLLOUT_DIR" --port 10206 --seed 0 --horizon 600
```

不把生成 XML、截图、原始轨迹/日志/权重放本仓库。当前代码不提供自动 expert、自动训练、完整交叉关系标签或恢复半途回合能力；这些不是隐含已完成模块。

运行边界：collector 的 --horizon 是外层最大步数；内部环境提前 done/truncated 也会终止，必须核查每回合 steps 与终止原因后再声称同预算。本轮600步仅作该任务工程采样上限，不作为官方榜单结果。GPU0单服务首轮热启动后推理约0.28秒/请求，端到端每16步约6–8秒，说明当前主要时间在客户端仿真/渲染/记录而不是模型；扩大规模前应测分项时间并优化客户端并发和I/O，不通过重复加载模型占满显存。

## 首批真实采样进展

首个新物体 rigid_00、seed=0：原专用策略官方成功，189/600步，82.8秒；完整轨迹读回确认189×12动作、189×14状态、三路189×256×256 RGB及200维仿真状态，时长一致、数值有限。关键帧已人工看到接近、抓取搬运和放入水槽。终态稳定/全包含独立诊断仍待审核，保持 review pending，不能仅凭该值纳入成功监督。原任务同seed也已执行成功174步，但目标是peach、布局采样不严格等价，因此只作接口验证，不作几何难度或改进效果比较。

独立物理审查指出初版落稳脚本虽然构造了 MJCFObject，却编译原始XML；现已修为实际加载后对象并重测六例全通过。实际加载后质量21.6至43.2g，宽长变化也改变质量和惯量；不以此声称纯几何因果。8cm长边接近Panda夹爪名义极限，必须检查接近方向及短边可抓性，不盲调摩擦来刷成功。
