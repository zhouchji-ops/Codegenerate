# 抬升历史与水槽缓存复核

独立工具 `scripts/review_rollout_history.py`，不修改冻结的 v2 文件或历史报告。

## 水槽缓存

`fixture_consistency(env)` 只检查任务实际使用的 sink interior：
- Python fixture 位置与编译后的 body 世界位置；
- region 本地中心、半尺寸；
- Python 世界区域四角与编译 region geom 的四角。

阈值 50 微米用于容纳保存 XML 的数值舍入，错误或缺失返回失败。此范围不声称全部场景物体缓存都已验证；不同回合须各自检查。检查失败时不自动改官方指标，保留证据并使用正确 episode metadata 重新恢复后复核。

首盒体及 RL bridge04 after 的 sink 最大误差约 2.44 微米，没有观察到跨布局错位。它们的 6.7 mm bbox 底界差不是该微米舍入造成。

## 实时 v3 辅助终奖

```python
from review_rollout_history import PickupStableSuccessGate
gate = PickupStableSuccessGate()
gate.reset(raw_env)  # 在每次真实 episode reset 完成后
verified, diagnostics = gate.update(raw_env)  # 每个控制步，时间必须递增
```

在 v2 连续释放稳定门槛前，要求历史上至少 0.1 秒：
- 两指均有 active 正法向接触力；
- 无非机器人接触支撑；
- 整个碰撞形状最低点比初始底部高超过 2 cm。

这比 body 原点升高更能排除仅旋转翘起的误判。只支持当前单根刚体、box compound 碰撞资产，不可直接泛化到任意 mesh 或关节物体。门槛是保守工程定义，仍可能误拒其他合法抓法。官方成功指标始终独立记录。

## 离线历史

```bash
python review_rollout_history.py EXTERNAL_ROLLOUT EXTERNAL_OUTPUT_JSON
```

从每个 pre-action simulator state 加终态恢复，检查双指接触、最低点抬升及持续时间。接触力由 mj_forward 重算，没有保存的 controls，因此不等于完整动力学重放或严格力闭合证明。输出排他创建，不能覆盖旧证据。

已核验 bridge04 after：121–142 步连续 1.05 秒双指抬升，最低点最大升高 5.381 cm；终态释放且在槽内受到接触力。不是仅把物体推入槽。

14 个逻辑测试覆盖缓存精度/错位/尺寸、双指/单指、推入、支撑、inactive、NaN、历史门槛、释放及 reset。

实际 v3 类对保存状态序列的回归：bridge04 首次抬升门槛第124步、v3成功第197步；T03分别第227和300步。这是状态恢复检查，不是动作重新执行或新增live评测。离线与实时统一拒绝非box碰撞类型，避免把mesh尺寸误当box半尺寸。
