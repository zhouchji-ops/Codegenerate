# 物理成功辅助审核

此工具保留官方成功判定，额外检查稳定、释放、碰撞和位置，不把辅助指标冒充官方指标。

## 运行

在服务器已安装 RoboCasa 的客户端环境运行：

```bash
python review_rollout_physics.py /absolute/external/rollout_directory
```

输入为同一回合的 `result.json`、`model.xml`、`trajectory.hdf5/terminal_sim_state`。
默认同目录排他新建 `physical_review_v2.json`，已有结果不覆盖。无渲染、无模型推理。
保存模型和终态摘要，记录初始与最终诊断及采样序列。

离线测试固定其他关节，只让目标刚体继续演化 1 秒，不能替代真实策略后续控制或多人/可动干扰物稳定性评估。

## 实时使用

```python
gate = StableSuccessGate()
gate.reset()  # 每次环境重置
verified, diagnostics = gate.update(raw_robocasa_env)  # 每个真实控制步
```

连续至少 0.5 秒：
- 官方成功；
- 水平两轴完整 bbox 在槽内；
- 目标与水槽有 active 接触约束（efc_address >= 0）且法向接触力 > 1e-8 N，并且不接触机器人或夹爪；
- 最大穿透 < 1 mm、线速度 < 1 cm/s、角速度 < 0.1 rad/s；
- 窗口内相对于起点的位移 < 5 mm。

上述是保守工程阈值，不是通用真实物理认证。只有槽内干扰物支撑的合法堆叠可能被拒绝。当前接触提取针对单根 body 的刚体，不适合直接用于多子体关节资产。实时环境必须每次 episode reset 重置 gate。

三轴严格 bbox 使用 `th=0` 单独记录，不作为唯一否决项。首个盒体轨迹最低角比槽内部 region 底界低 6.714 mm，但真实接触穿透仅微米级且稳定，故标记 `stable_but_bbox_region_mismatch`，不称真实穿透或虚假成功。需要进一步核对真实碰撞几何与 region 语义。

## 验证范围

首盒体终态实际重建及 1 秒 CPU 稳定诊断通过。22 个纯逻辑单测覆盖抱持、快速经过、旋转、悬空、仅干扰物支撑、深穿透、水平外突、官方失败、NaN、稳定窗口、位移和 reset。单测不替代上述反例在真实模拟器中的系统性构造测试。

## v2 加固

v2 拒绝 inactive、零力接触作为槽支持证据，并验证稳定窗口有限且大于零；CLI 拒绝在代码仓库内输出。v1 历史结果保留，不覆盖。接触力门槛说明有活跃受力，不等于已验证完整力矩平衡；仍结合低速和连续稳定窗口。
