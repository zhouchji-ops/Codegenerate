# RoboCasa 厨房物体的程序建模与物理验证

2026-11-14。三件程序建模资产已经交付，并在 RoboCasa 的训练厨房加载。本文是换对话后的接手入口。它们是独立的资产研究样本，此次为原固定 24 项新增的过审轨迹数为零。

本轮补充了三件资产的净重、质心、干湿摩擦、载荷挠度与薄边间隙预测，可从[物理预测与新模型入口](物理预测.md)继续。下文的 290、105、81 g 和八次编程用量属于首次交付。

## 本地工作目录

在工作区的 `kitchen_asset_trial/` 下按用途查找。根目录的两个 ZIP 是已发布的阶段交付包，包内的路径对应当时的版本。

| 目录 | 内容 |
| --- | --- |
| `token_usage/` | 用量总账、逐次原始事件与提示词、重建脚本；`snapshots/` 留存两个阶段的原始汇总 |
| `modeling/` | 三维建模、材质、碰撞和物理预测的 Python 源码 |
| `refs/` | 实物参考图、来源及旧版和官方同类资产 |
| `output/` | 初版模型与实物、旧版、官方资产的四列对照图 |
| `calibrated/` | 容量与接触校准后的模型、厨房图和物理检查 |
| `predicted/` | 写入预测质量、质心及惯量的 MJCF，参数和复核结果 |
| `logs/` | `build/` 为构建与导出日志，`render/` 为渲染日志 |
| `.runtime/` | 工具、Git 工作树、传输包及历史生成副本 |

## 从哪里接手

- 交付标识为 `kitchen_visual_trial_20261114/calibrated`。授权实验机的用户主目录下有同名目录。物理资产在 `calibrated/sim/{mug,scoop,turner}/model.xml`，相邻 OBJ 和 PNG 需要整目录保留。
- 本地工作区的 `kitchen_asset_trial/calibrated` 有最终 GLB、可编辑的 Blender 文件、MJCF 和三相机截图。`output/comparison.jpg` 依次展示品牌实物、旧模型、RoboCasa 同类资产和本次模型。各列按外形独立取景，图片不能用来比较实际尺寸。`calibrated/kitchen_views.jpg` 是最终资产在仿真里的实际大小和外观。
- 对照旧模型为 `handled_cup_v02`、`measuring_scoop_v02`、`slotted_turner_v02`。RoboCasa 对照选用 `objaverse/mug/mug_0`、`lightwheel/measuring_cup/MeasuringCup001`、`objaverse/spatula/spatula_0`。代码使用单张 2000 × 2000 商品图作各自的外形参考，从头生成网格。官方样例属于相同用途类别，具体款式不同。
- 原始与修正后的物理测试分别见 `calibrated/physics_before.json` 和 `calibrated/physics_ablation.json`。基础检查见 `calibrated/sim/validation.json`，场景与首帧见 `calibrated/kitchen.json` 及三个厨房 JPG。逐次调用与总数见 [Token 用量账本](token_usage/README.md)，原始事件位于外部实验目录的 `token_usage/calls/`。

## 尺寸和功能

| 资产 | 最终外包络，长 × 宽 × 高 | 厂商规格 | 独立检查 |
| --- | --- | --- | --- |
| 带柄杯 | 130.1 × 88.9 × 102.8 mm | 129.5 × 94.0 × 106.7 mm，14 美制液量盎司，约 414.0 mL | 碰撞腔从 267.6 校准到 414.0 mL，杯底封闭、杯柄可穿过 |
| 斜口不锈钢量勺 | 113.3 × 74.8 × 160.1 mm | 108.0 × 76.2 × 152.4 mm，约 1.5 美制杯，约 354.9 mL | 按最低溢出口计算，碰撞腔从 140.8 校准到 354.5 mL |
| 开槽鱼铲 | 307.1 × 78.8 × 14.0 mm | 285.8 × 76.2 × 25.4 mm，钢与硅胶，厂家说明铲片可弯 | 六个槽和挂孔贯通，五条内肋有碰撞；当前铲片为刚体 |

容量由成品的 48 个凸碰撞扇区重建内壁多边形，积分到最低唇边。测的是几何容积。量勺的第一版计算把同一角度在圆周接缝处重复计数，缩放使用的旧值为 140.98 mL；修正测量后如表所列，最终与厂家近似标称相差 0.34 mL。杯、量勺、鱼铲分别假定为 290、105、81 g，三者表面滑动摩擦系数均先设为 0.65。实物净重、质心、转动惯量、干湿摩擦与鱼铲载荷挠度都待测。来源为 [Le Creuset](https://www.lecreuset.com/vancouver-mug-flame-14-fl-oz/70317140090000.html)、[RSVP 量勺](https://rsvp-intl.com/products/wholesale-endurance-scoop-and-measure-cup) 和 [RSVP 鱼铲](https://rsvp-intl.com/products/endurance-fish-spatula)。

## 薄底穿透与求解器对照

RoboCasa 使用的 MuJoCo 3.3.1 和本地 4.0.1 分开运行。对两种容器各投一枚 5 g、直径 12 mm 的球，球心从台面上方 150 mm 起自由落下。每种容器交叉运行 Newton、PGS 与 1、2 ms 步长。检查末态球心相对杯体的位置、容器底部正向接触力、桌面是否也承重。

- 在服务器的原始模型中，8/8 组合发生穿底。保持校准后的形状、质量、姿态相同，单独恢复默认底部接触，仍为 0/8 接住。对容器底部单独赋予 `priority=1, solref="0.008 1"`，8/8 接住，桌面均未直接接触小球。本地版本也重现 0/8 与 8/8。两种求解算法和两档步长都复现现象，接触参数对该用例有决定性影响。8 ms 为此负载条件下的工程值，未经过实物标定。
- 在专用平面上以 0.9 倍重力大小的水平力作用于质心 150 ms，随后放开。服务器四种设置下，杯的最终水平位移 32.8–38.1 mm，量勺 33.9–35.8 mm，鱼铲 38.48–38.49 mm。求解设置会影响精确位移，尤其是杯；实物滑移距离尚未知。接触模型可参照 [MuJoCo 官方文档](https://mujoco.readthedocs.io/en/stable/computation/index.html)。
- 最终资产合计通过 18 项射线检查和 9 次落台稳定检查。在项目的 `RigidPilotCounterToSink` 训练适配环境中，固定 layout 14、style 12、seed 8110，三件均成功 reset，首帧状态有限，任务初始成功均为 false。没有用这些结果计算 XR1 抓放成功率。

## 复现与计量

以下从本实验的源码目录执行。设置 `BLENDER` 为 Blender 4.5.9 可执行文件，`ORIG` 和 `FINAL` 为绝对路径、仓库外的两个不同目录。首次执行应给 `FINAL` 一个全新路径。

```bash
mkdir -p "$ORIG"
"$BLENDER" -b --python-exit-code 2 --python modeling/repair.py -- "$ORIG"
python modeling/finish_pbr.py "$ORIG"
"$BLENDER" -b --python-exit-code 2 --python modeling/simfix.py -- "$ORIG"
python modeling/adapter.py "$ORIG"
"$BLENDER" -b --python-exit-code 2 --python modeling/kitchen_surface.py -- "$ORIG"
"$BLENDER" -b --python-exit-code 2 --python modeling/physical_calibration.py -- "$ORIG" "$FINAL"
python physical_audit.py "$ORIG/sim" "$ORIG/physics_before.json"
python physical_audit.py "$FINAL/sim" "$FINAL/physics_ablation.json"
python test_sim.py "$FINAL/sim"
```

`physical_audit.py` 与 `test_sim.py` 需要 NumPy 和 MuJoCo。厨房复测沿用已有训练适配器：把 `test_kitchen.py` 放到与最终 `sim/` 同级的目录，在实验机已有 RoboCasa 客户端环境运行 `MUJOCO_GL=egl python test_kitchen.py`。复现时须逐项读回 JSON 的布尔值，脚本完成不等于检查全部通过。

网页找图单独进行。八次 3D 编程调用合计输入 166,725 token（其中缓存命中 114,432）、输出 24,122 token，输入输出合计 190,847。首次生成三件外观用了 17,822 输入和 7,095 输出。完整用量含导出修复、碰撞建模、材质与物理校准；排除外层找图、评审、测试执行和交接。三件共享调用，无法从记录拆出单件用量。

## 下一次优先做什么

采用已交付的预测参数及其区间作训练场开发对照。后续如取得实物，对净重、干湿滑移、夹持和铲片力–挠度逐项校准。在独立训练场用固定初态验抓取、搬运、释放，保存全过程接触与成功判定，才可增加合格示范。原固定批次的教师采集与单任务评测继续按既有交接推进。
