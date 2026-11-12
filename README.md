# Codegenerate

GPT 操作证据驱动的三维环境扩增研究：生成资产与场景代码 → 执行验证与示范采集 → VLA 微调 → 泛化评测。

首轮平台：RoboCasa365 + Xiaomi-Robotics-1。未来通过适配层扩展到其他 benchmark；本仓库初始化不表示实验已完成。

## 存放边界

这里只放代码、可复现配置、测试和文档。模型权重、数据集、生成三维资产、视频、轨迹、原始评测 JSON、训练日志及缓存必须放在仓库之外。文档可以写经过核验的汇总结果与产物索引，但不能把原始产物搬进来。

本地对应目录：`E:\A_research\RSI_Agent_WM\Codegenerate\正式实验`。GitHub 是版本化发布端，本地是工作副本；禁止维护互不关联的两套实验代码。

## 导航

- [AGENTS.md](AGENTS.md)：代理与人工协作规则。
- [仓库组织与同步](docs/repository.md)：目录职责、提交流程、外部产物边界。
- [实验文档模板](experiments/TEMPLATE.md)：每个实验的设计、复现和结论记录。
- [代码与文档准入检查](scripts/check_repository.py)：标准库实现，可在 Windows/Linux 运行。

## 目录约定

- `docs/`：环境、研究决策和使用说明。
- `experiments/<experiment_id>/`：实验 README、配置和必要的实验专用小脚本。
- `src/`：跨实验复用实现，按需增加 `benchmarks/`、`policies/`、`generation/`、`evaluation/` 等模块，不提前堆空目录。
- `configs/`：跨实验共享配置；私有路径使用环境变量，示例不含凭证。
- `scripts/`：运行、转换、审计入口；真实输出目录必须位于仓库之外。
- `tests/`：单元与接口测试。测试产物写系统临时目录并清理。

## 本地检查

```powershell
python scripts/check_repository.py --all
python -B -m unittest discover -s tests -v
```

使用 Git 后，提交前还须运行 `python scripts/check_repository.py --staged` 和 `git diff --cached --check`。准入检查是防误入的基础检查，不替代秘密扫描、代码审查和实验正确性验证。
