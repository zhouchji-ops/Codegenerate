"""Build the kitchen asset token ledger from independently recorded CLI calls."""
import argparse
import hashlib
import json
from pathlib import Path


FIELDS = ("input_tokens", "cached_input_tokens", "output_tokens",
          "reasoning_output_tokens", "cache_write_input_tokens")


def collect(root):
    rows = []
    seen = set()
    directories = sorted(p for p in (root / "calls").glob("*/*") if p.is_dir())
    if not directories:
        raise ValueError(f"No call directories under {root / 'calls'}")
    for folder in directories:
        path = folder / "events.jsonl"
        raw = path.read_bytes()
        events = [json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
        starts = [e["thread_id"] for e in events if e["type"] == "thread.started"]
        turns = [e["usage"] for e in events if e["type"] == "turn.completed"]
        if len(starts) != 1 or len(turns) != 1 or starts[0] in seen:
            raise ValueError(f"Expected one unique, completed independent call in {path}")
        if any(e["type"] == "turn.failed" for e in events):
            raise ValueError(f"Review failed turn in {path}")
        seen.add(starts[0])
        usage = {k: turns[0][k] for k in FIELDS}
        if any(type(v) is not int or v < 0 for v in usage.values()):
            raise ValueError(f"Invalid usage counts in {path}")
        if usage["cached_input_tokens"] > usage["input_tokens"] or usage["reasoning_output_tokens"] > usage["output_tokens"]:
            raise ValueError(f"Invalid included token counts in {path}")
        rows.append(dict(call=folder.name, phase=folder.parent.name, **usage,
                         total_tokens=usage["input_tokens"]+usage["output_tokens"],
                         event_file=path.relative_to(root).as_posix(),
                         sha256=hashlib.sha256(raw).hexdigest()))
    return rows


def total(rows):
    return {"calls": len(rows), **{k: sum(r[k] for r in rows) for k in (*FIELDS, "total_tokens")}}


def markdown(rows, phases):
    totals = total(rows)
    lines = [
        "# 厨房资产实验的 token 账本", "",
        f"已记录 **{len(rows)} 次独立编程调用，合计 {totals['total_tokens']:,} token**。"
        "从每次 CLI 输出的 `turn.completed.usage` 取数，合计为输入加输出。缓存命中已计入输入，推理已计入输出。", "",
        "| 阶段 | 调用数 | 输入 | 其中缓存命中 | 输出 | 输入＋输出 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for phase, t in [*phases.items(), ("合计", totals)]:
        label = phase.partition("_")[2] or phase
        lines.append(f"| {label} | {t['calls']} | {t['input_tokens']:,} | {t['cached_input_tokens']:,} | "
                     f"{t['output_tokens']:,} | {t['total_tokens']:,} |")
    lines += ["", "## 逐次调用", "",
              "| 序号 | 用途 | 输入 | 其中缓存命中 | 输出 | 输入＋输出 |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for r in rows:
        number, _, label = r["call"].partition("_")
        lines.append(f"| {number} | {label} | {r['input_tokens']:,} | {r['cached_input_tokens']:,} | "
                     f"{r['output_tokens']:,} | {r['total_tokens']:,} |")
    lines += ["", "## 查证与续记", "",
              "- 原始记录位于本地工作区的 `kitchen_asset_trial/token_usage/`。"
              "`ledger.json` 保存逐次用量、相对路径与原始事件文件的 SHA-256。"
              "`calls/阶段/编号_用途/` 中的 `events.jsonl`、`prompt.txt` 和 `stderr.log` 对应同一次调用。",
              "- 下一次独立调用建一个新的编号目录，保存原始事件和提示词，再运行 "
              "`python rebuild.py`。用 `python rebuild.py --check` 校验账本与原始事件是否一致。"
              "脚本也接受一个外部计量目录作为路径参数。",
              "- 本账本覆盖资产建模、适配修复及物理属性预测。找图、外层对话、评审与测试编排的用量未纳入；"
              "缺少独立计量的部分记为未知。三件资产共用调用，无法据此拆出单件用量。"
              "金额还取决于模型、费率及缓存计费方式。"]
    return "\n".join(lines)+"\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path(__file__).resolve().parent)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rows = collect(args.directory)
    phases = {p: total([r for r in rows if r["phase"] == p]) for p in dict.fromkeys(r["phase"] for r in rows)}
    ledger = {"scope": "独立编程调用", "source": "turn.completed.usage", "phases": phases,
              "total": total(rows), "calls": rows}
    files = {"ledger.json": json.dumps(ledger, ensure_ascii=False, indent=2)+"\n",
             "README.md": markdown(rows, phases)}
    for name, content in files.items():
        path = args.directory / name
        if args.check:
            if path.read_text(encoding="utf-8") != content:
                raise SystemExit(f"Ledger differs: {path}")
        else:
            path.write_text(content, encoding="utf-8", newline="\n")
    print(f"{len(rows)} calls, {ledger['total']['total_tokens']} tokens")


if __name__ == "__main__":
    main()
