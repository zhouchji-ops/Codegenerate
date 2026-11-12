"""Guard code/document-only repositories. Standard library; no files modified."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys

ALLOWED = {".md", ".rst", ".txt", ".py", ".ps1", ".sh", ".bash", ".yml", ".yaml",
           ".json", ".toml", ".ini", ".cfg", ".xml", ".mjcf", ".urdf", ".patch",
           ".diff", ".c", ".cpp", ".h", ".hpp", ".js", ".ts", ".tsx", ".jsx",
           ".html", ".css", ".sql", ".lock"}
SPECIAL = {".gitignore", ".gitattributes", ".editorconfig", "Dockerfile", "Makefile",
           "LICENSE", ".env.example"}
FORBIDDEN_DIRS = {"data", "datasets", "weights", "checkpoints", "outputs", "results",
                  "runs", "logs", "artifacts", "videos", "trajectories", "assets",
                  ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache"}
LIMIT = 1024 * 1024
SECRETS = [
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(rb"github_pat_[A-Za-z0-9_]{30,}"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
]


def inspect_content(name: str, content: bytes) -> list[str]:
    path = Path(name)
    problems = []
    if any(part.lower() in FORBIDDEN_DIRS for part in path.parts[:-1]):
        problems.append("runtime/data directory")
    if path.name not in SPECIAL and path.suffix.lower() not in ALLOWED:
        problems.append("not an approved code/document extension")
    if path.name.startswith(".env") and path.name != ".env.example":
        problems.append("local environment file")
    if len(content) > LIMIT:
        problems.append("larger than 1 MiB")
    if b"\0" in content:
        problems.append("binary/NUL content")
    try:
        content.decode("utf-8-sig")
    except UnicodeDecodeError:
        problems.append("not UTF-8 text")
    if any(pattern.search(content) for pattern in SECRETS):
        problems.append("possible credential (value withheld)")
    return problems


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)


def check(root: Path, staged: bool) -> int:
    failures = []
    checked = 0
    try:
        if staged:
            top = Path(git(root, "rev-parse", "--show-toplevel").decode().strip()).resolve()
            if top != root:
                raise ValueError("--staged requires the repository root")
            names = git(root, "diff", "--cached", "--name-only", "--diff-filter=ACMR",
                        "-z").decode("utf-8").split("\0")
            for name in filter(None, names):
                checked += 1
                # Read index bytes, not a potentially different working tree version.
                mode = git(root, "ls-files", "--stage", "--", name).split(b" ", 1)[0]
                issues = inspect_content(name, git(root, "show", ":" + name))
                if mode != b"100644" and mode != b"100755":
                    issues.append("symlink/submodule is not allowed")
                failures.extend((name, issue) for issue in issues)
        else:
            for path in sorted(root.rglob("*")):
                relative = path.relative_to(root)
                if ".git" in relative.parts:
                    continue
                if path.is_symlink():
                    failures.append((str(relative), "symlink is not allowed"))
                elif path.is_file():
                    checked += 1
                    failures.extend((str(relative), issue)
                                    for issue in inspect_content(str(relative), path.read_bytes()))
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        print("CHECK ERROR: " + type(exc).__name__, file=sys.stderr)
        return 2
    for name, issue in failures:
        print(f"REJECT {name}: {issue}")
    print(f"Checked {checked} files; {len(failures)} issue(s).")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--all", action="store_true")
    mode.add_argument("--staged", action="store_true")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    return check(args.root.resolve(), args.staged)


if __name__ == "__main__":
    raise SystemExit(main())
