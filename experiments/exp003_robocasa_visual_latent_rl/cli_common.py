"""Standard-library CLI checks; no model import or filesystem creation at --help."""
import argparse
from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parents[2]

def external_output(value):
    path = Path(value).expanduser().resolve()
    if path == REPO_ROOT or REPO_ROOT in path.parents:
        raise ValueError("Artifacts must be outside the code repository")
    if path.exists() and not path.is_dir():
        raise ValueError("Output root must be a directory")
    return path

def component(value):
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", value):
        raise argparse.ArgumentTypeError("Use one safe path component")
    return value

def port(value):
    n = int(value)
    if not 1024 <= n <= 65535:
        raise argparse.ArgumentTypeError("Port must be 1024..65535")
    return n

def parse_args(kind):
    p = argparse.ArgumentParser(description="Experimental visual latent SAC; trusted local inputs only.")
    p.add_argument("--model", type=Path, required=True, help="Reviewed local XR1 checkpoint; trust_remote_code executes its Python")
    p.add_argument("--port", type=port, required=True)
    p.add_argument("--run-root", type=Path, required=True, help="External artifact root, never repository")
    if kind == "server":
        p.add_argument("--trust-local-pickle", action="store_true", required=True,
                       help="Acknowledge unsafe pickle: trusted same-user loopback clients ONLY; no tunnels")
    else:
        p.add_argument("--xr1-repo", type=Path, required=True)
        p.add_argument("--adapter-dir", type=Path, required=True, help="Reviewed smoke_rigid_kitchen and review_rollout_history modules")
        p.add_argument("--source-root", type=Path, required=True)
        p.add_argument("--assets-root", type=Path, required=True)
        if kind == "train":
            p.add_argument("--asset", choices=["t_grip_00","t_grip_04","bridge_00","bridge_04"], required=True)
            p.add_argument("--episode", type=int, required=True)
            p.add_argument("--noise", type=int, required=True)
        else:
            p.add_argument("--mode", choices=["native","before","after"], required=True)
            p.add_argument("--scenario", choices=["t_grip_00","t_grip_04","bridge_00","bridge_04","t_grip_01","bridge_01"], required=True)
            p.add_argument("--label", type=component, required=True)
            p.add_argument("--actor-checkpoint", type=Path)
    args = p.parse_args()
    args.run_root = external_output(args.run_root)
    for name in ("model","xr1_repo","adapter_dir","source_root","assets_root"):
        value = getattr(args, name, None)
        if value is not None and not value.is_dir():
            p.error(name + " must be an existing reviewed directory")
    if kind == "train":
        if args.episode < 0: p.error("episode must be nonnegative")
        args.mode = "train"
    if kind == "eval" and args.mode == "after":
        if args.actor_checkpoint is None or not args.actor_checkpoint.is_file():
            p.error("after requires --actor-checkpoint")
    return args

def new_output(path):
    path = external_output(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def require_selected_gpu():
    import os
    selected = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not selected or selected == "-1":
        raise ValueError("Set CUDA_VISIBLE_DEVICES to an explicitly authorized GPU before serving")
    return selected
