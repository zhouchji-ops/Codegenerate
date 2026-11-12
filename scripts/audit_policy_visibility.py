"""Offline target-geometry visibility diagnostic, never an extra policy input.

Run one snapshot/rollout per process. Masks are rendered with antialiasing disabled;
crop counts are BEFORE RGB bilinear resize and are not semantic recognition scores.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

CAMERAS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")

def history_indices(step, length=4, interval=2):
    if any(type(v) is not int for v in (step, length, interval)):
        raise ValueError("History arguments must be integers")
    if step < 0 or length < 1 or interval < 1:
        raise ValueError("Invalid history arguments")
    return [max(0, step - (length - 1 - i) * interval) for i in range(length)]

def crop_bounds(width, height, ratio=.95):
    if type(width) is not int or type(height) is not int or min(width, height) < 1:
        raise ValueError("Image dimensions must be positive integers")
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("Crop ratio must be finite and positive")
    cw = width if ratio >= 1 else max(1, int(width * ratio))
    ch = height if ratio >= 1 else max(1, int(height * ratio))
    left, top = (width - cw) // 2, (height - ch) // 2
    return left, top, left + cw, top + ch

def query_schedule(total_steps, replan=16):
    if type(total_steps) is not int or type(replan) is not int or total_steps < 1 or replan < 1:
        raise ValueError("Positive integer step count and replan interval required")
    return list(range(0, total_steps, replan))

def validate_output(output, repo):
    output, repo = Path(output).resolve(), Path(repo).resolve()
    if output == repo or repo in output.parents:
        raise ValueError("Diagnostic artifacts must be outside the code repository")
    if output.exists():
        raise FileExistsError(output)
    return output

def audit(args):
    # Heavy dependencies load only after CLI parsing, not during --help/tests.
    import numpy as np
    import mujoco

    root = (args.snapshot or args.rollout).resolve()
    model_file = root / "model.xml"
    model_bytes = model_file.read_bytes()
    model_hash = hashlib.sha256(model_bytes).hexdigest()
    model = mujoco.MjModel.from_xml_path(str(model_file))
    model.vis.quality.offsamples = 0
    data = mujoco.MjData(model)
    target_ids = [
        i for i in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith(args.target_geom_prefix)
    ]
    if not target_ids:
        raise ValueError("No target geoms matched; do not report this as invisibility")
    for camera in args.cameras:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera) < 0:
            raise ValueError("Missing camera: " + camera)
    left, top, right, bottom = crop_bounds(args.width, args.height, args.crop)
    handle = None
    metadata_verified = None
    try:
        if args.snapshot:
            states = np.load(root / "initial_sim_state.npy", allow_pickle=False)[None, :]
            metadata_verified = False
            metadata_path = root / "snapshot_metadata.json"
            if metadata_path.exists():
                meta = json.loads(metadata_path.read_text())
                if meta.get("model_sha256") != model_hash:
                    raise ValueError("Snapshot model hash mismatch")
                if meta.get("sim_state_sha256") != hashlib.sha256(states[0].tobytes()).hexdigest():
                    raise ValueError("Snapshot state hash mismatch")
                metadata_verified = True
            queries = [0]
            schedule_source = "snapshot initial state only"
        else:
            import h5py
            handle = h5py.File(root / "trajectory.hdf5", "r")
            states = handle["sim_states"]
            if "actions" in handle and len(handle["actions"]) != len(states):
                raise ValueError("Pre-action states/actions length mismatch")
            queries = query_schedule(len(states), args.replan)
            schedule_source = "assumed fixed replan schedule; confirm against collector before calling these actual queries"
            if args.query_steps:
                queries = json.loads(args.query_steps.read_text())
                if (not isinstance(queries, list) or not queries
                    or any(type(t) is not int or not 0 <= t < len(states) for t in queries)
                    or queries != sorted(set(queries))):
                    raise ValueError("Query steps must be unique, increasing, in-range integers")
                schedule_source = "explicit user-supplied query indices"
        expected = 1 + model.nq + model.nv + model.na
        if len(states.shape) != 2 or states.shape[1] != expected:
            raise ValueError("Expected flattened MuJoCo time/qpos/qvel/act states")
        indices = sorted({f for q in queries for f in history_indices(q, args.history, args.interval)})
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        renderer.enable_segmentation_rendering()
        rows = []
        state_hash = hashlib.sha256()
        try:
            for index in indices:
                state = np.asarray(states[index])
                if not np.isfinite(state).all():
                    raise ValueError("Nonfinite simulator state")
                state_hash.update(state.tobytes())
                mujoco.mj_resetData(model, data)
                data.time = state[0]
                data.qpos[:] = state[1:1 + model.nq]
                data.qvel[:] = state[1 + model.nq:1 + model.nq + model.nv]
                if model.na:
                    data.act[:] = state[-model.na:]
                mujoco.mj_forward(model, data)
                row = {"step": index, "cameras": {}}
                for camera in args.cameras:
                    renderer.update_scene(data, camera=camera)
                    seg = renderer.render()
                    if seg.shape != (args.height, args.width, 2):
                        raise ValueError("Unexpected geom segmentation shape")
                    geom_ids = seg[:, :, 0][seg[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)]
                    if ((geom_ids < 0) | (geom_ids >= model.ngeom)).any():
                        raise ValueError("Out-of-range geom ID in diagnostic renderer")
                    mask = np.isin(seg[:, :, 0], target_ids) & (
                        seg[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
                    row["cameras"][camera] = {
                        "raw_pixels": int(mask.sum()),
                        "crop_retained_pixels_before_resize": int(mask[top:bottom, left:right].sum()),
                    }
                rows.append(row)
        finally:
            renderer.close()
        lookup = {row["step"]: row for row in rows}
        query_rows = []
        for step in queries:
            hist = history_indices(step, args.history, args.interval)
            cams = [c for index in hist for c in lookup[index]["cameras"].values()]
            query_rows.append({
                "step": step, "history_indices": hist,
                "any_raw_visible": any(c["raw_pixels"] > 0 for c in cams),
                "any_cropped_visible": any(c["crop_retained_pixels_before_resize"] > 0 for c in cams),
            })
        return {
            "schema_version": 1, "input_kind": "snapshot" if args.snapshot else "rollout",
            "snapshot_metadata_verified": metadata_verified,
            "query_schedule_source": schedule_source, "history": args.history,
            "interval": args.interval, "replan": args.replan, "crop_ratio": args.crop,
            "crop_bounds_xyxy": [left, top, right, bottom],
            "render_size_wh": [args.width, args.height], "antialias_samples": 0,
            "model_sha256": model_hash, "selected_state_rows_sha256": state_hash.hexdigest(),
            "target_geom_prefix": args.target_geom_prefix, "target_geom_count": len(target_ids),
            "mujoco_version": mujoco.__version__, "frames": rows, "queries": query_rows,
            "summary": {
                "rendered_history_frames": len(rows), "queries": len(query_rows),
                "raw_visible_queries": sum(r["any_raw_visible"] for r in query_rows),
                "cropped_visible_queries": sum(r["any_cropped_visible"] for r in query_rows),
            },
            "limitation": (
                "Offline geom-ID masks with antialiasing disabled, not recorded RGB or model input. "
                "Counts are before bilinear resize; tiny-edge semantics and recognizability are not established. "
                "Only listed histories are audited; initial invisibility does not imply episode-long invisibility. "
                "State restoration omits controls and does not re-execute actions or reproduce policy internals."
            ),
        }
    finally:
        if handle is not None:
            handle.close()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--snapshot", type=Path)
    group.add_argument("--rollout", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--crop", type=float, default=.95)
    p.add_argument("--history", type=int, default=4)
    p.add_argument("--interval", type=int, default=2)
    p.add_argument("--replan", type=int, default=16)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--cameras", nargs="+", default=list(CAMERAS))
    p.add_argument("--target-geom-prefix", default="obj_")
    p.add_argument("--query-steps", type=Path, help="Optional external JSON list of actual query indices")
    args = p.parse_args()
    if not args.target_geom_prefix:
        p.error("Target prefix cannot be empty")
    if args.snapshot and args.query_steps:
        p.error("Explicit query indices are only for rollouts")
    history_indices(0, args.history, args.interval)
    query_schedule(1, args.replan)
    crop_bounds(args.width, args.height, args.crop)
    output = validate_output(args.output, Path(__file__).resolve().parents[1])
    report = audit(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report["summary"]))

if __name__ == "__main__":
    main()
