"""Train-only initial observation-pose engineering, not dynamic reachability.

Source roles are bound by a separate frozen registry. Fixed yaw candidates are
audited independently; failures are retained, never replaced with searched angles.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    with Path(path).open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, default=lambda x: x.item())

def validate_inputs(args):
    import numpy as np
    from observation_pose_checks import require_train_asset
    root = args.source.resolve()
    meta_path = root / "snapshot_metadata.json"
    meta = json.loads(meta_path.read_text())
    registry = json.loads(args.source_role_manifest.read_text())
    matches = [r for r in registry["sources"] if Path(r["snapshot_path"]).resolve() == root]
    if len(matches) != 1 or matches[0]["role"] != args.role:
        raise ValueError("Source role absent, ambiguous, or different from requested role")
    if matches[0]["snapshot_metadata_sha256"] != digest(meta_path):
        raise ValueError("Source metadata differs from frozen role registry")
    if matches[0].get("episode_metadata_sha256") != digest(root / "episode_metadata.json"):
        raise ValueError("Source episode metadata missing from or different from frozen role registry")
    asset = Path(meta["asset_xml"]).resolve()
    manifest = json.loads(args.asset_manifest.read_text())
    asset_root = args.asset_manifest.resolve().parent
    candidates = []
    for record in manifest["candidates"]:
        path = (asset_root / record["path"]).resolve()
        if path == asset:
            if asset_root not in path.parents:
                raise ValueError("Asset escaped its manifest directory")
            candidates.append(record)
    if len(candidates) != 1:
        raise ValueError("Asset must match exactly one manifest entry")
    entry = candidates[0]
    require_train_asset(entry)
    if digest(asset) != entry["xml_sha256"] or digest(asset) != meta["asset_sha256"]:
        raise ValueError("Asset checksum mismatch")
    if digest(root / "model.xml") != meta["model_sha256"]:
        raise ValueError("Source compiled model checksum mismatch")
    initial = np.load(root / "initial_sim_state.npy", allow_pickle=False)
    if initial.ndim != 1 or not np.isfinite(initial).all():
        raise ValueError("Malformed/nonfinite source state")
    if hashlib.sha256(initial.tobytes()).hexdigest() != meta["sim_state_sha256"]:
        raise ValueError("Source state checksum mismatch")
    with np.load(root / "initial_observation.npz", allow_pickle=False) as obs:
        if set(obs.files) != set(meta["observation_hashes"]):
            raise ValueError("Source observation keys differ")
        for name in obs.files:
            if hashlib.sha256(obs[name].tobytes()).hexdigest() != meta["observation_hashes"][name]:
                raise ValueError("Source observation checksum mismatch: " + name)
    if meta.get("initial_success") is not False:
        raise ValueError("Source must explicitly be an initially unsuccessful task")
    episode = json.loads((root / "episode_metadata.json").read_text())
    lineage = {
        "snapshot_path": str(root), "snapshot_metadata_sha256": digest(meta_path),
        "episode_metadata_sha256": digest(root / "episode_metadata.json"),
        "asset_manifest_sha256": digest(args.asset_manifest),
        "source_role_manifest_sha256": digest(args.source_role_manifest),
        "source_role": args.role, "source_asset_id": entry["id"],
        "capture_code_sha256": digest(__file__),
        "checks_code_sha256": digest(Path(__file__).with_name("observation_pose_checks.py")),
        "source_model_sha256": meta["model_sha256"],
        "source_sim_state_sha256": meta["sim_state_sha256"],
    }
    return root, meta, episode, initial, lineage

def capture(args, validated):
    import sys
    import types
    import numpy as np
    import mujoco
    sys.path.insert(0, str(args.adapter_dir.resolve()))
    from smoke_rigid_kitchen import RigidPilotCounterToSink
    from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv
    from review_rollout_history import fixture_consistency
    from observation_pose_checks import retreat_delta, contact_changes, changed_indices_exact

    root, meta, episode, initial, lineage = validated
    env = RoboCasaGymEnv(
        env_name="RigidPilotCounterToSink", pilot_xml=meta["asset_xml"], split=None,
        obj_instance_split="pretrain", layout_ids=[meta["layout_id"]],
        style_ids=[meta["style_id"]], seed=meta["seed"])
    try:
        raw = env.env
        original = raw.sample_object
        def sample(this, groups, **kwargs):
            if isinstance(groups, str) and Path(groups).resolve() == Path(meta["asset_xml"]).resolve():
                return original("__rigid_pilot__", **kwargs)
            return original(groups, **kwargs)
        raw.sample_object = types.MethodType(sample, raw)
        raw.set_ep_meta(copy.deepcopy(episode))
        env.reset(seed=meta["seed"])
        if (int(raw.layout_id), int(raw.style_id)) != (meta["layout_id"], meta["style_id"]):
            raise ValueError("Episode layout/style restoration failed")
        for name in ("sink", "counter"):
            if getattr(raw, name).name != episode["fixture_refs"][name]:
                raise ValueError("Episode fixture reference restoration failed")
        raw.reset_from_xml_string((root / "model.xml").read_text())
        raw.sim.set_state_from_flattened(initial)
        raw.sim.forward()
        if not np.array_equal(raw.sim.get_state().flatten(), initial):
            raise ValueError("Source simulator state did not restore exactly")
        if not fixture_consistency(raw)["passed"]:
            raise ValueError("Source sink cache disagrees with compiled model")
        m, d = raw.sim.model, raw.sim.data
        yaw_joint = m.joint_name2id("mobilebase0_joint_mobile_yaw")
        forward_joint = m.joint_name2id("mobilebase0_joint_mobile_forward")
        if int(m.jnt_type[yaw_joint]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError("Expected scalar hinge yaw joint")
        if int(m.jnt_type[forward_joint]) != int(mujoco.mjtJoint.mjJNT_SLIDE):
            raise ValueError("Expected scalar slide forward joint")
        qy, qf = int(m.jnt_qposadr[yaw_joint]), int(m.jnt_qposadr[forward_joint])
        vy, vf = int(m.jnt_dofadr[yaw_joint]), int(m.jnt_dofadr[forward_joint])
        target = raw.obj_body_id["obj"]
        baseline_qpos, baseline_qvel, baseline_act = d.qpos.copy(), d.qvel.copy(), d.act.copy()
        target_pose = np.r_[d.body_xpos[target], d.body_xquat[target]]
        forward_body = int(m.jnt_bodyid[forward_joint])
        base_position = d.body_xpos[forward_body].copy()
        axis = d.xaxis[forward_joint].copy()
        toward = np.asarray(raw.sink.pos) - base_position
        retreat = retreat_delta(axis.tolist(), toward.tolist(), args.retreat) if args.retreat else 0.0
        away_axis = -np.sign(np.dot(axis, toward)) * axis if args.retreat else np.zeros(3)

        def robot_body(body):
            while body:
                if (m.body_id2name(body) or "").startswith(("robot", "gripper", "mobilebase")):
                    return True
                body = int(m.body_parentid[body])
            return False

        def contacts():
            obstacles, target_contacts = [], []
            for cid, c in enumerate(d.contact):
                bodies = [int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])]
                names = [m.geom_id2name(c.geom1) or "", m.geom_id2name(c.geom2) or ""]
                robots = [robot_body(b) for b in bodies]
                if sum(robots) == 1:
                    obstacles.append({"pair": names, "distance_m": float(c.dist), "penetration_m": max(0.0, -float(c.dist))})
                if target in bodies:
                    other_index = 1 if bodies[0] == target else 0
                    force = np.zeros(6)
                    active = int(c.efc_address) >= 0
                    if active:
                        mujoco.mj_contactForce(getattr(m, "_model", m), getattr(d, "_data", d), cid, force)
                    target_contacts.append({
                        "other": names[other_index], "robot": robots[other_index],
                        "active": active, "normal_force_N": float(force[0])})
            return obstacles, target_contacts

        def limited_joints():
            rows = []
            for joint in range(m.njnt):
                if m.jnt_limited[joint] and int(m.jnt_type[joint]) in (2, 3):
                    value = float(d.qpos[int(m.jnt_qposadr[joint])])
                    lo, hi = map(float, m.jnt_range[joint])
                    excess = max(lo - value, value - hi, 0.0)
                    if excess > 0:
                        rows.append({"joint": m.joint_id2name(joint), "excess": excess})
            return rows

        baseline_contacts, _ = contacts()
        baseline_limits = limited_joints()
        cases = []
        for index, yaw in enumerate(args.yaw):
            folder = args.output / ("case_%03d" % index)
            folder.mkdir()
            record = {
                "case_id": folder.name, "role": args.role, "yaw_delta_rad": yaw,
                "retreat_joint_delta_m": float(retreat), "source_lineage": lineage,
                "baseline_robot_contacts": baseline_contacts, "baseline_limit_violations": baseline_limits,
                "camera_bodies": {cam: m.body_id2name(int(m.cam_bodyid[m.camera_name2id(cam)]))
                    for cam in ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")},
                "scope": "initial pose and discrete kinematic sweep only; not dynamic/continuous reachability",
            }
            try:
                nr, ny = math.ceil(abs(retreat) / .005), math.ceil(abs(yaw) / .01)
                points = [(0.0, 0.0)]
                points += [(retreat * i / nr, 0.0) for i in range(1, nr + 1)]
                points += [(retreat, yaw * i / ny) for i in range(1, ny + 1)]
                sweep = []
                record["planned_sweep_points"] = len(points)
                record["sweep"] = sweep
                for move, angle in points:
                    raw.sim.set_state_from_flattened(initial)
                    d.qpos[qf] += move
                    d.qpos[qy] += angle
                    d.qvel[vy] = 0
                    if args.retreat:
                        d.qvel[vf] = 0
                    raw.sim.forward()
                    current, support = contacts()
                    changes = contact_changes(baseline_contacts, current)
                    pose = np.r_[d.body_xpos[target], d.body_xquat[target]]
                    allowed_q = [qy, qf] if args.retreat else [qy]
                    allowed_v = [vy, vf] if args.retreat else [vy]
                    limit_ok = all(not m.jnt_limited[j] or
                        m.jnt_range[j, 0] <= d.qpos[int(m.jnt_qposadr[j])] <= m.jnt_range[j, 1]
                        for j in ([yaw_joint, forward_joint] if args.retreat else [yaw_joint]))
                    qpos_changes = changed_indices_exact(baseline_qpos.tolist(), d.qpos.tolist(), allowed_q)
                    qvel_changes = changed_indices_exact(baseline_qvel.tolist(), d.qvel.tolist(), allowed_v)
                    positive = [c for c in support if c["active"] and c["normal_force_N"] > 1e-8]
                    target_supported = any(not c["robot"] for c in positive) and not any(c["robot"] for c in positive)
                    retreat_projection = float(np.dot(d.body_xpos[forward_body] - base_position, away_axis))
                    row = {
                        "forward_delta": move, "yaw_delta": angle, "contact_changes": changes,
                        "all_other_qpos_exact": True, "all_other_qvel_exact": True,
                        "changed_qpos_indices": qpos_changes, "changed_qvel_indices": qvel_changes,
                        "act_exact": bool(np.array_equal(d.act, baseline_act)),
                        "target_pose_exact": bool(np.array_equal(pose, target_pose)),
                        "finite": bool(np.isfinite(raw.sim.get_state().flatten()).all()),
                        "modified_joints_within_limits": bool(limit_ok),
                        "target_supported_without_robot": target_supported,
                        "target_contacts": support, "robot_contacts": current,
                        "away_projection_m": retreat_projection,
                    }
                    sweep.append(row)
                    if (not all(row[k] for k in ("all_other_qpos_exact", "all_other_qvel_exact",
                            "act_exact", "target_pose_exact", "finite", "modified_joints_within_limits",
                            "target_supported_without_robot")) or changes["hard_reject"]):
                        raise ValueError("Kinematic safety check failed at predefined sweep point")
                    if args.retreat and move and retreat_projection <= 0:
                        raise ValueError("Forward translation did not move away from sink")
                fixture = fixture_consistency(raw)
                record["fixture_consistency"] = fixture
                record["initial_success"] = bool(raw._check_success())
                if not fixture["passed"] or record["initial_success"]:
                    raise ValueError("Invalid fixture cache or initially successful task")
                for robot in raw.robots:
                    robot.composite_controller.reset()
                changed_indices_exact(baseline_qpos.tolist(), d.qpos.tolist(), allowed_q)
                changed_indices_exact(baseline_qvel.tolist(), d.qvel.tolist(), allowed_v)
                if not np.array_equal(d.act, baseline_act):
                    raise ValueError("Controller reset changed actuator state")
                obs = env.get_observation(raw._get_observations(force_update=True))
                if obs["annotation.human.task_description"] != meta["instruction"]:
                    raise ValueError("Instruction changed during source reconstruction")
                state = raw.sim.get_state().flatten()
                xml = raw.sim.model.get_xml().encode("utf8")
                changed = copy.deepcopy(meta)
                changed.update(
                    role=args.role, split=args.role,
                    purpose="train-asset observation-pose engineering; not official benchmark",
                    source_lineage=lineage, yaw_delta_rad=yaw, retreat_joint_delta_m=float(retreat),
                    model_sha256=hashlib.sha256(xml).hexdigest(),
                    sim_state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),
                    observation_hashes={k: hashlib.sha256(np.asarray(v).tobytes()).hexdigest() for k, v in obs.items()},
                    initial_success=False)
                (folder / "model.xml").write_bytes(xml)
                np.save(folder / "initial_sim_state.npy", state, allow_pickle=False)
                np.savez_compressed(folder / "initial_observation.npz", **obs)
                write_json(folder / "episode_metadata.json", episode)
                write_json(folder / "snapshot_metadata.json", changed)
                record["status"] = "initial_pose_candidate_not_rollout_validated"
            except Exception as error:
                record["status"] = "rejected"
                record["error"] = str(error)
            write_json(folder / "initial_safety.json", record)
            cases.append(record)
            print(json.dumps({"case_id": folder.name, "status": record["status"]}), flush=True)
        return cases
    finally:
        env.close()

def main():
    from observation_pose_checks import validate_pose_plan
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--asset-manifest", type=Path, required=True)
    p.add_argument("--source-role-manifest", type=Path, required=True)
    p.add_argument("--adapter-dir", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--yaw", nargs="+", type=float, required=True)
    p.add_argument("--retreat", type=float, default=0.0, help="Nonnegative distance away from sink, metres")
    p.add_argument("--role", choices=["observation_train", "observation_validation_only"], required=True)
    args = p.parse_args()
    args.yaw, args.retreat = validate_pose_plan(args.yaw, args.retreat)
    args.output = args.output.resolve()
    repo = Path(__file__).resolve().parents[1]
    if args.output == repo or repo in args.output.parents:
        p.error("Outputs must be outside the code repository")
    validated = validate_inputs(args)
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "schema_version": 1, "role": args.role, "yaw_list_rad": args.yaw,
        "retreat_distance_m": args.retreat, "source_lineage": validated[-1],
        "kinematic_sweep_resolution": {"retreat_m": .005, "yaw_rad": .01},
        "scope": "predefined initial-state engineering; no automatic angle search or dynamic reachability claim",
    }
    write_json(args.output / "preregistered_protocol.json", protocol)
    try:
        cases = capture(args, validated)
        write_json(args.output / "summary.json", {"cases": [
            {"case_id": c["case_id"], "status": c["status"], "error": c.get("error")} for c in cases]})
    except Exception as error:
        write_json(args.output / "fatal_error.json", {"error": str(error), "scope": "shared setup failed"})
        raise

if __name__ == "__main__":
    main()
