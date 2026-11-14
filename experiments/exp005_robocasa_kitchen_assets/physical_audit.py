"""Measure authored cavity geometry and run matched rigid-contact tests on final MJCF."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np


def vertices(path):
    with path.open() as stream:
        return np.array([line.split()[1:4] for line in stream if line.startswith("v ")], dtype=float)


def capacity(path, root):
    # Each wall sector of these two assets has inner/outer vertices at floor and lip.
    # Reconstruct the inner polygon at each height up to the lowest spill point.
    lower, upper = {}, {}
    for mesh in root.findall("asset/mesh"):
        if not any(s in mesh.get("name") for s in ("integral_wall_and_foot", "oblique_integral_shell")):
            continue
        points = vertices(path.parent / mesh.get("file"))
        for p in points:
            angle = round(float(np.arctan2(p[1], p[0]))*1e6) % round(2*np.pi*1e6)
            if abs(angle-round(np.pi*1e6)) <= 1:
                angle = round(np.pi*1e6)
            if angle <= 1 or angle >= round(2*np.pi*1e6)-1:
                angle = 0
            ring = lower if abs(p[2]-min(points[:, 2])) < 1e-7 else upper
            if angle not in ring or np.linalg.norm(p[:2]) < np.linalg.norm(ring[angle][:2]):
                ring[angle] = p
    assert lower.keys() == upper.keys() and len(lower) >= 24
    keys = sorted(lower)
    bottom, rim = np.array([lower[k] for k in keys]), np.array([upper[k] for k in keys])
    floor, spill = max(bottom[:, 2]), min(rim[:, 2])
    def area(z):
        t = ((z-bottom[:, 2])/(rim[:, 2]-bottom[:, 2]))[:, None]
        x, y = (bottom[:, :2] + t*(rim[:, :2]-bottom[:, :2])).T
        return abs(np.dot(x,np.roll(y,-1))-np.dot(y,np.roll(x,-1)))/2
    # The area is quadratic since each polygon vertex interpolates linearly.
    volume = (spill-floor)*(area(floor)+4*area((spill+floor)/2)+area(spill))/6
    return dict(wall_sectors=len(keys), spill_height_mm=spill*1000, geometric_capacity_ml=volume*1e6)


def fixture(path, with_ball):
    root = ET.parse(path).getroot()
    for tag in ("mesh", "texture"):
        for node in root.findall("asset/"+tag):
            if node.get("file"):
                node.set("file", str((path.parent/node.get("file")).resolve()))
    world = root.find("worldbody")
    obj = world.find("body")
    ET.SubElement(obj, "freejoint", name="fixture_free")
    ET.SubElement(world, "geom", name="plane", type="plane", size="1 1 .02", friction=".65 .005 .0001")
    if with_ball:
        b = ET.SubElement(world, "body", name="probe", pos="0.006 0 .15")
        ET.SubElement(b, "freejoint", name="probe_free")
        ET.SubElement(b, "geom", name="ball", type="sphere", size=".006", mass=".005")
    return mj.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def episode(model, name, solver, dt, with_ball):
    model.opt.timestep = dt
    model.opt.solver = getattr(mj.mjtSolver, "mjSOL_"+solver)
    d = mj.MjData(model)
    physical = model.body("object").id
    q = model.jnt_qposadr[model.joint("fixture_free").id]
    d.qpos[q+2] += .002
    for _ in range(round(.5/dt)):
        mj.mj_step(model,d)
    start = d.xipos[physical].copy()
    if with_ball:
        for _ in range(round(.7/dt)):
            mj.mj_step(model,d)
        local = d.xmat[physical].reshape(3,3).T @ (d.body("probe").xpos-d.xpos[physical])
        ball_id = model.geom("ball").id
        supports = []
        for i, c in enumerate(d.contact):
            if ball_id in (c.geom1,c.geom2):
                other = c.geom2 if c.geom1==ball_id else c.geom1
                force = np.zeros(6)
                mj.mj_contactForce(model,d,i,force)
                if force[0] > .001:
                    supports.append(model.geom(int(other)).name)
        inside = np.hypot(*local[:2])<.024 and .005<local[2]<.03
        return dict(ball_center_local_mm=(local*1000).tolist(), supporting_geoms=supports,
                    retained=bool(inside and "plane" not in supports and any("bottom_disk" in s for s in supports)))
    # A prescribed horizontal force at the body COM allows both slip and tilt.
    # 0.9 mg for 150 ms, then free deceleration on the test surface.
    for _ in range(round(.15/dt)):
        d.xfrc_applied[physical,0] = .9*model.body_subtreemass[model.body(name).id]*9.81
        mj.mj_step(model,d)
    d.xfrc_applied[:] = 0
    for _ in range(round(.6/dt)):
        mj.mj_step(model,d)
    return dict(travel_mm=float((d.xipos[physical,0]-start[0])*1000),
                final_tilt_deg=float(np.degrees(np.arccos(np.clip(d.xmat[physical,8],-1,1)))),
                final_speed_m_s=float(np.linalg.norm(d.qvel[:3])),
                finite=bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()))


def audit(folder):
    result = {"mujoco": mj.__version__, "assets": {}}
    for name in ("mug","scoop","turner"):
        path = folder/name/"model.xml"
        xml = ET.parse(path).getroot()
        bbox = xml.find(".//geom[@name='reg_bbox']")
        entry = {"visual_bbox_mm": (2e3*np.fromstring(bbox.get("size"),sep=" ")).tolist()}
        if name != "turner":
            entry.update(capacity(path, xml))
        push = fixture(path,False)
        ball = fixture(path,True) if name != "turner" else None
        # Same calibrated scale and inertia. Change just the floor contact override.
        default_floor = None
        if ball is not None:
            disk = next(i for i in range(ball.ngeom) if "bottom_disk" in (ball.geom(i).name or ""))
            if ball.geom_priority[disk] > 0:
                default_floor = fixture(path,True)
                default_floor.geom_priority[disk] = 0
                default_floor.geom_solref[disk] = [.02, 1.]
        entry["trials"] = []
        for solver in ("NEWTON","PGS"):
            for dt in (.002,.001):
                trial = dict(solver=solver, timestep_s=dt, iterations=int(push.opt.iterations),
                             push=episode(push,name,solver,dt,False))
                if ball is not None:
                    trial["catch"] = episode(ball,name,solver,dt,True)
                if default_floor is not None:
                    trial["catch_default_floor_same_scale"] = episode(default_floor,name,solver,dt,True)
                entry["trials"].append(trial)
        result["assets"][name] = entry
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("sim_folder", type=Path)
    p.add_argument("report", type=Path)
    args = p.parse_args()
    report = audit(args.sim_folder)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, a in report["assets"].items():
        print(name, a.get("geometric_capacity_ml"), [(t["push"]["travel_mm"],t.get("catch",{}).get("retained")) for t in a["trials"]])
