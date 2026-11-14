"""Exercise the predicted MJCF and measure the actual leading-edge collision envelope."""
import argparse
import itertools
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np
from physical_audit import fixture


def bench(model, mu, dt, solver, speed=0., force_ratio=0.):
    model.opt.timestep = dt
    model.opt.solver = getattr(mj.mjtSolver, "mjSOL_" + solver)
    # Pair hypotheses: both surfaces use the same value, including the priority-1 foot.
    model.geom_friction[:, 0] = mu
    body = model.body("object").id
    data = mj.MjData(model)
    for _ in range(round(.5/dt)):
        mj.mj_step(model, data)
    start = data.xipos[body].copy()
    data.qvel[model.jnt_dofadr[model.joint("fixture_free").id]] = speed
    data.xfrc_applied[body, 0] = force_ratio*model.body_mass[body]*9.81
    for _ in range(round(.35/dt)):
        mj.mj_step(model, data)
    mj.mj_forward(model,data)
    return float((data.xipos[body,0]-start[0])*1000)


def clip_face(points, x_limit):
    clipped = []
    for a, b in zip(points, np.roll(points,-1,axis=0)):
        if a[0] <= x_limit:
            clipped.append(a)
        if (a[0] <= x_limit) != (b[0] <= x_limit):
            clipped.append(a+(b-a)*((x_limit-a[0])/(b[0]-a[0])))
    return clipped


def edge_envelope(path):
    # This asset exports its annular slots/frames as OBJ hulls with zero geom pose.
    root = ET.parse(path).getroot()
    files = {m.get("name"):m.get("file") for m in root.findall("asset/mesh")}
    pieces = []
    for g in root.findall(".//geom[@group='0'][@type='mesh']"):
        assert g.get("quat") is None and not np.any(np.fromstring(g.get("pos","0 0 0"),sep=" "))
        vs, fs = [], []
        with (path.parent/files[g.get("mesh")]).open() as stream:
            for line in stream:
                if line.startswith("v "):
                    vs.append([float(x) for x in line.split()[1:4]])
                elif line.startswith("f "):
                    fs.append([int(x.split("/")[0])-1 for x in line.split()[1:]])
        pieces.extend(np.asarray(vs)[fs])
    faces = np.array(pieces)
    front = faces[:,:,0].min()
    section = np.asarray([p for tri in faces for p in clip_face(tri,front+.005)])
    result = {}
    for angle in (-5.,0.,5.):
        a = np.deg2rad(angle)
        heights = -np.sin(a)*section[:,0]+np.cos(a)*section[:,2]
        result[str(angle)] = float(np.ptp(heights)*1000)
    return result


def run(folder):
    result = {"mujoco": mj.__version__, "compiled": {}, "mug_sliding": []}
    for name in ("mug","scoop","turner"):
        path = folder/name/"model.xml"
        model = fixture(path, False)
        node = ET.parse(path).find(".//body[@name='object']/inertial")
        body = model.body("object").id
        rotation = np.empty(9)
        mj.mju_quat2Mat(rotation, model.body_iquat[body])
        rot = rotation.reshape(3,3)
        tensor = rot @ np.diag(model.body_inertia[body]) @ rot.T
        xx,yy,zz,xy,xz,yz = np.fromstring(node.get("fullinertia"),sep=" ")
        np.testing.assert_allclose(tensor,[[xx,xy,xz],[xy,yy,yz],[xz,yz,zz]], atol=3e-9)
        np.testing.assert_allclose(model.body_ipos[body], np.fromstring(node.get("pos"),sep=" "), atol=1e-8)
        np.testing.assert_allclose(model.body_subtreemass[model.body(name).id],float(node.get("mass")),rtol=1e-6)
        result["compiled"][name] = {"mass_kg":float(model.body_mass[body]),"center_mm":(model.body_ipos[body]*1000).tolist()}
        if name == "mug":
            for (surface,mus,muk), dt, solver in itertools.product(
                [("dry",.45,.35),("wet",.25,.18)], [.001,.002], ["NEWTON","PGS"]):
                result["mug_sliding"].append(dict(surface=surface,solver=solver,dt=dt,
                    coast_mm=bench(model,muk,dt,solver,speed=.20),
                    below_breakaway_mm=bench(model,mus,dt,solver,force_ratio=.85*mus),
                    above_breakaway_mm=bench(model,mus,dt,solver,force_ratio=1.15*mus)))
        if name == "turner":
            result["leading_5mm_gap_at_pitch_mm"] = edge_envelope(path)
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("sim",type=Path)
    parser.add_argument("report",type=Path)
    args=parser.parse_args()
    result=run(args.sim)
    args.report.write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(result))
