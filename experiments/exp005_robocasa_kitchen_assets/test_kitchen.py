"""Load candidate objects through the existing training overlay in a real RoboCasa kitchen."""
from pathlib import Path
import sys
import json
import traceback
import numpy as np
from PIL import Image

root = Path(__file__).resolve().parent
xr = Path.home() / "xr1"
sys.path[:0] = [str(xr / "rigid_pilot_20261112_v1"), str(xr / "robocasa365_official_20261112/robocasa_source")]
from real_batch_adapter_v1 import RigidPilotCounterToSink
from robocasa.utils.env_utils import create_env

rows = []
for name in ("mug", "scoop", "turner"):
    p = root / "sim" / name
    env = None
    try:
        env = create_env("RigidPilotCounterToSink", pilot_xml=str(p / "model.xml"), seed=8110, split=None,
                         obj_instance_split="pretrain", layout_ids=[14], style_ids=[12], camera_widths=512, camera_heights=512)
        obs = env.reset()
        pos = env.sim.data.body_xpos[env.obj_body_id["obj"]].copy()
        info = dict(asset=name,initialized=True,object_position=pos.tolist(),finite=bool(np.isfinite(pos).all()),
                    initial_success=bool(env._check_success()),instruction=env.get_ep_meta()["lang"])
        predictions = root / "predictions.json"
        if predictions.is_file():
            expected = json.loads(predictions.read_text())["objects"][name]["central"]
            sim_model = env.sim.model
            target = env.obj_body_id["obj"]
            mass = float(sim_model.body_subtreemass[target])
            info["loaded_mass_kg"] = mass
            info["predicted_mass_matches"] = bool(np.isclose(mass, expected["mass_kg"], rtol=1e-5))
        frames = [obs["robot0_"+c+"_image"][::-1] for c in ["agentview_left","agentview_right","eye_in_hand"]]
        Image.fromarray(np.concatenate(frames,axis=1)).save(root/(name+"_kitchen.jpg"),quality=94)
        rows.append(info)
    except Exception as exc:
        rows.append(dict(asset=name,initialized=False,error=repr(exc),trace=traceback.format_exc()[-1600:]))
    finally:
        if env is not None:
            env.close()
(root/"kitchen.json").write_text(json.dumps(rows,indent=2))
for row in rows:
    print(row)
