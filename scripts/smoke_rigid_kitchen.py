"""Isolated target-object adapter and real kitchen initialization smoke.

Does not change native registries, official task, or success predicate.
No teacher or task success is claimed by a successful initialization.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from robocasa.environments.kitchen.atomic.kitchen_pick_place import PickPlaceCounterToSink
from robocasa.utils.env_utils import create_env

class RigidPilotCounterToSink(PickPlaceCounterToSink):
    def __init__(self, pilot_xml, *args, **kwargs):
        self.pilot_xml = str(Path(pilot_xml).resolve())
        super().__init__(obj_groups='__rigid_pilot__', *args, **kwargs)

    def sample_object(self, groups, **kwargs):
        if groups == '__rigid_pilot__':
            return dict(mjcf_path=self.pilot_xml, scale=1.0, density=300), dict(
                groups_containing_sampled_obj=[], groups=groups, cat='rigid_box',
                split='engineering_pilot', mjcf_path=self.pilot_xml)
        return super().sample_object(groups, **kwargs)

    def get_obj_lang(self, obj_name='obj', get_preposition=False):
        if obj_name == 'obj':
            return 'blue block'
        return super().get_obj_lang(obj_name, get_preposition)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--index', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    out = args.output.resolve()
    repo = Path(__file__).resolve().parents[1]
    if out == repo or repo in out.parents:
        raise ValueError('runtime output must be outside code tree')
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.assets / 'manifest.json').read_text())
    record = manifest['candidates'][args.index]
    xml = args.assets / record['path']
    if hashlib.sha256(xml.read_bytes()).hexdigest() != record['xml_sha256']:
        raise ValueError('checksum mismatch')
    env = create_env('RigidPilotCounterToSink', pilot_xml=str(xml), seed=args.seed,
                     split='pretrain', camera_widths=256, camera_heights=256)
    try:
        obs = env.reset()
        from PIL import Image
        for name in ['robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand']:
            Image.fromarray(obs[name + '_image'][::-1]).save(out / (name + '.png'))
        body_id = env.obj_body_id['obj']
        pos = env.sim.data.body_xpos[body_id].copy()
        result = dict(scope='kitchen initialization only; no grasp/teacher/training',
            asset=record['id'], seed=args.seed, xml_sha256=record['xml_sha256'],
            target_position_m=pos.tolist(), initial_success=bool(env._check_success()),
            finite=bool(np.isfinite(env.sim.data.qpos).all()),
            instruction=env.get_ep_meta()['lang'], action_dim=int(env.action_dim),
            layout_id=int(env.layout_id), style_id=int(env.style_id))
        (out / 'initialization.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        if result['initial_success'] or not result['finite']:
            raise RuntimeError('invalid initialization')
    finally:
        env.close()

if __name__ == '__main__': main()

