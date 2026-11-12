"""Auxiliary physical diagnostics. Never replaces the official benchmark metric.
Frozen-other-joint continuation is offline evidence, not a live-policy rollout.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import numpy as np

def physical_candidate(official, horizontal, contacts, penetration, linear, angular):
    """Conservative engineering gate, independently testable for reward hacking."""
    return bool(official and horizontal and contacts
                and any(c['sink'] and c.get('active', False)
                        and c.get('normal_force_N', 0.) > 1e-8 for c in contacts)
                and not any(c['robot'] for c in contacts)
                and 0 <= penetration < .001 and 0 <= linear < .01
                and 0 <= angular < .1)

def snapshot(env):
    import mujoco
    import robosuite.utils.transform_utils as T
    from robocasa.utils import object_utils as OU
    m, d = env.sim.model, env.sim.data
    b = env.obj_body_id['obj']
    j = int(m.body_jntadr[b])
    va = int(m.jnt_dofadr[j])
    pos = d.body_xpos[b].copy()
    pts = np.asarray(env.objects['obj'].get_bbox_points(
        trans=pos, rot=T.convert_quat(d.body_xquat[b], to='xyzw')))
    boundaries = []
    for name, (p0, px, py, pz) in env.sink.get_int_sites(relative=False).items():
        axes = []
        for end in (px, py, pz):
            axis = end-p0
            length = np.linalg.norm(axis)
            v = (pts-p0) @ (axis/length)
            axes.append(dict(length_m=float(length), min_m=float(v.min()), max_m=float(v.max())))
        boundaries.append(dict(region=name, axes=axes))
    horizontal = any(all(a['min_m'] >= 0 and a['max_m'] <= a['length_m']
                        for a in r['axes'][:2]) for r in boundaries)
    contacts = []
    for contact_id, c in enumerate(d.contact):
        if b not in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]):
            continue
        other = c.geom2 if m.geom_bodyid[c.geom1] == b else c.geom1
        other_body = int(m.geom_bodyid[other])
        names = []
        while other_body:
            names.append(m.body_id2name(other_body) or '')
            other_body = int(m.body_parentid[other_body])
        other_name = m.geom_id2name(other) or ''
        robot = any(n.startswith(('robot', 'gripper')) for n in names+[other_name])
        sink = other_name.startswith(env.sink.naming_prefix)
        active = int(c.efc_address) >= 0
        force = np.zeros(6)
        if active:
            mujoco.mj_contactForce(getattr(m, '_model', m), getattr(d, '_data', d), contact_id, force)
        contacts.append(dict(other=other_name, distance_m=float(c.dist),
                             robot=robot, sink=sink, dim=int(c.dim),
                             active=active, normal_force_N=float(force[0])))
    official = bool(env._check_success())
    full = bool(OU.obj_inside_of(env, 'obj', env.sink, partial_check=False, th=0.0))
    linear = float(np.linalg.norm(d.qvel[va:va+3]))
    angular = float(np.linalg.norm(d.qvel[va+3:va+6]))
    penetration = max([max(0, -c['distance_m']) for c in contacts], default=0.)
    # Deliberately conservative: direct sink support required. May reject
    # valid stacking on a distractor. Thresholds are engineering diagnostics.
    candidate = physical_candidate(official, horizontal, contacts, penetration, linear, angular)
    return dict(official_success=official, strict_bbox_th0=full,
                horizontal_containment=horizontal, position_m=pos.tolist(),
                linear_speed_m_s=linear, angular_speed_rad_s=angular,
                maximum_penetration_m=penetration, contacts=contacts,
                region_boundaries=boundaries, instantaneous_candidate=bool(candidate))

class StableSuccessGate:
    """Call after every real control step; reset for every episode.
    Sample window does not replace time coverage: >=0.5 simulation seconds.
    """
    def __init__(self, minimum_seconds=.5):
        if not np.isfinite(minimum_seconds) or minimum_seconds <= 0:
            raise ValueError('minimum_seconds must be finite and positive')
        self.minimum_seconds = minimum_seconds
        self.samples = collections.deque()
    def reset(self):
        self.samples.clear()
    def update(self, env):
        r = snapshot(env)
        t = float(env.sim.data.time)
        if not r['instantaneous_candidate']:
            self.reset()
            return False, r
        if self.samples and t <= self.samples[-1][0]:
            self.reset()
        self.samples.append((t, np.array(r['position_m'])))
        while len(self.samples)>1 and t-self.samples[1][0] >= self.minimum_seconds:
            self.samples.popleft()
        displacement = max(np.linalg.norm(p-self.samples[0][1]) for _,p in self.samples)
        verified = t-self.samples[0][0] >= self.minimum_seconds and displacement < .005
        return bool(verified), r

def validate_output_folder(folder, repository):
    folder, repository = Path(folder).resolve(), Path(repository).resolve()
    if folder == repository or repository in folder.parents:
        raise ValueError('review output must be outside code repository')
    return folder

def audit(folder, adapter_dir, seconds=1.0):
    import sys, h5py, robosuite
    sys.path.insert(0, str(adapter_dir))
    from smoke_rigid_kitchen import RigidPilotCounterToSink
    from robocasa.utils.env_utils import create_env
    result=json.loads((folder/'result.json').read_text())
    original_make=robosuite.make
    def no_render(**kw):
        kw.update(use_camera_obs=False,has_offscreen_renderer=False,has_renderer=False)
        return original_make(**kw)
    robosuite.make=no_render
    try:
        kw=dict(pilot_xml=result['xml']) if result.get('xml') else {}
        env=create_env('RigidPilotCounterToSink' if kw else 'PickPlaceCounterToSink',
                       seed=result['seed'],split='pretrain',**kw)
    finally:
        robosuite.make=original_make
    try:
        env.reset()
        env.reset_from_xml_string((folder/'model.xml').read_text())
        with h5py.File(folder/'trajectory.hdf5') as h:
            terminal=h['terminal_sim_state'][:]
        env.sim.set_state_from_flattened(terminal); env.sim.forward()
        first=snapshot(env)
        m,d=env.sim.model,env.sim.data
        b=env.obj_body_id['obj']; j=int(m.body_jntadr[b])
        qa=int(m.jnt_qposadr[j]);va=int(m.jnt_dofadr[j])
        q=d.qpos.copy();qm=np.ones(m.nq,bool);qm[qa:qa+7]=False
        vm=np.ones(m.nv,bool);vm[va:va+6]=False
        gate=StableSuccessGate(); gate.update(env)
        samples=[];verified=False
        every=max(1,round(.05/m.opt.timestep))
        for step in range(round(seconds/m.opt.timestep)):
            d.qpos[qm]=q[qm];d.qvel[vm]=0
            env.sim.forward();env.sim.step()
            if (step+1)%every==0:
                verified,r=gate.update(env);samples.append(r)
        final=snapshot(env)
        return dict(schema_version=2, scope='offline target dynamics; other joints held fixed',
                    model_sha256=hashlib.sha256((folder/'model.xml').read_bytes()).hexdigest(),
                    terminal_sha256=hashlib.sha256(terminal.tobytes()).hexdigest(),
                    thresholds=dict(speed_m_s=.01,angular_rad_s=.1,penetration_m=.001,
                                    displacement_m=.005,stable_seconds=.5),
                    initial=first,final=final,elapsed_sim_seconds=seconds,
                    displacement_m=float(np.linalg.norm(np.array(final['position_m'])-first['position_m'])),
                    offline_stability_verified=verified,
                    status=('stable_but_bbox_region_mismatch' if verified and not final['strict_bbox_th0']
                            else 'stable_physical_candidate' if verified else 'needs_review'),
                    samples=samples)
    finally:
        env.close()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path)
    p.add_argument('--adapter-dir',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--output-name',default='physical_review_v2.json')
    args=p.parse_args()
    if Path(args.output_name).name != args.output_name: raise ValueError('basename required')
    args.folder=validate_output_folder(args.folder, Path(__file__).resolve().parents[1])
    dest=args.folder/args.output_name
    if dest.exists(): raise FileExistsError(dest)
    report=audit(args.folder,args.adapter_dir)
    with dest.open('x') as f: json.dump(report,f,indent=2)
    print(json.dumps(dict(path=str(dest),status=report['status'],initial=report['initial'],
                         final=report['final'],displacement_m=report['displacement_m'])))
