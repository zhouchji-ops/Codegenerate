"""Independent fixture cache and pickup-history audit; does not modify v2."""
import argparse
import json
from pathlib import Path
import numpy as np

def fixture_consistency(env, tolerance_m=5e-5):
    m,d=env.sim.model,env.sim.data
    records=[]
    for name,fixture in [(env.sink.name, env.sink)]:
        try:
            b=m.body_name2id(fixture.root_body)
            pos_error=float(np.max(np.abs(np.asarray(fixture.pos)-d.body_xpos[b])))
            # All interior region geoms are axis-aligned in the fixture's model.
            for region in fixture.get_reset_region_names():
                cached=fixture._regions.get(region)
                if cached is None:continue
                g=m.geom_name2id(cached['elem'].get('name'))
                cp=np.fromstring(cached['elem'].get('pos'),sep=' ')
                cs=np.fromstring(cached['elem'].get('size'),sep=' ')
                region_error=max(float(np.max(np.abs(cp-m.geom_pos[g]))),
                                 float(np.max(np.abs(cs-m.geom_size[g]))))
                # Cached world region corners compared to compiled geom corners.
                points=np.asarray(fixture.get_int_sites(relative=False)[region])
                signs=np.array([[-1,-1,-1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])
                actual=d.geom_xpos[g]+(signs*m.geom_size[g])@d.geom_xmat[g].reshape(3,3).T
                world_error=float(np.max(np.abs(points-actual)))
                records.append(dict(fixture=name,region=region,position_error_m=pos_error,
                                    region_error_m=region_error,world_corner_error_m=world_error,
                                    passed=max(pos_error,region_error,world_error)<=tolerance_m))
        except (ValueError,KeyError,AttributeError) as error:
            records.append(dict(fixture=name,passed=False,error=str(error)))
    return dict(scope='task sink interior cache only', tolerance_m=tolerance_m,layout_id=int(env.layout_id),style_id=int(env.style_id),
                passed=bool(records) and all(r['passed'] for r in records),records=records)

def inspect_history(folder, adapter_dir):
    import sys,h5py,robosuite,mujoco
    sys.path.insert(0,str(adapter_dir))
    from smoke_rigid_kitchen import RigidPilotCounterToSink
    from robocasa.utils.env_utils import create_env
    from review_rollout_physics import snapshot
    r=json.loads((folder/'result.json').read_text())
    make=robosuite.make
    def nr(**kw):
        kw.update(use_camera_obs=False,has_offscreen_renderer=False,has_renderer=False)
        return make(**kw)
    robosuite.make=nr
    try:
        kw=dict(pilot_xml=r['xml']) if r.get('xml') else {}
        env=create_env('RigidPilotCounterToSink' if kw else 'PickPlaceCounterToSink',
                       seed=r['seed'],split='pretrain',**kw)
    finally:robosuite.make=make
    try:
        env.reset();env.reset_from_xml_string((folder/'model.xml').read_text())
        with h5py.File(folder/'trajectory.hdf5') as h:
            states=h['sim_states'][:];terminal=h['terminal_sim_state'][:]
        env.sim.set_state_from_flattened(states[0]);env.sim.forward()
        consistency=fixture_consistency(env)
        m,d=env.sim.model,env.sim.data;b=env.obj_body_id['obj']
        def bottom():
            return collision_bottom(env)
        initial_bottom=bottom()
        rows=[]
        for step,state in enumerate(list(states)+[terminal]):
            env.sim.set_state_from_flattened(state);env.sim.forward()
            contacts=[]
            for cid,c in enumerate(d.contact):
                if b not in (m.geom_bodyid[c.geom1],m.geom_bodyid[c.geom2]):continue
                other=c.geom2 if m.geom_bodyid[c.geom1]==b else c.geom1
                name=m.geom_id2name(other) or ''
                force=np.zeros(6)
                if c.efc_address>=0:mujoco.mj_contactForce(getattr(m,'_model',m),getattr(d,'_data',d),cid,force)
                if c.efc_address>=0 and force[0]>1e-8:
                    contacts.append(dict(name=name,normal_force_N=float(force[0])))
            names=[c['name'] for c in contacts]
            both=any('finger1' in n for n in names) and any('finger2' in n for n in names)
            nonrobot=any(not n.startswith(('robot','gripper')) for n in names)
            lift=bottom()-initial_bottom
            rows.append(dict(step=step,time=float(d.time),bottom_lift_m=lift,
                             both_fingers=both,nonrobot_support=nonrobot,contacts=contacts,
                             pickup_candidate=bool(both and not nonrobot and lift>.02)))
        runs=[];start=None
        for i,row in enumerate(rows+[dict(pickup_candidate=False)]):
            if row['pickup_candidate'] and start is None:start=i
            if not row['pickup_candidate'] and start is not None:
                end=i-1;runs.append(dict(start_step=start,end_step=end,
                                        duration_s=rows[end]['time']-rows[start]['time']))
                start=None
        return dict(schema_version=1,scope='restored pre-action states; not counterfactual execution',
                    fixture_consistency=consistency,initial_bottom_m=initial_bottom,
                    max_bottom_lift_m=max(x['bottom_lift_m'] for x in rows),
                    sustained_dual_finger_lift=any(x['duration_s']>=.1 for x in runs),
                    pickup_runs=runs,terminal=snapshot(env),history=rows,
                    caveat='Contact forces recomputed by mj_forward without saved controls; spatial two-finger contact and lift corroborate, not force-closure proof. Primitive-box collision assets only.')
    finally:env.close()


def collision_bottom(env):
    """Current generated candidates contain only rigid box collision geoms."""
    m,d=env.sim.model,env.sim.data
    b=env.obj_body_id['obj']
    geoms=[i for i in range(m.ngeom) if m.geom_bodyid[i]==b and m.geom_contype[i]]
    if not geoms or any(int(m.geom_type[g]) != 6 for g in geoms):
        raise ValueError('pickup gate supports box compound collision assets only')
    signs=np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])
    return min(float((d.geom_xpos[g]+(signs*m.geom_size[g])@d.geom_xmat[g].reshape(3,3).T)[:,2].min()) for g in geoms)

def pickup_candidate(contacts, lift_m):
    active=[c for c in contacts if c.get('active',False) and c.get('normal_force_N',0)>1e-8]
    names=[c['other'] for c in active]
    return bool(np.isfinite(lift_m) and lift_m>.02
                and any('finger1' in n for n in names)
                and any('finger2' in n for n in names)
                and not any(not c['robot'] for c in active))

class PickupStableSuccessGate:
    """v3 auxiliary live gate. reset(env) after reset, update each control step.
    This requires a real grasp/lift history plus v2 terminal stability.
    Never changes the benchmark's official metric.
    """
    def __init__(self):
        from review_rollout_physics import StableSuccessGate
        self.release_gate=StableSuccessGate()
        self.initial_bottom=None
        self.lift_start=None
        self.last_time=None
        self.pickup_seen=False
        self.cache_ok=False
    def reset(self,env):
        self.initial_bottom=collision_bottom(env)
        self.cache_ok=fixture_consistency(env)['passed']
        self.lift_start=None;self.last_time=None;self.pickup_seen=False
        self.release_gate.reset()
    def update(self,env):
        if self.initial_bottom is None:raise RuntimeError('reset(env) required')
        t=float(env.sim.data.time)
        if self.last_time is not None and t<=self.last_time:
            raise RuntimeError('time reset or repeated step: call reset(env)')
        self.last_time=t
        release,r=self.release_gate.update(env)
        lift=collision_bottom(env)-self.initial_bottom
        if pickup_candidate(r['contacts'],lift):
            if self.lift_start is None:self.lift_start=t
            if t-self.lift_start>=.1:self.pickup_seen=True
        else:self.lift_start=None
        r.update(pickup_seen=self.pickup_seen,bottom_lift_m=lift,fixture_cache_passed=self.cache_ok)
        return bool(self.cache_ok and self.pickup_seen and release),r

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--adapter-dir',type=Path,default=Path(__file__).resolve().parent)
    args=p.parse_args()
    from review_rollout_physics import validate_output_folder
    validate_output_folder(args.output.parent,Path(__file__).resolve().parents[1])
    if args.output.exists():raise FileExistsError(args.output)
    result=inspect_history(args.folder,args.adapter_dir)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='history'}))
