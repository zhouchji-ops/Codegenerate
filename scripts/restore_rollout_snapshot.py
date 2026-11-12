"""Restore an external rollout's initial scene/state before policy evaluation."""
import argparse, importlib.util, json, sys
from pathlib import Path
import h5py,numpy as np
p=argparse.ArgumentParser(add_help=False)
p.add_argument('--snapshot',type=Path,required=True)
p.add_argument('--collector',type=Path,required=True)
p.add_argument('--probe-only',action='store_true')
p.add_argument('--canonical-reference',type=Path)
a,remaining=p.parse_known_args()
code_root=Path(__file__).resolve().parents[1]
for artifact in (a.snapshot,a.canonical_reference):
    if artifact is not None and (artifact.resolve()==code_root or code_root in artifact.resolve().parents):
        raise ValueError('Snapshot and canonical artifacts must be outside the code repository')
spec=importlib.util.spec_from_file_location('pilot_collector',a.collector)
sys.path.insert(0,str(a.collector.parent))
collector=importlib.util.module_from_spec(spec);spec.loader.exec_module(collector)
expected=json.loads((a.snapshot/'result.json').read_text())
class SnapshotEnv(collector.RoboCasaGymEnv):
    def reset(self,seed=None,options=None):
        super().reset(seed=seed,options=options)
        raw=self.env
        raw.reset_from_xml_string((a.snapshot/'model.xml').read_text())
        with h5py.File(a.snapshot/'trajectory.hdf5','r') as h:
            initial=h['sim_states'][0]
            raw.sim.set_state_from_flattened(initial)
            raw.sim.forward()
            for robot in raw.robots:robot.composite_controller.reset()
            obs=self.get_observation(raw._get_observations(force_update=True))
            check={'sim_state_max_abs':float(np.max(np.abs(raw.sim.get_state().flatten()-initial))),
                'image_equal':{key:bool(np.array_equal(obs[key],h['observations'][key][0])) for key in h['observations']},
                'image_max_abs':{key:int(np.abs(obs[key].astype(np.int16)-h['observations'][key][0].astype(np.int16)).max()) for key in h['observations']},
                'image_mean_abs':{key:float(np.abs(obs[key].astype(np.float32)-h['observations'][key][0]).mean()) for key in h['observations']},
                'instruction_equal':obs['annotation.human.task_description']==expected['instruction']}
            (a.snapshot.parent/'restored_probe_model.xml').write_text(raw.sim.model.get_xml())
            print('RESTORE_DIAGNOSTIC '+json.dumps(check),flush=True)
            assert check['sim_state_max_abs']==0,check
            assert check['instruction_equal'],check
            if a.canonical_reference:
                if not a.canonical_reference.exists():
                    with a.canonical_reference.open('xb') as f:
                        np.savez_compressed(f,**{k:obs[k] for k in h['observations']})
                    print('CANONICAL_INITIAL_CREATED',flush=True)
                canonical=np.load(a.canonical_reference)
                matches={k:bool(np.array_equal(obs[k],canonical[k])) for k in h['observations']}
                assert all(matches.values()),matches
                print('CANONICAL_INITIAL_VERIFIED '+json.dumps(matches),flush=True)
            else:
                assert all(check['image_equal'].values()),check
        print('RESTORE_VERIFIED '+json.dumps(check),flush=True)
        return obs,{'success':bool(raw._check_success())}
collector.RoboCasaGymEnv=SnapshotEnv
if a.probe_only:
    env=SnapshotEnv(env_name='PickPlaceCounterToSink',split='pretrain',seed=expected['seed'])
    try:
        obs,_=env.reset(seed=expected['seed'])
        diffs=[]; replay_states=[]
        with h5py.File(a.snapshot/'trajectory.hdf5','r') as h:
            for i in range(5):
                env.step(collector.convert_action(h['actions'][i]))
                replay_states.append(env.env.sim.get_state().flatten().copy())
                diffs.append(float(np.max(np.abs(replay_states[-1]-h['sim_states'][i+1]))))
        print('REPLAY_FIRST5 '+json.dumps(diffs),flush=True)
        if a.canonical_reference:
            rp=a.canonical_reference.with_suffix('.replay.npz')
            current=np.stack(replay_states)
            if not rp.exists():
                with rp.open('xb') as f:np.savez_compressed(f,states=current)
            old=np.load(rp)['states']
            delta=float(np.max(np.abs(current-old)))
            print('CANONICAL_REPLAY_MAX_ABS '+str(delta),flush=True)
            assert np.array_equal(current,old),'Canonical dynamics are not reproducible'
    finally:env.close()
else:
    sys.argv=[str(a.collector)]+remaining
    collector.main()

