"""Collect bounded XR1 policy rollouts, not privileged expert demonstrations.

Images/state are pre-action; simulator state and actions are saved per step.
Success must pass review before inclusion in supervised training.
"""
import argparse
import collections
import importlib.util
import json
import time
from pathlib import Path
import sys
import h5py
import imageio.v2 as imageio
import numpy as np
from smoke_rigid_kitchen import RigidPilotCounterToSink
from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv
from robocasa.utils.env_utils import convert_action


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--xr1-repo', type=Path, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--xml', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--port', type=int, default=10206)
    p.add_argument('--horizon', type=int, default=600)
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    out=a.output.resolve()
    if out==root or root in out.parents: raise ValueError('external output required')
    out.mkdir(parents=True, exist_ok=False)
    spec=importlib.util.spec_from_file_location('xr1_eval', a.xr1_repo/'eval_robocasa365/entry.py')
    ev=importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
    client=ev.EvalClient(a.model,'127.0.0.1',a.port,'robocasa365',.95)
    kwargs=dict(pilot_xml=str(a.xml.resolve())) if a.xml else {}
    env=RoboCasaGymEnv(env_name='RigidPilotCounterToSink' if a.xml else 'PickPlaceCounterToSink', split='pretrain',seed=a.seed,**kwargs)
    start=time.monotonic(); success=False; steps=0
    timings=dict(inference_s=0.0, record_s=0.0, video_s=0.0, simulation_s=0.0)
    try:
        obs,_=env.reset(seed=a.seed)
        instruction=obs['annotation.human.task_description']
        histories={key:collections.deque(maxlen=7) for key in ev.CAMERA_KEYS}
        states=collections.deque(maxlen=7); plan=collections.deque()
        (out/'model.xml').write_text(env.env.sim.model.get_xml())
        with h5py.File(out/'trajectory.hdf5','x') as h, imageio.get_writer(str(out/'rollout.mp4'),fps=20) as video:
            h.attrs['source']='XR1 policy rollout; success is not expert certification'
            h.attrs['instruction']=instruction
            h.attrs['seed']=a.seed
            h.attrs['horizon']=a.horizon
            h.attrs['checkpoint']=a.model
            h.attrs['observation_alignment']='pre_action'
            datasets={}
            def append(name,value):
                value=np.asarray(value)
                if name not in datasets:
                    datasets[name]=h.create_dataset(name,shape=(0,)+value.shape,maxshape=(None,)+value.shape,dtype=value.dtype,compression='gzip',compression_opts=1,chunks=(1,)+value.shape if value.ndim==3 else None)
                d=datasets[name]; d.resize(d.shape[0]+1,axis=0); d[-1]=value
            for step in range(a.horizon):
                for key in ev.CAMERA_KEYS: histories[key].append(obs[key])
                states.append(ev.observation_to_state(obs))
                if not plan:
                    tick=time.monotonic()
                    chunk=client.infer(ev.sample_history(states,4,2),{k:ev.sample_history(v,4,2) for k,v in histories.items()},instruction)
                    timings['inference_s']+=time.monotonic()-tick
                    plan.extend(chunk[:16])
                action=np.asarray(plan.popleft(),dtype=np.float32)
                if action.shape!=(12,) or not np.isfinite(action).all(): raise ValueError('invalid policy action')
                tick=time.monotonic()
                for key in ev.CAMERA_KEYS: append('observations/'+key,obs[key])
                append('state14',states[-1]); append('actions',action)
                append('sim_states',env.env.sim.get_state().flatten())
                timings['record_s']+=time.monotonic()-tick
                tick=time.monotonic()
                video.append_data(ev.make_video_frame(obs))
                timings['video_s']+=time.monotonic()-tick
                tick=time.monotonic()
                obs,_,done,truncated,info=env.step(convert_action(action))
                timings['simulation_s']+=time.monotonic()-tick
                steps=step+1; success=bool(info.get('success',False))
                append('success',np.array(success))
                if success or done or truncated: break
            h.attrs['success']=success
            h.create_dataset('terminal_sim_state',data=env.env.sim.get_state().flatten())
            video.append_data(ev.make_video_frame(obs))
        report=dict(source='policy_rollout_not_expert',success=success,steps=steps,seed=a.seed,
                    horizon=a.horizon,xml=str(a.xml) if a.xml else None,elapsed_s=time.monotonic()-start,
                    instruction=instruction,review_status='pending',timings=timings)
        (out/'result.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report),flush=True)
    finally:
        env.close(); client.close()

if __name__=='__main__': main()
