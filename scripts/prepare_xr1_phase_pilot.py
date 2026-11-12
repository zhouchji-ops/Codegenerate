"""Matched new-scene uniform-vs-contact/lift replay; no development examples."""
import argparse,copy,hashlib,json
from pathlib import Path
import h5py
import numpy as np
from prepare_xr1_mixed_pilot import digest

def anchors(length,contact,lift,count,phase,approach_count=None):
    hi=length-16
    if not (6<=hi and 0<=contact<=lift<length and count>=2):
        raise ValueError('Invalid trajectory phase boundaries')
    def quantiles(lo,up,n):
        lo=int(np.clip(lo,6,hi));up=int(np.clip(up,lo,hi))
        return [lo+int((up-lo)*(i+.5)/n) for i in range(n)]
    if not phase:return quantiles(6,hi,count)
    n=(count+1)//2 if approach_count is None else approach_count
    if not 0<n<count:raise ValueError('Each phase must receive anchors')
    return quantiles(6,contact,n)+quantiles(max(6,contact-8),min(hi,lift+16),count-n)

def main():
    p=argparse.ArgumentParser()
    for key in ('accepted-manifest','quarter-manifest','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();root=Path(__file__).resolve().parents[1]
    if out==root or root in out.parents:raise ValueError('External outputs required')
    reviewed=json.loads(a.accepted_manifest.read_text());rows=reviewed['accepted']
    if not 2<=len(rows)<=32:raise ValueError('Require 2..32 reviewed training scenes')
    for r in rows:
        assert r['split']=='train_candidate' and r['source']=='policy_rollout_not_expert'
        for key in ('trajectory','physical_review','history_review'):
            assert digest(r[key+'_path'])==r[key+'_sha256']
        assert 0<=r['first_target_contact_step']<=r['grasp_phase_start_step']<=r['first_sustained_lift_step']<r['steps']
        assert r['physical_status'] in ('stable','stable_but_bbox_region_mismatch')
    base=json.loads(a.quarter_manifest.read_text())
    train=[r for r in base['samples'] if r['split']=='train']
    dev=[r for r in base['samples'] if r['split']=='dev']
    assert len(train)==256 and len({r['episode_index'] for r in dev})==8
    assert sum(r['source']=='policy_rollout_not_expert' for r in train)==64
    out.mkdir(parents=True,exist_ok=False)
    report={'accepted_manifest_sha256':digest(a.accepted_manifest),
            'quarter_manifest_sha256':digest(a.quarter_manifest),'steps':256,'human_slots':192,
            'generated_slots':64,'scene_count':len(rows),'budget':'identical original checkpoint/lr/noise/final-step rule',
            'phase_rule':'32 approach anchors frame6 to metadata grasp-phase start; 32 contact/lift anchors grasp-start-8 to sustained-lift+16; complete future16; explicit fallback labels retained'}
    for arm in ('uniform','phase'):
        schedule=copy.deepcopy(train);slot=0;selection=[];odd_seen=0
        folder=out/arm;folder.mkdir()
        for i,r in enumerate(rows):
            count=64//len(rows)+(i<64%len(rows))
            n_approach=count//2
            if count%2:
                n_approach+=int(odd_seen%2==0);odd_seen+=1
            chosen=anchors(r['steps'],r['grasp_phase_start_step'],r['first_sustained_lift_step'],count,arm=='phase',n_approach)
            with h5py.File(r['trajectory_path'],'r') as f:
                assert len(f['actions'])==r['steps']
                for j,t in enumerate(chosen):
                    ids=np.array([t-6,t-4,t-2,t])
                    data={'state_history_raw14':f['state14'][ids],'actions_raw12':f['actions'][t:t+16]}
                    for camera in ('robot0_agentview_left','robot0_agentview_right','robot0_eye_in_hand'):
                        data['images_'+camera]=f['observations/video.'+camera][ids]
                    assert data['state_history_raw14'].shape==(4,14) and data['actions_raw12'].shape==(16,12)
                    assert all(np.isfinite(v).all() for v in data.values())
                    for k,v in data.items():
                        if k.startswith('images_'):assert v.shape==(4,256,256,3) and v.dtype==np.uint8
                    path=folder/('sample_%02d_%02d.npz'%(i,j));np.savez(path,**data)
                    path.with_suffix('.json').write_text(json.dumps({**r,'task':str(f.attrs['instruction']),
                        'sample_step':t,'history_indices':ids.tolist(),'action_indices':[t,t+16],
                        'sample_sha256':digest(path),'arm':arm},indent=2))
                    row=dict(path=str(path),split='train',episode_index='new_geometry:'+r['run'],
                        sample_step=t,source='policy_rollout_not_expert')
                    schedule[4*slot+3]=row;selection.append(row);slot+=1
        assert slot==64
        assert all(schedule[i]==train[i] for i in range(256) if i%4!=3)
        assert {r['episode_index'] for r in schedule}.isdisjoint({r['episode_index'] for r in dev})
        (out/(arm+'_manifest.json')).write_text(json.dumps({**report,'arm':arm,'samples':schedule+dev},indent=2))
    (out/'protocol.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
if __name__=='__main__':main()
