"""Prepare frozen 50% policy-rollout / human replay pilot; outputs stay external."""
import argparse, hashlib, json
from pathlib import Path
import h5py
import numpy as np

def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--review-manifest',required=True)
    p.add_argument('--human-manifest',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    out=Path(a.output).resolve()
    root=Path(__file__).resolve().parents[1]
    if out==root or root in out.parents: raise ValueError('External outputs required')
    out.mkdir(parents=True,exist_ok=False)
    review=json.loads(Path(a.review_manifest).read_text())
    candidates=review['accepted']
    assert len(candidates)>=16
    for row in candidates:
        assert row['split'] in ('engineering_pilot','train_candidate')
        assert row['source']=='policy_rollout_not_expert'
        assert row['physical_status'] in ('stable_but_bbox_region_mismatch','stable')
        assert digest(row['trajectory_path'])==row['trajectory_sha256']
        assert digest(row['physical_review_path'])==row['physical_review_sha256']
    selected=[candidates[i] for i in np.linspace(0,len(candidates)-1,16,dtype=int)]
    humanpath=Path(a.human_manifest)
    human=json.loads(humanpath.read_text())['samples']
    for row in human:
        q=Path(row['path'])
        row['path']=str(q if q.is_absolute() else humanpath.parent/q)
        row['source']='human_dataset'
    train=[r for r in human if r['split']=='train']
    dev=[r for r in human if r['split']=='dev']
    assert len(train)==32 and len(dev)==8
    mixed=[]
    for i,row in enumerate(selected):
        path=Path(row['trajectory_path'])
        with h5py.File(path,'r') as f:
            n=len(f['actions'])
            assert n==row['steps'] and n>=22
            t=6+int((n-16-6)*((i+.5)/16))
            ids=np.maximum(0,np.array([t-6,t-4,t-2,t]))
            data={'state_history_raw14':f['state14'][ids],
                  'actions_raw12':f['actions'][t:t+16]}
            for camera in ('robot0_agentview_left','robot0_agentview_right','robot0_eye_in_hand'):
                data['images_'+camera]=f['observations/video.'+camera][ids]
            task=str(f.attrs['instruction'])
        assert data['state_history_raw14'].shape==(4,14)
        assert data['actions_raw12'].shape==(16,12)
        assert all(np.isfinite(x).all() for x in data.values())
        for k,v in data.items():
            if k.startswith('images_'): assert v.shape==(4,256,256,3) and v.dtype==np.uint8
        dst=out/('geometry_%02d.npz'%i)
        np.savez(dst,**data)
        meta={**row,'task':task,'history_indices':ids.tolist(),'action_indices':[t,t+16],
              'state_layout':'already adapted EE-first14; identity + zero-pad60',
              'sample_sha256':digest(dst)}
        dst.with_suffix('.json').write_text(json.dumps(meta,indent=2))
        # Keep original human samples at even optimizer steps; substitute odd slots only.
        mixed.append(train[2*i])
        mixed.append({'path':str(dst),'split':'train','episode_index':'geometry:'+row['run'],
                      'sample_step':t,'source':'policy_rollout_not_expert'})
    result={'protocol':'32 steps; even slots unchanged human; odd slots reviewed generated policy rollouts',
            'review_manifest_sha256':digest(a.review_manifest),'human_manifest_sha256':digest(a.human_manifest),
            'samples':mixed+dev,'limitations':review['limitations']}
    (out/'manifest.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'train':32,'human':16,'generated':16,'dev':8,'manifest':str(out/'manifest.json')}))
if __name__=='__main__': main()
