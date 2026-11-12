"""Freeze matched-budget human and generated replay schedules; no final-test input."""
import argparse,json
from pathlib import Path
import h5py
import numpy as np
from prepare_xr1_mixed_pilot import digest

def schedules(human,geometry,steps,seed):
    if steps<=0 or steps%2:raise ValueError('Positive even step budget required')
    if len(human)<steps//2 or len(geometry)<steps//2:raise ValueError('Insufficient unique anchors')
    rng=np.random.default_rng(seed)
    order=rng.permutation(len(human))
    baseline=[human[int(order[i%len(order)])].copy() for i in range(steps)]
    gs=rng.permutation(len(geometry))[:steps//2]
    mixed=[r.copy() for r in baseline]
    for i,index in enumerate(gs):mixed[2*i+1]=geometry[int(index)].copy()
    return baseline,mixed

def main():
    p=argparse.ArgumentParser()
    for name in ('review-manifest','fixture-summary','human-manifest','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--steps',type=int,default=256);p.add_argument('--seed',type=int,default=20261112)
    a=p.parse_args();out=Path(a.output).resolve();root=Path(__file__).resolve().parents[1]
    if out==root or root in out.parents:raise ValueError('External outputs required')
    out.mkdir(parents=True,exist_ok=False)
    review=json.loads(Path(a.review_manifest).read_text())
    fixtures={r['run']:r for r in json.loads(Path(a.fixture_summary).read_text())}
    rows=review['accepted']
    for r in rows:
        assert r['split'] in ('engineering_pilot','train_candidate')
        assert r['physical_status'] in ('stable','stable_but_bbox_region_mismatch')
        assert r['source']=='policy_rollout_not_expert'
        assert fixtures[r['run']]['fixture_consistency']['passed']
        assert fixtures[r['run']]['sustained_dual_finger_lift']
        assert digest(r['trajectory_path'])==r['trajectory_sha256']
        assert digest(r['physical_review_path'])==r['physical_review_sha256']
    geometry=[]
    for ri,r in enumerate(rows):
        with h5py.File(r['trajectory_path'],'r') as f:
            n=len(f['actions']);assert n==r['steps'] and n>=30
            for j in range(8):
                t=6+int((n-22)*(j+.5)/8)
                ids=np.array([t-6,t-4,t-2,t])
                data={'state_history_raw14':f['state14'][ids],'actions_raw12':f['actions'][t:t+16]}
                for camera in ('robot0_agentview_left','robot0_agentview_right','robot0_eye_in_hand'):
                    data['images_'+camera]=f['observations/video.'+camera][ids]
                assert data['state_history_raw14'].shape==(4,14) and data['actions_raw12'].shape==(16,12)
                assert all(np.isfinite(v).all() for v in data.values())
                for k,v in data.items():
                    if k.startswith('images_'):assert v.shape==(4,256,256,3) and v.dtype==np.uint8
                dst=out/('geometry_%02d_%d.npz'%(ri,j));np.savez(dst,**data)
                dst.with_suffix('.json').write_text(json.dumps({**r,'task':str(f.attrs['instruction']),
                    'history_indices':ids.tolist(),'action_indices':[t,t+16],'sample_sha256':digest(dst)},indent=2))
                geometry.append(dict(path=str(dst),split='train',episode_index='geometry:'+r['run'],
                    sample_step=t,source='policy_rollout_not_expert'))
    hp=Path(a.human_manifest);samples=json.loads(hp.read_text())['samples']
    for r in samples:
        path=Path(r['path']);r['path']=str(path if path.is_absolute() else hp.parent/path);r['source']='human_dataset'
    train=[r for r in samples if r['split']=='train'];dev=[r for r in samples if r['split']=='dev']
    assert len(dev)==8
    assert {r['episode_index'] for r in train}.isdisjoint({r['episode_index'] for r in dev})
    baseline,mixed=schedules(train,geometry,a.steps,a.seed)
    protocol=dict(steps=a.steps,seed=a.seed,geometry_candidates=len(geometry),
        human_unique_anchors=len(train),human_episodes=len({r['episode_index'] for r in train}),
        dev_episodes=len(dev),source_manifest_sha256=digest(a.review_manifest),
        fixture_summary_sha256=digest(a.fixture_summary),human_manifest_sha256=digest(hp),
        hyperparameters='original checkpoint, expert only, batch1, lr3e-6, repeat1, no async prefix',
        evaluation='final step only; fixed 6 cross development initial states; elbow test untouched')
    for name,schedule in [('human',baseline),('mixed',mixed)]:
        (out/(name+'_manifest.json')).write_text(json.dumps({**protocol,'arm':name,'samples':schedule+dev},indent=2))
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    print(json.dumps(protocol),flush=True)
if __name__=='__main__':main()
