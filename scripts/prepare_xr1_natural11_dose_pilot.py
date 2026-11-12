"""Frozen natural-training pilot: 11-group full-dose refresh: replace all64 generated slots, preserve192human."""
import argparse, copy, hashlib, json
from pathlib import Path
import h5py
import numpy as np
from prepare_xr1_phase_pilot import anchors, validate_quarter_slots, digest

ALLOWED = {f"train_t_grip_{i:02d}_seed{6001+i}" for i in (0,1,3,4,5)} | {f"train_bridge_{i:02d}_seed{6007+i}" for i in range(6)}
def require(ok, message):
    if not ok: raise ValueError(message)
def read(path):
    return json.loads(Path(path).read_text())
def checked(path, expected):
    require(digest(path)==expected, "Hash mismatch: "+str(path))
def validate_groups(rows):
    require(len(rows)==11 and {r['scene_group'] for r in rows}==ALLOWED, 'Exact eleven frozen natural train groups required')
    require(len({r['run'] for r in rows})==11, 'Duplicate run')
    for r in rows:
        require(r['split']=='train' and r['source']=='policy_rollout_not_expert', 'Natural schema/source mismatch')
        require(r['run']=='original_'+r['scene_group'], 'Run/group mismatch')
def review_row(r):
    for key in ('trajectory','physical_review','history_review','initial_safety'):
        checked(r[key+'_path'],r[key+'_sha256'])
    root=Path(r['snapshot_path'])
    checked(root/'snapshot_metadata.json',r['snapshot_metadata_sha256'])
    checked(root/'episode_metadata.json',r['episode_metadata_sha256'])
    m=read(root/'snapshot_metadata.json')
    require(m['role']=='train' and m['seed']==int(r['scene_group'].split('seed')[-1]),'Snapshot role/seed mismatch')
    require(m['episode_metadata_sha256']==r['episode_metadata_sha256'] and not m['initial_success'],'Snapshot task mismatch')
    checked(m['asset_xml'],m['asset_sha256']); checked(root/'model.xml',m['model_sha256'])
    state=np.load(root/'initial_sim_state.npy',allow_pickle=False)
    require(np.isfinite(state).all() and hashlib.sha256(state.tobytes()).hexdigest()==m['sim_state_sha256'],'State mismatch')
    with np.load(root/'initial_observation.npz',allow_pickle=False) as obs:
        for k,v in m['observation_hashes'].items():
            require(hashlib.sha256(obs[k].tobytes()).hexdigest()==v,'Observation mismatch')
    p=read(r['physical_review_path']); h=read(r['history_review_path']); s=read(r['initial_safety_path'])
    require(r['official_success'] and r['integrity']['integrity_pass'],'Failed rollout')
    require(p['physics']['offline_stability_verified'] and p['physics']['status'] in ('stable','stable_physical_candidate','stable_but_bbox_region_mismatch'),'Physical gate failed')
    require(h['history']['sustained_dual_finger_lift'],'Pickup gate failed')
    require(s['supplemental_static_pass'] and s['finite'] and s['counter_support'] and not s['positive_target_robot_contact'],'Initial safety gate failed')
    checked(Path(r['trajectory_path']).parent/'model.xml',r['integrity']['model_xml_sha256'])
    for report in (p,h):
        require(report['fixture_consistency']['passed'] and report['pair_validation']['initial_state_exact'],'Fixture/pair mismatch')
        require(report['pair_validation']['snapshot_xml_sha256']==m['model_sha256'] and report['pair_validation']['rollout_xml_sha256']==r['integrity']['model_xml_sha256'],'Review/model mismatch')
    require(h['labels']['label_alignment']=='pre-action state indices; not post-action labels','Label alignment mismatch')

def replace_slots(train, new):
    validate_quarter_slots(train)
    require(len(new)==64,'Exactly 64 natural samples required')
    result=copy.deepcopy(train)
    for j,row in enumerate(new): result[4*j+3]=row
    require(all(result[i]==train[i] for i in range(256) if i%4!=3),'Preserved 192 human slots changed')
    validate_quarter_slots(result)
    return result

def uniform_anchors(length, count):
    require(isinstance(count, int) and count >= 1 and length >= 22, 'Invalid uniform anchor count/length')
    return [6 + int((length - 22) * (j + .5) / count) for j in range(count)]

def family_quota(rows, row):
    family = [x for x in rows if ('train_bridge_' in x['scene_group']) == ('train_bridge_' in row['scene_group'])]
    rank = next(i for i,x in enumerate(family) if x['scene_group'] == row['scene_group'])
    return 32 // len(family) + int(rank < 32 % len(family))

def main():
    ap=argparse.ArgumentParser()
    for name in ('accepted-manifest','quarter-manifest','output'): ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--accepted-sha256',required=True)
    a=ap.parse_args(); out=a.output.resolve(); repo=Path(__file__).resolve().parents[1]
    require(out!=repo and repo not in out.parents,'External output required')
    require(not out.exists(),'Refuse overwrite')
    checked(a.accepted_manifest,a.accepted_sha256)
    doc=read(a.accepted_manifest); rows=sorted(doc['accepted'],key=lambda r:r['scene_group'])
    require(doc['schema_version']==1,'Unknown natural schema')
    validate_groups(rows)
    require({r['scene_group'] for r in rows}.isdisjoint({r['scene_group'] for r in doc['excluded']}),'Accepted/excluded overlap')
    for row in rows: review_row(row)
    base=read(a.quarter_manifest); train=[r for r in base['samples'] if r['split']=='train']; dev=[r for r in base['samples'] if r['split']=='dev']
    validate_quarter_slots(train)
    require(len({r['episode_index'] for r in dev})==8,'Expected frozen 8 human dev episodes')
    require({r['episode_index'] for r in train}.isdisjoint({r['episode_index'] for r in dev}),'Base episode leakage')
    out.mkdir(parents=True,exist_ok=False); samples=[]; counts={}
    for i,r in enumerate(rows):
        count=family_quota(rows, r); counts[r['scene_group']]=count
        chosen=uniform_anchors(r['steps'], count)
        with h5py.File(r['trajectory_path'],'r') as f:
            require(f['actions'].shape==(r['steps'],12) and f['state14'].shape==(r['steps'],14),'Trajectory shape mismatch')
            for j,t in enumerate(chosen):
                ids=np.array([t-6,t-4,t-2,t])
                data={'state_history_raw14':f['state14'][ids],'actions_raw12':f['actions'][t:t+16]}
                for camera in ('robot0_agentview_left','robot0_agentview_right','robot0_eye_in_hand'):
                    v=f['observations/video.'+camera][ids]
                    require(v.shape==(4,256,256,3) and v.dtype==np.uint8,'Image shape/dtype mismatch')
                    data['images_'+camera]=v
                require(all(np.isfinite(v).all() for v in data.values()),'Nonfinite sample')
                path=out/f'natural_{i:02d}_{j:02d}.npz'; np.savez(path,**data)
                path.with_suffix('.json').write_text(json.dumps({**r,'task':str(f.attrs['instruction']),'sample_step':t,'history_indices':ids.tolist(),'action_indices':[t,t+16],'sample_sha256':digest(path)},indent=2))
                samples.append(dict(path=str(path),split='train',episode_index='natural:'+r['scene_group'],sample_step=t,source='policy_rollout_not_expert'))
    schedule=replace_slots(train,samples)
    require({r['episode_index'] for r in schedule}.isdisjoint({r['episode_index'] for r in dev}),'Episode leakage')
    report=dict(schema_version=1,arm='natural11_refresh64',accepted_manifest_sha256=digest(a.accepted_manifest),quarter_manifest_sha256=digest(a.quarter_manifest),steps=256,human_slots=192,old_generated_slots=0,natural_generated_slots=64,group_counts=counts,sampling='full episode quantile anchors 6..N-16; complete future16; pre-action raw14',comparison='same original checkpoint, lr 3e-6, matched step budget; retain all 192 human slots exactly; policy pseudo-demonstrations not experts',samples=schedule+dev)
    (out/'natural11_64_manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='samples'}))
if __name__=='__main__': main()
