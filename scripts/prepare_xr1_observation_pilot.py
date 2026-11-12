"""Exploratory two-scene-group visible-start versus hidden-start replay preparation."""
import argparse,copy,json
from pathlib import Path
import h5py
import numpy as np
from prepare_xr1_mixed_pilot import digest
from prepare_xr1_phase_pilot import validate_quarter_slots,require_training_family

def history_indices(t):
    if t<0:raise ValueError('Negative anchor')
    return [max(0,t-6),max(0,t-4),max(0,t-2),t]

def pair_rows(rows,expected_groups):
    if len(expected_groups)!=2 or len(set(expected_groups))!=2 or len(rows)!=4:
        raise ValueError('Exactly two distinct groups and four trajectories required')
    result={g:{} for g in expected_groups}
    for r in rows:
        group=r['scene_group'];role=r['role']
        if group not in result or role not in ('visible_start','hidden_start'):
            raise ValueError('Unexpected group/role; held-out groups forbidden')
        if role in result[group]:raise ValueError('Duplicate role within a scene group')
        result[group][role]=r
    if any(set(v)!={'visible_start','hidden_start'} for v in result.values()):
        raise ValueError('Each group requires a matched pair')
    return result

def validate_visibility(evidence,role,first_contact):
    queries={int(q['step']):q for q in evidence['queries']}
    if len(queries)!=len(evidence['queries']) or 0 not in queries:raise ValueError('Invalid query timeline')
    max_pixels={}
    for t,q in queries.items():
        counts=np.asarray(q['counts_history_by_camera'])
        if counts.shape!=(4,3) or not np.isfinite(counts).all() or (counts<0).any():
            raise ValueError('Expected finite nonnegative 4x3 model-crop counts')
        if q['history_indices']!=history_indices(t):raise ValueError('History/query mismatch')
        max_pixels[t]=int(counts.max())
    first20=min((t for t,n in max_pixels.items() if n>=20),default=None)
    if role=='hidden_start' and max_pixels[0]!=0:raise ValueError('Hidden arm is not initially geometric-zero')
    if role=='visible_start' and max_pixels[0]<20:raise ValueError('Visible control lacks nontrivial initial target')
    if first20 is None or first20>=first_contact:raise ValueError('No nontrivial discovery before target contact')
    for t in range(0,128,16):
        if t not in queries:raise ValueError('Missing selected policy query')
    return queries,first20

def main():
    p=argparse.ArgumentParser()
    for k in ('paired-manifest','quarter-manifest','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--expected-groups',nargs=2,required=True)
    p.add_argument('--group-families',nargs=2,choices=('t_grip','bridge'),required=True)
    a=p.parse_args();out=a.output.resolve();root=Path(__file__).resolve().parents[1]
    if out==root or root in out.parents:raise ValueError('External outputs required')
    paired=json.loads(a.paired_manifest.read_text())
    groups=pair_rows(paired['accepted'],a.expected_groups)
    family_map=dict(zip(a.expected_groups,a.group_families))
    for group,roles in groups.items():
        for role,row in roles.items():
            require_training_family(family_map[group])
            if 'family' in row:assert row['family']==family_map[group]
            row['family']=family_map[group]
            assert row['split']=='observation_train'
            for key in ('trajectory','physical_review','history_review','query_evidence'):
                assert digest(row[key+'_path'])==row[key+'_sha256']
            phys=json.loads(Path(row['physical_review_path']).read_text())
            hist=json.loads(Path(row['history_review_path']).read_text())
            assert phys['physics']['offline_stability_verified'] and phys['fixture_consistency']['passed']
            assert hist['fixture_consistency']['passed'] and hist['history']['sustained_dual_finger_lift']
            assert phys['pair_validation']['initial_state_exact'] and hist['pair_validation']['initial_state_exact']
            evidence=json.loads(Path(row['query_evidence_path']).read_text())
            queries,first20=validate_visibility(evidence,role,hist['labels']['first_target_contact_step'])
            row['_query_source']=evidence['query_schedule_source']
            row['_first20']=first20
            assert row['steps']>=128
    base=json.loads(a.quarter_manifest.read_text())
    train=[r for r in base['samples'] if r['split']=='train'];dev=[r for r in base['samples'] if r['split']=='dev']
    validate_quarter_slots(train)
    assert len(dev)==8 and len({r['episode_index'] for r in dev})==8
    out.mkdir(parents=True,exist_ok=False)
    protocol={'scope':'exploratory TWO-group pilot; original three-group gate was not met',
        'groups':a.expected_groups,'independent_training_groups':2,'trajectories_per_arm':2,
        'steps':256,'human_slots':192,'old_generated_slots':48,'new_slots':16,
        'new_group_slots':{g:8 for g in a.expected_groups},
        'query_anchors':list(range(0,128,16)),'history':'first-frame clamp, four frames at interval2',
        'paired_manifest_sha256':digest(a.paired_manifest),'base_manifest_sha256':digest(a.quarter_manifest),
        'limits':['pose and visibility jointly differ; not pure visibility causal effect',
                  'two groups cannot establish broad generalization','six cross cases are reused development, not final test']}
    for role in ('visible_start','hidden_start'):
        folder=out/role;folder.mkdir();schedule=copy.deepcopy(train)
        for qi,t in enumerate(range(0,128,16)):
            for gi,group in enumerate(a.expected_groups):
                row=groups[group][role];ids=history_indices(t)
                with h5py.File(row['trajectory_path'],'r') as f:
                    assert len(f['actions'])==row['steps'] and t+16<=row['steps']
                    data={'state_history_raw14':np.stack([f['state14'][i] for i in ids]),
                          'actions_raw12':f['actions'][t:t+16]}
                    for camera in ('robot0_agentview_left','robot0_agentview_right','robot0_eye_in_hand'):
                        data['images_'+camera]=np.stack([f['observations/video.'+camera][i] for i in ids])
                    task=str(f.attrs['instruction'])
                assert data['state_history_raw14'].shape==(4,14) and data['actions_raw12'].shape==(16,12)
                assert all(np.isfinite(v).all() for v in data.values())
                for k,v in data.items():
                    if k.startswith('images_'):assert v.shape==(4,256,256,3) and v.dtype==np.uint8
                path=folder/('group%d_query%03d.npz'%(gi,t));np.savez(path,**data)
                path.with_suffix('.json').write_text(json.dumps({**row,'task':task,'history_indices':ids,
                    'sample_step':t,'sample_sha256':digest(path)},indent=2))
                slot=2*qi+gi;index=16*slot+15
                schedule[index]=dict(path=str(path),split='train',episode_index='observation:'+group+':'+role,
                    scene_group=group,observation_role=role,sample_step=t,source='policy_rollout_not_expert')
        validate_quarter_slots(schedule)
        assert all(schedule[i]==train[i] for i in range(256) if i%16!=15)
        assert {r['episode_index'] for r in schedule}.isdisjoint({r['episode_index'] for r in dev})
        (out/(role+'_manifest.json')).write_text(json.dumps({**protocol,'arm':role,'samples':schedule+dev},indent=2))
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    print(json.dumps(protocol),flush=True)
if __name__=='__main__':main()
