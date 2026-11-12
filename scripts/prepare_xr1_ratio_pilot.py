"""Freeze a quarter-generated replay ablation from existing matched schedules."""
import argparse,copy,hashlib,json
from pathlib import Path

def quarter_schedule(human,mixed):
    if len(human)!=len(mixed) or len(human)%4:
        raise ValueError('Schedules must have equal length divisible by four')
    if any(r.get('source')!='human_dataset' for r in human):
        raise ValueError('Control must contain human samples only')
    if human[::2]!=mixed[::2]:
        raise ValueError('Prior common human slots differ')
    if any(r.get('source')!='policy_rollout_not_expert' for r in mixed[1::2]):
        raise ValueError('Prior odd slots must be reviewed generated samples')
    result=copy.deepcopy(human)
    for i in range(3,len(result),4):result[i]=copy.deepcopy(mixed[i])
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--human-manifest',type=Path,required=True)
    p.add_argument('--mixed-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();root=Path(__file__).resolve().parents[1]
    if out==root or root in out.parents:raise ValueError('External output required')
    hm=json.loads(a.human_manifest.read_text());mm=json.loads(a.mixed_manifest.read_text())
    h=[r for r in hm['samples'] if r['split']=='train']
    m=[r for r in mm['samples'] if r['split']=='train']
    hd=[r for r in hm['samples'] if r['split']=='dev']
    md=[r for r in mm['samples'] if r['split']=='dev']
    assert hd==md and len({r['episode_index'] for r in hd})==8
    schedule=quarter_schedule(h,m)
    assert len(schedule)==256
    assert {r['episode_index'] for r in schedule}.isdisjoint({r['episode_index'] for r in hd})
    out.mkdir(parents=True,exist_ok=False)
    result={**hm,'arm':'quarter_generated','samples':schedule+hd,
        'recipe':'slots i%4==3 retained from frozen 50% mix; all others from human control',
        'training_budget':'256 steps, original checkpoint, identical optimizer/noise settings',
        'selection_status':'development follow-up after observing 50% mix, not confirmatory final test',
        'human_manifest_sha256':hashlib.sha256(a.human_manifest.read_bytes()).hexdigest(),
        'mixed_manifest_sha256':hashlib.sha256(a.mixed_manifest.read_bytes()).hexdigest()}
    (out/'manifest.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'human':192,'generated':64,'steps':256,'generated_episodes':len({r['episode_index'] for r in schedule if r['source']=='policy_rollout_not_expert'})}),flush=True)
if __name__=='__main__':main()
