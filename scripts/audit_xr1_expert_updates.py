"""Read-only expert-update norm/cosine diagnostics; not evidence of causal forgetting."""
import argparse,json,math
from pathlib import Path
import torch
from safetensors import safe_open

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--expert',action='append',required=True,help='name=external expert checkpoint')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();root=Path(__file__).resolve().parents[1]
    if out==root or root in out.parents:raise ValueError('External output required')
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(8)
    pairs_input=[s.split('=',1) for s in a.expert]
    if len({name for name,path in pairs_input})!=len(pairs_input):raise ValueError('Expert names must be unique')
    experts={name:torch.load(path,map_location='cpu',weights_only=True) for name,path in pairs_input}
    keys=set(next(iter(experts.values())))
    assert all(set(d)==keys for d in experts.values())
    index=json.loads((a.checkpoint/'model.safetensors.index.json').read_text())['weight_map']
    names=list(experts);pairs=[(x,y) for i,x in enumerate(names) for y in names[i+1:]]
    stats={name:{'delta_sq':0.,'base_sq':0.,'max_abs_delta':0.,'changed_elements':0} for name in names}
    dots={x+'|'+y:0. for x,y in pairs};groups={}
    handles={}
    try:
        for shard in sorted({index[k.removeprefix('model.')] for k in keys}):
            handles[shard]=safe_open(str(a.checkpoint/shard),framework='pt',device='cpu')
        for key in sorted(keys):
            refkey=key.removeprefix('model.')
            base=handles[index[refkey]].get_tensor(refkey).float()
            assert torch.isfinite(base).all()
            ds={}
            for name,weights in experts.items():
                v=weights[key].float();assert v.shape==base.shape and torch.isfinite(v).all()
                delta=v-base;ds[name]=delta
                sq=float(torch.sum(delta.double().square()))
                bq=float(torch.sum(base.double().square()))
                stats[name]['delta_sq']+=sq;stats[name]['base_sq']+=bq
                stats[name]['max_abs_delta']=max(stats[name]['max_abs_delta'],float(delta.abs().max()))
                stats[name]['changed_elements']+=int(torch.count_nonzero(delta))
                group=refkey.split('.')[0]
                record=groups.setdefault(group,{}).setdefault(name,{'delta_sq':0.,'base_sq':0.})
                record['delta_sq']+=sq;record['base_sq']+=bq
            for x,y in pairs:dots[x+'|'+y]+=float(torch.sum(ds[x].double()*ds[y].double()))
        for value in stats.values():
            value['delta_l2']=math.sqrt(value['delta_sq'])
            value['relative_delta_l2']=math.sqrt(value['delta_sq']/value['base_sq']) if value['base_sq'] else None
        cosine={}
        for k,v in dots.items():
            denom=math.sqrt(stats[k.split('|')[0]]['delta_sq']*stats[k.split('|')[1]]['delta_sq'])
            cosine[k]=v/denom if denom else None
        result={'keys':len(keys),'updates':stats,'update_cosines':cosine,'groups':groups,
                'limitation':'Parameter direction/magnitude only; not proof of skill conflict or catastrophic forgetting.'}
        out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2))
        print(json.dumps({'keys':len(keys),'updates':stats,'cosines':cosine}),flush=True)
    finally:
        handles.clear()
if __name__=='__main__':main()
