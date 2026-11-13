"""Loopback-only official XR1 server with an optional validated expert checkpoint."""
import argparse, importlib.util
from pathlib import Path
import torch
p=argparse.ArgumentParser()
p.add_argument('--xr1-repo',type=Path,required=True)
p.add_argument('--model',required=True)
p.add_argument('--expert',type=Path)
p.add_argument('--port',type=int,required=True)
p.add_argument('--seed',type=int,required=True)
a=p.parse_args()
spec=importlib.util.spec_from_file_location('official_xr1_server',a.xr1_repo/'deploy/server.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
server=module.Server(a.model,'127.0.0.1',a.port)
if a.expert:
    delta=torch.load(a.expert,map_location='cpu',weights_only=True)
    permitted=('dit.','state_projector.','action_projector.','action_output_layer.','t_embedder.','t_projector.','sink.')
    stripped={}
    for key,val in delta.items():
        if not key.startswith('model.'):raise ValueError('unexpected expert prefix')
        target=key[len('model.'):]
        if not target.startswith(permitted):raise ValueError('non-expert weight')
        stripped[target]=val
    expected={n for n,_ in server.model.named_parameters() if n.startswith(permitted)}
    assert set(stripped)==expected
    current=server.model.state_dict()
    assert all(current[k].shape==v.shape and torch.isfinite(v).all() for k,v in stripped.items())
    result=server.model.load_state_dict(stripped,strict=False)
    assert not result.unexpected_keys
    print('EXPERT_LOADED',len(stripped),flush=True)
server.model.eval()
torch.manual_seed(a.seed)
print('SEEDED',a.seed,flush=True)
server.serve()

