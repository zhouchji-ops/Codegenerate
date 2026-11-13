"""Fixed-observation, same-RNG XR1 action diagnostic; NEVER updates model weights.

Requires reviewed local checkpoint/adapter code (trust_remote_code executes it).
Uses official processor, crop .95, state14 padded60, history t-6/-4/-2/t,
bf16 flash_attention_2 and strict complete expert coverage. This is not a
counterfactual rollout, success-rate estimate, or proof of catastrophic forgetting.
Original diagnostic ran 36 forwards; this portable copy only CPU/CLI checked.
"""
import argparse
import os
from pathlib import Path

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("xr1-repo","model","source","human-expert","uniform-expert","phase-expert","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--steps",type=int,nargs="+",default=[0,120,180])
    p.add_argument("--seeds",type=int,nargs="+",default=[2001,2002,2003])
    args=p.parse_args()
    if not os.environ.get("CUDA_VISIBLE_DEVICES","").strip():
        p.error("Set CUDA_VISIBLE_DEVICES to your explicitly authorized free GPU")
    for name in ("xr1_repo","model","source"):
        value=getattr(args,name).expanduser().resolve()
        if not value.is_dir():p.error(name+" must exist")
        setattr(args,name,value)
    for name in ("human_expert","uniform_expert","phase_expert"):
        value=getattr(args,name).expanduser().resolve()
        if not value.is_file():p.error(name+" must exist")
        setattr(args,name,value)
    args.output=args.output.expanduser().resolve()
    for root in (Path(__file__).resolve().parents[1],args.xr1_repo,args.model,args.source):
        if args.output==root or root in args.output.parents:
            p.error("Output must be outside repository, model, and source input directories")
    if args.output.exists():p.error("Output must be a new directory; overwrite forbidden")
    if len(set(args.steps))!=len(args.steps) or any(t<0 for t in args.steps):
        p.error("Steps must be unique nonnegative indices")
    if len(set(args.seeds))!=len(args.seeds) or any(s<0 for s in args.seeds):
        p.error("Seeds must be unique nonnegative integers")
    for file in (args.source/"trajectory.hdf5",args.source/"result.json",args.xr1_repo/"eval_robocasa365/entry.py"):
        if not file.is_file():p.error("Missing required input: "+str(file))
    return args

def main():
    args=parse_args()
    import os,time,json,hashlib,importlib.util,sys
    from pathlib import Path
    import numpy as np,torch,h5py
    from transformers import AutoModel,AutoProcessor
    out=args.output;out.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.set_num_threads(8)
    repo=args.xr1_repo;modelpath=str(args.model)
    source=args.source
    spec=importlib.util.spec_from_file_location('entry',repo/'eval_robocasa365/entry.py');ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
    processor=AutoProcessor.from_pretrained(modelpath,trust_remote_code=True,local_files_only=True,use_fast=False)
    builder=object.__new__(ev.EvalClient);builder.crop_ratio=.95
    instruction=json.loads((source/'result.json').read_text())['instruction'];inputs={};provenance=[]
    with h5py.File(source/'trajectory.hdf5') as h:
     for t in args.steps:
      assert 0 <= t < len(h['state14']), 'step outside trajectory'
      ids=[max(0,t-6),max(0,t-4),max(0,t-2),t]
      images={k:np.stack([h['observations/'+k][i] for i in ids]) for k in ev.CAMERA_KEYS}
      raw=np.stack([h['state14'][i] for i in ids]);assert raw.shape==(4,14) and np.isfinite(raw).all()
      state=np.zeros((1,4,60),np.float32);state[0,:,:14]=raw
      inp=processor.apply_chat_template(builder._build_messages(images,instruction),tokenize=True,return_dict=True,return_tensors='pt',do_resize=False,state=state,robot_type='robocasa365')
      inputs[t]=dict(inp)
      provenance.append({'step':t,'history_indices':ids,'state_sha256':hashlib.sha256(raw.tobytes()).hexdigest(),'camera_sha256':{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in images.items()},'actual_next16_gripper':h['actions'][t:t+16,6].tolist()})
    model=AutoModel.from_pretrained(modelpath,trust_remote_code=True,local_files_only=True,dtype=torch.bfloat16,attn_implementation='flash_attention_2').cuda().eval()
    prefix=('dit.','state_projector.','action_projector.','action_output_layer.','t_embedder.','t_projector.','sink.')
    expected={n for n,_ in model.named_parameters() if n.startswith(prefix)}
    arms={'original':None,'human256':args.human_expert,'new_uniform256':args.uniform_expert,'new_phase256':args.phase_expert}
    arrays={};rows=[];checkpoint_hash={}
    for arm,path in arms.items():
     if path:
      weights=torch.load(path,map_location='cpu',weights_only=True)
      assert all(k.startswith('model.') for k in weights)
      weights={k[6:]:v for k,v in weights.items()}
      assert set(weights)==expected and all(torch.isfinite(v).all() for v in weights.values())
      state=model.state_dict();assert all(state[k].shape==v.shape for k,v in weights.items())
      res=model.load_state_dict(weights,strict=False);assert not res.unexpected_keys
      checkpoint_hash[arm]=hashlib.sha256(path.read_bytes()).hexdigest()
      del weights,state
     for t,raw in inputs.items():
      data={k:v.to('cuda',dtype=torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point() else v.to('cuda') if torch.is_tensor(v) else v for k,v in raw.items()}
      data['task_id']='robocasa365'
      for seed in args.seeds:
       torch.manual_seed(seed);rng_before=torch.cuda.get_rng_state().clone()
       tick=time.monotonic()
       with torch.inference_mode():outputs=model(**data).actions.cpu()
       actions=processor.decode_action(outputs,robot_type='robocasa365')[0,:,:12].float().numpy()
       assert actions.shape==(16,12) and np.isfinite(actions).all()
       key=f'{arm}_t{t}_seed{seed}';arrays[key]=actions
       rows.append({'arm':arm,'step':t,'seed':seed,'rng_sha256':hashlib.sha256(rng_before.numpy().tobytes()).hexdigest(),'gripper':actions[:,6].tolist(),'mean_gripper':float(actions[:,6].mean()),'inference_s':time.monotonic()-tick})
     print('finished',arm,flush=True)
    for t in inputs:
     for seed in args.seeds:
      hashes={r['rng_sha256'] for r in rows if r['step']==t and r['seed']==seed}
      assert len(hashes)==1, 'initial CUDA RNG differs across experts'
    comparisons=[]
    for t in inputs:
     for seed in args.seeds:
      base=arrays[f'original_t{t}_seed{seed}']
      uniform=arrays[f'new_uniform256_t{t}_seed{seed}']
      for arm in arms:
       a=arrays[f'{arm}_t{t}_seed{seed}'];delta=a-base;du=a-uniform
       comparisons.append({'arm':arm,'step':t,'seed':seed,'max_abs_vs_original':float(abs(delta).max()),'mean_abs_vs_original':float(abs(delta).mean()),'gripper_max_abs_vs_original':float(abs(delta[:,6]).max()),'gripper_mean_delta_vs_uniform':float(du[:,6].mean()),'gripper_sign_disagreement_vs_uniform':int(((a[:,6]>0)!=(uniform[:,6]>0)).sum()),'first_action':a[0].tolist()})
    np.savez_compressed(out/'decoded_actions.npz',**arrays)
    report={'source':str(source),'observation_scope':'fixed uniform trajectory pre-action observations; diagnostic dev only, no training','instruction':instruction,'history':provenance,'checkpoint_sha256':checkpoint_hash,'attention':'flash_attention_2','dtype':'bf16','seeds':args.seeds,'rows':rows,'comparisons':comparisons,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'elapsed_s':time.monotonic()-start,'source_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'limit':'Same observations and reset RNG compare action tendencies, not counterfactual rollout success nor forgetting proof; t120/t180 may not coincide with original live chunk boundaries.'}
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:report[k] for k in ['peak_allocated_gib','elapsed_s']}),flush=True)

if __name__=="__main__":
    main()
