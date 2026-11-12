"""Isolated real-data XR1 short SFT pilot. Not an official benchmark recipe."""
import argparse, json, time, gc, hashlib, os
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoProcessor, AutoModel, Qwen3VLConfig
from opentau.policies.xr1.configuration_xr1 import XR1Config
from opentau.policies.xr1.modeling_xr1 import XR1Policy
from opentau.policies.xr1.state_dict_remap import remap_reference_state_dict, assert_full_coverage
import importlib.util

p=argparse.ArgumentParser()
p.add_argument('--checkpoint',required=True);p.add_argument('--batch',required=True)
p.add_argument('--xr1-repo',required=True);p.add_argument('--output',required=True)
p.add_argument('--manifest',required=True)
p.add_argument('--steps',type=int,default=32)
p.add_argument('--lr',type=float,default=3e-6)
p.add_argument('--monitor-every',type=int,default=0)
a=p.parse_args()
out=Path(a.output).resolve()
code_root=Path(__file__).resolve().parents[1]
if out==code_root or code_root in out.parents: raise ValueError('External outputs required')
if not os.environ.get('CUDA_VISIBLE_DEVICES'): raise ValueError('Select an authorized GPU explicitly')
out.mkdir(parents=True,exist_ok=False)
report={'scope':'real human or reviewed-policy mixed short SFT pilot; no improvement claim'}
def record(**kw):
    report.update(kw);(out/'diagnostic.json').write_text(json.dumps(report,indent=2));print(json.dumps(kw),flush=True)
torch.set_num_threads(8);torch.manual_seed(20261112)
ckpt=Path(a.checkpoint);raw=np.load(a.batch)
instruction=str(raw['prompt'].item()) if 'prompt' in raw else json.loads(Path(a.batch).with_suffix('.json').read_text())['task']
record(batch_shapes={k:list(raw[k].shape) for k in raw.files},batch_sha256=hashlib.sha256(Path(a.batch).read_bytes()).hexdigest())
names=['robot0_agentview_left','robot0_agentview_right','robot0_eye_in_hand']
batch={'state':torch.from_numpy(raw['state_history_raw16']).unsqueeze(0),
       'actions':torch.from_numpy(raw['actions_raw12']).unsqueeze(0),
       'real_action_dim':torch.tensor([12]),'prompt':[instruction]}
for i,n in enumerate(names):batch['camera'+str(i)]=torch.from_numpy(raw['images_'+n]).permute(0,3,1,2).float().unsqueeze(0)/255.
cfg=XR1Config(pretrained_backbone_repo_id=str(ckpt),load_pretrained_backbone=False,
    state_adapter='robocasa365_dataset',train_expert_only=True,enable_choice_heads=False,
    gradient_checkpointing=True,training_repeat=1,train_prefix_prob=0.,
    attention_implementation='eager')
qcfg=Qwen3VLConfig(**json.loads((ckpt/'config.json').read_text())['vlm_config'])
start=time.monotonic()
torch.set_default_dtype(torch.bfloat16)
from transformers.modeling_utils import no_init_weights
with no_init_weights(): policy=XR1Policy(cfg,qwen3vl_config=qcfg)
torch.set_default_dtype(torch.float32)
expected=set(policy.state_dict()); loaded=set();unexp=set()
index=json.loads((ckpt/'model.safetensors.index.json').read_text())
for shard in sorted(set(index['weight_map'].values())):
    d=remap_reference_state_dict(load_file(str(ckpt/shard)))
    r=policy.load_state_dict(d,strict=False,assign=True);loaded.update(d);unexp.update(r.unexpected_keys)
    del d
missing=sorted(expected-loaded)
assert_full_coverage(missing,sorted(unexp))
policy.model.vlm.tie_weights()
assert not any(x.is_meta for x in policy.parameters())
policy.tokenizer=AutoTokenizer.from_pretrained(ckpt,local_files_only=True)
policy=policy.to('cuda',dtype=torch.bfloat16)
record(load_seconds=time.monotonic()-start,loaded_keys=len(loaded),missing_keys=missing,
       unexpected_keys=sorted(unexp),trainable_parameters=sum(x.numel() for x in policy.parameters() if x.requires_grad))

# Use the official evaluator's PIL crop + HF processor, not OpenTau's approximate tensor resize.
spec=importlib.util.spec_from_file_location('ref_entry',Path(a.xr1_repo)/'eval_robocasa365/entry.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
proc=AutoProcessor.from_pretrained(ckpt,trust_remote_code=True,local_files_only=True,use_fast=False)
client=object.__new__(entry.EvalClient);client.crop_ratio=.95
norm=proc.action_config['robocasa365']
assert (norm['mean']==0).all() and (norm['std'][...,:12]==1).all() and (norm['std'][...,12:]==0).all()
manifest_path=Path(a.manifest)
manifest=json.loads(manifest_path.read_text())
prepared=[]
for item in manifest['samples']:
    path=Path(item['path'])
    if not path.is_absolute():path=manifest_path.parent/path
    z=np.load(path)
    meta=json.loads(path.with_suffix('.json').read_text())
    task=meta.get('task',meta.get('instruction'))
    assert task and np.isfinite(z['actions_raw12']).all()
    from opentau.policies.xr1.obs_adapter import adapt_state
    is14='state_history_raw14' in z
    rawstate=z['state_history_raw14' if is14 else 'state_history_raw16']
    assert rawstate.shape==(4,14 if is14 else 16) and np.isfinite(rawstate).all()
    assert z['actions_raw12'].shape==(16,12)
    b={'state':torch.from_numpy(rawstate).unsqueeze(0).cuda(),
       'actions':torch.from_numpy(z['actions_raw12']).unsqueeze(0).cuda(),
       'real_action_dim':torch.tensor([12],device='cuda')}
    if is14:
        assert item.get('source')=='policy_rollout_not_expert'
        independent=rawstate
        state=adapt_state(b['state'],state_adapter='identity',state_token_dim=60,quat_order='xyzw')
    else:
        independent=np.stack([np.concatenate([s[:3],entry.quat_xyzw_to_axis_angle(s[3:7]),
            s[14:16],s[7:10],entry.quat_xyzw_to_axis_angle(s[10:14])]) for s in rawstate])
        state=policy.prepare_state(b)
    assert np.max(np.abs(state[0,:,:14].cpu().numpy()-independent))<1e-5
    assert torch.count_nonzero(state[:,:,14:])==0
    hist={k:z['images_'+n] for k,n in zip(entry.CAMERA_KEYS,names)}
    inputs=proc.apply_chat_template(client._build_messages(hist,task),tokenize=True,return_dict=True,
        return_tensors='pt',do_resize=False,state=state.float().cpu().numpy(),robot_type='robocasa365')
    mm={k:inputs[k].cuda() for k in ('input_ids','attention_mask','pixel_values_videos','video_grid_thw')}
    prepared.append((item,b,mm,state))
train=[x for x in prepared if x[0]['split']=='train']
dev=[x for x in prepared if x[0]['split']=='dev']
assert len(train)>=8 and len(dev)>=4
assert {x[0]['episode_index'] for x in train}.isdisjoint({x[0]['episode_index'] for x in dev})
record(manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),train_samples=len(train),dev_samples=len(dev),steps_planned=a.steps,lr=a.lr,
       preprocessing='official evaluator PIL crop + HF AutoProcessor use_fast=False',
       training_repeat=1,train_prefix_prob=0,normalization='raw state dataset adapter; identity actions')
params=policy.get_optim_params()
for param in params:param.data=param.data.float()
optimizer=torch.optim.AdamW(params,lr=a.lr,betas=(.9,.95),weight_decay=.1,foreach=False)
def forward_one(sample, seed, training):
    _,b,mm,state=sample
    torch.manual_seed(seed)
    action=torch.nn.functional.pad(b['actions'],(0,48))
    mask=torch.zeros_like(action);mask[...,:12]=1
    kwargs=dict(**mm,state=state,actions=action,action_mask=mask,real_action_dim=b['real_action_dim'])
    if not training:
        kwargs['noise']=torch.randn_like(action)
        kwargs['time']=torch.tensor([.5],device='cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):
        losses=policy.model(**kwargs)
    return losses['MSE']+losses['CE']
def evaluate(samples=None):
    samples=dev if samples is None else samples
    policy.eval()
    with torch.no_grad():values=[float(forward_one(s,9000+i,False)) for i,s in enumerate(samples)]
    assert np.isfinite(values).all()
    return values
probe=train[::2][:8]  # Shared human slots in both matched-budget arms.
before=evaluate();record(dev_baseline_losses=before,dev_baseline_mean=float(np.mean(before)),
    train_probe_baseline=evaluate(probe),monitor_every=a.monitor_every)
monitor=[]
def checkpoint_metrics(step):
    row={'step':step,'train_probe':evaluate(probe),'dev':evaluate()}
    monitor.append(row)
    (out/'monitor.json').write_text(json.dumps(monitor,indent=2))
    print('MONITOR '+json.dumps(row),flush=True)
if a.monitor_every:checkpoint_metrics(0)
rows=[]
start=time.monotonic()
for step in range(a.steps):
    sample=train[step%len(train)]
    policy.train();optimizer.zero_grad(set_to_none=True)
    loss=forward_one(sample,10000+step,True)
    assert torch.isfinite(loss)
    loss.backward();grad=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
    optimizer.step();torch.cuda.synchronize()
    row={'step':step+1,'episode':sample[0]['episode_index'],'sample_step':sample[0]['sample_step'],
         'source':sample[0].get('source','human_dataset'),'loss':float(loss.detach()),'grad_norm':float(grad),'elapsed_s':time.monotonic()-start,
         'gpu_allocated_gib':torch.cuda.memory_allocated()/1024**3}
    rows.append(row)
    with (out/'steps.jsonl').open('a') as f:f.write(json.dumps(row)+chr(10))
    print(json.dumps(row),flush=True)
after=evaluate()
record(real_optimizer_steps=a.steps,dev_after_losses=after,dev_after_mean=float(np.mean(after)),
    train_seconds=time.monotonic()-start,peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3,
    limitation='Held-out demonstration flow loss only; no task success improvement claim.')
torch.save({n:p.detach().cpu() for n,p in policy.named_parameters() if p.requires_grad},out/'expert_final.pt')
torch.save(optimizer.state_dict(),out/'optimizer_final.pt')
record(saved_checkpoint=True,optimizer_saved=True)

