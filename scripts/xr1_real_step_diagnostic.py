"""Isolated real-data XR1 one-step diagnostic. Not a benchmark or training recipe."""
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
p.add_argument('--verify-expert',type=Path,help='Reload a saved expert checkpoint without training')
a=p.parse_args()
out=Path(a.output).resolve()
code_root=Path(__file__).resolve().parents[1]
if out==code_root or code_root in out.parents: raise ValueError('Outputs must be outside code repository')
if not os.environ.get('CUDA_VISIBLE_DEVICES'): raise ValueError('Explicitly select an authorized GPU via CUDA_VISIBLE_DEVICES')
out.mkdir(parents=True,exist_ok=False)
report={'scope':'one real human batch optimizer diagnostic; not benchmark or improved policy'}
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
batch={k:v.to('cuda') if torch.is_tensor(v) else v for k,v in batch.items()}
policy.eval()
with torch.no_grad():
    state=policy.prepare_state(batch)
    mm=policy.prepare_multimodal_inputs(batch)
# Independent raw-state semantics via official quaternion conversion.
spec=importlib.util.spec_from_file_location('ref_entry',Path(a.xr1_repo)/'eval_robocasa365/entry.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
independent=[]
for rawstate in raw['state_history_raw16']:
    independent.append(np.concatenate([rawstate[:3],entry.quat_xyzw_to_axis_angle(rawstate[3:7]),
        rawstate[14:16],rawstate[7:10],entry.quat_xyzw_to_axis_angle(rawstate[10:14])]))
state_delta=float(np.max(np.abs(np.stack(independent)-state[0,:,:14].cpu().numpy())))
assert state_delta<1e-5,'state layout/rotation mismatch'

record(independent_state_max_abs=state_delta)
if a.verify_expert:
    torch.manual_seed(1717)
    with torch.no_grad(): before_actions=policy.sample_actions(batch).float().cpu()
    expert_path=a.verify_expert
    saved=torch.load(expert_path,map_location='cpu',weights_only=True)
    expected_expert={n for n,p in policy.named_parameters() if p.requires_grad}
    assert set(saved)==expected_expert
    result=policy.load_state_dict(saved,strict=False,assign=True)
    assert not result.unexpected_keys
    assert set(result.missing_keys)==set(policy.state_dict())-set(saved)
    for n,p in policy.named_parameters():
        if n in saved: assert torch.equal(p.detach().cpu(),saved[n])
    policy=policy.to('cuda',dtype=torch.bfloat16).eval()
    torch.manual_seed(1717)
    with torch.no_grad(): after_actions=policy.sample_actions(batch).float().cpu()
    assert torch.isfinite(after_actions).all()
    record(reloaded_expert_keys=len(saved),exact_fp32_reload=True,
        independent_state_max_abs=state_delta,
        inference_after_bf16_reload_finite=True,
        inference_change_max_abs=float((after_actions-before_actions).abs().max()),
        inference_change_mean_abs=float((after_actions-before_actions).abs().mean()),
        note='Fresh process base+expert reload, then bf16 inference. Not a rollout or success improvement.')
    np.savez_compressed(out/'reload_actions.npz',before=before_actions.numpy(),after=after_actions.numpy())
    
    raise SystemExit(0)

# Build reference input through the actual official evaluator and processor.
spec=importlib.util.spec_from_file_location('ref_entry',Path(a.xr1_repo)/'eval_robocasa365/entry.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
proc=AutoProcessor.from_pretrained(ckpt,trust_remote_code=True,local_files_only=True)
client=object.__new__(entry.EvalClient);client.crop_ratio=.95
hist={k:raw['images_'+n] for k,n in zip(entry.CAMERA_KEYS,names)}
messages=client._build_messages(hist,instruction)
refinputs=proc.apply_chat_template(messages,tokenize=True,return_dict=True,return_tensors='pt',
    do_resize=False,state=state.float().cpu().numpy(),robot_type='robocasa365')
preproc={}
for k in mm:
    if k not in refinputs:continue
    x,y=mm[k].float().cpu(),refinputs[k].float().cpu()
    preproc[k]={'shape_equal':x.shape==y.shape,'max_abs':float((x-y).abs().max()) if x.shape==y.shape else None}
record(preprocessing_comparison=preproc)
assert preproc['input_ids']['max_abs']==0,'Prompt token mismatch'
assert preproc['video_grid_thw']['max_abs']==0,'Video grid mismatch'
refinputs={k:v.to('cuda') if torch.is_tensor(v) else v for k,v in refinputs.items()}
for k,v in refinputs.items():
    if torch.is_tensor(v) and v.is_floating_point():refinputs[k]=v.to(torch.bfloat16)
# Same seed, same bf16 noise. Real official forward is inference-only.
reference=AutoModel.from_pretrained(ckpt,trust_remote_code=True,local_files_only=True,dtype=torch.bfloat16,
    attn_implementation='eager').to('cuda').eval()
torch.manual_seed(1717)
with torch.no_grad():refactions=reference(**refinputs,num_steps=5).actions.float().cpu()
del reference;gc.collect();torch.cuda.empty_cache()
torch.manual_seed(1717)
with torch.no_grad():portactions=policy.sample_actions(batch).float().cpu()
# sample_actions returns unpadded actions; reference has padded 60.
refactions=refactions[...,:12];portactions=portactions[...,:12]
record(action_parity_max_abs=float((portactions-refactions).abs().max()),
       action_parity_mean_abs=float((portactions-refactions).abs().mean()),
       action_parity_exact=bool(torch.equal(portactions,refactions)),
       parity_note='Numerical comparison only; no rollout equivalence claim.')
np.savez_compressed(out/'fixed_input_actions.npz',reference=refactions.numpy(),opentau=portactions.numpy())
policy.train()
params=policy.get_optim_params()
# fp32 optimizer parameters avoid rounding away a 3e-5 update in bf16 weights.
for param in params:param.data=param.data.float()
optimizer=torch.optim.AdamW(params,lr=3e-5,betas=(.9,.95),weight_decay=.1,foreach=False)
probe=next((n,p) for n,p in policy.named_parameters() if p.requires_grad and 'action_output_layer' in n)
before=probe[1].detach().cpu().clone()
optimizer.zero_grad(set_to_none=True)
torch.manual_seed(2323);start=time.monotonic()
with torch.autocast('cuda',dtype=torch.bfloat16):losses=policy(batch)
loss=losses['MSE']+losses['CE']
assert torch.isfinite(loss),'nonfinite loss'
loss.backward()
norm=torch.nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=True)
optimizer.step();torch.cuda.synchronize()
delta=(probe[1].detach().cpu()-before).abs()
assert delta.max()>0,'no measured parameter update'
record(real_optimizer_steps=1,loss=float(loss.detach()),grad_norm=float(norm),step_seconds=time.monotonic()-start,
       probe_name=probe[0],probe_max_abs_update=float(delta.max()),probe_changed_elements=int((delta>0).sum()),
       peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3,
       limitation='One batch/one step; no measured task improvement. Data not independent expert certification.')
torch.save({n:p.detach().cpu() for n,p in policy.named_parameters() if p.requires_grad},out/'expert_step1.pt')
record(saved_expert_checkpoint=True)

