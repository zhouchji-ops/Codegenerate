"""Parameterized research prototype; algorithm unchanged, see README."""
from cli_common import parse_args, new_output

def main():
    args = parse_args("train")
    """One train-candidate scene, isolated visual latent-SAC. Not held-out evaluation."""
    import argparse,collections,copy,hashlib,importlib.util,json,random,sys,time
    from pathlib import Path
    import numpy as np,torch,h5py
    from torch import nn
    from visual_actor import VisualActor
    from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv
    from robocasa.utils.env_utils import convert_action
    sys.path.insert(0,str(args.adapter_dir.resolve()))
    from smoke_rigid_kitchen import RigidPilotCounterToSink
    from review_rollout_history import PickupStableSuccessGate,fixture_consistency
    torch.set_num_threads(4);torch.manual_seed(719);np.random.seed(719);random.seed(719)
    root=args.run_root;out=root/("episode_%02d_"%args.episode+args.asset);new_output(out)
    source=args.source_root/(args.asset+"_seed_0")
    actor=VisualActor().eval()
    if args.mode=="after":actor.load_state_dict(torch.load(root/"train/actor.pt",weights_only=True,map_location="cpu"))
    actor_before={k:v.clone() for k,v in actor.state_dict().items()}
    class Q(nn.Module):
     def __init__(self):
      super().__init__();self.net=nn.Sequential(nn.Linear(2766,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,1))
     def forward(self,s,a):return self.net(torch.cat([s,a],-1))
    q1=Q();q2=Q();t1=copy.deepcopy(q1);t2=copy.deepcopy(q2);q_before={k:v.clone() for k,v in q1.state_dict().items()}
    ao=torch.optim.Adam(actor.parameters(),lr=1e-4);qo=torch.optim.Adam(list(q1.parameters())+list(q2.parameters()),lr=3e-4)
    replay=[];updates=[];gamma=.99;alpha=.001;rng=np.random.default_rng(719)
    if (root/"latest.pt").exists():
     resume=torch.load(root/"latest.pt",weights_only=True,map_location="cpu")
     assert resume["episode"]==args.episode-1
     actor.load_state_dict(resume["actor"]);q1.load_state_dict(resume["q1"]);q2.load_state_dict(resume["q2"])
     t1.load_state_dict(resume["target1"]);t2.load_state_dict(resume["target2"])
     ao.load_state_dict(resume["actor_optimizer"]);qo.load_state_dict(resume["critic_optimizer"])
     torch.set_rng_state(resume["torch_rng"]);rng.bit_generator.state=json.loads(resume["numpy_rng_json"])
     saved=np.load(root/("episode_%02d_"%resume["episode"]+resume["asset"])/"replay.npz")
     assert len(saved["state"])==resume["replay_size"],"incomplete replay checkpoint"
     replay=[tuple(saved[k][i] for k in ["state","latent","reward","next_state","terminal","discount"]) for i in range(len(saved["state"]))]
    elif args.episode!=0:raise RuntimeError("missing previous episode checkpoint")
    actor_before={k:v.clone() for k,v in actor.state_dict().items()};q_before={k:v.clone() for k,v in q1.state_dict().items()}
    initial_replay_size=len(replay)
    def update():
     if args.mode!="train" or len(replay)<8:return
     ids=rng.choice(len(replay),8,replace=False)
     ss,aa,rr,ns,dd,gg=[torch.from_numpy(np.stack([replay[i][col] for i in ids]).astype("float32")) for col in range(6)]
     rr=rr[:,None];dd=dd[:,None];gg=gg[:,None]
     with torch.no_grad():
      na,lp=actor.sample(ns);target=rr+gg*(1-dd)*(torch.minimum(t1(ns,na),t2(ns,na))-alpha*lp)
     qloss=((q1(ss,aa)-target).square()+(q2(ss,aa)-target).square()).mean()
     qo.zero_grad();qloss.backward();qn=torch.nn.utils.clip_grad_norm_(list(q1.parameters())+list(q2.parameters()),10,error_if_nonfinite=True);qo.step()
     for q in (q1,q2):
      for par in q.parameters():par.requires_grad_(False)
     pa,lp=actor.sample(ss);mu,ls=actor.net(ss).chunk(2,-1);ls=ls.clamp(-5,1);kl=.5*(mu.square()+torch.exp(2*ls)-1-2*ls).mean();aloss=(alpha*lp-torch.minimum(q1(ss,pa),q2(ss,pa))).mean()+.01*kl
     ao.zero_grad();aloss.backward();an=torch.nn.utils.clip_grad_norm_(actor.parameters(),10,error_if_nonfinite=True);ao.step()
     for q in (q1,q2):
      for par in q.parameters():par.requires_grad_(True)
     with torch.no_grad():
      for q,t in ((q1,t1),(q2,t2)):
       for pp,tt in zip(q.parameters(),t.parameters()):tt.lerp_(pp,.005)
     assert torch.isfinite(qloss) and torch.isfinite(aloss)
     updates.append({"q_loss":float(qloss),"actor_loss":float(aloss),"actor_grad":float(an),"critic_grad":float(qn),"replay_size":len(replay),"pre_tanh_kl":float(kl)})
    spec=importlib.util.spec_from_file_location("entry",str(args.xr1_repo/"eval_robocasa365/entry.py"))
    ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
    model=str(args.model)
    client=ev.EvalClient(model,"127.0.0.1",args.port,"robocasa365",.95)
    asset=str(args.assets_root/args.asset/"model.xml")
    env=RoboCasaGymEnv(env_name="RigidPilotCounterToSink",pilot_xml=asset,split="pretrain",seed=0)
    with h5py.File(source/"trajectory.hdf5") as h:initial=h["sim_states"][0]
    xml=(source/"model.xml").read_text()
    def restore():
     env.reset(seed=0);raw=env.env;raw.reset_from_xml_string(xml);raw.sim.set_state_from_flattened(initial);raw.sim.forward()
     for robot in raw.robots:robot.composite_controller.reset()
     obs=env.get_observation(raw._get_observations(force_update=True))
     assert np.array_equal(raw.sim.get_state().flatten(),initial)
     assert fixture_consistency(raw)["passed"],"sink cache mismatch"
     return obs
    def potential():
     pos=env.env.sim.data.body_xpos[env.env.obj_body_id["obj"]]
     return -float(np.clip(np.linalg.norm(pos[:2]-np.asarray(env.env.sink.pos)[:2]),0,1))
    def request(hist,states,instruction,seed):
     state=np.zeros((1,4,60),dtype=np.float32);state[0,:,:14]=ev.sample_history(states,4,2)
     inp=client.processor.apply_chat_template(client._build_messages({k:ev.sample_history(v,4,2) for k,v in hist.items()},instruction),tokenize=True,return_dict=True,return_tensors="pt",do_resize=False,state=state,robot_type="robocasa365")
     client.client._send_with_length_prefix(dict(inp,actor_weights=actor.state_dict(),latent_seed=seed))
     response=client.client._recv_with_length_prefix();assert response["encoder_calls"]==1
     actions=client.processor.decode_action(response["actions"],robot_type="robocasa365")[0,:,:12].float().numpy()
     assert actions.shape==(16,12) and np.isfinite(actions).all()
     return actions,response["actor_input"][0].numpy(),response["latent"][0].numpy()
    try:
     obs=restore();canon_dir=root/"canonical"/args.asset;canon_dir.mkdir(parents=True,exist_ok=True);canonical=canon_dir/"canonical.npz"
     if not canonical.exists():np.savez_compressed(canonical,**{k:obs[k] for k in ev.CAMERA_KEYS},state=initial)
     saved=np.load(canonical);assert all(np.array_equal(saved[k],obs[k]) for k in ev.CAMERA_KEYS)
     with h5py.File(source/"trajectory.hdf5") as h:probe_actions=h["actions"][:5]
     probe=[]
     for action in probe_actions:
      env.step(convert_action(action));probe.append(env.env.sim.get_state().flatten().copy())
     probe=np.stack(probe);probe_path=canon_dir/"canonical_probe.npy"
     if not probe_path.exists():np.save(probe_path,probe)
     assert np.array_equal(probe,np.load(probe_path))
     obs=restore();assert all(np.array_equal(saved[k],obs[k]) for k in ev.CAMERA_KEYS)
     instruction=obs["annotation.human.task_description"]
     hist={k:collections.deque([obs[k]],maxlen=7) for k in ev.CAMERA_KEYS};states=collections.deque([ev.observation_to_state(obs)],maxlen=7)
     gate=PickupStableSuccessGate();gate.reset(env.env);steps=0;official_first=None;verified_first=None;pending=None
     logs=[];physics=[];simstates=[];executed=[];feature_log=[];noise_log=[];start=time.monotonic();infer_s=0;end_reason="budget_truncation"
     seedbase=args.noise
     for chunk in range(38):
      tick=time.monotonic();actions,feature,noise=request(hist,states,instruction,seedbase+chunk);infer_s+=time.monotonic()-tick
      if pending is not None:
       f,n,r,d,g=pending;replay.append((f,n,r,feature,d,g));update()
      feature_log.append(feature);noise_log.append(noise);total=0.;done=False
      for j,action in enumerate(actions[:600-steps]):
       old_phi=potential();simstates.append(env.env.sim.get_state().flatten().copy());executed.append(action.copy())
       obs,_,env_done,trunc,info=env.step(convert_action(action));steps+=1
       verified,diag=gate.update(env.env);diag.update(step=steps,verified_stable=verified);physics.append(diag)
       finite=bool(np.isfinite(env.env.sim.get_state().flatten()).all());fell=bool(diag["position_m"][2]<.05)
       done=bool(verified or not finite or fell or env_done)
       if info.get("success",False) and official_first is None:official_first=steps
       if verified and verified_first is None:verified_first=steps
       sparse=1. if verified else -1. if fell or not finite else 0.
       shaped=.1*(gamma*(0. if done else potential())-old_phi)
       total+=(gamma**j)*(sparse+shaped)
       for k in hist:hist[k].append(obs[k])
       states.append(ev.observation_to_state(obs))
       if done or trunc:
        end_reason="verified_success" if verified else "fell_below_5cm" if fell else "invalid" if not finite else "environment_terminal"
        break
      k=j+1;pending=(feature,noise,total,float(done),gamma**k)
      logs.append({"chunk":chunk,"sim_steps":steps,"executed_steps":k,"discounted_return":total,"verified":verified,"fell":fell,"done":done,"actor_update_count":len(updates)})
      if done:
       f,n,r,d,g=pending;replay.append((f,n,r,np.zeros_like(f),d,g));update();pending=None;break
      if steps>=600:break
     if pending is not None:
      tick=time.monotonic();_,nextfeature,_=request(hist,states,instruction,seedbase+1000);infer_s+=time.monotonic()-tick
      f,n,r,d,g=pending;replay.append((f,n,r,nextfeature,d,g));update()
     ad=max(float((v-actor_before[k]).abs().max()) for k,v in actor.state_dict().items())
     qd=max(float((v-q_before[k]).abs().max()) for k,v in q1.state_dict().items())
     if args.mode=="train" and len(replay)>=8:assert updates and ad>0 and qd>0
     else:assert not updates and ad==0
     (out/"model.xml").write_text(xml)
     with h5py.File(out/"trajectory.hdf5","w") as h:
      h.create_dataset("sim_states",data=np.stack(simstates));h.create_dataset("actions",data=np.stack(executed));h.create_dataset("terminal_sim_state",data=env.env.sim.get_state().flatten())
      h.create_dataset("actor_features",data=np.stack(feature_log));h.create_dataset("latents",data=np.stack(noise_log))
     np.savez_compressed(out/"replay.npz",**{name:np.stack([r[i] for r in replay]) for i,name in enumerate(["state","latent","reward","next_state","terminal","discount"])})
     report={"mode":args.mode,"gate_version":"v3","seed":0,"latent_seedbase":seedbase,"asset":args.asset,"asset_split":"train_candidate","success":official_first is not None,"official_first_step":official_first,"auxiliary_verified_first_step":verified_first,"steps":steps,"horizon":600,"end_reason":end_reason,"xml":asset,"instruction":instruction,"elapsed_s":time.monotonic()-start,"inference_s":infer_s,"updates":len(updates),"actor_parameter_max_update":ad,"critic_parameter_max_update":qd,"canonical_initial_state_images_equal":True,"canonical_probe5_dynamics_equal":True,"actor_input_dim":2574,"visual_tokens":384,"single_encoder_pass":True,"xr1_frozen":True,"reward":"v3 grasp/lift history then stable release +1 once; floor/invalid -1; potential beta=.1 gamma=.99; initial-steering pre-tanh KL coefficient .01","physics_module_sha256":hashlib.sha256((args.adapter_dir/"review_rollout_history.py").read_bytes()).hexdigest(),"scope":"fixed mixed training pool, no dev/final-test data","episode":args.episode,"initial_replay_size":initial_replay_size,"final_replay_size":len(replay),"concurrent_gpu_workload":True}
     for name,data in [("result",report),("chunks",logs),("physics_steps",physics),("updates",updates)]:(out/(name+".json")).write_text(json.dumps(data,indent=2))
     torch.save(actor.state_dict(),out/"actor.pt")
     checkpoint={"episode":args.episode,"asset":args.asset,"replay_size":len(replay),"actor":actor.state_dict(),"q1":q1.state_dict(),"q2":q2.state_dict(),"target1":t1.state_dict(),"target2":t2.state_dict(),"actor_optimizer":ao.state_dict(),"critic_optimizer":qo.state_dict(),"torch_rng":torch.get_rng_state(),"numpy_rng_json":json.dumps(rng.bit_generator.state)}
     torch.save(checkpoint,out/"checkpoint.pt");torch.save(checkpoint,root/"latest.tmp.pt");(root/"latest.tmp.pt").replace(root/"latest.pt")
     np.savez_compressed(root/"latest_replay.npz",**{name:np.stack([r[i] for r in replay]) for i,name in enumerate(["state","latent","reward","next_state","terminal","discount"])})
     print(json.dumps(report),flush=True)
    finally:env.close();client.close()

if __name__ == "__main__":
    main()
