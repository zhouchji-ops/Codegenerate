# Copyright (C) 2026 Xiaomi Corporation
# SPDX-License-Identifier: Apache-2.0
# Modified: current-feature steering, local-only CLI and request checks.
"""WARNING: unsafe pickle protocol. Trusted same-user loopback clients only; never expose/tunnel.
Model trust_remote_code executes reviewed local checkpoint code. No authentication.
"""
from cli_common import parse_args, new_output, require_selected_gpu

def main():
    args = parse_args("server")
    require_selected_gpu()
    new_output(args.run_root)
    """Loopback-only current-VLM-feature latent actor server, single encoder pass."""
    import socket,struct,pickle,json,torch
    from transformers import AutoModel
    from visual_actor import VisualActor
    from initial_noise_adapter_visual import bind_instance
    torch.set_num_threads(8)
    model=AutoModel.from_pretrained(str(args.model),trust_remote_code=True,local_files_only=True,dtype=torch.bfloat16,attn_implementation="eager").cuda().eval()
    actor=VisualActor().eval();context={};native_rng=torch.Generator(device="cuda")
    # Native Gaussian stream parity: same seed, shape, device and bf16 across multiple chunks.
    torch.manual_seed(314159)
    mask=torch.ones((1,16,60),device="cuda",dtype=torch.bfloat16)
    reference=[torch.randn_like(mask) for _ in range(3)]
    g=torch.Generator(device="cuda").manual_seed(314159)
    candidate=[torch.randn(mask.shape,device=mask.device,dtype=mask.dtype,generator=g) for _ in range(3)]
    assert all(torch.equal(a,b) for a,b in zip(reference,candidate))
    from pathlib import Path
    (args.run_root/"native_rng_parity.json").write_text(json.dumps({"same_seed_3_chunks_bitwise_equal":True,"shape":[1,16,60],"dtype":"bf16","device":"cuda","purpose":"native persistent noise stream equivalence; full forward explicit-noise equivalence validated separately"}))

    def hook(module,args,out):
     mask=context["video_mask"]
     h=out.last_hidden_state.detach().float()
     pooled=((h*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True)).cpu()
     context["feature"]=torch.nn.functional.layer_norm(pooled,pooled.shape[-1:])
     context["hook_calls"]+=1
    handle=model.vlm.model.register_forward_hook(hook)
    def select_noise(mask,unused=None):
     assert context["hook_calls"]==1,"must use exactly one current VLM feature"
     feat=torch.cat([context["feature"],context["state14"]],-1)
     if context["sampling_mode"]=="native":
      noise=torch.randn(mask.shape,device=mask.device,dtype=mask.dtype,generator=native_rng)
      latent=noise[:,:,:12].float().cpu().reshape(1,192);lp=torch.full((1,1),float("nan"))
     else:
      torch.manual_seed(context["seed"])
      with torch.no_grad():latent,lp=actor.sample(feat)
      noise=torch.zeros_like(mask);noise[:,:,:12]=latent.reshape(1,16,12).to(mask.device,mask.dtype)
     context.update(actor_input=feat,latent=latent,logprob=lp)
     return noise
    bind_instance(model,noise_selector=select_noise)
    def recv(sock,n):
     chunks=[]
     while n:
      data=sock.recv(n)
      if not data:return None
      chunks.append(data);n-=len(data)
     return b"".join(chunks)
    with socket.socket() as server:
     server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);server.bind(("127.0.0.1",args.port));server.listen(1)
     print(f"VISUAL_NOISE_READY_{args.port}",flush=True)
     while True:
      conn,_=server.accept()
      with conn:
       while True:
        header=recv(conn,4)
        if header is None:break
        size=struct.unpack(">I",header)[0]
        if size>64*1024*1024:raise ValueError("oversize local request")
        data=pickle.loads(recv(conn,size))
        weights=data.pop("actor_weights");seed=int(data.pop("latent_seed"))
        mode=data.pop("sampling_mode","steering");assert mode in ("native","steering")
        reset_seed=data.pop("reset_noise_seed",None)
        if reset_seed is not None:native_rng.manual_seed(int(reset_seed))
        assert set(weights)==set(actor.state_dict()) and all(torch.isfinite(v).all() for v in weights.values())
        actor.load_state_dict(weights,strict=True);actor.eval()
        data.pop("task_id",None)
        context.clear();context.update(seed=seed,sampling_mode=mode,hook_calls=0,state14=data["state"][:,-1,:14].float().cpu())
        inp={k:v.to("cuda",dtype=torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point() else v.to("cuda") if torch.is_tensor(v) else v for k,v in data.items()}
        context["video_mask"]=inp["input_ids"]==model.config.vlm_config.video_token_id
        assert context["video_mask"].any()
        with torch.no_grad():actions=model(**inp,num_steps=5).actions.cpu()
        response={"actions":actions,"actor_input":context["actor_input"],"latent":context["latent"],"logprob":context["logprob"],"encoder_calls":context["hook_calls"]}
        payload=pickle.dumps(response);conn.sendall(struct.pack(">I",len(payload))+payload)

if __name__ == "__main__":
    main()
