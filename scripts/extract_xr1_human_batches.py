"""CPU-only extraction from locally verified fixed HF revision, no downloads.
Spec JSON: {"samples":[{"episode_index":12127,"sample_step":6,"split":"train"},...]}
Output is external runtime data, not a repository file.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd,av
p=argparse.ArgumentParser()
p.add_argument("--spec-json",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
p.add_argument("--source",type=Path,required=True)
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
if a.out.resolve()==root or root in a.out.resolve().parents:raise ValueError('External outputs required')
a.out.mkdir(parents=True,exist_ok=False)
spec=json.loads(a.spec_json.read_text());items=spec["samples"]
episodes={x["episode_index"]:x for x in json.loads((a.source/"smoke_episode_manifest.json").read_text())}
df=pd.read_parquet(a.source/"counter_to_sink_selected.parquet")
cams={"robot0_agentview_left":"left-251.mp4","robot0_agentview_right":"right-251.mp4","robot0_eye_in_hand":"wrist-154.mp4"}
samples=[];seen=set();splits={}
for item in items:
 ei=int(item["episode_index"]);t=int(item["sample_step"]);split=item["split"];key=(ei,t)
 assert key not in seen;seen.add(key)
 assert ei in episodes and (ei not in splits or splits[ei]==split),"episode split leakage"
 splits[ei]=split;e=episodes[ei]
 d=df[df.episode_index==ei].sort_values("frame_index");length=len(d)
 assert length==e["length"] and np.array_equal(d.frame_index.to_numpy(),np.arange(length))
 h=np.array([t-6,t-4,t-2,t]);future=np.arange(t,t+16)
 assert h[0]>=0 and future[-1]<length
 data={"state_history_raw16":np.stack(d.iloc[h]["observation.state"]).astype("float32"),"actions_raw12":np.stack(d.iloc[future]["action"]).astype("float32"),"history_frame_indices":h,"action_frame_indices":future}
 assert data["state_history_raw16"].shape==(4,16) and data["actions_raw12"].shape==(16,12)
 assert np.isfinite(data["state_history_raw16"]).all() and np.isfinite(data["actions_raw12"]).all()
 meta=dict(episode_index=ei,sample_step=t,split=split,task=e["tasks"][0],history_indices=h.tolist(),action_indices=future.tolist(),source_revision="3f72c9307b4a132dd55e1433e2dda0f54bb8c1d0",raw_only=True,video_timestamps={},video_actual_timestamps={},source_hashes={})
 samples.append((data,meta))
for cam,file in cams.items():
 targets={}
 for n,(data,meta) in enumerate(samples):
  e=episodes[meta["episode_index"]];start=e["videos/observation.images."+cam+"/from_timestamp"]
  ts=[start+int(i)/20 for i in data["history_frame_indices"]]
  meta["video_timestamps"][cam]=ts;meta["video_actual_timestamps"][cam]=[None]*4
  data["images_"+cam]=np.empty((4,256,256,3),dtype="uint8")
  for k,t in enumerate(ts):targets.setdefault(round(t*20),[]).append((n,k,t))
 digest=hashlib.sha256((a.source/file).read_bytes()).hexdigest()
 container=av.open(str(a.source/file));stream=container.streams.video[0]
 for frame in container.decode(stream):
  seconds=float(frame.pts*frame.time_base);index=round(seconds*20)
  if index not in targets:continue
  rgb=frame.to_ndarray(format="rgb24");assert rgb.shape==(256,256,3)
  for n,k,want in targets.pop(index):
   assert abs(seconds-want)<.001,(seconds,want)
   data,meta=samples[n];data["images_"+cam][k]=rgb
   meta["video_actual_timestamps"][cam][k]=seconds;meta["source_hashes"][file]=digest
  if not targets:break
 container.close();assert not targets,"missing target frames"
 print("decoded",cam,flush=True)
manifest={"samples":[],"source_revision":"3f72c9307b4a132dd55e1433e2dda0f54bb8c1d0","spec_sha256":hashlib.sha256(a.spec_json.read_bytes()).hexdigest(),"note":"raw16 dataset order, raw12 actions, RGB THWC. No normalization/padding/crop. Development is not final test."}
for data,meta in samples:
 stem=meta["split"]+"_ep"+str(meta["episode_index"])+"_t"+str(meta["sample_step"])
 np.savez_compressed(a.out/(stem+".npz"),**data);(a.out/(stem+".json")).write_text(json.dumps(meta,indent=2))
 manifest["samples"].append(dict(path=stem+".npz",split=meta["split"],episode_index=meta["episode_index"],sample_step=meta["sample_step"]))
(a.out/"manifest.json").write_text(json.dumps(manifest,indent=2));print("DONE",a.out/"manifest.json",len(samples),flush=True)
