"""RGB-only bounded observation prefix, not a semantic target detector."""
import numpy as np
CAMERAS=('video.robot0_agentview_left','video.robot0_agentview_right','video.robot0_eye_in_hand')
def blue_counts(obs):
 counts=[]
 for key in CAMERAS:
  im=np.asarray(obs[key]);assert im.shape==(256,256,3) and im.dtype==np.uint8
  v=im[6:249,6:249].astype(np.float32);r,g,b=v[...,0],v[...,1],v[...,2]
  counts.append(int(((b>80)&(b>1.3*g)&(b>1.5*r)).sum()))
 return counts
class ObservationPrefix:
 def __init__(self,mode):
  assert mode in ('none','fixed20','rgb80');self.mode=mode;self.used=0;self.done=mode=='none';self.decisions=[]
 def action(self,obs):
  if self.done:return None
  counts=blue_counts(obs)
  if self.mode=='rgb80' and max(counts)>=64:
   self.done=True;self.decisions.append(dict(prefix_step=self.used,counts=counts,decision='stop_rgb_threshold'));return None
  limit=20 if self.mode=='fixed20' else 80
  if self.used>=limit:
   self.done=True;self.decisions.append(dict(prefix_step=self.used,counts=counts,decision='budget_exhausted'));return None
  yaw=.25 if self.mode=='fixed20' or self.used<20 or self.used>=60 else -.25
  a=np.zeros(12,dtype=np.float32);a[9]=yaw;a[11]=1.
  self.decisions.append(dict(prefix_step=self.used,counts=counts,decision='yaw_prefix',yaw=yaw));self.used+=1
  return a
