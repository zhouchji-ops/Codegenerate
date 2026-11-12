import torch,numpy as np
from torch import nn
class VisualActor(nn.Module):
 def __init__(self,feature_dim=2560):
  super().__init__();self.net=nn.Sequential(nn.Linear(feature_dim+14,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,384))
  nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
 def sample(self,s):
  mean,ls=self.net(s).chunk(2,-1);ls=ls.clamp(-5,1)
  dist=torch.distributions.Normal(mean,ls.exp());z=dist.rsample();u=torch.tanh(z)
  lp=(dist.log_prob(z)-torch.log(1-u.square()+1e-6)-np.log(1.5)).sum(-1,keepdim=True)
  return 1.5*u,lp
