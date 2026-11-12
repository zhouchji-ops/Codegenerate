import importlib.util
from pathlib import Path
import types,unittest
import numpy as np
from xml.etree.ElementTree import Element
p=Path(__file__).resolve().parents[1]/'scripts'/'review_rollout_history.py'
s=importlib.util.spec_from_file_location('history_audit',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class FixtureTests(unittest.TestCase):
 def env(self,shift=0):
  signs=np.array([[-1,-1,-1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])
  elem=Element('geom',name='reg',pos='0 0 0',size='1 1 1')
  sink=types.SimpleNamespace(name='sink',root_body='sinkroot',pos=np.zeros(3),_regions={'basin':{'elem':elem}},
       get_reset_region_names=lambda:['basin'],get_int_sites=lambda relative=False:{'basin':signs})
  model=types.SimpleNamespace(body_name2id=lambda n:0,geom_name2id=lambda n:0,geom_pos=np.zeros((1,3)),geom_size=np.ones((1,3)))
  data=types.SimpleNamespace(body_xpos=np.array([[shift,0,0]]),geom_xpos=np.array([[shift,0,0]]),geom_xmat=np.eye(3).reshape(1,9))
  return types.SimpleNamespace(sink=sink,sim=types.SimpleNamespace(model=model,data=data),layout_id=1,style_id=1)
 def test_equal(self):self.assertTrue(m.fixture_consistency(self.env())['passed'])
 def test_rounding(self):self.assertTrue(m.fixture_consistency(self.env(2e-6))['passed'])
 def test_offset_rejected(self):self.assertFalse(m.fixture_consistency(self.env(.1))['passed'])
 def test_local_size_rejected(self):
  e=self.env();e.sim.model.geom_size[0,0]=1.01
  self.assertFalse(m.fixture_consistency(e)['passed'])

from unittest.mock import patch
import sys
class PickupTests(unittest.TestCase):
 def contacts(self):
  return [dict(other='gripper_finger1',robot=True,active=True,normal_force_N=.1),
          dict(other='gripper_finger2',robot=True,active=True,normal_force_N=.1)]
 def test_dual_lift(self):self.assertTrue(m.pickup_candidate(self.contacts(),.03))
 def test_push(self):self.assertFalse(m.pickup_candidate(self.contacts(),.001))
 def test_single_finger(self):self.assertFalse(m.pickup_candidate(self.contacts()[:1],.03))
 def test_supported(self):
  self.assertFalse(m.pickup_candidate(self.contacts()+[dict(other='table',robot=False,active=True,normal_force_N=.1)],.03))
 def test_inactive(self):
  c=self.contacts();c[0]['active']=False;self.assertFalse(m.pickup_candidate(c,.03))
 def test_nan(self):self.assertFalse(m.pickup_candidate(self.contacts(),float('nan')))
 def test_temporal_then_release(self):
  release=types.SimpleNamespace(reset=lambda:None,update=lambda e:(False,dict(contacts=self.contacts())))
  fake=types.SimpleNamespace(StableSuccessGate=lambda:release)
  env=types.SimpleNamespace(sim=types.SimpleNamespace(data=types.SimpleNamespace(time=0.)))
  with patch.dict(sys.modules,{'review_rollout_physics':fake}),patch.object(m,'collision_bottom',return_value=1.) as bottom,patch.object(m,'fixture_consistency',return_value={'passed':True}):
   g=m.PickupStableSuccessGate();g.reset(env);bottom.return_value=1.03
   self.assertFalse(g.update(env)[0]);env.sim.data.time=.11
   self.assertFalse(g.update(env)[0]);self.assertTrue(g.pickup_seen)
   release.update=lambda e:(True,dict(contacts=[]));env.sim.data.time=.7
   self.assertTrue(g.update(env)[0]);g.reset(env);self.assertFalse(g.pickup_seen)
 def test_release_without_pickup(self):
  release=types.SimpleNamespace(reset=lambda:None,update=lambda e:(True,dict(contacts=[])))
  fake=types.SimpleNamespace(StableSuccessGate=lambda:release)
  env=types.SimpleNamespace(sim=types.SimpleNamespace(data=types.SimpleNamespace(time=0.)))
  with patch.dict(sys.modules,{'review_rollout_physics':fake}),patch.object(m,'collision_bottom',return_value=1.),patch.object(m,'fixture_consistency',return_value={'passed':True}):
   g=m.PickupStableSuccessGate();g.reset(env);self.assertFalse(g.update(env)[0])


class CollisionTypeTests(unittest.TestCase):
 def test_nonbox_rejected(self):
  model=types.SimpleNamespace(ngeom=1,geom_bodyid=np.array([1]),geom_contype=np.array([1]),geom_type=np.array([7]))
  env=types.SimpleNamespace(obj_body_id={'obj':1},sim=types.SimpleNamespace(model=model,data=types.SimpleNamespace()))
  with self.assertRaises(ValueError):m.collision_bottom(env)
 def test_empty_collision_rejected(self):
  model=types.SimpleNamespace(ngeom=0,geom_bodyid=np.array([]),geom_contype=np.array([]),geom_type=np.array([]))
  env=types.SimpleNamespace(obj_body_id={'obj':1},sim=types.SimpleNamespace(model=model,data=types.SimpleNamespace()))
  with self.assertRaises(ValueError):m.collision_bottom(env)

if __name__=='__main__':unittest.main()
