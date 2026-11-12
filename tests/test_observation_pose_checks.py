import importlib.util,math,unittest
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'scripts'/'observation_pose_checks.py'
s=importlib.util.spec_from_file_location('pose_checks',p);c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
def contact(a='robot',b='cabinet',gap=0):
    return {'pair':[a,b],'distance_m':gap}
class PlanTests(unittest.TestCase):
 def test_valid(self):self.assertEqual(c.validate_pose_plan([-.2,.2],.05),([-.2,.2],.05))
 def test_limits(self):c.validate_pose_plan([-math.pi,math.pi],.5)
 def test_empty(self):
  with self.assertRaises(ValueError):c.validate_pose_plan([],0)
 def test_many(self):
  with self.assertRaises(ValueError):c.validate_pose_plan([i*.01 for i in range(33)],0)
 def test_duplicate(self):
  with self.assertRaises(ValueError):c.validate_pose_plan([0,-0.],0)
 def test_bad_yaw(self):
  for x in [float('nan'),float('inf'),math.pi+.01,True]:
   with self.assertRaises(ValueError):c.validate_pose_plan([x],0)
 def test_bad_retreat(self):
  for x in [-.01,.51,float('nan'),float('inf'),False]:
   with self.assertRaises(ValueError):c.validate_pose_plan([0],x)
class RetreatTests(unittest.TestCase):
 def test_positive_axis(self):self.assertEqual(c.retreat_delta([10,0,0],[2,0,0],.05),-.05)
 def test_negative_axis(self):self.assertEqual(c.retreat_delta([-10,0,0],[2,0,0],.05),.05)
 def test_opposite_target(self):self.assertEqual(c.retreat_delta([1,0,0],[-2,0,0],.05),.05)
 def test_near_orthogonal(self):
  with self.assertRaises(ValueError):c.retreat_delta([1,0,0],[.01,1,0],.05)
 def test_degenerate(self):
  for v in [[0,0,0],[float('nan'),1,0],[1,2]]:
   with self.assertRaises(ValueError):c.retreat_delta(v,[1,0,0],.05)
class ContactTests(unittest.TestCase):
 def test_new_shallow_recorded(self):
  r=c.contact_changes([],[contact(gap=-.0001)])
  self.assertEqual(len(r['new_contacts']),1);self.assertFalse(r['hard_reject'])
 def test_new_deep_rejected(self):self.assertTrue(c.contact_changes([],[contact(gap=-.0006)])['hard_reject'])
 def test_same_deep_baseline_not_rejected(self):
  self.assertFalse(c.contact_changes([contact(gap=-.01)],[contact(gap=-.01)])['hard_reject'])
 def test_existing_deepens_rejected(self):
  self.assertTrue(c.contact_changes([contact(gap=-.0001)],[contact(gap=-.00016)])['hard_reject'])
 def test_tiny_deepening_recorded(self):
  r=c.contact_changes([contact(gap=-.0001)],[contact(gap=-.00011)])
  self.assertEqual(len(r['deepened_contacts']),1);self.assertFalse(r['hard_reject'])
 def test_pair_order_and_duplicate_points(self):
  r=c.contact_changes([contact(gap=-.0003)],[
   contact('cabinet','robot',-.0002),contact(gap=-.0003)])
  self.assertFalse(r['new_contacts']);self.assertFalse(r['deepened_contacts'])
 def test_removed(self):self.assertFalse(c.contact_changes([contact(gap=-.1)],[])['hard_reject'])
 def test_invalid(self):
  for row in [{'pair':['x'],'distance_m':0},contact(gap=float('nan'))]:
   with self.assertRaises(ValueError):c.contact_changes([],[row])
 def test_negative_threshold(self):
  with self.assertRaises(ValueError):c.contact_changes([],[],threshold=-1)
class ScopeTests(unittest.TestCase):
 def test_train(self):
  for family in ['t_grip','bridge']:c.require_train_asset({'family':family,'split':'train_candidate'})
 def test_reserved_rejected(self):
  for entry in [{'family':'bridge','split':'test_reserved'},{'family':'cross_grip','split':'train_candidate'},{'family':'elbow_grip','split':'train_candidate'},{}]:
   with self.assertRaises(ValueError):c.require_train_asset(entry)
 def test_exact_unchanged(self):self.assertEqual(c.changed_indices_exact([1,2],[1,2],[0]),[])
 def test_declared(self):self.assertEqual(c.changed_indices_exact([1,2],[3,2],[0]),[0])
 def test_undeclared(self):
  with self.assertRaises(ValueError):c.changed_indices_exact([1,2],[1,3],[0])
 def test_bad_index(self):
  for index in [-1,2,True]:
   with self.assertRaises(ValueError):c.changed_indices_exact([1,2],[1,2],[index])
 def test_nonfinite(self):
  with self.assertRaises(ValueError):c.changed_indices_exact([1],[float('nan')],[0])
 def test_size(self):
  with self.assertRaises(ValueError):c.changed_indices_exact([1],[1,2],[0])
if __name__=='__main__':unittest.main()
