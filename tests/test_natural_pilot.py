import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import prepare_xr1_natural_pilot as n
class NaturalTest(unittest.TestCase):
    def rows(self):
        return [dict(scene_group=g,run='original_'+g,split='train',source='policy_rollout_not_expert') for g in sorted(n.ALLOWED)]
    def test_groups(self): n.validate_groups(self.rows())
    def test_reserved(self):
        r=self.rows();r[0]['scene_group']='train_bridge_05_seed6024'
        with self.assertRaises(ValueError): n.validate_groups(r)
    def test_old_schema(self):
        r=self.rows();r[0]['split']='train_candidate'
        with self.assertRaises(ValueError): n.validate_groups(r)
    def test_duplicate(self):
        r=self.rows();r[0]=r[1]
        with self.assertRaises(ValueError): n.validate_groups(r)
    def test_slots(self):
        rows=[dict(source='policy_rollout_not_expert' if i%4==3 else 'human_dataset',identity=i) for i in range(256)]
        new=[dict(source='policy_rollout_not_expert',natural=i) for i in range(16)]
        out=n.replace_slots(rows,new)
        self.assertEqual(sum(out[i]==rows[i] for i in range(256)),240)
        self.assertEqual([out[i] for i in range(15,256,16)],new)
    def test_incomplete(self):
        with self.assertRaises(ValueError): n.replace_slots([],[])
    def test_full_span(self):
        a=n.anchors(320,0,0,4,False)
        self.assertTrue(6<=min(a) and max(a)<=304 and max(a)>240)
if __name__=='__main__':unittest.main()
