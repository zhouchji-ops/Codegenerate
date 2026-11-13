import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prepare_xr1_natural11_pilot as n

class Natural11Test(unittest.TestCase):
    def rows(self):
        return [dict(scene_group=g, run='original_'+g, split='train', source='policy_rollout_not_expert') for g in sorted(n.ALLOWED)]
    def test_exact_groups(self):
        self.assertEqual(len(self.rows()), 11)
        n.validate_groups(self.rows())
    def test_no_holdout(self):
        rows=self.rows(); rows[0]['scene_group']='final_holdout_bridge_01_seed6204'
        with self.assertRaises(ValueError): n.validate_groups(rows)
    def test_uniform_single(self):
        self.assertEqual(n.uniform_anchors(130, 1), [60])
    def test_legacy_formula_unchanged(self):
        for size in (30, 130, 294, 530):
            for count in (2, 3, 4):
                self.assertEqual(n.uniform_anchors(size, count), n.anchors(size, 0, 0, count, False))
    def test_invalid(self):
        for size,count in ((20,1),(130,0),(130,-1)):
            with self.assertRaises(ValueError): n.uniform_anchors(size,count)
    def test_balanced_quota(self):
        rows=self.rows(); counts=[n.family_quota(rows,r) for r in rows]
        self.assertEqual(sum(counts),16)
        for family in ('train_bridge_','train_t_grip_'):
            self.assertEqual(sum(n.family_quota(rows,r) for r in rows if r['scene_group'].startswith(family)),8)
        self.assertTrue(all(c in (1,2) for c in counts))
    def test_preserve_slots(self):
        old=[dict(source='policy_rollout_not_expert' if i%4==3 else 'human_dataset',identity=i) for i in range(256)]
        new=[dict(source='policy_rollout_not_expert',new=i) for i in range(16)]
        out=n.replace_slots(old,new)
        self.assertTrue(all(out[i]==old[i] for i in range(256) if i%16!=15))
        self.assertEqual([out[i] for i in range(15,256,16)],new)
if __name__=='__main__': unittest.main()
