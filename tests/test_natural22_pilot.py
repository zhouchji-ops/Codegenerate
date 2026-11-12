import sys, unittest, copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prepare_xr1_natural22_dose_pilot as m

class Natural22Tests(unittest.TestCase):
    def rows(self):
        return [dict(scene_group=g, run='original_'+g, split='train', source='policy_rollout_not_expert') for g in sorted(m.ALLOWED)]
    def test_exact_groups(self):
        self.assertEqual(len(m.ALLOWED),22); m.validate_groups(self.rows())
    def test_no_heldout(self):
        self.assertFalse(any('elbow' in g or '610' in g or '620' in g or '640' in g for g in m.ALLOWED))
    def test_excluded_failures(self):
        self.assertNotIn('train_t_grip_02_seed6003',m.ALLOWED)
        self.assertNotIn('train_bridge_05_seed6024',m.ALLOWED)
    def test_reject_missing(self):
        with self.assertRaises(ValueError): m.validate_groups(self.rows()[:-1])
    def test_reject_expert_label(self):
        rows=self.rows();rows[0]['source']='expert'
        with self.assertRaises(ValueError):m.validate_groups(rows)
    def test_unique_quantiles(self):
        for count in (2,3):
            ids=m.uniform_anchors(141,count);self.assertEqual(len(set(ids)),count)
            self.assertTrue(all(6<=x<=125 for x in ids))
    def test_family_quota(self):
        rows=self.rows();self.assertEqual(sum(m.family_quota(rows,r) for r in rows),64)
        for prefix in ('train_bridge_','train_t_grip_'):
            family=[r for r in rows if r['scene_group'].startswith(prefix)]
            self.assertEqual(len(family),11);self.assertEqual(sum(m.family_quota(rows,r) for r in family),32)
            self.assertEqual(sorted(m.family_quota(rows,r) for r in family),[2]+[3]*10)
    def test_seeded_training_source(self):
        s=(Path(__file__).resolve().parents[1]/'scripts/xr1_seeded_training_probe.py').read_text()
        self.assertIn('torch.manual_seed(a.seed)',s)
        self.assertIn('forward_one(sample,a.train_noise_base+step,True)',s)
        self.assertIn('forward_one(s,9000+i,False)',s)
if __name__=='__main__':unittest.main()
