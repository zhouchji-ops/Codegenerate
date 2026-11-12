"""Unit checks for the quarter-generated ablation."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_xr1_ratio_pilot import quarter_schedule
class RatioTests(unittest.TestCase):
    def test_exact_ratio_and_control_slots(self):
        human=[{'source':'human_dataset','id':i} for i in range(256)]
        mixed=[r.copy() if i%2==0 else {'source':'policy_rollout_not_expert','id':i} for i,r in enumerate(human)]
        result=quarter_schedule(human,mixed)
        self.assertEqual(sum(r['source']=='human_dataset' for r in result),192)
        self.assertEqual(sum(r['source']=='policy_rollout_not_expert' for r in result),64)
        for i,row in enumerate(result):self.assertEqual(row,mixed[i] if i%4==3 else human[i])
    def test_mismatched_control_rejected(self):
        h=[{'source':'human_dataset','id':i} for i in range(4)]
        m=[r.copy() if i%2==0 else {'source':'policy_rollout_not_expert','id':i} for i,r in enumerate(h)]
        m[0]['id']=99
        with self.assertRaises(ValueError):quarter_schedule(h,m)
if __name__=='__main__':unittest.main()
