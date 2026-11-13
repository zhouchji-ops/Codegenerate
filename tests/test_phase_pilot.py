"""Phase sampling boundaries and allocation tests, using no research artifacts."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_xr1_phase_pilot import anchors,require_training_family,validate_unique_runs,validate_quarter_slots,balanced_anchors
class PhaseTests(unittest.TestCase):
    def test_balanced_stage_quotas_and_holding(self):
        total={};offset=0
        for i in range(9):
            count=8 if i==0 else 7
            chosen,labels=balanced_anchors(240,100,140,count,offset,list(range(140,180)))
            self.assertEqual(len(chosen),count)
            for t,label in zip(chosen,labels):
                self.assertTrue(6<=t<=224)
                if label=='verified_held_transport':self.assertTrue(140<=t<180)
                total[label]=total.get(label,0)+1
            offset+=count
        self.assertEqual(set(total.values()),{16})
    def test_unverified_transport_refused(self):
        with self.assertRaises(ValueError):balanced_anchors(240,100,140,8,0,[])
    def test_duplicate_runs_rejected(self):
        validate_unique_runs([{'run':'a'},{'run':'b'}])
        with self.assertRaises(ValueError):validate_unique_runs([{'run':'a'},{'run':'a'}])
    def test_shifted_quarter_pattern_rejected(self):
        rows=[{'source':'policy_rollout_not_expert' if i%4==3 else 'human_dataset'} for i in range(256)]
        validate_quarter_slots(rows)
        rows[0],rows[3]=rows[3],rows[0]
        self.assertEqual(sum(r['source']=='policy_rollout_not_expert' for r in rows),64)
        with self.assertRaises(ValueError):validate_quarter_slots(rows)
    def test_reserved_families_rejected(self):
        require_training_family('t_grip');require_training_family('bridge')
        for family in ('cross','elbow'):
            with self.assertRaises(ValueError):require_training_family(family)
    def test_uniform_and_phase_same_count(self):
        u=anchors(240,100,140,8,False)
        p=anchors(240,100,140,8,True,4)
        self.assertEqual(len(u),len(p))
        self.assertTrue(all(6<=t<=224 for t in u+p))
        self.assertTrue(all(6<=t<=100 for t in p[:4]))
        self.assertTrue(all(92<=t<=156 for t in p[4:]))
    def test_short_episode_safe(self):
        p=anchors(30,2,8,4,True,2)
        self.assertTrue(all(6<=t<=14 for t in p))
    def test_invalid_phase(self):
        with self.assertRaises(ValueError):anchors(240,140,100,8,True)
        with self.assertRaises(ValueError):anchors(240,100,140,8,True,0)
    def test_uneven_scenes_still_split32(self):
        for scenes in range(2,33):
            odd=0;approach=0;total=0
            for i in range(scenes):
                count=64//scenes+(i<64%scenes);n=count//2
                if count%2:n+=int(odd%2==0);odd+=1
                approach+=n;total+=count
            self.assertEqual(total,64);self.assertEqual(approach,32)
if __name__=='__main__':unittest.main()
