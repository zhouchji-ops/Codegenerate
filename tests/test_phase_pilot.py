"""Phase sampling boundaries and allocation tests, using no research artifacts."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_xr1_phase_pilot import anchors,require_training_family
class PhaseTests(unittest.TestCase):
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
