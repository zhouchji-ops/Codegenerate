"""CPU tests for clamped initial observations and two-group admission."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_xr1_observation_pilot import history_indices,pair_rows,validate_visibility
class ObservationPrepTests(unittest.TestCase):
    def evidence(self,hidden):
        queries=[]
        for t in range(0,128,16):
            count=0 if hidden and t<48 else 25
            queries.append(dict(step=t,history_indices=history_indices(t),counts_history_by_camera=[[count,0,0]]*4))
        return {'queries':queries}
    def test_initial_clamp(self):
        self.assertEqual(history_indices(0),[0,0,0,0])
        self.assertEqual(history_indices(4),[0,0,2,4])
        self.assertEqual(history_indices(16),[10,12,14,16])
        with self.assertRaises(ValueError):history_indices(-1)
    def test_group_identity_not_trajectory_count(self):
        rows=[dict(scene_group=g,role=r) for g in ('g1','g2') for r in ('visible_start','hidden_start')]
        self.assertEqual(len(pair_rows(rows,['g1','g2'])),2)
        rows[-1]['scene_group']='heldout'
        with self.assertRaises(ValueError):pair_rows(rows,['g1','g2'])
    def test_duplicate_role_rejected(self):
        rows=[dict(scene_group=g,role=r) for g in ('g1','g2') for r in ('visible_start','hidden_start')]
        rows[-1]['role']='visible_start'
        with self.assertRaises(ValueError):pair_rows(rows,['g1','g2'])
    def test_discovery_before_contact(self):
        _,t=validate_visibility(self.evidence(True),'hidden_start',123)
        self.assertEqual(t,48)
        with self.assertRaises(ValueError):validate_visibility(self.evidence(True),'hidden_start',32)
    def test_small_pixel_is_not_visible_control(self):
        d=self.evidence(False);d['queries'][0]['counts_history_by_camera']=[[2,0,0]]*4
        with self.assertRaises(ValueError):validate_visibility(d,'visible_start',123)
    def test_visible_not_hidden(self):
        with self.assertRaises(ValueError):validate_visibility(self.evidence(False),'hidden_start',123)
if __name__=='__main__':unittest.main()
