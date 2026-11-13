import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from rgb_observation_prefix import ObservationPrefix,blue_counts,CAMERAS
class RGBPrefixTests(unittest.TestCase):
 def obs(self):return {k:np.zeros((256,256,3),dtype=np.uint8) for k in CAMERAS}
 def test_none(self):self.assertIsNone(ObservationPrefix('none').action(self.obs()))
 def test_fixed_budget(self):
  p=ObservationPrefix('fixed20');a=[p.action(self.obs()) for _ in range(21)];self.assertEqual(sum(x is not None for x in a),20)
 def test_rgb_budget(self):
  p=ObservationPrefix('rgb80');a=[p.action(self.obs()) for _ in range(81)];self.assertEqual(sum(x is not None for x in a),80);self.assertAlmostEqual(sum(x[9] for x in a if x is not None),0)
 def test_rgb_stops(self):
  o=self.obs();o[CAMERAS[0]][10:20,10:20,2]=255;self.assertEqual(blue_counts(o)[0],100);self.assertIsNone(ObservationPrefix('rgb80').action(o))
 def test_crop(self):
  o=self.obs();o[CAMERAS[0]][:6,:,2]=255;self.assertEqual(blue_counts(o)[0],0)
 def test_legal_actions(self):
  p=ObservationPrefix('rgb80')
  for _ in range(80):
   a=p.action(self.obs());self.assertEqual(a.shape,(12,));self.assertTrue(np.isfinite(a).all());self.assertLessEqual(np.max(np.abs(a)),1);self.assertTrue(np.all(a[[0,1,2,3,4,5,6,7,8,10]]==0))
 def test_only_rgb_inputs(self):
  o=self.obs();self.assertIsNotNone(ObservationPrefix('rgb80').action(o))
 def test_reject_mode(self):
  with self.assertRaises(AssertionError):ObservationPrefix('oracle')
if __name__=='__main__':unittest.main()
