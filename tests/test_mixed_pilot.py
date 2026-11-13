"""CPU tests with generated fixtures; no research data committed."""
import importlib.util, json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import h5py
import numpy as np
spec=importlib.util.spec_from_file_location('mixed_prep',Path(__file__).resolve().parents[1]/'scripts/prepare_xr1_mixed_pilot.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

class MixedPilotTests(unittest.TestCase):
    def fixture(self,p):
        h=p/'fixture.h5'
        with h5py.File(h,'w') as f:
            f.create_dataset('actions',data=np.arange(32*12,dtype=np.float32).reshape(32,12))
            f.create_dataset('state14',data=np.arange(32*14,dtype=np.float32).reshape(32,14))
            for camera in ('robot0_agentview_left','robot0_agentview_right','robot0_eye_in_hand'):
                f.create_dataset('observations/video.'+camera,shape=(32,256,256,3),dtype='uint8',compression='gzip')
            f.attrs['instruction']='test-only generated task'
        review=p/'review.json';review.write_text('{}')
        rows=[dict(run='synthetic_'+str(i),split='train_candidate',source='policy_rollout_not_expert',
              physical_status='stable',trajectory_path=str(h),trajectory_sha256=mod.digest(h),
              physical_review_path=str(review),physical_review_sha256=mod.digest(review),steps=32) for i in range(16)]
        manifest=p/'review_manifest.json'
        manifest.write_text(json.dumps(dict(accepted=rows,limitations='synthetic fixture only')))
        human=p/'human.json'
        human.write_text(json.dumps({'samples':[dict(path='sample_'+str(i)+'.npz',
             split='train' if i<32 else 'dev',episode_index=i,sample_step=6) for i in range(40)]}))
        return manifest,human
    def test_boundary_and_equal_budget(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);r,h=self.fixture(p);out=p/'out'
            with patch('sys.argv',['prep','--review-manifest',str(r),'--human-manifest',str(h),'--output',str(out)]):mod.main()
            m=json.loads((out/'manifest.json').read_text())['samples']
            self.assertEqual(len(m),40)
            self.assertEqual([x['episode_index'] for x in m[:32:2]],list(range(0,32,2)))
            z=np.load(out/'geometry_00.npz')
            self.assertEqual(z['state_history_raw14'].shape,(4,14))
            self.assertEqual(z['actions_raw12'].shape,(16,12))
            meta=json.loads((out/'geometry_00.json').read_text())
            t=meta['action_indices'][0]
            self.assertEqual(z['actions_raw12'][0,0],t*12)
            self.assertEqual(z['state_history_raw14'][-1,0],t*14)
            z.close()
    def test_reserved_split_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);r,h=self.fixture(p)
            m=json.loads(r.read_text());m['accepted'][0]['split']='test';r.write_text(json.dumps(m))
            with patch('sys.argv',['prep','--review-manifest',str(r),'--human-manifest',str(h),'--output',str(p/'out')]):
                with self.assertRaises(AssertionError):mod.main()
if __name__=='__main__':unittest.main()
