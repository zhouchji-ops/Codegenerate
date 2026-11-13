"""Small tensor-only tests; no model weights or research data stored in the repo."""
import importlib.util,json,subprocess,sys,tempfile,unittest
from pathlib import Path
HAS_TORCH=importlib.util.find_spec('torch') is not None and importlib.util.find_spec('safetensors') is not None
@unittest.skipUnless(HAS_TORCH,'optional torch/safetensors dependencies')
class ExpertAuditTests(unittest.TestCase):
    def test_zero_update_cosine_and_changed_norm(self):
        import torch
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);ck=root/'checkpoint';ck.mkdir()
            save_file({'dit.weight':torch.ones(2,3)},str(ck/'model.safetensors'))
            (ck/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'dit.weight':'model.safetensors'}}))
            torch.save({'model.dit.weight':torch.ones(2,3)},root/'same.pt')
            torch.save({'model.dit.weight':torch.ones(2,3)*2},root/'changed.pt')
            script=Path(__file__).resolve().parents[1]/'scripts/audit_xr1_expert_updates.py'
            out=root/'audit.json'
            subprocess.run([sys.executable,'-B',str(script),'--checkpoint',str(ck),
                '--expert','same='+str(root/'same.pt'),'--expert','changed='+str(root/'changed.pt'),
                '--output',str(out)],check=True,capture_output=True,text=True)
            result=json.loads(out.read_text())
            self.assertEqual(result['updates']['same']['delta_l2'],0)
            self.assertEqual(result['updates']['changed']['relative_delta_l2'],1)
            self.assertIsNone(result['update_cosines']['same|changed'])
if __name__=='__main__':unittest.main()
