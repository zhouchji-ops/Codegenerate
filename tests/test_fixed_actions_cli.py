"""CPU-only CLI safety checks; do not import model libraries or run inference."""
import contextlib,importlib.util,io,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
SCRIPT=Path(__file__).resolve().parents[1]/'scripts/audit_xr1_fixed_actions.py'
spec=importlib.util.spec_from_file_location('fixed_actions_cli',SCRIPT)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class FixedActionsCliTests(unittest.TestCase):
    def inputs(self,p):
        for n in ('repo','model','source'):(p/n).mkdir()
        for n in ('human','uniform','phase'):(p/(n+'.pt')).touch()
        return [str(SCRIPT),'--xr1-repo',str(p/'repo'),'--model',str(p/'model'),
            '--source',str(p/'source'),'--human-expert',str(p/'human.pt'),
            '--uniform-expert',str(p/'uniform.pt'),'--phase-expert',str(p/'phase.pt')]
    def test_help_without_cuda(self):
        env=os.environ.copy();env.pop('CUDA_VISIBLE_DEVICES',None)
        r=subprocess.run([sys.executable,'-B',str(SCRIPT),'--help'],capture_output=True,text=True,env=env)
        self.assertEqual(r.returncode,0);self.assertIn('--human-expert',r.stdout)
    def test_gpu_selection_required(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);args=self.inputs(p)+['--output',str(p/'out')]
            with patch.object(sys,'argv',args),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}),contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):module.parse_args()
            self.assertFalse((p/'out').exists())
    def test_source_nested_output_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);args=self.inputs(p)+['--output',str(p/'source'/'out')]
            with patch.object(sys,'argv',args),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}),contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):module.parse_args()
            self.assertFalse((p/'source'/'out').exists())
    def test_existing_output_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);args=self.inputs(p);(p/'out').mkdir()
            with patch.object(sys,'argv',args+['--output',str(p/'out')]),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}),contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):module.parse_args()
if __name__=='__main__':unittest.main()
