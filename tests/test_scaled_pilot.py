"""Regression tests for frozen matched-budget schedules."""
import ast,importlib.util,sys,unittest
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'scripts'))
from prepare_xr1_scaled_pilot import schedules

class ScaledScheduleTests(unittest.TestCase):
    def test_matched_even_slots_and_deterministic(self):
        h=[dict(source='human',id=i) for i in range(180)]
        g=[dict(source='generated',id=i) for i in range(152)]
        b,m=schedules(h,g,256,7)
        self.assertEqual((b,m),schedules(h,g,256,7))
        self.assertEqual(len(b),256);self.assertEqual(len(m),256)
        self.assertEqual(b[::2],m[::2])
        self.assertEqual(len({x['id'] for x in m[1::2]}),128)
        self.assertEqual(len({x['id'] for x in b}),180)
        self.assertTrue(all(x['source']=='human' for x in b))
        self.assertTrue(all(x['source']=='generated' for x in m[1::2]))
    def test_monitor_call_is_inside_optimizer_loop(self):
        tree=ast.parse((root/'scripts/xr1_short_train.py').read_text())
        loops=[n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='step']
        self.assertEqual(len(loops),1)
        calls=[n for n in ast.walk(loops[0]) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='checkpoint_metrics']
        self.assertEqual(len(calls),1)
    def test_invalid_budget(self):
        with self.assertRaises(ValueError):schedules([{}]*10,[{}]*10,3,7)
        with self.assertRaises(ValueError):schedules([{}]*10,[{}]*10,256,7)
if __name__=='__main__':unittest.main()
