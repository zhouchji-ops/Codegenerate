"""Unit checks of duration, episode reset and rejection logic (no simulator)."""
import importlib.util
from pathlib import Path
import types
import unittest
from unittest.mock import patch
p=Path(__file__).resolve().parents[1]/'scripts'/'review_rollout_physics.py'
spec=importlib.util.spec_from_file_location('physics_review',p)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
class GateTests(unittest.TestCase):
    def setUp(self):
        self.env=types.SimpleNamespace(sim=types.SimpleNamespace(data=types.SimpleNamespace(time=0.)))
        self.gate=mod.StableSuccessGate()
    def step(self,t,ok=True,pos=(0,0,0)):
        self.env.sim.data.time=t
        with patch.object(mod,'snapshot',return_value=dict(instantaneous_candidate=ok,position_m=pos)):
            return self.gate.update(self.env)[0]
    def test_requires_duration(self):
        self.assertFalse(self.step(0));self.assertFalse(self.step(.49));self.assertTrue(self.step(.5))
    def test_failed_sample_resets(self):
        self.step(0);self.step(.4,False);self.assertFalse(self.step(.5))
    def test_motion_rejected(self):
        self.step(0);self.assertFalse(self.step(.5,pos=(.006,0,0)))
    def test_episode_reset(self):
        self.step(1);self.step(1.5);self.gate.reset();self.assertFalse(self.step(0))
    def test_time_reversal_resets(self):
        self.step(1);self.assertFalse(self.step(.5))

class CandidateTests(unittest.TestCase):
    def check(self, **kw):
        args=dict(official=True,horizontal=True,contacts=[dict(sink=True,robot=False,active=True,normal_force_N=.1)],
                  penetration=.00001,linear=.001,angular=.01)
        args.update(kw);return mod.physical_candidate(**args)
    def test_valid_support(self):self.assertTrue(self.check())
    def test_held(self):self.assertFalse(self.check(contacts=[dict(sink=True,robot=True)]))
    def test_fast_flyby(self):self.assertFalse(self.check(linear=.1))
    def test_spinning(self):self.assertFalse(self.check(angular=1.))
    def test_no_support(self):self.assertFalse(self.check(contacts=[]))
    def test_distractor_only(self):self.assertFalse(self.check(contacts=[dict(sink=False,robot=False)]))
    def test_deep_penetration(self):self.assertFalse(self.check(penetration=.005))
    def test_xy_protrusion(self):self.assertFalse(self.check(horizontal=False))
    def test_official_failure(self):self.assertFalse(self.check(official=False))
    def test_nan(self):self.assertFalse(self.check(linear=float('nan')))


class SafetyTests(unittest.TestCase):
    def test_bad_duration(self):
        for value in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):mod.StableSuccessGate(value)
    def test_output_inside(self):
        with self.assertRaises(ValueError):mod.validate_output_folder(Path.cwd()/'out',Path.cwd())
    def test_output_equal(self):
        with self.assertRaises(ValueError):mod.validate_output_folder(Path.cwd(),Path.cwd())
    def test_output_external(self):
        root=Path.cwd()/'code'
        self.assertEqual(mod.validate_output_folder(Path.cwd()/'data',root),(Path.cwd()/'data').resolve())
    def test_inactive_contact(self):
        self.assertFalse(mod.physical_candidate(True,True,[dict(sink=True,robot=False,active=False,normal_force_N=.1)],0,0,0))
    def test_zero_force(self):
        self.assertFalse(mod.physical_candidate(True,True,[dict(sink=True,robot=False,active=True,normal_force_N=0)],0,0,0))
    def test_nan_force(self):
        self.assertFalse(mod.physical_candidate(True,True,[dict(sink=True,robot=False,active=True,normal_force_N=float('nan'))],0,0,0))

if __name__=='__main__':unittest.main()
