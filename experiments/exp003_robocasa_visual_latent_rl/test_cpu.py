"""Synthetic CPU tests, no upstream function extraction."""
import unittest,tempfile
from pathlib import Path
from cli_common import external_output,new_output,component,REPO_ROOT,require_selected_gpu
from unittest.mock import patch
import os
from reward_reset_contract import EpisodeContract
import torch
from visual_actor import VisualActor
from initial_noise_adapter_visual import adapt_forward,choose_noise

def synthetic_forward(action_mask):
    x = torch.randn_like(action_mask)
    return x * action_mask

class Contracts(unittest.TestCase):
    def test_gpu_selection_required(self):
        for value in ("", "-1", "  "):
            with patch.dict(os.environ,{"CUDA_VISIBLE_DEVICES":value}):
                with self.assertRaises(ValueError):require_selected_gpu()
    def test_gpu_selection_explicit(self):
        with patch.dict(os.environ,{"CUDA_VISIBLE_DEVICES":"1"}):
            self.assertEqual(require_selected_gpu(),"1")
    def test_repository_rejected(self):
        with self.assertRaises(ValueError):external_output(REPO_ROOT/"forbidden")
    def test_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/"new";new_output(p)
            with self.assertRaises(FileExistsError):new_output(p)
    def test_component(self):
        with self.assertRaises(Exception):component("../escape")
    def test_noise_parity(self):
        f,_=adapt_forward(synthetic_forward);mask=torch.ones(1,16,60)
        torch.manual_seed(17);expected=synthetic_forward(mask)
        torch.manual_seed(17);self.assertTrue(torch.equal(expected,f(mask)))
        self.assertTrue(torch.equal(expected,f(mask,initial_noise=expected)))
        self.assertFalse(torch.equal(expected,f(mask,initial_noise=expected+.25)))
    def test_noise_validation(self):
        mask=torch.ones(1,16,60)
        for bad in (torch.ones(1),mask.double(),mask*float("nan")):
            with self.assertRaises(ValueError):choose_noise(mask,bad)
        self.assertNotEqual(mask.data_ptr(),choose_noise(mask,mask).data_ptr())
    def test_actor(self):
        torch.manual_seed(719);actor=VisualActor();a,lp=actor.sample(torch.zeros(2,2574))
        self.assertEqual(tuple(a.shape),(2,192));self.assertTrue(torch.isfinite(lp).all())
        self.assertTrue((a.abs()<=1.5).all())
        (a.square().mean()+lp.mean()*.001).backward()
        self.assertTrue(any(p.grad is not None for p in actor.parameters()))
    def test_contract(self):
        c=EpisodeContract("episode","hash",1)
        self.assertEqual(c.step(official_success=True,physical_verified=True,invalid_sim=True)["reward"],0)
        with self.assertRaises(RuntimeError):c.step(official_success=True,physical_verified=True)
        c.reset("next","hash",2)
        self.assertEqual(c.step(official_success=True,physical_verified=True)["reward"],1)
if __name__=="__main__":unittest.main()
