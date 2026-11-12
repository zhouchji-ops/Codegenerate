"""Input-lineage tests without MuJoCo or generated research artifacts."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
import numpy as np

scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("capture_pose", scripts / "capture_observation_pose.py")
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)

class InputLineageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        source = root / "source"; source.mkdir()
        assets = root / "assets"; assets.mkdir()
        asset = assets / "model.xml"; asset.write_text("<test/>")
        (source / "model.xml").write_text("<compiled/>")
        state = np.array([0., 1., 2.]); np.save(source / "initial_sim_state.npy", state)
        obs = np.array([1., 2.]); np.savez(source / "initial_observation.npz", state=obs)
        meta = dict(asset_xml=str(asset), asset_sha256=c.digest(asset),
            model_sha256=c.digest(source / "model.xml"),
            sim_state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),
            observation_hashes={"state": hashlib.sha256(obs.tobytes()).hexdigest()},
            initial_success=False)
        (source / "snapshot_metadata.json").write_text(json.dumps(meta))
        (source / "episode_metadata.json").write_text("{}")
        manifest = assets / "manifest.json"
        manifest.write_text(json.dumps({"candidates": [dict(id="t0", family="t_grip",
            split="train_candidate", path="model.xml", xml_sha256=c.digest(asset))]}))
        roles = root / "roles.json"
        roles.write_text(json.dumps({"sources": [dict(snapshot_path=str(source),
            snapshot_metadata_sha256=c.digest(source / "snapshot_metadata.json"),
            episode_metadata_sha256=c.digest(source / "episode_metadata.json"),
            role="observation_train")]}))
        self.args = types.SimpleNamespace(source=source, asset_manifest=manifest,
            source_role_manifest=roles, role="observation_train")
    def test_valid_source(self):
        self.assertEqual(c.validate_inputs(self.args)[-1]["source_role"], "observation_train")
    def test_role_change_rejected(self):
        self.args.role = "observation_validation_only"
        with self.assertRaises(ValueError): c.validate_inputs(self.args)
    def test_missing_role_rejected(self):
        self.args.source_role_manifest.write_text('{"sources":[]}')
        with self.assertRaises(ValueError): c.validate_inputs(self.args)
    def test_metadata_mutation_rejected(self):
        p=self.args.source / "snapshot_metadata.json"
        p.write_text(p.read_text()+" ")
        with self.assertRaises(ValueError): c.validate_inputs(self.args)
    def test_model_mutation_rejected(self):
        (self.args.source / "model.xml").write_text("<changed/>")
        with self.assertRaises(ValueError): c.validate_inputs(self.args)
    def test_duplicate_role_rejected(self):
        p=self.args.source_role_manifest; d=json.loads(p.read_text())
        d["sources"].append(d["sources"][0]); p.write_text(json.dumps(d))
        with self.assertRaises(ValueError): c.validate_inputs(self.args)
    def test_episode_metadata_mutation_rejected(self):
        (self.args.source / "episode_metadata.json").write_text("{\"fixture_refs\":{}}")
        with self.assertRaises(ValueError): c.validate_inputs(self.args)
    def test_missing_episode_hash_rejected(self):
        p=self.args.source_role_manifest; d=json.loads(p.read_text())
        del d["sources"][0]["episode_metadata_sha256"]; p.write_text(json.dumps(d))
        with self.assertRaises(ValueError): c.validate_inputs(self.args)
    def test_asset_mutation_rejected(self):
        (self.args.asset_manifest.parent / "model.xml").write_text("<changed/>")
        with self.assertRaises(ValueError): c.validate_inputs(self.args)
    def test_state_mutation_rejected(self):
        np.save(self.args.source / "initial_sim_state.npy", np.array([0., 2., 3.]))
        with self.assertRaises(ValueError): c.validate_inputs(self.args)
    def test_reserved_asset_rejected(self):
        p=self.args.asset_manifest; d=json.loads(p.read_text())
        d["candidates"][0]["family"]="elbow_grip"
        p.write_text(json.dumps(d))
        with self.assertRaises(ValueError): c.validate_inputs(self.args)
    def test_observation_mutation_rejected(self):
        np.savez(self.args.source / "initial_observation.npz", state=np.array([9., 2.]))
        with self.assertRaises(ValueError): c.validate_inputs(self.args)

if __name__ == "__main__":
    unittest.main()
