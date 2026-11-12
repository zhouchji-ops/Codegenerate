"""Pure-function tests; no MuJoCo, rendering, or experiment artifacts."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

p = Path(__file__).resolve().parents[1] / "scripts" / "audit_policy_visibility.py"
spec = importlib.util.spec_from_file_location("visibility", p)
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)

class VisibilityTests(unittest.TestCase):
    def test_initial_history_repeats(self):
        self.assertEqual(v.history_indices(0), [0, 0, 0, 0])
    def test_early_and_late_history(self):
        self.assertEqual(v.history_indices(3), [0, 0, 1, 3])
        self.assertEqual(v.history_indices(16), [10, 12, 14, 16])
    def test_full_query_budget(self):
        q = v.query_schedule(600)
        self.assertEqual((len(q), q[-1]), (38, 592))
        self.assertEqual(v.query_schedule(16), [0])
    def test_crop_official_floor(self):
        self.assertEqual(v.crop_bounds(256, 256), (6, 6, 249, 249))
    def test_crop_non_square(self):
        self.assertEqual(v.crop_bounds(5, 7, .5), (1, 2, 3, 5))
    def test_crop_identity_and_tiny(self):
        self.assertEqual(v.crop_bounds(256, 256, 1), (0, 0, 256, 256))
        self.assertEqual(v.crop_bounds(5, 7, .001), (2, 3, 3, 4))
    def test_invalid_crop(self):
        for ratio in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError): v.crop_bounds(256, 256, ratio)
    def test_invalid_history_and_budget(self):
        for args in ((-1, 4, 2), (0, 0, 2), (0, 4, 0), (0.5, 4, 2)):
            with self.assertRaises(ValueError): v.history_indices(*args)
        with self.assertRaises(ValueError): v.query_schedule(0)
    def test_output_guard(self):
        with tempfile.TemporaryDirectory() as root:
            repo = Path(root) / "repo"
            with self.assertRaises(ValueError): v.validate_output(repo / "audit.json", repo)
            output = Path(root) / "audit.json"
            self.assertEqual(v.validate_output(output, repo), output.resolve())
            output.write_text("{}")
            with self.assertRaises(FileExistsError): v.validate_output(output, repo)
if __name__ == "__main__":
    unittest.main()
