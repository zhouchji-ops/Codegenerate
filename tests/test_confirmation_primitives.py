import importlib.util
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'generate_confirmation_primitives.py'
spec = importlib.util.spec_from_file_location('confirmation_primitives', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ConfirmationPrimitiveTest(unittest.TestCase):
    def test_three_fixed_lineages(self):
        self.assertEqual(set(module.PARTS), {'open_fork', 'stepped_solid', 'double_end'})
        for parts in module.PARTS.values():
            module.validate(parts)

    def test_crossed_lineages(self):
        self.assertEqual(set(module.CROSSED_PARTS), {'planar_ring', 'staggered_z'})
        for parts in module.CROSSED_PARTS.values():
            module.validate(parts)

    def test_reject_overlap(self):
        with self.assertRaises(ValueError):
            module.validate([((1, 1, 1), (0, 0, 0)), ((1, 1, 1), (.1, 0, 0))])

    def test_reject_disconnected(self):
        with self.assertRaises(ValueError):
            module.validate([((1, 1, 1), (0, 0, 0)), ((1, 1, 1), (2, 0, 0))])

    def test_reject_nonpositive(self):
        with self.assertRaises(ValueError):
            module.validate([((0, 1, 1), (0, 0, 0))])


if __name__ == '__main__':
    unittest.main()
