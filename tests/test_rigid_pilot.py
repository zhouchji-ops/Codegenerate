import importlib.util
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

spec = importlib.util.spec_from_file_location('rigid', Path(__file__).resolve().parents[1] / 'scripts/generate_rigid_pilot.py')
rigid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rigid)

class RigidPilotTests(unittest.TestCase):
    def test_dimensions_and_collision_match(self):
        root = ET.fromstring(rigid.make_box((.04,.06,.03)))
        geoms = root.findall('.//geom')
        self.assertEqual(len(geoms), 3)
        self.assertEqual({g.get('size') for g in geoms}, {'0.02 0.03 0.015'})
    def test_reject_bad_dimensions(self):
        for size in ((0,.04,.03), (float('nan'),.04,.03), (.2,.04,.03)):
            with self.assertRaises(ValueError): rigid.make_box(size)
    def test_no_overwrite_and_external_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError): rigid.generate(root / 'repo' / 'assets', root / 'repo')
            out = root / 'external'
            self.assertEqual(len(rigid.generate(out, root / 'repo')), 6)
            with self.assertRaises(FileExistsError): rigid.generate(out, root / 'repo')
