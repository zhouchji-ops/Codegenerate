import importlib.util
import itertools
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

spec=importlib.util.spec_from_file_location('grips',Path(__file__).resolve().parents[1]/'scripts/generate_grip_families.py')
grips=importlib.util.module_from_spec(spec);spec.loader.exec_module(grips)

class GripFamilyTests(unittest.TestCase):
    def test_mass_and_nonoverlap_and_connected(self):
        for family in ('t_grip','bridge','cross_grip','elbow_grip'):
            for w,t in itertools.product((.052,.06,.068),(.008,.012)):
                parts=grips.components(family,w,t)
                xml,_,_=grips.build(family,w,t)
                geoms=ET.fromstring(xml).findall('.//geom')
                self.assertAlmostEqual(sum(float(g.get('mass',0)) for g in geoms),.04)
                edges={i:set() for i in range(len(parts))}
                for i,j in itertools.combinations(range(len(parts)),2):
                    si,pi=parts[i]; sj,pj=parts[j]
                    overlaps=[min(pi[a]+si[a]/2,pj[a]+sj[a]/2)-max(pi[a]-si[a]/2,pj[a]-sj[a]/2) for a in range(3)]
                    self.assertFalse(all(v>1e-9 for v in overlaps), (family,i,j,'overlapping volumes'))
                    if min(overlaps)>=-1e-9 and sum(v>1e-9 for v in overlaps)>=2:
                        edges[i].add(j);edges[j].add(i)
                seen={0}
                for _ in parts:
                    seen.update(set.union(*(edges[i] for i in list(seen))))
                self.assertEqual(len(seen),len(parts),(family,'disconnected'))

    def test_unknown_family_rejected(self):
        with self.assertRaises(ValueError):grips.components('unknown',.06,.01)
