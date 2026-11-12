"""Generate a tiny external MJCF pilot, never mutate the RoboCasa registry.

All dimensions are metres. Primitive rigid boxes are engineering controls,
not a strong procedural baseline or evidence of useful robot demonstrations.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def make_box(size, color=(0.3, 0.6, 0.8, 1.0)):
    if len(size) != 3 or any(not math.isfinite(v) or not 0.01 <= v <= 0.12 for v in size):
        raise ValueError('full box dimensions must be finite and in [0.01, 0.12] m')
    if len(color) != 4 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in color):
        raise ValueError('invalid rgba')
    root = ET.Element('mujoco', model='pilot_rigid_box')
    ET.SubElement(root, 'asset')
    body = ET.SubElement(ET.SubElement(ET.SubElement(root, 'worldbody'), 'body'), 'body', name='object')
    half = ' '.join(str(v / 2) for v in size)
    common = dict(type='box', size=half, pos='0 0 0')
    ET.SubElement(body, 'geom', name='collision', group='0', density='300', friction='0.95 0.3 0.1', rgba='0.8 0.8 0.8 0', **common)
    ET.SubElement(body, 'geom', name='visual', group='1', contype='0', conaffinity='0', mass='0', rgba=' '.join(map(str, color)), **common)
    ET.SubElement(body, 'geom', name='reg_bbox', group='1', contype='0', conaffinity='0', mass='0', rgba='0 1 0 0', **common)
    return ET.tostring(root, encoding='unicode')


def generate(output, forbidden_root):
    output, forbidden_root = Path(output).resolve(), Path(forbidden_root).resolve()
    if output == forbidden_root or forbidden_root in output.parents:
        raise ValueError('generated assets must be outside the code repository')
    # Fail closed: no deletion, reuse, or overwrite of any existing directory.
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for i, size in enumerate(((.04,.06,.03),(.05,.06,.03),(.06,.06,.03),(.04,.08,.03),(.05,.08,.03),(.06,.08,.03))):
        name = 'rigid_%02d' % i
        folder = output / name
        folder.mkdir()
        content = make_box(size).encode('utf-8')
        (folder / 'model.xml').write_bytes(content)
        records.append(dict(id=name, family='box_engineering_pilot', full_size_m=size,
                            xml_sha256=hashlib.sha256(content).hexdigest(),
                            path=name + '/model.xml', status='generated_not_task_validated'))
    (output / 'manifest.json').write_text(json.dumps(dict(schema_version=1, task='PickPlaceCounterToSink',
        note='No teacher, robot execution, relation label, or training success is implied.', candidates=records), indent=2), encoding='utf-8')
    return records


def smoke(output):
    """Real MJCFObject construction and isolated gravity/plane settling only."""
    import mujoco
    import numpy as np
    from robocasa.models.objects.objects import MJCFObject
    root = Path(output)
    records = json.loads((root / 'manifest.json').read_text())['candidates']
    results = []
    for record in records:
        path = root / record['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['xml_sha256']:
            raise ValueError('asset checksum mismatch')
        obj = MJCFObject(name=record['id'], mjcf_path=str(path))
        xml = ET.fromstring(path.read_text())
        world = xml.find('worldbody')
        inner = world.find('body/body')
        world.remove(world.find('body'))
        world.append(inner)
        inner.set('pos', '0 0 0.2')
        ET.SubElement(inner, 'freejoint')
        ET.SubElement(world, 'geom', name='floor', type='plane', size='1 1 0.1')
        model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding='unicode'))
        data = mujoco.MjData(model)
        for _ in range(2000):
            mujoco.mj_step(model, data)
        expected_z = record['full_size_m'][2] / 2
        ok = bool(np.isfinite(data.qpos).all() and abs(data.qpos[2]-expected_z) < .002 and np.linalg.norm(data.qvel) < .01)
        results.append(dict(id=record['id'], mjcf_object_loaded=True, settle_pass=ok,
                            z_m=float(data.qpos[2]), speed=float(np.linalg.norm(data.qvel))))
    report = dict(scope='MJCFObject load + isolated plane settling; NOT kitchen task or grasp validation', results=results)
    (root / 'smoke_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not all(r['settle_pass'] for r in results):
        raise RuntimeError('settling check failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    if args.smoke_only:
        smoke(args.output)
    else:
        generate(args.output, Path(__file__).resolve().parents[1])
