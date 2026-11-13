"""Generate three frozen construction-lineage assets outside the code repository.

These are compound rigid primitives, not external-model output. Each family is
one cluster; mirrored or rescaled copies would not count as new clusters.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

PARTS = {
    'open_fork': [((.060,.012,.016),(0,-.019,.008)), ((.012,.038,.016),(-.024,.006,.008)), ((.012,.038,.016),(.024,.006,.008))],
    'stepped_solid': [((.060,.050,.012),(0,0,.006)), ((.040,.050,.012),(-.010,0,.018)), ((.020,.050,.012),(-.020,0,.030))],
    'double_end': [((.036,.014,.018),(0,0,.009)), ((.012,.050,.018),(-.024,0,.009)), ((.012,.050,.018),(.024,0,.009))],
}


CROSSED_PARTS = {
    'planar_ring': [((.060,.012,.016),(0,-.019,.008)), ((.060,.012,.016),(0,.019,.008)), ((.012,.026,.016),(-.024,0,.008)), ((.012,.026,.016),(.024,0,.008))],
    'staggered_z': [((.036,.012,.018),(0,0,.009)), ((.012,.034,.018),(-.024,.008,.009)), ((.012,.034,.018),(.024,-.008,.009))],
}


def validate(parts):
    edges = []
    for i, (size, pos) in enumerate(parts):
        if min(size) <= 0:
            raise ValueError('nonpositive dimensions')
        for j, (other_size, other_pos) in enumerate(parts[:i]):
            overlap = [min(pos[k]+size[k]/2, other_pos[k]+other_size[k]/2) - max(pos[k]-size[k]/2, other_pos[k]-other_size[k]/2) for k in range(3)]
            if all(v > 1e-9 for v in overlap):
                raise ValueError('volumetric overlap')
            if sum(v > 1e-9 for v in overlap) == 2 and all(v >= -1e-9 for v in overlap):
                edges.append((i, j))
    seen = {0}
    for _ in parts:
        for i, j in edges:
            if i in seen or j in seen:
                seen.update((i, j))
    if len(seen) != len(parts):
        raise ValueError('disconnected components')


def generate(output, parts_by_family=None):
    # Reuse the same XML serializer as the earlier asset families, retaining
    # mass, friction, color, collision/visual separation and registration bbox.
    parts_by_family = PARTS if parts_by_family is None else parts_by_family
    from generate_grip_families import build
    import generate_grip_families as serializer
    output = Path(output).resolve()
    repo = Path(__file__).resolve().parents[1]
    if output == repo or repo in output.parents:
        raise ValueError('external output required')
    output.mkdir(parents=True, exist_ok=False)
    records = []
    original = serializer.components
    try:
        serializer.components = lambda family, width, thickness: parts_by_family[family]
        for family, parts in parts_by_family.items():
            validate(parts)
            xml, size, count = build(family, 0, 0)
            folder = output / family
            folder.mkdir()
            data = xml.encode('utf-8')
            (folder / 'model.xml').write_bytes(data)
            records.append(dict(cluster=family, full_size_m=size, components=count,
                                xml_sha256=hashlib.sha256(data).hexdigest(), nominal_mass_kg=.04))
    finally:
        serializer.components = original
    (output / 'manifest.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--family-set', choices=('initial', 'crossed'), default='initial')
    args = parser.parse_args()
    generate(args.output, PARTS if args.family_set == 'initial' else CROSSED_PARTS)
