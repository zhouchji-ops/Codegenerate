import bpy
import json
import math
import os
import re
import shutil
import struct
import sys
from pathlib import Path
from mathutils import Matrix

SCALES = {
    "mug": (414.029413875 / 267.56303865) ** (1.0 / 3.0),
    "scoop": (354.88235475 / 140.97639983) ** (1.0 / 3.0),
    "turner": 1.0,
}
CAPACITIES = {
    "mug": {"old_ml": 267.56303865, "target_ml": 414.029413875},
    "scoop": {"old_ml": 140.97639983, "target_ml": 354.88235475},
    "turner": {"old_ml": None, "target_ml": None},
}
ATTR = re.compile(r"""([:\w.-]+)\s*=\s*(["'])(.*?)\2""", re.S)
TOKENS = re.compile(r"<!--.*?-->|<!\[CDATA\[.*?\]\]>|<[^>]*>", re.S)


def inside(path, parent):
    return path == parent or parent in path.parents


def number(value):
    return format(value, ".17g")


def scaled_numbers(value, scale, expected=None):
    parts = value.split()
    if expected is not None and len(parts) not in expected:
        raise ValueError("Unexpected coordinate count: " + value)
    values = [float(x) * scale for x in parts]
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Non-finite geometry")
    return " ".join(number(x) for x in values)


def diagnostic(path):
    return any("diagnostic" in part.lower() for part in path.parts)


def scale_obj(path, scale):
    if scale == 1.0:
        return
    data = path.read_bytes()
    pattern = re.compile(
        rb"(?m)^([ \t]*v[ \t]+)([^ \t\r\n]+)([ \t]+)"
        rb"([^ \t\r\n]+)([ \t]+)([^ \t\r\n]+)"
    )

    def replace(match):
        return (
            match[1] + number(float(match[2]) * scale).encode("ascii")
            + match[3] + number(float(match[4]) * scale).encode("ascii")
            + match[5] + number(float(match[6]) * scale).encode("ascii")
        )

    path.write_bytes(pattern.sub(replace, data))


def scale_xml(path, scale):
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    if re.search(r"<!DOCTYPE", text, re.I):
        raise ValueError("Unsupported XML DOCTYPE: " + str(path))

    def transform(match):
        tag = match[0]
        opening = re.match(r"<(geom|site|body)\b", tag)
        if not opening:
            return tag
        kind = opening[1]
        matches = list(ATTR.finditer(tag))
        attrs = {m[1]: m[3] for m in matches}
        updates = {}
        if scale != 1.0:
            if "pos" in attrs:
                updates["pos"] = scaled_numbers(attrs["pos"], scale, (3,))
            if "fromto" in attrs:
                updates["fromto"] = scaled_numbers(attrs["fromto"], scale, (6,))
            primitive = kind == "site" or (
                kind == "geom"
                and attrs.get("type") not in ("mesh", "hfield")
                and "mesh" not in attrs
                and "hfield" not in attrs
            )
            if "size" in attrs and primitive:
                updates["size"] = scaled_numbers(attrs["size"], scale)

        bottom = (
            kind == "geom"
            and "bottom_disk" in attrs.get("name", "")
            and attrs.get("group") == "0"
        )
        if bottom:
            updates.update(priority="1", solref="0.008 1")

        for attr in reversed(matches):
            if attr[1] in updates:
                tag = tag[:attr.start(3)] + updates[attr[1]] + tag[attr.end(3):]
        missing = [(k, v) for k, v in updates.items() if k not in attrs]
        if missing:
            end = tag.rfind("/>") if tag.rstrip().endswith("/>") else tag.rfind(">")
            tag = tag[:end] + "".join(
                ' {}="{}"'.format(k, v) for k, v in missing
            ) + tag[end:]
        return tag

    updated = TOKENS.sub(transform, text)
    if updated != text:
        path.write_bytes(updated.encode("utf-8"))


def scale_glb(source, destination, scale):
    data = source.read_bytes()
    if len(data) < 12:
        raise ValueError("Truncated GLB: " + str(source))
    magic, version, total = struct.unpack_from("<4sII", data)
    if magic != b"glTF" or version != 2 or total != len(data):
        raise ValueError("Invalid GLB2: " + str(source))
    chunks = []
    offset = 12
    while offset < total:
        if offset + 8 > total:
            raise ValueError("Truncated GLB chunk")
        length, kind = struct.unpack_from("<II", data, offset)
        end = offset + 8 + length
        if length % 4 or end > total:
            raise ValueError("Invalid GLB chunk length")
        chunks.append((kind, data[offset + 8:end], data[offset:end]))
        offset = end
    if not chunks or chunks[0][0] != 0x4E4F534A:
        raise ValueError("GLB JSON must be first")
    if sum(kind == 0x4E4F534A for kind, _, _ in chunks) != 1:
        raise ValueError("Multiple GLB JSON chunks")
    if scale == 1.0:
        destination.write_bytes(data)
        return
    document = json.loads(chunks[0][1].decode("utf-8"))
    nodes = document.get("nodes", [])
    roots = {
        index
        for scene in document.get("scenes", [])
        for index in scene.get("nodes", [])
    }
    for index in roots:
        node = nodes[index]
        if "matrix" in node:
            matrix = list(node["matrix"])
            if len(matrix) != 16:
                raise ValueError("Invalid node matrix")
            for column in range(4):
                for row in range(3):
                    i = column * 4 + row
                    matrix[i] *= scale
            node["matrix"] = matrix
        else:
            node["scale"] = [v * scale for v in node.get("scale", [1, 1, 1])]
            if "translation" in node:
                node["translation"] = [v * scale for v in node["translation"]]
    payload = json.dumps(
        document, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")
    payload += b" " * (-len(payload) % 4)
    body = struct.pack("<II", len(payload), 0x4E4F534A) + payload
    body += b"".join(original for _, _, original in chunks[1:])
    destination.write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body)


def scale_blend(source, destination, scale):
    bpy.ops.wm.open_mainfile(filepath=str(source))
    transform = Matrix.Scale(scale, 4)
    roots = [
        obj for obj in bpy.data.objects
        if obj.type == "MESH" and obj.parent is None
    ]
    for obj in roots:
        obj.matrix_world = transform @ obj.matrix_world
    bpy.context.view_layer.update()
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(destination), check_existing=False)


def main():
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if len(args) != 2:
        raise ValueError("Usage: blender -b --python physical_calibration.py -- SOURCE_OUTPUT NEW_OUTPUT")
    source = Path(args[0]).expanduser().resolve()
    destination = Path(args[1]).expanduser().resolve()
    if not source.is_dir():
        raise ValueError("SOURCE_OUTPUT must be a directory")
    if destination == source or inside(destination, source / "sim"):
        raise ValueError("NEW_OUTPUT must be distinct and outside SOURCE_OUTPUT/sim")
    if destination.exists() and (
        not destination.is_dir() or any(destination.iterdir())
    ):
        raise ValueError("NEW_OUTPUT must be absent or empty")
    for name in SCALES:
        for extension in ("blend", "glb"):
            if not (source / (name + "." + extension)).is_file():
                raise FileNotFoundError(name + "." + extension)
        if not (source / "sim" / name / "model.xml").is_file():
            raise FileNotFoundError("sim/" + name + "/model.xml")
    destination.mkdir(parents=True, exist_ok=True)
    for name, scale in SCALES.items():
        asset_source = source / "sim" / name
        asset_destination = destination / "sim" / name

        def ignore(directory, names):
            return [
                entry for entry in names
                if diagnostic((Path(directory) / entry).relative_to(asset_source))
            ]

        shutil.copytree(asset_source, asset_destination, ignore=ignore)
        for path in sorted(asset_destination.rglob("*")):
            if path.is_file():
                if path.suffix.lower() == ".obj":
                    scale_obj(path, scale)
                elif path.suffix.lower() == ".xml":
                    scale_xml(path, scale)
        scale_glb(
            source / (name + ".glb"), destination / (name + ".glb"), scale
        )
        scale_blend(
            source / (name + ".blend"), destination / (name + ".blend"), scale
        )
    report = {
        "scales": SCALES,
        "capacities": CAPACITIES,
        "capacity_basis": "Labeled US capacity; collision cavity to lowest spill lip.",
        "mass_note": "Measured SKU mass is unknown; current assumed masses are unchanged.",
        "turner_note": "Scale exactly 1; no flex introduced.",
        "contact_note": "Only group 0 bottom_disk geoms receive priority 1 and solref 0.008 1.",
    }
    (destination / "calibration.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="ascii",
    )


if __name__ == "__main__":
    main()