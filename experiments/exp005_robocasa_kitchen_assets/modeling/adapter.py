import json
import math
import os
import stat
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


DISPLAY_NAMES = {
    "mug": "Mug",
    "scoop": "Scoop",
    "turner": "Turner",
}


def check_path(path):
    path = Path(os.path.abspath(path))
    for part in reversed((path, *path.parents)):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or (
            getattr(info, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        ):
            raise ValueError("Symlinks and reparse points are not allowed: " + str(part))
    return path


def numbers(value, count=None):
    result = [float(item) for item in value.split()]
    if count is not None and len(result) != count:
        raise ValueError("Unexpected number of components: " + value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError("Non-finite numeric attribute: " + value)
    return result


def formatted(values):
    return " ".join(format(value if value else 0.0, ".17g") for value in values)


def capsule_pose(endpoints):
    start = endpoints[:3]
    end = endpoints[3:]
    delta = [b - a for a, b in zip(start, end)]
    length = math.hypot(*delta)
    if not math.isfinite(length) or length <= 0:
        raise ValueError("Capsule segment must have finite, positive length")

    position = [a * 0.5 + b * 0.5 for a, b in zip(start, end)]
    x, y, z = [component / length for component in delta]
    transverse = math.hypot(x, y)

    if transverse == 0:
        quaternion = [1.0, 0.0, 0.0, 0.0] if z >= 0 else [0.0, 1.0, 0.0, 0.0]
    else:
        angle = math.atan2(transverse, z)
        sine = math.sin(angle * 0.5)
        quaternion = [
            math.cos(angle * 0.5),
            -(y / transverse) * sine,
            (x / transverse) * sine,
            0.0,
        ]

    return position, quaternion, length * 0.5


def repair_xml(path):
    parser = ET.XMLParser(
        target=ET.TreeBuilder(insert_comments=True, insert_pis=True)
    )
    tree = ET.parse(path, parser=parser)
    changed = False

    for geom in tree.iter("geom"):
        if (
            geom.get("type") == "capsule"
            and geom.get("fromto") is not None
        ):
            sizes = numbers(geom.get("size", ""))
            if len(sizes) not in (1, 2) or sizes[0] <= 0:
                raise ValueError("Capsule requires a positive radius: " + str(path))

            position, quaternion, half_length = capsule_pose(
                numbers(geom.attrib["fromto"], 6)
            )
            del geom.attrib["fromto"]
            for attribute in ("axisangle", "euler", "xyaxes", "zaxis"):
                geom.attrib.pop(attribute, None)
            geom.set("pos", formatted(position))
            geom.set("quat", formatted(quaternion))
            geom.set("size", formatted([sizes[0], half_length]))
            changed = True

        if (
            geom.get("type") == "mesh" or geom.get("mesh") is not None
        ) and geom.get("pos") is None:
            geom.set("pos", "0 0 0")
            changed = True

    if not changed:
        return None
    return ET.tostring(tree.getroot(), encoding="utf-8", xml_declaration=True)


def atomic_write(path, content):
    check_path(path)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".adapter-", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        check_path(path)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    if len(sys.argv) != 2:
        raise ValueError("Usage: python adapter.py OUTPUT_DIR")

    supplied = Path(sys.argv[1])
    if ".." in supplied.parts:
        raise ValueError("Parent traversal is not allowed")
    root = check_path(supplied)
    if not root.is_dir():
        raise ValueError("OUTPUT_DIR must be an existing directory")

    pending = []
    for name, display_name in DISPLAY_NAMES.items():
        folder = check_path(root / "sim" / name)
        if not folder.is_dir():
            raise ValueError("Missing asset directory: " + str(folder))

        model = check_path(folder / "model.xml")
        if not model.is_file():
            raise ValueError("Missing model.xml: " + str(model))
        repaired = repair_xml(model)
        if repaired is not None:
            pending.append((model, repaired))

        metadata_path = check_path(folder / "metadata.json")
        metadata = {}
        if metadata_path.exists():
            if not metadata_path.is_file():
                raise ValueError("Metadata must be a regular file")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
            if not isinstance(metadata, dict):
                raise ValueError("Metadata must contain a JSON object")

        if metadata.get("display_name") != display_name:
            metadata["display_name"] = display_name
            content = (
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
            ).encode("utf-8")
            pending.append((metadata_path, content))

    for path, content in pending:
        atomic_write(path, content)


if __name__ == "__main__":
    main()