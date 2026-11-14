import bpy
import sys
import math
import json
import struct
import zlib
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from mathutils import Vector
from mathutils.kdtree import KDTree

MATCH_TOLERANCE = 2.0e-6
CELL_SIZE = 4
ATLAS_NAME = "mug_glaze_atlas.png"
TEXTURE_NAME = "mug_vertex_glaze_texture"
MATERIAL_NAME = "mug_vertex_glaze_material"


def fail(message):
    raise RuntimeError(message)


def srgb_byte(value):
    value = max(0.0, min(1.0, float(value)))
    if value <= 0.0031308:
        encoded = 12.92 * value
    else:
        encoded = 1.055 * math.pow(value, 1.0 / 2.4) - 0.055
    return max(0, min(255, int(encoded * 255.0 + 0.5)))


def quantize(rgba):
    return (
        srgb_byte(rgba[0]),
        srgb_byte(rgba[1]),
        srgb_byte(rgba[2]),
        max(0, min(255, int(float(rgba[3]) * 255.0 + 0.5))),
    )


def png_chunk(kind, data):
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    )


def make_png(palette, columns, rows):
    width = columns * CELL_SIZE
    height = rows * CELL_SIZE
    scanlines = bytearray()
    fallback = palette[-1]
    for png_y in range(height):
        # PNG rows run downward; UV rows run upward.
        cell_y = (height - 1 - png_y) // CELL_SIZE
        scanlines.append(0)
        for cell_x in range(columns):
            index = cell_y * columns + cell_x
            color = palette[index] if index < len(palette) else fallback
            scanlines.extend(bytes(color) * CELL_SIZE)
    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", header)
        + png_chunk(b"sRGB", b"\x00")
        + png_chunk(b"IDAT", zlib.compress(bytes(scanlines), 9))
        + png_chunk(b"IEND", b"")
    )


def load_source(source):
    bpy.ops.wm.open_mainfile(filepath=str(source))
    positions = []
    colors = []
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        attribute = obj.data.color_attributes.get("Color")
        if attribute is None or attribute.domain != "POINT":
            fail("Missing POINT Color attribute on source mesh: " + obj.name)
        for vertex in obj.data.vertices:
            positions.append(obj.matrix_world @ vertex.co)
            # Read float linear colors directly, including alpha.
            colors.append(tuple(float(x) for x in attribute.data[vertex.index].color))
    if not positions:
        fail("No source mug vertices found")
    tree = KDTree(len(positions))
    for index, position in enumerate(positions):
        tree.insert(position, index)
    tree.balance()
    return tree, colors, len(positions)


def resolve_path(base, directory, filename, strip=False):
    filename = filename.replace("\\", "/")
    if strip:
        filename = filename.rsplit("/", 1)[-1]
    path = Path(filename)
    if path.is_absolute():
        return path.resolve()
    return (base / directory / path).resolve()


def effective_geom_attributes(element, inherited, defaults):
    class_name = element.get("class", inherited)
    result = dict(defaults.get(class_name, {}))
    result.update(element.attrib)
    return result


def geom_defaults(root):
    defaults = {"": {}}

    def visit(element, inherited):
        current = dict(inherited)
        geom = element.find("geom")
        if geom is not None:
            current.update(geom.attrib)
        name = element.get("class", "")
        defaults[name] = current
        for child in element.findall("default"):
            visit(child, current)

    for element in root.findall("default"):
        visit(element, {})
    return defaults


def gather_geoms(root, defaults):
    result = []

    def visit(element, inherited=""):
        if element.tag == "body":
            inherited = element.get("childclass", inherited)
        if element.tag == "geom":
            result.append((
                element,
                effective_geom_attributes(element, inherited, defaults),
            ))
        for child in element:
            visit(child, inherited)

    world = root.find("worldbody")
    if world is not None:
        visit(world)
    return result


def parse_obj(path, tree, source_colors):
    lines = path.read_text(encoding="utf-8").splitlines()
    vertices = []
    faces = []
    vertex_colors = []
    maximum = 0.0
    for line in lines:
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "v":
            if len(fields) < 4:
                fail("Invalid OBJ vertex in " + str(path))
            point = Vector(tuple(float(x) for x in fields[1:4]))
            _, source_index, distance = tree.find(point)
            maximum = max(maximum, distance)
            if distance > MATCH_TOLERANCE:
                fail(
                    "Mug vertex mismatch in %s: %.12g m exceeds %.12g m"
                    % (path, distance, MATCH_TOLERANCE)
                )
            vertices.append(point)
            vertex_colors.append(source_colors[source_index])
        elif fields[0] == "f":
            if len(fields) != 4:
                fail("Expected triangle faces in " + str(path))
            indices = []
            normals = []
            for token in fields[1:]:
                parts = token.split("/")
                index = int(parts[0])
                index = index - 1 if index > 0 else len(vertices) + index
                if index < 0 or index >= len(vertices):
                    fail("Invalid OBJ face index in " + str(path))
                indices.append(index)
                normals.append(parts[2] if len(parts) > 2 else "")
            faces.append((indices, normals))
    if not vertices or not faces:
        fail("Empty visual OBJ: " + str(path))
    face_colors = []
    for indices, _ in faces:
        average = tuple(
            sum(vertex_colors[index][channel] for index in indices) / 3.0
            for channel in range(4)
        )
        face_colors.append(quantize(average))
    return {
        "path": path,
        "lines": lines,
        "vertices": len(vertices),
        "faces": faces,
        "colors": face_colors,
        "max_distance_m": maximum,
    }


def rewrite_obj(record, palette_indices, columns, rows):
    # Separate UVs per face prevent interpolation between palette cells.
    # All three corners of each triangle sample exactly the same cell center.
    uv_lines = []
    for color in record["colors"]:
        index = palette_indices[color]
        u = ((index % columns) + 0.5) / columns
        v = ((index // columns) + 0.5) / rows
        uv_lines.append("vt %.12g %.12g" % (u, v))
    output = []
    inserted = False
    face_index = 0
    for line in record["lines"]:
        fields = line.split()
        if fields and fields[0] == "vt":
            continue
        if fields and fields[0] == "f":
            if not inserted:
                output.extend(uv_lines)
                inserted = True
            indices, normals = record["faces"][face_index]
            uv_index = face_index + 1
            tokens = []
            for index, normal in zip(indices, normals):
                token = "%d/%d" % (index + 1, uv_index)
                if normal:
                    token += "/" + normal
                tokens.append(token)
            output.append("f " + " ".join(tokens))
            face_index += 1
        else:
            output.append(line)
    return "\n".join(output) + "\n"


def upsert(asset, tag, name):
    matches = [node for node in asset.findall(tag) if node.get("name") == name]
    if len(matches) > 1:
        fail("Duplicate MJCF asset: " + name)
    return matches[0] if matches else ET.SubElement(asset, tag, {"name": name})


def numbers(text):
    try:
        return tuple(float(value) for value in text.split())
    except (ValueError, AttributeError):
        return ()


def is_silver(rgba):
    return len(rgba) >= 3 and all(
        abs(rgba[i] - target) <= 0.065
        for i, target in enumerate((0.66, 0.69, 0.72))
    )


def main():
    arguments = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if len(arguments) != 1:
        fail("Usage: blender -b --python kitchen_surface.py -- OUTPUT_DIR")
    output_dir = Path(arguments[0]).resolve()
    source = output_dir / "mug.blend"
    sim_dir = output_dir / "sim"
    if not source.is_file() or not sim_dir.is_dir():
        fail("Expected OUTPUT_DIR/mug.blend and OUTPUT_DIR/sim")
    tree, source_colors, source_count = load_source(source)
    documents = []
    selected_paths = set()
    other_paths = set()

    for xml_path in sorted(sim_dir.rglob("*.xml")):
        parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
        document = ET.parse(xml_path, parser=parser)
        root = document.getroot()
        if root.tag != "mujoco":
            continue
        compiler = root.find("compiler")
        compiler_attributes = compiler.attrib if compiler is not None else {}
        assetdir = compiler_attributes.get("assetdir", "")
        meshdir = compiler_attributes.get("meshdir", assetdir)
        texturedir = compiler_attributes.get("texturedir", assetdir)
        strip = compiler_attributes.get("strippath", "false").lower() == "true"
        assets = {}
        for mesh in root.findall("./asset/mesh"):
            filename = mesh.get("file")
            if filename:
                assets[mesh.get("name", Path(filename).stem)] = (
                    mesh,
                    resolve_path(xml_path.parent, meshdir, filename, strip),
                )
        defaults = geom_defaults(root)
        geoms = gather_geoms(root, defaults)
        mug_geoms = []
        for geom, attributes in geoms:
            mesh_name = attributes.get("mesh")
            if mesh_name not in assets:
                continue
            _, path = assets[mesh_name]
            visual = (
                attributes.get("type", "mesh") == "mesh"
                and int(attributes.get("group", "0")) == 1
            )
            identifier = " ".join((
                mesh_name,
                geom.get("name", ""),
                path.as_posix(),
            )).lower()
            mug = "mug" in identifier
            if visual and mug:
                if path.suffix.lower() != ".obj":
                    fail("Mug visual mesh is not OBJ: " + str(path))
                if not path.is_relative_to(sim_dir):
                    fail("Refusing to rewrite mesh outside sim directory: " + str(path))
                selected_paths.add(path)
                mug_geoms.append(geom)
            else:
                other_paths.add(path)
        documents.append({
            "path": xml_path,
            "tree": document,
            "root": root,
            "geoms": geoms,
            "mug_geoms": mug_geoms,
            "texturedir": texturedir,
            "strip": strip,
        })

    if not selected_paths:
        fail("No group 1 mug visual OBJ meshes found")
    shared = selected_paths & other_paths
    if shared:
        fail("Visual OBJ is also referenced by non-mug or collision geometry: "
             + ", ".join(str(path) for path in sorted(shared)))

    records = [
        parse_obj(path, tree, source_colors)
        for path in sorted(selected_paths)
    ]
    palette = sorted(
        {color for record in records for color in record["colors"]},
        key=lambda color: (
            0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2],
            color,
        ),
    )
    columns = max(1, math.ceil(math.sqrt(len(palette))))
    rows = math.ceil(len(palette) / columns)
    palette_indices = {color: index for index, color in enumerate(palette)}
    png_bytes = make_png(palette, columns, rows)
    pending_objs = [
        (record["path"], rewrite_obj(record, palette_indices, columns, rows))
        for record in records
    ]
    atlas_paths = set()
    modified_documents = []
    metal_count = 0
    grip_count = 0
    mug_geom_count = 0

    for info in documents:
        root = info["root"]
        changed = False
        asset = root.find("asset")
        if asset is None:
            asset = ET.SubElement(root, "asset")
        materials = {
            material.get("name"): material
            for material in asset.findall("material")
        }
        if info["mug_geoms"]:
            atlas_path = resolve_path(
                info["path"].parent, info["texturedir"], ATLAS_NAME, info["strip"]
            )
            if not atlas_path.is_relative_to(sim_dir):
                fail("Atlas destination lies outside sim directory: " + str(atlas_path))
            atlas_paths.add(atlas_path)
            texture = upsert(asset, "texture", TEXTURE_NAME)
            texture.attrib.clear()
            texture.attrib.update({
                "name": TEXTURE_NAME,
                "type": "2d",
                "file": ATLAS_NAME,
            })
            material = upsert(asset, "material", MATERIAL_NAME)
            material.attrib.clear()
            material.attrib.update({
                "name": MATERIAL_NAME,
                "texture": TEXTURE_NAME,
                "texrepeat": "1 1",
                "texuniform": "false",
                "rgba": "1 1 1 1",
                "specular": ".65",
                "shininess": ".55",
                "reflectance": "0",
            })
            for geom in info["mug_geoms"]:
                geom.set("material", MATERIAL_NAME)
                geom.set("rgba", "1 1 1 1")
                mug_geom_count += 1
            changed = True

        for geom, attributes in info["geoms"]:
            if geom in info["mug_geoms"]:
                continue
            if int(attributes.get("group", "0")) != 1:
                continue
            original_material = materials.get(attributes.get("material"))
            rgba = numbers(attributes.get("rgba"))
            if not rgba and original_material is not None:
                rgba = numbers(original_material.get("rgba", "1 1 1 1"))
            identity = " ".join((
                attributes.get("name", ""),
                attributes.get("mesh", ""),
                attributes.get("material", ""),
            )).lower()
            tokens = set(re.findall(r"[a-z]+", identity))
            silver = is_silver(rgba) or (
                bool(tokens & {"scoop", "blade"})
                and not bool(tokens & {"grip", "handle", "black"})
            )
            grip = (
                bool(tokens & {"grip", "handle"})
                and (
                    "black" in tokens
                    or (len(rgba) >= 3 and max(rgba[:3]) < 0.18)
                )
            )
            if silver:
                name = "kitchen_brushed_silver"
                settings = {
                    "rgba": ".66 .69 .72 1",
                    "specular": ".85",
                    "shininess": ".72",
                    "reflectance": "0",
                }
                metal_count += 1
            elif grip:
                name = "kitchen_black_grip"
                settings = {
                    "rgba": ".035 .035 .04 1",
                    "specular": ".12",
                    "shininess": ".18",
                    "reflectance": "0",
                }
                grip_count += 1
            else:
                continue
            material = upsert(asset, "material", name)
            material.attrib.clear()
            material.attrib.update({"name": name, **settings})
            geom.set("material", name)
            geom.set("rgba", "1 1 1 1")
            changed = True
        if changed:
            modified_documents.append(info)

    # All source matching and validation complete before modifying any outputs.
    for path in sorted(atlas_paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(png_bytes)
    for path, content in pending_objs:
        path.write_text(content, encoding="ascii")
    for info in modified_documents:
        info["tree"].write(
            info["path"], encoding="utf-8", xml_declaration=True
        )

    manifest = {
        "source_blend": str(source),
        "source_vertices": source_count,
        "match_tolerance_m": MATCH_TOLERANCE,
        "maximum_match_distance_m": max(
            record["max_distance_m"] for record in records
        ),
        "meshes_transferred": len(records),
        "vertices_matched": sum(record["vertices"] for record in records),
        "faces_transferred": sum(len(record["faces"]) for record in records),
        "palette_colors": len(palette),
        "atlas_size_pixels": [columns * CELL_SIZE, rows * CELL_SIZE],
        "palette_cell_size_pixels": CELL_SIZE,
        "sampling": "linear face-average RGB encoded to sRGB; constant UV per face",
        "atlases": [str(path) for path in sorted(atlas_paths)],
        "mjcf_files": [str(info["path"]) for info in modified_documents],
        "mug_visual_geoms": mug_geom_count,
        "silver_visual_geoms": metal_count,
        "black_grip_visual_geoms": grip_count,
        "meshes": [
            {
                "file": str(record["path"]),
                "vertices": record["vertices"],
                "faces_transferred": len(record["faces"]),
                "maximum_match_distance_m": record["max_distance_m"],
            }
            for record in records
        ],
    }
    manifest_path = sim_dir / "kitchen_surface_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
