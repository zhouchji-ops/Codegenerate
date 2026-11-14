import os
import re
import sys

_wrapper_file = os.path.abspath(__file__)
_original_argv = sys.argv[:]
_source_file = os.path.join(os.path.dirname(_wrapper_file), "collision.py")

with open(_source_file, "r", encoding="utf-8") as _stream:
    _source = _stream.read()

_replacement = r'''def tube_rings(name):
    import math
    import bpy
    from mathutils import Vector

    asset = str(name).casefold()
    if "tubular_d_handle" in asset or "scoop" in asset:
        token = "tubular_d_handle"
    elif "loop_handle" in asset or "mug" in asset or "cup" in asset:
        token = "loop_handle"
    else:
        raise ValueError("Unknown tube asset: %r" % (name,))

    matches = [
        obj for obj in bpy.context.scene.objects
        if obj.type == "MESH" and token in obj.name.casefold()
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one mesh containing %r; found %r"
            % (token, [obj.name for obj in matches])
        )

    obj = matches[0]
    count = len(obj.data.vertices)
    if count % 40 != 0 or count < 80:
        raise ValueError(
            "%s: expected complete rings of 40 vertices, got %d"
            % (obj.name, count)
        )

    points = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
    if any(not math.isfinite(value) for point in points for value in point):
        raise ValueError("%s: non-finite world coordinates" % obj.name)

    centers = []
    radii = []
    for start in range(0, count, 40):
        ring = points[start:start + 40]
        center = sum(ring, Vector((0, 0, 0))) / 40.0
        radius = sum((point - center).length for point in ring) / 40.0
        if not math.isfinite(radius) or not 1.0e-5 < radius < 1.0:
            raise ValueError(
                "%s: implausible world-space ring radius %r"
                % (obj.name, radius)
            )
        centers.append(center)
        radii.append(radius)

    lengths = [
        (centers[index + 1] - centers[index]).length
        for index in range(len(centers) - 1)
    ]
    total_length = sum(lengths)
    if (
        any(not math.isfinite(length) or length <= 1.0e-8
            for length in lengths)
        or not 1.0e-4 < total_length < 10.0
    ):
        raise ValueError(
            "%s: implausible world-space tube lengths %r"
            % (obj.name, lengths)
        )

    return centers, radii


def _simfix_convex_hull(vertices):
    import math
    import bmesh
    from mathutils import Vector

    coordinates = sorted(set(tuple(float(x) for x in v) for v in vertices))
    if len(coordinates) < 4:
        raise ValueError("A convex hull requires four distinct vertices")
    if any(len(v) != 3 or not all(math.isfinite(x) for x in v)
           for v in coordinates):
        raise ValueError("Invalid convex hull coordinates")

    bm = bmesh.new()
    try:
        input_vertices = [bm.verts.new(v) for v in coordinates]
        bmesh.ops.convex_hull(
            bm, input=input_vertices, use_existing_faces=False
        )
        hull_faces = list(bm.faces)
        if not hull_faces:
            raise ValueError("Convex hull has no faces")

        bmesh.ops.triangulate(
            bm,
            faces=hull_faces,
            quad_method="FIXED",
            ngon_method="EAR_CLIP",
        )

        faces = list(bm.faces)
        used = {v for face in faces for v in face.verts}
        ordered = sorted(used, key=lambda v: tuple(v.co))
        output_vertices = [tuple(float(x) for x in v.co) for v in ordered]
        indices = {v: index for index, v in enumerate(ordered)}
        center = sum(
            (Vector(v) for v in output_vertices), Vector((0, 0, 0))
        ) / len(output_vertices)

        triangles = set()
        for face in faces:
            if len(face.verts) != 3:
                raise ValueError("Convex hull triangulation failed")
            triangle = [indices[v] for v in face.verts]
            a, b, c = [Vector(output_vertices[i]) for i in triangle]
            normal = (b - a).cross(c - a)
            if normal.length_squared == 0:
                raise ValueError("Degenerate convex hull triangle")
            if normal.dot((a + b + c) / 3.0 - center) < 0:
                triangle[1], triangle[2] = triangle[2], triangle[1]
            first = triangle.index(min(triangle))
            triangle = triangle[first:] + triangle[:first]
            triangles.add(tuple(triangle))

        output_faces = sorted(triangles)
        edges = {}
        volume6 = 0.0
        for triangle in output_faces:
            a, b, c = [
                Vector(output_vertices[i]) - center for i in triangle
            ]
            volume6 += a.dot(b.cross(c))
            for i in range(3):
                edge = tuple(sorted((triangle[i], triangle[(i + 1) % 3])))
                edges[edge] = edges.get(edge, 0) + 1

        if not output_faces or any(n != 2 for n in edges.values()):
            raise ValueError("Convex hull is not a closed triangular surface")
        if not math.isfinite(volume6) or volume6 <= 0:
            raise ValueError("Convex hull must have positive volume")

        return output_vertices, output_faces
    finally:
        bm.free()


'''

_start = _source.find("def tube_rings(name):")
_end = _source.find("def vessel(export):", _start + 1)
if _start < 0 or _end < 0:
    raise RuntimeError("Cannot locate tube_rings replacement boundaries")
_source = _source[:_start] + _replacement + _source[_end:]

_hull_pattern = re.compile(
    r"^(?P<indent>[ \t]*)key = self\.mesh\(label, vertices, \[\]\)[ \t]*$",
    re.MULTILINE,
)

def _replace_hull(match):
    indent = match.group("indent")
    return (
        indent + "_hull_vertices, _hull_faces = "
        "_simfix_convex_hull(vertices)\n"
        + indent + "key = self.mesh(label, _hull_vertices, _hull_faces)"
    )

_source, _count = _hull_pattern.subn(_replace_hull, _source)
if _count != 1:
    raise RuntimeError("Expected exactly one point-only hull export")

_source, _count = re.subn(
    r"(?<![\d.])0\.45[ \t]+0\.55[ \t]+0\.65[ \t]+0(?![\d.])",
    "0.45 0.55 0.65 1",
    _source,
)
if _count == 0:
    raise RuntimeError("Cannot locate transparent collision geom color")

_source, _count = re.subn(
    r"""name\s*=\s*self\.name\s*\+\s*(['"])_reg_bbox\1""",
    'name="reg_bbox"',
    _source,
)
if _count == 0:
    raise RuntimeError("Cannot locate reg_bbox name")

sys.argv[:] = _original_argv
_namespace = {
    "__name__": "__main__",
    "__file__": _source_file,
    "__package__": None,
    "__cached__": None,
}
exec(compile(_source, _source_file, "exec"), _namespace, _namespace)