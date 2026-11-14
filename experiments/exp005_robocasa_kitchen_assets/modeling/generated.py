import bpy
import bmesh
import math
import os
import sys
from mathutils import Vector
from mathutils.geometry import tessellate_polygon

# Blender 4.5. All dimensions are meters.
args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
if not args:
    raise RuntimeError("Pass an output directory after --")
output_dir = os.path.abspath(args[0])
os.makedirs(output_dir, exist_ok=True)

TAU = math.tau


def reset_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in list(bpy.data.materials):
        bpy.data.materials.remove(block)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0


def make_mesh(name, vertices, faces, material=None):
    mesh = bpy.data.meshes.new(name + "_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    if material:
        mesh.materials.append(material)
    return obj


def activate(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def apply_modifier(obj, modifier):
    activate(obj)
    bpy.ops.object.modifier_apply(modifier=modifier.name)


def bevel(obj, width, segments=4, angle=0.45):
    mod = obj.modifiers.new("Machined edge radii", "BEVEL")
    mod.width = width
    mod.segments = segments
    mod.limit_method = "ANGLE"
    mod.angle_limit = angle
    apply_modifier(obj, mod)


def shade(obj, weighted=False):
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    if weighted:
        mod = obj.modifiers.new("Surface normals", "WEIGHTED_NORMAL")
        mod.keep_sharp = True
        mod.weight = 35
        apply_modifier(obj, mod)


def pbr(name, color, metallic=0.0, roughness=0.25):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    mat.diffuse_color = (*color, 1)
    return mat


def fine_finish(mat, roughness, scale, strength, anisotropy=0.0):
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Anisotropic IOR Level"].default_value = anisotropy
    tex = nodes.new("ShaderNodeTexNoise")
    tex.inputs["Scale"].default_value = scale
    tex.inputs["Detail"].default_value = 2.0
    coord = nodes.new("ShaderNodeTexCoord")
    links.new(coord.outputs["Generated"], tex.inputs["Vector"])
    ramp = nodes.new("ShaderNodeMapRange")
    ramp.inputs["From Min"].default_value = 0
    ramp.inputs["From Max"].default_value = 1
    ramp.inputs["To Min"].default_value = roughness - strength
    ramp.inputs["To Max"].default_value = roughness + strength
    links.new(tex.outputs["Fac"], ramp.inputs["Value"])
    links.new(ramp.outputs["Result"], bsdf.inputs["Roughness"])


def srgb(value):
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def ceramic_colors(obj, interior=False):
    colors = obj.data.color_attributes.new(name="Color", type="FLOAT_COLOR", domain="POINT")
    low = (0.69, 0.055, 0.012)
    high = (1.0, 0.345, 0.035)
    cream = tuple(srgb(c) for c in (0.96, 0.925, 0.82))
    for vertex, entry in zip(obj.data.vertices, colors.data):
        if interior:
            color = cream
        else:
            t = max(0.0, min(1.0, vertex.co.z / 0.09))
            t = t * t * (3 - 2 * t)
            color = tuple(srgb(low[k] * (1 - t) + high[k] * t) for k in range(3))
        entry.color = (*color, 1)
    obj.data.color_attributes.active_color = colors


def ceramic_material():
    mat = pbr("Fired orange ceramic glaze", (0.8, 0.12, 0.01), roughness=0.19)
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Coat Weight"].default_value = 0.30
    bsdf.inputs["Coat Roughness"].default_value = 0.15
    attr = mat.node_tree.nodes.new("ShaderNodeVertexColor")
    attr.layer_name = "Color"
    mat.node_tree.links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def lathe(name, profile, material, segments=192):
    # Profile starts and ends on the axis. No coincident pole rings.
    vertices = []
    rings = []
    for radius, height in profile:
        if abs(radius) < 1e-10:
            rings.append([len(vertices)])
            vertices.append((0, 0, height))
        else:
            ring = []
            for j in range(segments):
                a = TAU * j / segments
                ring.append(len(vertices))
                vertices.append((radius * math.cos(a), radius * math.sin(a), height))
            rings.append(ring)
    faces = []
    for a, b in zip(rings[:-1], rings[1:]):
        for j in range(segments):
            k = (j + 1) % segments
            if len(a) == 1:
                faces.append((a[0], b[k], b[j]))
            elif len(b) == 1:
                faces.append((a[j], a[k], b[0]))
            else:
                faces.append((a[j], a[k], b[k], b[j]))
    return make_mesh(name, vertices, faces, material)


def bezier_points(p0, p1, p2, p3, count=32):
    result = []
    for i in range(count):
        t = i / (count - 1)
        p = (Vector(p0) * (1 - t) ** 3 +
             Vector(p1) * (3 * (1 - t) ** 2 * t) +
             Vector(p2) * (3 * (1 - t) * t * t) +
             Vector(p3) * t ** 3)
        result.append(p)
    return result


def tube(name, points, radius, material, sides=32, flatten=1.0):
    vertices = []
    for i, point in enumerate(points):
        tangent = points[min(i + 1, len(points) - 1)] - points[max(i - 1, 0)]
        tangent.normalize()
        across = Vector((0, 1, 0))
        other = tangent.cross(across).normalized()
        r = radius(i / (len(points) - 1)) if callable(radius) else radius
        for j in range(sides):
            a = TAU * j / sides
            v = point + across * (r * math.cos(a) * flatten) + other * (r * math.sin(a))
            vertices.append(tuple(v))
    faces = []
    for i in range(len(points) - 1):
        for j in range(sides):
            k = (j + 1) % sides
            faces.append((i * sides + j, i * sides + k,
                          (i + 1) * sides + k, (i + 1) * sides + j))
    faces.append(tuple(reversed(range(sides))))
    faces.append(tuple((len(points) - 1) * sides + j for j in range(sides)))
    obj = make_mesh(name, vertices, faces, material)
    shade(obj)
    return obj


def ellipsoid(name, center, size, material):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, location=center)
    obj = bpy.context.object
    obj.name = name
    obj.scale = size
    activate(obj)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)
    shade(obj)
    return obj


def prism(name, outline, bottom, top, material=None):
    n = len(outline)
    vertices = [(x, y, bottom) for x, y in outline] + [(x, y, top) for x, y in outline]
    vectors = [Vector((x, y, 0)) for x, y in outline]
    lookup = {tuple(v): i for i, v in enumerate(vectors)}
    triangles = tessellate_polygon([vectors])
    faces = []
    for triangle in triangles:
        ids = [lookup[tuple(v)] for v in triangle]
        faces.append(tuple(reversed(ids)))
        faces.append(tuple(i + n for i in ids))
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, j + n, i + n))
    return make_mesh(name, vertices, faces, material)


def capsule(a, b, radius, steps=24):
    a, b = Vector(a), Vector(b)
    angle = math.atan2(b.y - a.y, b.x - a.x)
    outline = []
    for i in range(steps + 1):
        theta = angle - math.pi / 2 + math.pi * i / steps
        outline.append((b.x + radius * math.cos(theta), b.y + radius * math.sin(theta)))
    for i in range(steps + 1):
        theta = angle + math.pi / 2 + math.pi * i / steps
        outline.append((a.x + radius * math.cos(theta), a.y + radius * math.sin(theta)))
    return outline


def cut(obj, cutter):
    mod = obj.modifiers.new("Through opening", "BOOLEAN")
    mod.operation = "DIFFERENCE"
    mod.solver = "EXACT"
    mod.object = cutter
    apply_modifier(obj, mod)
    bpy.data.objects.remove(cutter, do_unlink=True)


def curve_outline(points, samples=8):
    # Closed Catmull-Rom spline.
    result = []
    n = len(points)
    for i in range(n):
        p0 = Vector(points[(i - 1) % n])
        p1 = Vector(points[i])
        p2 = Vector(points[(i + 1) % n])
        p3 = Vector(points[(i + 2) % n])
        for j in range(samples):
            t = j / samples
            v = 0.5 * ((2 * p1) + (-p0 + p2) * t +
                       (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t +
                       (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t)
            result.append((v.x, v.y))
    return result


def validate_and_export(name):
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    for obj in meshes:
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        if any(not edge.is_manifold for edge in bm.edges):
            bm.free()
            raise RuntimeError("Non-manifold mesh: " + obj.name)
        bm.free()
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.context.scene["asset_units"] = "meters"
    bpy.context.scene["asset_name"] = name
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(output_dir, name + ".blend"))
    bpy.ops.export_scene.gltf(
        filepath=os.path.join(output_dir, name + ".glb"),
        export_format="GLB",
        use_selection=True,
        export_apply=True,
        export_materials="EXPORT",
        export_normals=True,
        export_texcoords=True,
        export_attributes=True,
        export_all_vertex_colors=True,
        export_active_vertex_color_when_no_material=True,
        export_cameras=False,
        export_lights=False,
        export_yup=True
    )


# ---------------------------------------------------------------------------
# Mug: integral ceramic wall, rolled rim, cream interior and molded foot bands.
# ---------------------------------------------------------------------------
reset_scene()
glaze = ceramic_material()
profile = [
    (0, 0.0032), (0.023, 0.0032), (0.027, 0.0030),
    (0.0285, 0.0010), (0.0292, 0.0002), (0.0305, 0.0000),
    (0.0316, 0.0004), (0.0323, 0.0014), (0.0327, 0.0030)
]
for i in range(1, 97):
    z = 0.003 + 0.015 * i / 96
    base = 0.0327 + 0.0006 * (z - 0.003) / 0.015
    ridges = sum(0.00072 * math.exp(-((z - c) / 0.00080) ** 2)
                 for c in (0.0052, 0.0094, 0.0136))
    profile.append((base + ridges, z))
for i in range(1, 81):
    t = i / 80
    profile.append((0.0333 + 0.0049 * t + 0.00025 * t * t, 0.018 + 0.0695 * t))
# Round over a substantial ceramic lip.
rim_center = 0.03710
rim_radius = 0.00135
for i in range(1, 33):
    angle = math.pi * i / 32
    profile.append((rim_center + rim_radius * math.cos(angle),
                    0.0875 + rim_radius * math.sin(angle)))
inner_start = len(profile) - 17
for i in range(1, 89):
    t = i / 88
    profile.append((0.03575 - 0.00625 * t, 0.0875 - 0.0735 * t))
for i in range(1, 25):
    a = -math.pi / 2 * i / 24
    profile.append((0.0245 + 0.005 * math.cos(a), 0.014 + 0.005 * math.sin(a)))
profile.extend([(0.020, 0.009), (0.010, 0.009), (0, 0.009)])
mug = lathe("Mug_integral_wall_and_foot", profile, glaze)
ceramic_colors(mug)
cream = tuple(srgb(c) for c in (0.96, 0.925, 0.82))
# Profile rings use one vertex at each pole.
for ring_index in range(inner_start, len(profile)):
    start = 1 + (ring_index - 1) * 192
    count = 1 if profile[ring_index][0] == 0 else 192
    for index in range(start, min(start + count, len(mug.data.vertices))):
        mug.data.color_attributes["Color"].data[index].color = (*cream, 1)
shade(mug)

upper = bezier_points((0.0350, 0, 0.071), (0.056, 0, 0.083),
                      (0.071, 0, 0.070), (0.069, 0, 0.052), 48)
lower = bezier_points((0.069, 0, 0.052), (0.068, 0, 0.035),
                      (0.049, 0, 0.023), (0.0327, 0, 0.027), 48)
handle = tube("Mug_elliptic_loop_handle", upper + lower[1:],
              lambda t: 0.0044 + 0.00045 * math.sin(math.pi * t),
              glaze, 40, 0.91)
ceramic_colors(handle)
for name, center, size in [
    ("Mug_upper_soft_lug", (0.0356, 0, 0.0715), (0.0045, 0.0054, 0.0055)),
    ("Mug_lower_soft_lug", (0.0330, 0, 0.0275), (0.0045, 0.0053, 0.0048))
]:
    lug = ellipsoid(name, center, size, glaze)
    ceramic_colors(lug)
validate_and_export("mug")

# ---------------------------------------------------------------------------
# Scoop: cylindrical closed shell with an oblique open mouth and rolled edge.
# ---------------------------------------------------------------------------
reset_scene()
silver = pbr("Satin brushed silver", (0.67, 0.69, 0.70), 1, 0.29)
fine_finish(silver, 0.29, 175, 0.025, 0.35)
# Coordinates are radius, absolute height, and contribution from oblique rim.
scoop_profile = [
    (0, 0.0018, 0), (0.020, 0.0018, 0),
    (0.0250, 0.0018, 0), (0.0265, 0.0021, 0),
    (0.0272, 0.0028, 0), (0.0275, 0.0040, 0),
    (0.0275, 0.0100, 0),
    (0.0275, -0.0010, 1)
]
# The rim center follows a plane. Its small circular section is physically solid.
for i in range(25):
    a = math.pi * i / 24
    scoop_profile.append((0.0269 + 0.0006 * math.cos(a),
                          -0.0008 + 0.0006 * math.sin(a), 1))
scoop_profile.extend([
    (0.0263, -0.0020, 1), (0.0263, 0.0060, 0),
    (0.02615, 0.0047, 0), (0.0257, 0.0037, 0),
    (0.0246, 0.0031, 0), (0.020, 0.0030, 0), (0, 0.0030, 0)
])
vertices = []
rings = []
segments = 224
for radius, offset, factor in scoop_profile:
    if radius == 0:
        rings.append([len(vertices)])
        vertices.append((0, 0, offset))
    else:
        ring = []
        for j in range(segments):
            a = TAU * j / segments
            rim_z = 0.094 - 0.026 * (radius / 0.0275) * math.cos(a)
            ring.append(len(vertices))
            vertices.append((radius * math.cos(a), radius * math.sin(a),
                             offset + factor * rim_z))
        rings.append(ring)
faces = []
for a, b in zip(rings[:-1], rings[1:]):
    for j in range(segments):
        k = (j + 1) % segments
        if len(a) == 1:
            faces.append((a[0], b[k], b[j]))
        elif len(b) == 1:
            faces.append((a[j], a[k], b[0]))
        else:
            faces.append((a[j], a[k], b[k], b[j]))
scoop = make_mesh("Scoop_oblique_integral_shell", vertices, faces, silver)
shade(scoop)
path1 = bezier_points((0.0268, 0, 0.064), (0.043, 0, 0.064),
                      (0.053, 0, 0.060), (0.053, 0, 0.043), 42)
path2 = bezier_points((0.053, 0, 0.043), (0.053, 0, 0.023),
                      (0.043, 0, 0.019), (0.0268, 0, 0.019), 42)
tube("Scoop_tubular_D_handle", path1 + path2[1:], 0.0028, silver, 40)
for z in (0.064, 0.019):
    ellipsoid("Scoop_welded_handle_seat", (0.0275, 0, z),
              (0.0013, 0.0035, 0.0036), silver)
validate_and_export("scoop")

# ---------------------------------------------------------------------------
# Turner: a thin slanted fan, six rounded through slots and a bent metal tang.
# ---------------------------------------------------------------------------
reset_scene()
steel = pbr("Champagne stainless", (0.65, 0.62, 0.54), 1, 0.23)
fine_finish(steel, 0.23, 210, 0.018, 0.25)
rubber = pbr("Black molded elastomer", (0.014, 0.016, 0.017), 0, 0.38)
fine_finish(rubber, 0.38, 130, 0.018)
rubber.node_tree.nodes.get("Principled BSDF").inputs["Specular IOR Level"].default_value = 0.30

head_outline = curve_outline([
    (-0.190, -0.032), (-0.188, -0.038), (-0.174, -0.0378),
    (-0.145, -0.0297), (-0.111, -0.0214), (-0.069, -0.0127),
    (-0.056, -0.0082), (-0.053, 0.0000), (-0.059, 0.0095),
    (-0.088, 0.0160), (-0.125, 0.0250), (-0.161, 0.0356),
    (-0.178, 0.0400), (-0.184, 0.0360), (-0.187, 0.0120)
], 8)
blade = prism("Turner_continuous_slotted_blade", head_outline, 0.0001, 0.0013, steel)

# Fan slots stop short of both ends, leaving a connected perimeter and ribs.
for i in range(6):
    f = (i + 0.5) / 6
    y_far = -0.029 + f * 0.060
    y_near = -0.0092 + f * 0.0184
    x_far = -0.178 + 0.003 * f
    x_near = -0.072 - 0.004 * abs(2 * f - 1)
    slot = prism("Temporary_slot", capsule((x_far, y_far), (x_near, y_near),
                                          0.00155, 20), -0.01, 0.02)
    cut(blade, slot)
bevel(blade, 0.00020, 3, 0.40)

# Deform the actual solid so that the far end lifts subtly from the worktop.
def blade_height(x):
    t = max(0.0, min(1.0, (-x - 0.055) / 0.135))
    return 0.001 + 0.0085 * t * t

for vertex in blade.data.vertices:
    vertex.co.z += blade_height(vertex.co.x)
blade.data.update()
shade(blade, True)

# Curved rectangular metal neck, embedded in the blade and elastomer grip.
neck_stations = [
    (-0.061, 0.0018, 0.0065),
    (-0.052, 0.0017, 0.0060),
    (-0.043, 0.0025, 0.0053),
    (-0.034, 0.0053, 0.0048),
    (-0.022, 0.0083, 0.0048),
    (-0.008, 0.0090, 0.0052),
    (0.012, 0.0090, 0.0052)
]
vertices = []
for x, z, width in neck_stations:
    vertices.extend([(x, -width, z - 0.0007), (x, width, z - 0.0007),
                     (x, width, z + 0.0007), (x, -width, z + 0.0007)])
faces = [(3, 2, 1, 0)]
for i in range(len(neck_stations) - 1):
    for j in range(4):
        faces.append((i * 4 + j, i * 4 + (j + 1) % 4,
                      (i + 1) * 4 + (j + 1) % 4, (i + 1) * 4 + j))
faces.append(tuple((len(neck_stations) - 1) * 4 + j for j in range(4)))
neck = make_mesh("Turner_bent_stainless_neck", vertices, faces, steel)
bevel(neck, 0.00045, 5, 0.2)
shade(neck, True)

# A softly contoured solid grip. The hanging eye is cut through the full solid.
grip_outline = curve_outline([
    (-0.006, -0.0070), (0.006, -0.0093), (0.024, -0.0086),
    (0.043, -0.0077), (0.061, -0.0098), (0.083, -0.0110),
    (0.104, -0.0095), (0.114, -0.0058), (0.117, 0),
    (0.114, 0.0058), (0.104, 0.0095), (0.083, 0.0110),
    (0.061, 0.0098), (0.043, 0.0077), (0.024, 0.0086),
    (0.006, 0.0093), (-0.006, 0.0070)
], 8)
grip = prism("Turner_elastomer_grip_with_open_eye", grip_outline, 0.0040, 0.0150, rubber)
# Broad outer rounding is applied before cutting to preserve the eye's shape.
bevel(grip, 0.0030, 7, 0.5)
eye = prism("Temporary_hanging_eye",
            capsule((0.0985, 0), (0.105, 0), 0.0032, 32), -0.008, 0.030)
cut(grip, eye)
bevel(grip, 0.00065, 5, 0.55)
shade(grip, True)

# Place the complete object on z=0 without changing its practical XY origin.
objects = [o for o in bpy.context.scene.objects if o.type == "MESH"]
min_z = min((obj.matrix_world @ vertex.co).z
            for obj in objects for vertex in obj.data.vertices)
for obj in objects:
    obj.location.z -= min_z
validate_and_export("turner")