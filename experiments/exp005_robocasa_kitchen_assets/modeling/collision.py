import bpy
import sys
import os
import math
import xml.etree.ElementTree as ET
from mathutils import Vector

OUT = os.path.abspath(sys.argv[sys.argv.index("--") + 1])
N = 48


def fmt(values):
    return " ".join("%.9g" % float(v) for v in values)


def obj_write(path, vertices, faces):
    with open(path, "w", encoding="ascii") as stream:
        for v in vertices:
            stream.write("v %s\n" % fmt(v))
        for face in faces:
            stream.write("f %s\n" % " ".join(str(i + 1) for i in face))


class Export:
    def __init__(self, name, folder):
        self.name = name
        self.folder = folder
        self.root = ET.Element("mujoco", model=name)
        ET.SubElement(self.root, "compiler", angle="radian",
                      meshdir=".", autolimits="true")
        self.asset = ET.SubElement(self.root, "asset")
        world = ET.SubElement(self.root, "worldbody")
        self.outer = ET.SubElement(world, "body", name=name)
        self.body = ET.SubElement(self.outer, "body", name="object")
        self.parts = []
        self.counter = 0

    def mesh(self, label, vertices, faces):
        self.counter += 1
        key = "%s_%s_%03d" % (self.name, label, self.counter)
        filename = key + ".obj"
        obj_write(os.path.join(self.folder, filename), vertices, faces)
        ET.SubElement(self.asset, "mesh", name=key, file=filename)
        return key

    def visual(self, vertices, faces, rgba):
        key = self.mesh("visual", vertices, faces)
        ET.SubElement(self.body, "geom", name=key, type="mesh", mesh=key,
                      group="1", contype="0", conaffinity="0",
                      mass="0", rgba=fmt(rgba))

    def collision(self, label, weight, **attributes):
        self.counter += 1
        attributes.update(name="%s_%s_%03d" %
                          (self.name, label, self.counter),
                          group="0", contype="1", conaffinity="1",
                          friction="0.65 0.005 0.0001",
                          rgba="0.45 0.55 0.65 0")
        geom = ET.SubElement(self.body, "geom", **attributes)
        self.parts.append((geom, max(weight, 1e-12)))

    def hull(self, label, vertices, weight):
        # MuJoCo constructs a convex hull from these noncoplanar points.
        key = self.mesh(label, vertices, [])
        self.collision(label, weight, type="mesh", mesh=key)

    def cylinder(self, label, radius, bottom, top):
        self.collision(label, math.pi * radius * radius * (top - bottom),
                       type="cylinder", pos=fmt((0, 0, (bottom + top) / 2)),
                       size=fmt((radius, (top - bottom) / 2)))

    def capsule(self, label, a, b, radius):
        length = (Vector(b) - Vector(a)).length
        if length < 1e-8:
            return
        volume = math.pi * radius * radius * length
        volume += 4 * math.pi * radius ** 3 / 3
        self.collision(label, volume, type="capsule",
                       fromto=fmt(tuple(a) + tuple(b)), size=fmt((radius,)))

    def box(self, label, center, half):
        self.collision(label, 8 * half[0] * half[1] * half[2],
                       type="box", pos=fmt(center), size=fmt(half))

    def finish(self, low, high, mass):
        total = sum(weight for _, weight in self.parts)
        for geom, weight in self.parts:
            geom.set("mass", "%.12g" % (mass * weight / total))
        center = (Vector(low) + Vector(high)) / 2
        half = (Vector(high) - Vector(low)) / 2
        ET.SubElement(self.body, "geom", name=self.name + "_reg_bbox",
                      type="box", pos=fmt(center), size=fmt(half),
                      group="1", rgba="0 0 0 0", mass="0",
                      contype="0", conaffinity="0")
        for label, pos in (
            ("bottom_site", (0, 0, low[2])),
            ("top_site", (0, 0, high[2])),
            ("horizontal_radius_site",
             (max(abs(low[0]), abs(high[0])),
              max(abs(low[1]), abs(high[1])), 0)),
        ):
            ET.SubElement(self.outer, "site", name=label, pos=fmt(pos),
                          size="0.001", rgba="0 0 0 0")
        ET.indent(self.root)
        ET.ElementTree(self.root).write(
            os.path.join(self.folder, "model.xml"),
            encoding="ascii", xml_declaration=True)


def physical_color(obj, material, name):
    text = (obj.name + " " + (material.name if material else "")).lower()
    if any(s in text for s in ("rubber", "grip", "black")):
        return (0.025, 0.028, 0.032, 1)
    if any(s in text for s in ("steel", "metal", "silver", "blade", "neck")):
        return (0.66, 0.69, 0.72, 1)
    if name == "turner":
        return (0.66, 0.69, 0.72, 1)
    if name == "scoop" or "orange" in text:
        return (0.91, 0.29, 0.065, 1)
    return (0.88, 0.82, 0.67, 1)


def inspect_visuals(export):
    points = []
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in list(bpy.context.scene.objects):
        if obj.type not in {"MESH", "CURVE", "SURFACE", "FONT", "META"}:
            continue
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            if not mesh or not mesh.vertices:
                continue
            vertices = [tuple(obj.matrix_world @ v.co) for v in mesh.vertices]
            points.extend(vertices)
            mesh.calc_loop_triangles()
            groups = {}
            for tri in mesh.loop_triangles:
                groups.setdefault(tri.material_index, []).append(
                    tuple(tri.vertices))
            for index, faces in groups.items():
                material = (mesh.materials[index]
                            if index < len(mesh.materials) else None)
                used = sorted({i for face in faces for i in face})
                remap = {old: new for new, old in enumerate(used)}
                export.visual([vertices[i] for i in used],
                              [tuple(remap[i] for i in f) for f in faces],
                              physical_color(obj, material, export.name))
        finally:
            evaluated.to_mesh_clear()
    if not points:
        raise RuntimeError("No visual geometry in " + export.name)
    low = tuple(min(v[i] for v in points) for i in range(3))
    high = tuple(max(v[i] for v in points) for i in range(3))
    return low, high


def tube_rings(name):
    candidates = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        verts = obj.data.vertices
        if len(verts) < 122 or (len(verts) - 2) % 40:
            continue
        count = (len(verts) - 2) // 40
        # The two cap centers can precede, follow, or bracket the rings.
        for offset in (0, 1, 2):
            centers, radii = [], []
            for k in range(count):
                ring = [obj.matrix_world @ verts[offset + 40*k+j].co
                        for j in range(40)]
                center = sum(ring, Vector()) / 40
                centers.append(center)
                radii.append(sum((p-center).length for p in ring) / 40)
            if not all(0.001 < r < 0.012 for r in radii):
                continue
            steps = [(b-a).length for a, b in zip(centers, centers[1:])]
            if not steps or max(steps) > 0.012:
                continue
            if max(c.x for c in centers) < (0.045 if name == "scoop" else 0.055):
                continue
            score = sum(abs(r - sum(radii)/count) for r in radii)
            candidates.append((score, centers, radii))
    if not candidates:
        raise RuntimeError("Cannot infer 40-vertex swept handle rings: " + name)
    return min(candidates, key=lambda item: item[0])[1:]


def vessel(export):
    mug = export.name == "mug"
    bottom = 0.009 if mug else 0.003
    rb = 0.0327 if mug else 0.0275
    ib = 0.0295 if mug else 0.0263
    rt = 0.03845 if mug else rb
    it = 0.03575 if mug else ib
    export.cylinder("bottom_disk", rb, 0, bottom)
    label = "integral_wall_and_foot" if mug else "oblique_integral_shell"
    for i in range(N):
        angles = [2*math.pi*i/N, 2*math.pi*(i+1)/N]
        vertices = []
        heights = []
        for angle in angles:
            top = 0.089 if mug else 0.094 - 0.026*math.cos(angle)
            heights.append(top - bottom)
            for radius, z in ((rb, bottom), (ib, bottom),
                              (rt, top), (it, top)):
                vertices.append((radius*math.cos(angle),
                                 radius*math.sin(angle), z))
        area0 = rb*rb - ib*ib
        area1 = rt*rt - it*it
        volume = math.pi / N * (area0 + area1) / 2 * sum(heights)/2
        export.hull(label, vertices, volume)
    centers, radii = tube_rings(export.name)
    indices = list(range(0, len(centers), 6))
    if indices[-1] != len(centers)-1:
        indices.append(len(centers)-1)
    for a, b in zip(indices, indices[1:]):
        radius = min(radii[a:b+1]) * 0.94
        export.capsule("loop_handle", centers[a], centers[b], radius)


def blade_z(x):
    t = max(0, min(1, (-x - 0.055)/0.135))
    return 0.0018 + 0.0085*t*t


def blade_strip(export, a, b, wa, wb, label):
    direction = Vector((b[0]-a[0], b[1]-a[1]))
    normal = Vector((-direction.y, direction.x)).normalized()
    # Subdivide the curved blade instead of using a thick enclosing hull.
    for k in range(8):
        vertices = []
        length = direction.length / 8
        widths = []
        for t in (k/8, (k+1)/8):
            p = Vector(a).lerp(Vector(b), t)
            width = wa*(1-t) + wb*t
            widths.append(width)
            for sign in (-1, 1):
                q = p + sign*width*normal
                for dz in (-0.0006, 0.0006):
                    vertices.append((q.x, q.y, blade_z(q.x)+dz))
        export.hull(label, vertices, length*sum(widths)*0.0012)


def turner(export):
    slots = []
    for i in range(6):
        f = (i+0.5)/6
        slots.append((Vector((-0.178+0.003*f, -0.029+0.060*f)),
                      Vector((-0.072-0.004*abs(2*f-1),
                              -0.0092+0.0184*f))))
    for i in range(5):
        a = (slots[i][0]+slots[i+1][0])/2
        b = (slots[i][1]+slots[i+1][1])/2
        d = (b-a).normalized()
        n = Vector((-d.y, d.x))
        wa = abs((slots[i+1][0]-slots[i][0]).dot(n))/2 - 0.0018
        wb = abs((slots[i+1][1]-slots[i][1]).dot(n))/2 - 0.0018
        blade_strip(export, a, b, max(wa, 0.0002),
                    max(wb, 0.0002), "between_slots")
    blade_strip(export, (-0.185, -0.038), (-0.060, -0.015),
                0.0015, 0.0015, "side_frame")
    blade_strip(export, (-0.185, 0.038), (-0.060, 0.015),
                0.0015, 0.0015, "side_frame")
    blade_strip(export, (-0.186, -0.036), (-0.186, 0.036),
                0.004, 0.004, "end_crossbar")
    blade_strip(export, (-0.062, -0.014), (-0.062, 0.014),
                0.006, 0.006, "neck_crossbar")
    export.capsule("steel_neck", (-0.057, 0, 0.002),
                   (0.005, 0, 0.006), 0.0025)
    export.box("rubber_grip", (0.048, 0, 0.006),
               (0.044, 0.006, 0.004))
    # Four bars leave the actual eye open through the grip.
    export.box("eye_left", (0.09365, 0, 0.006),
               (0.00165, 0.006, 0.004))
    export.box("eye_right", (0.1126, 0, 0.006),
               (0.0044, 0.006, 0.004))
    for sign in (-1, 1):
        export.box("eye_side", (0.10175, sign*0.0046, 0.006),
                   (0.00645, 0.0014, 0.004))


for name, mass in (("mug", 0.29), ("scoop", 0.105), ("turner", 0.081)):
    bpy.ops.wm.open_mainfile(filepath=os.path.join(OUT, name + ".blend"))
    folder = os.path.join(OUT, "sim", name)
    os.makedirs(folder, exist_ok=True)
    exporter = Export(name, folder)
    low, high = inspect_visuals(exporter)
    if name == "turner":
        turner(exporter)
    else:
        vessel(exporter)
    exporter.finish(low, high, mass)
