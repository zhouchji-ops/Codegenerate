import argparse
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np


G = 9.80665
NAMES = ("mug", "scoop", "turner")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(a, label):
    a = np.asarray(a, dtype=float)
    require(np.all(np.isfinite(a)), "Nonfinite " + label)
    return a


def read_obj(path):
    vertices, faces = [], []
    with path.open("r", encoding="utf-8", errors="strict") as stream:
        for line in stream:
            fields = line.split("#", 1)[0].split()
            if not fields:
                continue
            if fields[0] == "v":
                require(len(fields) >= 4, "Malformed vertex: " + str(path))
                vertices.append([float(x) for x in fields[1:4]])
            elif fields[0] == "f":
                require(len(fields) >= 4, "Malformed face: " + str(path))
                polygon = []
                for token in fields[1:]:
                    index = int(token.split("/")[0])
                    require(index != 0, "OBJ index zero")
                    index = index - 1 if index > 0 else len(vertices) + index
                    require(0 <= index < len(vertices), "OBJ index out of bounds")
                    polygon.append(index)
                for i in range(1, len(polygon) - 1):
                    faces.append([polygon[0], polygon[i], polygon[i + 1]])
    v = finite(vertices, "vertices")
    f = np.asarray(faces, dtype=np.int64)
    require(v.ndim == 2 and v.shape[1] == 3 and len(v) >= 4,
            "Insufficient vertices: " + str(path))
    require(f.ndim == 2 and f.shape[1] == 3 and len(f) >= 4,
            "Insufficient triangles: " + str(path))
    return v, f


def integrate(path):
    v, f = read_obj(path)
    edges = {}
    for tri in f:
        require(len(set(map(int, tri))) == 3, "Degenerate indexed triangle")
        for i, j in zip(tri, np.roll(tri, -1)):
            i, j = int(i), int(j)
            key = (min(i, j), max(i, j))
            count, direction = edges.get(key, (0, 0))
            edges[key] = (count + 1, direction + (1 if i < j else -1))
    require(all(n == 2 and direction == 0 for n, direction in edges.values()),
            "Mesh must be closed and consistently wound: " + str(path))
    a, b, c = (v[f[:, i]] for i in range(3))
    normal = np.cross(b - a, c - a)
    require(np.all(np.linalg.norm(normal, axis=1) > 1e-18),
            "Zero-area triangle: " + str(path))
    w = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    s = a + b + c
    volume = float(w.sum())
    require(abs(volume) > 1e-15, "Zero signed volume: " + str(path))
    first = np.einsum("i,ij->j", w / 4.0, s)
    second = np.einsum("i,ij,ik->jk", w / 20.0, s, s)
    for vertex in (a, b, c):
        second += np.einsum("i,ij,ik->jk", w / 20.0, vertex, vertex)
    if volume < 0:
        volume, first, second = -volume, -first, -second
    center = first / volume
    covariance = second / volume - np.outer(center, center)
    covariance = (covariance + covariance.T) / 2
    require(np.linalg.eigvalsh(covariance).min() > 0,
            "Nonpositive volume covariance: " + str(path))
    return dict(path=path, vertices=v, faces=f, volume=volume,
                center=center, covariance=covariance,
                bounds=np.array([v.min(axis=0), v.max(axis=0)]))


def check_inertia(inertia):
    inertia = finite(inertia, "inertia")
    require(np.allclose(inertia, inertia.T, atol=1e-15, rtol=1e-12),
            "Asymmetric inertia")
    eigenvalues = np.linalg.eigvalsh(inertia)
    require(eigenvalues[0] > 0, "Inertia must be positive definite")
    require(eigenvalues[2] <= eigenvalues[0] + eigenvalues[1] + 1e-13,
            "Inertia violates triangle inequalities")
    return eigenvalues


def aggregate(parts, masses):
    masses = finite(masses, "component masses")
    require(np.all(masses > 0), "Component mass must be positive")
    total = float(masses.sum())
    center = sum(m * p["center"] for p, m in zip(parts, masses)) / total
    covariance = np.zeros((3, 3))
    for p, m in zip(parts, masses):
        displacement = p["center"] - center
        covariance += m * (p["covariance"] +
                           np.outer(displacement, displacement))
    inertia = np.trace(covariance) * np.eye(3) - covariance
    eigenvalues = check_inertia(inertia)
    return dict(mass_kg=total, center_m=center.tolist(),
                center_mm=(1000 * center).tolist(),
                inertia_com_kg_m2=inertia.tolist(),
                principal_inertia_kg_m2=eigenvalues.tolist())


def envelope(parts, parameters, mass_function):
    keys = list(parameters)
    nominal = {key: parameters[key][1] for key in keys}
    central_masses = mass_function(nominal)
    central = aggregate(parts, central_masses)
    cases = []
    for values in itertools.product(*[(parameters[k][0], parameters[k][2])
                                      for k in keys]):
        p = dict(zip(keys, values))
        result = aggregate(parts, mass_function(p))
        cases.append((p, result))
    mass_values = [r["mass_kg"] for _, r in cases]
    centers = np.asarray([r["center_mm"] for _, r in cases])
    tensors = np.asarray([r["inertia_com_kg_m2"] for _, r in cases])
    lo, hi = int(np.argmin(mass_values)), int(np.argmax(mass_values))
    return dict(
        central=central,
        parameters_low_nominal_high=parameters,
        scenario_envelope=dict(
            label="Endpoint scenario envelope of explicit mechanical priors",
            endpoint_scenario_count=len(cases),
            mass_kg=[min(mass_values), max(mass_values)],
            center_mm_componentwise=[centers.min(axis=0).tolist(),
                                     centers.max(axis=0).tolist()],
            inertia_com_kg_m2_componentwise=[
                tensors.min(axis=0).tolist(), tensors.max(axis=0).tolist()],
            minimum_mass_parameters=cases[lo][0],
            maximum_mass_parameters=cases[hi][0],
            note="Componentwise bounds need not occur in one scenario."),
        components=[
            dict(file=p["path"].name, role=p["role"],
                 closed_mesh_volume_m3=p["volume"],
                 geometric_center_mm=(1000 * p["center"]).tolist(),
                 effective_mass_kg=float(m),
                 effective_fraction_of_rho_volume=float(
                     m / (p["volume"] * nominal[p["density_key"]])))
            for p, m in zip(parts, central_masses)])


def assign_roles(name, parts):
    if name == "mug":
        shell = max(parts, key=lambda p: p["volume"])
        remaining = [p for p in parts if p is not shell]
        handle = max(remaining, key=lambda p: p["volume"])
        for p in parts:
            p["role"] = ("main_shell" if p is shell else
                         "handle" if p is handle else "attachment")
            p["density_key"] = "density_kg_m3"
    elif name == "scoop":
        # The scoop bowl has the largest transverse span; its tube is slender.
        shell = max(parts, key=lambda p: float(np.max(
            p["bounds"][1] - p["bounds"][0]) *
            np.median(p["bounds"][1] - p["bounds"][0])))
        remaining = [p for p in parts if p is not shell]
        tube = max(remaining, key=lambda p: float(np.max(
            p["bounds"][1] - p["bounds"][0])))
        for p in parts:
            p["role"] = ("shell" if p is shell else
                         "tube" if p is tube else "welded_seat")
            p["density_key"] = "density_kg_m3"
    else:
        blades = [p for p in parts
                  if re.search(r"_visual_001\.obj$", p["path"].name)]
        require(len(blades) == 1, "Turner blade visual_001 is required")
        blade = blades[0]
        remaining = [p for p in parts if p is not blade]
        grip = max(remaining, key=lambda p: p["volume"])
        for p in parts:
            p["role"] = ("blade" if p is blade else
                         "grip" if p is grip else "neck")
            p["density_key"] = ("silicone_density_kg_m3"
                                if p is grip else "steel_density_kg_m3")


def predict(name, parts):
    assign_roles(name, parts)
    if name == "mug":
        parameters = dict(density_kg_m3=[2150, 2300, 2450],
                          main_shell_multiplier=[0.85, 1.0, 1.20],
                          handle_factor=[0.85, 1.0, 1.15],
                          attachment_1_factor=[0.25, 0.50, 0.75],
                          attachment_2_factor=[0.25, 0.50, 0.75])
        seats = [p for p in parts if p["role"] == "attachment"]

        def masses(q):
            result = []
            for p in parts:
                if p["role"] == "attachment":
                    index = next(i for i, s in enumerate(seats) if s is p)
                    factor = q["attachment_%d_factor" % (index + 1)]
                else:
                    factor = q["main_shell_multiplier" if
                               p["role"] == "main_shell" else "handle_factor"]
                result.append(p["volume"] * q["density_kg_m3"] * factor)
            return result
    elif name == "scoop":
        parameters = dict(density_kg_m3=[7800, 7900, 8000],
                          shell_gauge_mm=[0.35, 0.50, 0.75],
                          tube_wall_mm=[0.35, 0.50, 0.80],
                          seat_1_factor=[0.25, 0.50, 0.75],
                          seat_2_factor=[0.25, 0.50, 0.75])
        seats = [p for p in parts if p["role"] == "welded_seat"]

        def masses(q):
            result = []
            for p in parts:
                if p["role"] == "shell":
                    factor = q["shell_gauge_mm"] / 1.6324
                elif p["role"] == "tube":
                    factor = 1 - (1 - q["tube_wall_mm"] / 3.809) ** 2
                else:
                    index = next(i for i, s in enumerate(seats) if s is p)
                    factor = q["seat_%d_factor" % (index + 1)]
                result.append(p["volume"] * q["density_kg_m3"] * factor)
            return result
    else:
        parameters = dict(steel_density_kg_m3=[7800, 7900, 8000],
                          silicone_density_kg_m3=[1050, 1150, 1250],
                          blade_thickness_mm=[0.9, 1.2, 1.4])

        def masses(q):
            return [p["volume"] * q[p["density_key"]] *
                    (q["blade_thickness_mm"] / 1.2
                     if p["role"] == "blade" else 1.0) for p in parts]

    result = envelope(parts, parameters, masses)
    result["role_assignment_method"] = (
        "Mug: largest volume shell, next largest handle. "
        "Scoop: largest maximum-span times median-span bowl, longest remaining "
        "part tube. Turner: visual_001 blade, larger remaining volume grip. "
        "The component filenames and assigned roles are recorded for audit.")
    result["evidence_status"] = (
        "Calibrated closed-mesh integrals with explicit mechanical priors; "
        "densities, thicknesses, overlap factors and friction are not "
        "SKU-specific measurements. Intersections are approximately additive; "
        "effective fractions do not guarantee removal of double counting.")
    if name == "mug":
        result["coordinate_convention"] = (
            "Input calibrated coordinates retained: base z=0, "
            "origin on centerline, +X toward handle.")
    if name == "scoop":
        result["literal_solid_mesh_at_rho7900"] = aggregate(
            parts, [p["volume"] * 7900 for p in parts])
        result["shell_reference"] = dict(
            original_thickness_m=0.0012, calibration_scale=1.36033539,
            effective_average_thickness_mm=1.6324,
            tube_outer_radius_mm=3.809)
    return result


def friction_predictions(results):
    mug = results["mug"]
    mass = mug["central"]["mass_kg"]
    low, high = mug["scenario_envelope"]["mass_kg"]
    mug["countertop_translation"] = {}
    for condition, static, kinetic in (
        ("dry", [0.30, 0.45, 0.65], [0.20, 0.35, 0.50]),
        ("thin_water_film", [0.12, 0.25, 0.45], [0.08, 0.18, 0.35]),
    ):
        mug["countertop_translation"][condition] = dict(
            pair="Unglazed mug foot on smooth stone countertop",
            mu_static_low_nominal_high=static,
            mu_kinetic_low_nominal_high=kinetic,
            breakaway_horizontal_force_N=mass * G * static[1],
            breakaway_force_range_N=[low * G * static[0],
                                     high * G * static[2]],
            initial_speed_m_s=0.20,
            coast_distance_m=0.20 ** 2 / (2 * kinetic[1] * G),
            coast_distance_range_m=[
                0.20 ** 2 / (2 * kinetic[2] * G),
                0.20 ** 2 / (2 * kinetic[0] * G)],
            assumptions="Level pure translation and sliding; no tipping.")
    scoop = results["scoop"]
    mass = scoop["central"]["mass_kg"]
    low, high = scoop["scenario_envelope"]["mass_kg"]
    scoop["symmetric_pad_lift"] = {}
    for condition, mu in (("dry", [0.40, 0.65, 0.90]),
                          ("wet", [0.18, 0.35, 0.65])):
        scoop["symmetric_pad_lift"][condition] = dict(
            pair="Two silicone pads on opposite sides of vertical steel handle",
            mu_static_low_nominal_high=mu,
            normal_force_per_pad_N=mass * G / (2 * mu[1]),
            normal_force_per_pad_range_N=[
                low * G / (2 * mu[2]), high * G / (2 * mu[0])],
            assumptions="Slow empty lift, symmetric contact, no torque; "
                        "values are limiting equilibrium forces.")


def section_area(part, station):
    v, f = part["vertices"], part["faces"]
    signed_area = 0.0
    count = 0
    for tri in v[f]:
        distances = tri[:, 0] - station
        if distances.min() >= 0 or distances.max() <= 0:
            continue
        hits = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            if distances[i] * distances[j] < 0:
                hits.append(tri[i] + (tri[j] - tri[i]) *
                            (-distances[i] / (distances[j] - distances[i])))
            elif distances[i] == 0:
                hits.append(tri[i])
        unique = []
        for point in hits:
            if not any(np.linalg.norm(point - other) < 1e-12
                       for other in unique):
                unique.append(point)
        if len(unique) != 2:
            raise ValueError("Ambiguous triangle section")
        p, q = unique
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        direction = np.cross(normal, [1.0, 0.0, 0.0])
        if np.dot(q - p, direction) < 0:
            p, q = q, p
        signed_area += (p[1] * q[2] - q[1] * p[2]) / 2
        count += 1
    require(count >= 3, "Insufficient blade section segments")
    # A globally reversed winding reverses every contour, including holes.
    return abs(signed_area)


def beam_predictions(parts):
    blade = next(p for p in parts if p["role"] == "blade")
    load_x, clamp_x, n = -0.184, -0.060, 200
    require(blade["bounds"][0, 0] <= load_x and
            blade["bounds"][1, 0] >= clamp_x,
            "Blade mesh does not cover the specified beam interval")
    dx = (clamp_x - load_x) / n
    stations = load_x + (np.arange(n) + 0.5) * dx
    widths = finite([section_area(blade, x) / 0.0012 for x in stations],
                    "section widths")
    max_transverse_span = float(blade["bounds"][1, 1] -
                                blade["bounds"][0, 1])
    require(np.all(widths > 1e-5) and
            np.all(widths <= 2 * max_transverse_span),
            "Nonpositive or implausible net section width")
    print("turner net section width mm: %.6g .. %.6g" %
          (1000 * widths.min(), 1000 * widths.max()), file=sys.stderr)

    def compliance(young, thickness):
        inertia = widths * thickness ** 3 / 12
        value = float(np.sum((stations - load_x) ** 2 /
                             (young * inertia)) * dx)
        require(math.isfinite(value) and value > 0, "Invalid beam compliance")
        return value

    nominal = compliance(200e9, 0.0012)
    lower = compliance(210e9, 0.0014)
    upper = compliance(185e9, 0.0009)
    length = clamp_x - load_x
    return dict(
        model="Variable net-width slotted Euler-Bernoulli beam surrogate",
        clamp_x_m=clamp_x, point_load_x_m=load_x, midpoint_slices=n,
        width_method="Oriented triangle-plane section shoelace area, including "
                     "hole subtraction, divided by nominal 0.0012 m thickness",
        net_width_range_mm=[float(1000 * widths.min()),
                            float(1000 * widths.max())],
        stations_x_m=stations.tolist(), net_width_mm=(1000 * widths).tolist(),
        young_modulus_GPa_low_nominal_high=[185, 200, 210],
        thickness_mm_low_nominal_high=[0.9, 1.2, 1.4],
        compliance_m_per_N=nominal,
        point_loads=[
            dict(force_N=force, tip_deflection_mm=1000 * force * nominal,
                 tip_deflection_range_mm=[1000 * force * lower,
                                          1000 * force * upper],
                 maximum_deflection_over_span=force * upper / length,
                 small_deflection_screen_pass=bool(force * upper / length < 0.05))
            for force in (0.25, 0.5, 1.0)],
        validity="Requires small transverse deflection and rotations, linear "
                 "elastic behavior, and an ideal fixed clamp. The reported "
                 "delta/span < 0.05 screen is necessary guidance, not a "
                 "verification of rotations or stress. Local slot stress, "
                 "shear, curved-shell coupling and clamp compliance omitted.",
        rigid_mjcf_tip_flex_mm=0.0,
        relationship_to_mjcf="Independent analytical surrogate; MJCF remains rigid.")


def slot_predictions():
    cases = []
    for degrees in (0, 5):
        angle = math.radians(degrees)

        def gap(thickness):
            return thickness * math.cos(angle) + 5.0 * math.sin(angle)

        cases.append(dict(approach_angle_degrees=degrees,
                          required_gap_mm=gap(1.2),
                          required_gap_range_mm=[gap(0.9), gap(1.4)]))
    return dict(
        insertion_length_mm=5.0, leading_strip_thickness_mm=1.2,
        thickness_tolerance_mm=[0.9, 1.4], cases=cases,
        formula="gap = thickness*cos(theta) + insertion_length*sin(theta)",
        assumptions="Dry parallel rigid slot, nondeforming thin leading strip, "
                    "tested planar alignment independent of curved blade "
                    "global pose. Equality is geometric contact; practical "
                    "clearance is additional.",
        collision_geometry_modified=False)


def source_directory(argument):
    root = Path(argument).expanduser().resolve(strict=True)
    require(root.is_dir(), "Source must be a directory")
    candidates = [root, root / "sim"]
    matches = [p for p in candidates
               if all((p / name / "model.xml").is_file() for name in NAMES)]
    require(len(matches) == 1, "Supply a directory containing sim assets")
    return root, matches[0]


def collect_parts(asset_directory, name, expected):
    paths = sorted(p for p in asset_directory.rglob(name + "_visual_*.obj")
                   if re.fullmatch(re.escape(name) + r"_visual_\d{3}\.obj",
                                   p.name))
    require(len(paths) == expected,
            "%s requires exactly %d visual components; found %d" %
            (name, expected, len(paths)))
    return [integrate(p) for p in paths]


def inside(path, parent):
    return path == parent or parent in path.parents


def localize_assets(tree, source_model, target_directory):
    root = tree.getroot()
    require(not root.findall(".//include"),
            "Included MJCF requires flattening; refusing to alter unknown includes")
    compiler = root.find("compiler")
    assetdir = compiler.get("assetdir", "") if compiler is not None else ""
    meshdir = compiler.get("meshdir", assetdir) if compiler is not None else ""
    texturedir = compiler.get("texturedir", assetdir) if compiler is not None else ""
    used = {}
    for element in root.findall(".//asset/*"):
        value = element.get("file")
        if value is None:
            continue
        directory = meshdir if element.tag == "mesh" else (
            texturedir if element.tag == "texture" else assetdir)
        file_path = Path(value)
        if not file_path.is_absolute():
            file_path = source_model.parent / directory / file_path
        file_path = file_path.resolve(strict=True)
        require(file_path.is_file(), "Asset is not a file")
        key = str(file_path)
        if key not in used:
            suffix = file_path.suffix
            filename = "asset_%03d%s" % (len(used), suffix)
            shutil.copy2(file_path, target_directory / filename)
            used[key] = filename
        element.set("file", used[key])
    if compiler is not None:
        for attribute in ("assetdir", "meshdir", "texturedir"):
            compiler.attrib.pop(attribute, None)
        if compiler.get("inertiafromgeom", "").lower() == "true":
            compiler.set("inertiafromgeom", "auto")
    return used


def write_model(source_model, target_directory, prediction):
    target_directory.mkdir(parents=True)
    tree = ET.parse(source_model)
    root = tree.getroot()
    bodies = [b for b in root.findall(".//body") if b.get("name") == "object"]
    require(len(bodies) == 1, "Expected exactly one body named object")
    body = bodies[0]
    for old in list(body.findall("inertial")):
        body.remove(old)
    central = prediction["central"]
    inertia = np.asarray(central["inertia_com_kg_m2"])
    check_inertia(inertia)

    def numbers(values):
        return " ".join("%.17g" % float(v) for v in values)

    inertial = ET.Element("inertial", dict(
        pos=numbers(central["center_m"]),
        mass="%.17g" % central["mass_kg"],
        fullinertia=numbers([inertia[0, 0], inertia[1, 1], inertia[2, 2],
                            inertia[0, 1], inertia[0, 2], inertia[1, 2]])))
    body.insert(0, inertial)
    localize_assets(tree, source_model, target_directory)
    tree.write(target_directory / "model.xml", encoding="utf-8",
               xml_declaration=True)


VALIDATOR = r'''import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np

def validate(directory):
    directory = Path(directory)
    data = json.loads((directory / "predictions.json").read_text("utf-8"))
    for name in ("mug", "scoop", "turner"):
        prediction = data["objects"][name]["central"]
        model = directory / "sim" / name / "model.xml"
        root = ET.parse(model).getroot()
        bodies = [b for b in root.findall(".//body")
                  if b.get("name") == "object"]
        assert len(bodies) == 1, name
        inertials = bodies[0].findall("inertial")
        assert len(inertials) == 1, name
        node = inertials[0]
        mass = float(node.get("mass"))
        center = np.fromstring(node.get("pos"), sep=" ")
        full = np.fromstring(node.get("fullinertia"), sep=" ")
        assert center.shape == (3,) and full.shape == (6,), name
        xx, yy, zz, xy, xz, yz = full
        inertia = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
        assert np.isfinite(inertia).all() and np.isfinite(center).all()
        assert np.isfinite(mass) and mass > 0
        eigenvalues = np.linalg.eigvalsh(inertia)
        assert eigenvalues[0] > 0, name
        assert eigenvalues[2] <= eigenvalues[:2].sum() + 1e-13, name
        assert np.isclose(mass, prediction["mass_kg"], rtol=1e-12)
        assert np.allclose(center, prediction["center_m"],
                           rtol=1e-12, atol=1e-15)
        assert np.allclose(inertia, prediction["inertia_com_kg_m2"],
                           rtol=1e-12, atol=1e-15)
        compiler = root.find("compiler")
        assert compiler is None or compiler.get("inertiafromgeom") != "true"
        for node in root.findall(".//asset/*"):
            filename = node.get("file")
            if filename is not None:
                assert Path(filename).name == filename, filename
                assert (model.parent / filename).is_file(), filename
        print(name, "mass_kg", mass, "center_mm", (center * 1000).tolist(),
              "inertia_com_kg_m2", inertia.tolist())
    print("Validation passed")

if __name__ == "__main__":
    validate(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent)
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("calibrated_sim")
    parser.add_argument("out_new_dir")
    args = parser.parse_args()
    source_root, sim = source_directory(args.calibrated_sim)
    destination = Path(args.out_new_dir).expanduser().resolve()
    require(not inside(destination, source_root),
            "Destination must not be inside the supplied source")
    require(not inside(source_root, destination),
            "Destination must not contain the source")
    if destination.exists():
        require(destination.is_dir() and not any(destination.iterdir()),
                "Destination must be absent or empty")

    parts = {name: collect_parts(sim / name, name, count)
             for name, count in zip(NAMES, (4, 4, 3))}
    results = {name: predict(name, parts[name]) for name in NAMES}
    friction_predictions(results)
    results["turner"]["vertical_tip_deflection"] = beam_predictions(parts["turner"])
    results["turner"]["rigid_slot_entry"] = slot_predictions()
    document = dict(
        schema_version=1, gravity_m_s2=G,
        geometry_units="m",
        inertia_convention="COM tensor with off-diagonal inertia products "
                           "Ixy=-integral(x*y)dm, and similarly Ixz,Iyz",
        geometry_policy="Original calibrated geometry and contact parameters "
                        "retained. Explicit object inertial overrides geometry "
                        "inference. Asset file references localized.",
        uncertainty_policy="Endpoint scenario envelopes, not statistical "
                           "confidence intervals; no measured SKU claims.",
        objects=results)
    encoded = json.dumps(document, indent=2, allow_nan=False) + "\n"

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="physics_assets_",
                                     dir=str(destination.parent)))
    try:
        for name in NAMES:
            write_model(sim / name / "model.xml",
                        temporary / "sim" / name, results[name])
        (temporary / "predictions.json").write_text(encoded, encoding="ascii")
        (temporary / "validate_properties.py").write_text(
            VALIDATOR, encoding="ascii")
        namespace = {"__name__": "asset_validator"}
        exec(compile(VALIDATOR, "validate_properties.py", "exec"), namespace)
        namespace["validate"](temporary)
        if destination.exists():
            require(destination.is_dir() and not any(destination.iterdir()),
                    "Destination became nonempty")
            destination.rmdir()
        # On Windows rename refuses an existing destination.
        require(not destination.exists(), "Destination appeared during output")
        os.rename(str(temporary), str(destination))
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


if __name__ == "__main__":
    main()
