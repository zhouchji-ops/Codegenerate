import pathlib
import bpy

source_path = pathlib.Path(__file__).with_name("generated.py")
source = source_path.read_text(encoding="utf-8")

substitutions = [
    (
        'bsdf.inputs["Anisotropic IOR Level"].default_value = anisotropy',
        'bsdf.inputs["Anisotropic"].default_value = anisotropy',
    ),
    (
        'ids = [lookup[tuple(v)] for v in triangle]',
        'ids = [v if isinstance(v, int) else lookup[tuple(v)] for v in triangle]',
    ),
    (
        'rubber.node_tree.nodes.get("Principled BSDF").inputs["Specular IOR Level"].default_value = 0.30',
        'rubber.node_tree.nodes.get("Principled BSDF").inputs['
        '"Specular IOR Level" if "Specular IOR Level" in '
        'rubber.node_tree.nodes.get("Principled BSDF").inputs else '
        '"Specular"].default_value = 0.30',
    ),
]

for old, new in substitutions:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            "Expected exactly one occurrence of {!r}; found {}".format(old, count)
        )

for old, new in substitutions:
    source = source.replace(old, new, 1)

code = compile(source, str(source_path), "exec")
bpy.context.preferences.filepaths.save_version = 0

namespace = {
    "__name__": "__main__",
    "__file__": str(source_path),
    "__package__": None,
    "__spec__": None,
    "__builtins__": __builtins__,
}
exec(code, namespace, namespace)