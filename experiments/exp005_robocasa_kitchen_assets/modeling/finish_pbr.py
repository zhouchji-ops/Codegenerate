import json
import os
import struct
import sys
import tempfile

ROUGHNESS = {
    "Satin brushed silver": 0.29,
    "Champagne stainless": 0.23,
    "Black molded elastomer": 0.38,
}
JSON = 0x4E4F534A
BIN = 0x004E4942


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def bad_constant(value):
    raise ValueError("Invalid JSON constant: " + value)


DECODER = json.JSONDecoder(
    object_pairs_hook=unique_object,
    parse_constant=bad_constant,
)


def skip_space(text, pos):
    while pos < len(text) and text[pos] in " \t\r\n":
        pos += 1
    return pos


def members(text, pos):
    """Return object member value spans and the closing brace position."""
    require(text[pos] == "{", "Expected JSON object")
    result = {}
    pos = skip_space(text, pos + 1)
    while text[pos] != "}":
        key, pos = DECODER.raw_decode(text, pos)
        pos = skip_space(text, pos)
        require(text[pos] == ":", "Expected colon")
        start = skip_space(text, pos + 1)
        _, end = DECODER.raw_decode(text, start)
        result[key] = (start, end)
        pos = skip_space(text, end)
        if text[pos] == "}":
            break
        require(text[pos] == ",", "Expected comma")
        pos = skip_space(text, pos + 1)
    return result, pos


def elements(text, pos):
    require(text[pos] == "[", "Expected JSON array")
    result = []
    pos = skip_space(text, pos + 1)
    while text[pos] != "]":
        start = pos
        _, end = DECODER.raw_decode(text, start)
        result.append(start)
        pos = skip_space(text, end)
        if text[pos] == "]":
            break
        require(text[pos] == ",", "Expected comma")
        pos = skip_space(text, pos + 1)
    return result


def factor(value):
    return type(value) in (int, float) and 0 <= value <= 1


def read_glb(data):
    require(len(data) >= 20, "Truncated GLB")
    magic, version, total = struct.unpack_from("<4sII", data)
    require(magic == b"glTF", "Invalid GLB magic")
    require(version == 2, "Only GLB version 2 is supported")
    require(total == len(data), "Incorrect GLB length")
    require(total % 4 == 0, "Unaligned GLB length")

    chunks = []
    pos = 12
    while pos < total:
        require(pos + 8 <= total, "Truncated chunk header")
        length, kind = struct.unpack_from("<II", data, pos)
        require(length % 4 == 0, "Unaligned chunk length")
        end = pos + 8 + length
        require(end <= total, "Truncated chunk")
        chunks.append((kind, data[pos:end], data[pos + 8:end]))
        pos = end

    require(chunks[0][0] == JSON, "First chunk must be JSON")
    require(sum(c[0] == JSON for c in chunks) == 1,
            "Expected exactly one JSON chunk")
    bins = [i for i, c in enumerate(chunks) if c[0] == BIN]
    require(not bins or bins == [1], "BIN must occur once, second")

    text = chunks[0][2].decode("utf-8")
    require(not text.startswith("\ufeff"), "JSON BOM is forbidden")
    root, end = DECODER.raw_decode(text, skip_space(text, 0))
    require(all(c == " " for c in text[end:]),
            "JSON chunk padding must contain only spaces")
    require(isinstance(root, dict), "glTF root must be an object")
    asset = root.get("asset")
    require(isinstance(asset, dict), "Missing glTF asset")
    require(asset.get("version") == "2.0", "Only glTF 2.0 is supported")
    require(asset.get("minVersion", "2.0") == "2.0",
            "Unsupported glTF minimum version")

    buffers = root.get("buffers", [])
    require(isinstance(buffers, list), "buffers must be an array")
    for i, buffer in enumerate(buffers):
        require(isinstance(buffer, dict), "Invalid buffer")
        size = buffer.get("byteLength")
        require(type(size) is int and size > 0, "Invalid buffer byteLength")
        if "uri" in buffer:
            require(isinstance(buffer["uri"], str), "Invalid buffer URI")
        else:
            require(i == 0 and bins == [1], "Missing buffer URI or BIN")
            payload = chunks[1][2]
            require(size <= len(payload) <= size + 3,
                    "BIN length disagrees with buffer byteLength")
            require(payload[size:] == b"\0" * (len(payload) - size),
                    "Invalid BIN padding")
    if bins:
        require(buffers and "uri" not in buffers[0],
                "BIN requires an embedded first buffer")

    materials = root.get("materials")
    require(isinstance(materials, list) and materials,
            "Missing materials array")
    for material in materials:
        require(isinstance(material, dict), "Invalid material")
        name = material.get("name")
        require(isinstance(name, str) and name in ROUGHNESS,
                "Unknown material: " + repr(name))
        pbr = material.get("pbrMetallicRoughness", {})
        require(isinstance(pbr, dict), "Invalid pbrMetallicRoughness")
        for key in ("metallicFactor", "roughnessFactor"):
            require(key not in pbr or factor(pbr[key]),
                    "Invalid " + key)
    return text, root, chunks


def patched(data):
    text, root, chunks = read_glb(data)
    fields, _ = members(text, skip_space(text, 0))
    starts = elements(text, fields["materials"][0])
    edits = []

    for material, start in zip(root["materials"], starts):
        value = str(ROUGHNESS[material["name"]])
        fields, close = members(text, start)
        if "pbrMetallicRoughness" not in fields:
            addition = '"pbrMetallicRoughness":{"roughnessFactor":' + value + "}"
            edits.append((close, close, ("," if fields else "") + addition))
            continue
        pbr_start, _ = fields["pbrMetallicRoughness"]
        pbr_fields, pbr_close = members(text, pbr_start)
        if "roughnessFactor" in pbr_fields:
            if material["pbrMetallicRoughness"]["roughnessFactor"] != float(value):
                a, b = pbr_fields["roughnessFactor"]
                edits.append((a, b, value))
        else:
            addition = '"roughnessFactor":' + value
            edits.append((pbr_close, pbr_close,
                          ("," if pbr_fields else "") + addition))

    if not edits:
        return data
    for a, b, replacement in sorted(edits, reverse=True):
        text = text[:a] + replacement + text[b:]
    payload = text.encode("utf-8")
    payload += b" " * (-len(payload) % 4)
    tail = b"".join(chunk[1] for chunk in chunks[1:])
    total = 20 + len(payload) + len(tail)
    require(total <= 0xFFFFFFFF, "GLB exceeds 32-bit length")
    result = (
        struct.pack("<4sII", b"glTF", 2, total)
        + struct.pack("<II", len(payload), JSON)
        + payload
        + tail
    )
    read_glb(result)
    return result


def atomic_write(path, original, replacement):
    if replacement == original:
        return
    require(not os.path.islink(path), "Refusing symbolic link: " + path)
    mode = os.stat(path).st_mode
    fd, temporary = tempfile.mkstemp(
        prefix="." + os.path.basename(path) + ".",
        suffix=".tmp",
        dir=os.path.dirname(path),
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(replacement)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        with open(path, "rb") as stream:
            require(stream.read() == original,
                    "File changed during processing: " + path)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    require(len(sys.argv) == 2, "Usage: script.py FOLDER")
    folder = os.path.abspath(sys.argv[1])
    pending = []
    for name in ("scoop.glb", "turner.glb"):
        path = os.path.join(folder, name)
        require(not os.path.islink(path), "Refusing symbolic link: " + path)
        with open(path, "rb") as stream:
            original = stream.read()
        pending.append((path, original, patched(original)))
    # Each replacement is atomic; the pair is not a filesystem transaction.
    for path, original, replacement in pending:
        atomic_write(path, original, replacement)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, IndexError, struct.error) as error:
        sys.exit("Error: " + str(error))