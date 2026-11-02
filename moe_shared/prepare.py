"""Convert a pinned HF checkpoint once into addressable BF16 backing files.

Each expert is one raw binary file, with layouts and SHA256 in manifest.json.
Only one source tensor is exposed at a time; safetensors mappings are closed
between source shards. Conversion is outside experiment timing/pool accounting.
"""

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import re
import time
import uuid

FORMAT_VERSION = 1
EXPERT_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(.+)$")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_bytes(tensor):
    """A zero-copy Python view for contiguous CPU tensor I/O (including BF16)."""
    import torch
    if tensor.device.type != "cpu" or not tensor.is_contiguous():
        raise ValueError("raw I/O requires a contiguous CPU tensor")
    return memoryview(tensor.view(torch.uint8).numpy()).cast("B")


def read_tensor_file(path, tensor):
    view = tensor_bytes(tensor)
    with Path(path).open("rb", buffering=0) as stream:
        offset = 0
        while offset < len(view):
            count = stream.readinto(view[offset:])
            if not count:
                raise IOError(f"truncated backing file: {path}")
            offset += count
        if stream.read(1):
            raise IOError(f"oversized backing file: {path}")


def _space_guard(scratch, additional_bytes):
    from .audit import check_space
    return check_space(scratch, additional_bytes=additional_bytes)


def verify_backing(backing, manifest=None):
    backing = Path(backing)
    if manifest is None:
        manifest = json.loads((backing / "manifest.json").read_text())
    if manifest["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported backing format")
    for item in [*manifest["experts"].values(), *manifest["dense"].values()]:
        path = backing / item["file"]
        if path.stat().st_size != item["nbytes"] or sha256_file(path) != item["sha256"]:
            raise IOError(f"backing verification failed: {path}")
    return manifest


def prepare_model(model_dir, backing):
    import torch
    from safetensors import safe_open

    model_dir, backing = Path(model_dir), Path(backing)
    config_sha = sha256_file(model_dir / "config.json")
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
    else:
        files = sorted(model_dir.glob("*.safetensors"))
        if len(files) != 1:
            raise ValueError("expected HF safetensors checkpoint/index")
        with safe_open(files[0], framework="pt", device="cpu") as handle:
            weight_map = {key: files[0].name for key in handle.keys()}
    source_files = sorted(set(weight_map.values()))
    source_hashes = {name: sha256_file(model_dir / name) for name in source_files}
    if (backing / "manifest.json").is_file():
        old = json.loads((backing / "manifest.json").read_text())
        if old["config_sha256"] != config_sha or old["source_sha256"] != source_hashes:
            raise ValueError("existing backing belongs to a different model checkpoint")
        return verify_backing(backing, old)
    if backing.exists():
        raise FileExistsError(f"refusing to overwrite incomplete/unrelated backing: {backing}")

    # BF16 output is no larger than the BF16/FP32 source checkpoint used here.
    _space_guard(backing.parent, sum((model_dir / name).stat().st_size for name in source_files))
    staging = backing.with_name(backing.name + ".preparing-" + uuid.uuid4().hex[:10])
    staging.mkdir(parents=True, exist_ok=False)
    (staging / "experts").mkdir()
    (staging / "dense").mkdir()
    manifest = {"format_version": FORMAT_VERSION, "dtype": "bfloat16",
                "config_sha256": config_sha, "source_sha256": source_hashes,
                "experts": {}, "dense": {}, "created_unix": time.time(),
                "conversion_scope": "outside timed experiments and application host pool"}
    for candidate in ("model-provenance.json", "provenance.json", "acquisition.json"):
        if (model_dir / candidate).is_file():
            manifest["model_provenance"] = json.loads((model_dir / candidate).read_text())
            break

    groups = {}
    for name in sorted(weight_map):
        match = EXPERT_RE.match(name)
        if match:
            key = f"{match[1]}.{match[2]}"
            groups.setdefault(("expert", key), []).append((name, match[3]))
        else:
            groups[("dense", name)] = [(name, name)]

    for (kind, key), names in groups.items():
        relpath = (f"experts/{key}.bin" if kind == "expert"
                   else f"dense/{key}.bin")
        layout = []
        offset = 0
        with (staging / relpath).open("xb", buffering=0) as out:
            # Reopen mmap per tensor: no long-lived shard mapping or hidden state dict.
            for full_name, short_name in names:
                with safe_open(model_dir / weight_map[full_name], framework="pt", device="cpu") as src:
                    tensor = src.get_tensor(full_name)
                    if not tensor.is_floating_point():
                        raise ValueError(f"unexpected nonfloating model weight {full_name}")
                    tensor = tensor.to(torch.bfloat16).contiguous()
                    nbytes = tensor.numel() * tensor.element_size()
                    view = tensor_bytes(tensor)
                    written = 0
                    while written < nbytes:
                        count = out.write(view[written:])
                        if not count:
                            raise IOError("short backing write")
                        written += count
                    layout.append({"name": short_name, "shape": list(tensor.shape),
                                   "offset_bytes": offset, "nbytes": nbytes})
                    offset += nbytes
                    del view, tensor
            os.fsync(out.fileno())
        item = {"file": relpath, "nbytes": offset, "layout": layout,
                "sha256": sha256_file(staging / relpath)}
        manifest["experts" if kind == "expert" else "dense"][key] = item
    manifest["max_expert_bytes"] = max(e["nbytes"] for e in manifest["experts"].values())
    manifest["expert_bytes"] = sum(e["nbytes"] for e in manifest["experts"].values())
    manifest["dense_bytes"] = sum(e["nbytes"] for e in manifest["dense"].values())
    manifest["total_bytes"] = manifest["expert_bytes"] + manifest["dense_bytes"]
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    staging.rename(backing)
    _space_guard(backing.parent, 0)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = prepare_model(args.model_dir, args.output)
    print(json.dumps({key: manifest[key] for key in
                      ("format_version", "expert_bytes", "dense_bytes", "max_expert_bytes")}, indent=2))


if __name__ == "__main__":
    main()
