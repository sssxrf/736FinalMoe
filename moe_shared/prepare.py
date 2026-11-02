"""Convert a pinned HF checkpoint once into addressable BF16 backing files.

Each expert is one raw binary file, with layouts and SHA256 in manifest.json.
Only one source tensor is exposed at a time; safetensors mappings are closed
between source shards. Conversion is outside experiment timing/pool accounting.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
from .artifacts import atomic_json, contained_file, no_symlinks

FORMAT_VERSION = 1
CONVERTER_VERSION = 2
EXPERT_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(.+)$")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def converter_hash():
    return sha256_file(__file__)


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
    backing = no_symlinks(backing)
    if manifest is None:
        manifest = json.loads((backing / "manifest.json").read_text())
    if manifest["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported backing format")
    for item in [*manifest["experts"].values(), *manifest["dense"].values()]:
        path = contained_file(backing, item["file"])
        if path.stat().st_size != item["nbytes"] or sha256_file(path) != item["sha256"]:
            raise IOError(f"backing verification failed: {path}")
        expected_offset = 0
        for tensor in item["layout"]:
            count = 1
            for size in tensor["shape"]:
                count *= size
            if tensor["offset_bytes"] != expected_offset or tensor["nbytes"] != count * 2:
                raise ValueError(f"invalid tensor layout: {path}")
            expected_offset += tensor["nbytes"]
        if expected_offset != item["nbytes"]:
            raise ValueError(f"layout does not cover backing file: {path}")
    return manifest


def _valid_entry(directory, item):
    try:
        path = contained_file(directory, item["file"])
        return path.is_file() and path.stat().st_size == item["nbytes"] and sha256_file(path) == item["sha256"]
    except (OSError, ValueError, KeyError):
        return False


def verified_progress_bytes(backing):
    """Count verified completed payloads, never trust partial lengths for capacity planning."""
    backing = no_symlinks(backing)
    if (backing / "manifest.json").is_file():
        manifest = json.loads((backing / "manifest.json").read_text())
        entries = [*manifest["experts"].values(), *manifest["dense"].values()]
        return sum(item["nbytes"] for item in entries if _valid_entry(backing, item))
    staging = backing.with_name(backing.name + ".preparing")
    if not (staging / ".progress.json").is_file():
        return 0
    result = 0
    for receipt in (staging / ".receipts").glob("*.json"):
        try:
            item = json.loads(receipt.read_text())
            if _valid_entry(staging, item):
                result += item["nbytes"]
        except (OSError, ValueError):
            continue
    return result


def prepare_model(model_dir, backing, storage_root=None, *, _after_file=None):
    import torch
    from safetensors import safe_open

    model_dir, backing = no_symlinks(model_dir), no_symlinks(backing)
    storage_root = no_symlinks(storage_root or backing.parent)
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
    converter_source = converter_hash()
    if (backing / "manifest.json").is_file():
        old = json.loads((backing / "manifest.json").read_text())
        if old["config_sha256"] != config_sha or old["source_sha256"] != source_hashes:
            raise ValueError("existing backing belongs to a different model checkpoint")
        if old.get("converter_source_sha256") != converter_source:
            raise ValueError("existing backing uses a different converter; select a new converter-keyed directory")
        return verify_backing(backing, old)
    if backing.exists():
        raise FileExistsError(f"refusing to overwrite incomplete/unrelated backing: {backing}")

    # Stable task-owned staging and file receipts survive interruption. Existing
    # completed payloads are verified and reused; only unfinished files are written.
    staging = no_symlinks(backing.with_name(backing.name + ".preparing"))
    context = {"format_version": FORMAT_VERSION, "converter_version": CONVERTER_VERSION,
               "converter_source_sha256": converter_source,
               "config_sha256": config_sha, "source_sha256": source_hashes}
    progress = staging / ".progress.json"
    if staging.exists():
        if not progress.is_file() or json.loads(progress.read_text()) != context:
            raise ValueError("incomplete backing belongs to another source/converter or is unmanaged")
    else:
        staging.mkdir(parents=True)
        atomic_json(progress, context)
    for name in ("experts", "dense", ".receipts"):
        no_symlinks(staging / name).mkdir(exist_ok=True)
    remaining = max(0, sum((model_dir / name).stat().st_size for name in source_files)
                    - verified_progress_bytes(backing))
    _space_guard(storage_root, remaining + 256 * 1024 * 1024)
    manifest = {"format_version": FORMAT_VERSION, "dtype": "bfloat16",
                "converter_version": CONVERTER_VERSION, "converter_source_sha256": converter_source,
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

    reused_files, written_files = 0, 0
    for (kind, key), names in groups.items():
        relpath = (f"experts/{key}.bin" if kind == "expert"
                   else f"dense/{key}.bin")
        receipt = staging / ".receipts" / f"{kind}-{key}.json"
        category = "experts" if kind == "expert" else "dense"
        if receipt.is_file():
            try:
                existing = json.loads(receipt.read_text())
            except (OSError, ValueError):
                existing = {}
            if existing.get("file") == relpath and _valid_entry(staging, existing):
                manifest[category][key] = existing
                reused_files += 1
                continue
        layout = []
        offset = 0
        target = contained_file(staging, relpath)
        partial = target.with_name(target.name + ".partial")
        no_symlinks(partial)
        with partial.open("wb", buffering=0) as out:
            # Reopen mmap per tensor: no long-lived shard mapping or hidden state dict.
            for full_name, short_name in names:
                with safe_open(model_dir / weight_map[full_name], framework="pt", device="cpu") as src:
                    tensor = src.get_tensor(full_name)
                    if not tensor.is_floating_point():
                        raise ValueError(f"unexpected nonfloating model weight {full_name}")
                    tensor = tensor.to(torch.bfloat16).contiguous()
                    nbytes = tensor.numel() * tensor.element_size()
                    # Shared filesystem headroom can change during conversion.
                    if shutil.disk_usage(storage_root).free - nbytes < 20 * 1024**3:
                        raise RuntimeError("Filesystem headroom fell below 20 GiB during conversion")
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
                "sha256": sha256_file(partial)}
        os.replace(partial, target)
        atomic_json(receipt, item)
        manifest[category][key] = item
        written_files += 1
        if written_files % 64 == 0:
            _space_guard(storage_root, 0)
        if _after_file is not None:
            _after_file(target)
    manifest["max_expert_bytes"] = max(e["nbytes"] for e in manifest["experts"].values())
    manifest["expert_bytes"] = sum(e["nbytes"] for e in manifest["experts"].values())
    manifest["dense_bytes"] = sum(e["nbytes"] for e in manifest["dense"].values())
    manifest["total_bytes"] = manifest["expert_bytes"] + manifest["dense_bytes"]
    manifest["recovery"] = {"reused_verified_files": reused_files, "written_files": written_files}
    verify_backing(staging, manifest)
    atomic_json(staging / "manifest.json", manifest)
    staging.rename(backing)
    _space_guard(storage_root, 0)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--storage-root")
    args = parser.parse_args()
    manifest = prepare_model(args.model_dir, args.output, storage_root=args.storage_root)
    print(json.dumps({key: manifest[key] for key in
                      ("format_version", "expert_bytes", "dense_bytes", "max_expert_bytes")}, indent=2))


if __name__ == "__main__":
    main()
