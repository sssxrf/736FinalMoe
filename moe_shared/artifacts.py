"""Persistent, verified checkpoint/conversion store, independent of run staging."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil

from .audit import GiB, check_space, command, directory_bytes


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def no_symlinks(path):
    path = Path(path).absolute()
    for candidate in [*reversed(path.parents), path]:
        if candidate.is_symlink():
            raise ValueError(f"Canonical artifact path contains a symlink: {candidate}")
    return path


def contained_file(root, relative):
    root = no_symlinks(root)
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe manifest path: {relative}")
    path = no_symlinks(root / relative)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Artifact escapes its store: {path}")
    return path


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    no_symlinks(temporary)
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)


def sync_directory(path):
    if os.name == "posix":
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def publish_completion(path, complete):
    """Preserve a completed artifact identity byte-for-byte on verified reuse."""
    path = no_symlinks(path)
    observations = {"completed_at_utc", "artifact_guard"}
    identity = {key: value for key, value in complete.items() if key not in observations}
    if path.is_file():
        previous = json.loads(path.read_text())
        old_identity = {key: value for key, value in previous.items() if key not in observations}
        if identity == old_identity:
            return previous, True
    atomic_json(path, complete)
    return complete, False


def verify_model(model_dir, provenance=None):
    model_dir = no_symlinks(model_dir)
    if provenance is None:
        provenance = json.loads(contained_file(model_dir, "model-provenance.json").read_text())
    for name, entry in provenance["files"].items():
        path = contained_file(model_dir, name)
        if not path.is_file() or path.stat().st_size != entry["bytes"] or file_hash(path) != entry["sha256"]:
            raise IOError(f"Checkpoint verification failed: {path}")
    return provenance


def resolve_artifacts(root, verify=True):
    """Resolve the atomic completed store pointer; no /tmp links are permitted."""
    root = no_symlinks(root)
    complete = json.loads(contained_file(root, "complete.json").read_text())
    if complete.get("schema_version") != 1 or complete.get("complete") is not True:
        raise ValueError("Persistent artifact store has no completed preparation")
    model_dir = contained_file(root, complete["model_dir"])
    backing_dir = contained_file(root, complete["backing_dir"])
    for path, expected in ((model_dir / "model-provenance.json", complete["model_provenance_sha256"]),
                           (backing_dir / "manifest.json", complete["backing_manifest_sha256"])):
        no_symlinks(path)
        if file_hash(path) != expected:
            raise IOError(f"Completed artifact metadata changed: {path}")
    provenance = json.loads((model_dir / "model-provenance.json").read_text())
    manifest = json.loads((backing_dir / "manifest.json").read_text())
    if provenance["revision"] != complete["revision"] or provenance["model_id"] != complete["model_id"]:
        raise ValueError("Artifact completion/model revision mismatch")
    if manifest["converter_source_sha256"] != complete["converter_source_sha256"]:
        raise ValueError("Artifact converter identity mismatch")
    if verify:
        from .prepare import verify_backing
        verify_model(model_dir, provenance)
        verify_backing(backing_dir, manifest)
    return model_dir, backing_dir, complete


def prepare_store(root, config_path, output, scratch):
    from .prepare import CONVERTER_VERSION, FORMAT_VERSION, converter_hash, prepare_model
    from .lifecycle import identity_checkpoint
    # The acquisition script also supports direct CLI use; load its local source.
    import importlib.util
    script = Path(__file__).resolve().parents[1] / "scripts" / "acquire_model.py"
    spec = importlib.util.spec_from_file_location("acquire_model", script)
    acquire_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(acquire_module)

    root = no_symlinks(root)
    if os.name == "posix" and root.resolve().is_relative_to(Path("/tmp")):
        raise ValueError("Canonical artifacts must be outside disposable /tmp")
    root.mkdir(parents=True, exist_ok=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(config_path).read_text())
    source_hash = converter_hash()
    model_dir = contained_file(root, config["revision"] + "/model")
    backing_dir = contained_file(root, config["revision"] + "/backing-" + source_hash[:16])
    paths = {"artifacts": root, "model": model_dir, "backing": backing_dir, "scratch": Path(scratch)}
    identity_checkpoint(output, "preparation-before", {"artifacts": root, "scratch": Path(scratch)}, create_sentinels=True)
    usage = shutil.disk_usage(root)
    atomic_json(output / "artifact-storage-before.json", {
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "artifact_root": str(root), "artifact_bytes": directory_bytes(root),
        "filesystem_total_bytes": usage.total, "filesystem_used_bytes": usage.used,
        "filesystem_free_bytes": usage.free,
        "account_home_bytes": directory_bytes(Path.home()),
        "runtime_bytes": directory_bytes(root.parent),
        "quota_query": command(["quota", "-s"]),
        "quota_note": "statvfs reports mounted filesystem headroom; quota command may be unavailable. All other home usage is included in free bytes.",
        "required_headroom_bytes": 20 * GiB,
        "space_guard": check_space(root),
    })
    acquire_module.acquire(config_path, model_dir=model_dir, artifact_root=root, backing_dir=backing_dir)
    identity_checkpoint(output, "checkpoint-complete", paths)
    provenance = verify_model(model_dir)
    atomic_json(output / "model-provenance.json", provenance)
    atomic_json(output / "checkpoint-stage.json", {
        "complete": True, "model_id": provenance["model_id"], "revision": provenance["revision"],
        "model_dir": str(model_dir), "model_provenance_sha256": file_hash(model_dir / "model-provenance.json"),
        "verified_files": len(provenance["files"]),
    })
    manifest = prepare_model(model_dir, backing_dir, storage_root=root)
    # The converter atomically renamed its fully verified directory. Persist that
    # parent directory entry before publishing the store's completion pointer.
    sync_directory(backing_dir.parent)
    identity_checkpoint(output, "conversion-complete", paths)
    complete = {"schema_version": 1, "complete": True,
                "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "model_id": config["model_id"], "revision": config["revision"],
                "model_dir": model_dir.relative_to(root).as_posix(),
                "backing_dir": backing_dir.relative_to(root).as_posix(),
                "converter_format_version": FORMAT_VERSION, "converter_source_sha256": source_hash,
                "converter_version": CONVERTER_VERSION,
                "model_provenance_sha256": file_hash(model_dir / "model-provenance.json"),
                "backing_manifest_sha256": file_hash(backing_dir / "manifest.json"),
                "expert_count": len(manifest["experts"]), "backing_bytes": manifest["total_bytes"],
                "artifact_guard": check_space(root)}
    complete, reused_completion = publish_completion(root / "complete.json", complete)
    resolve_artifacts(root, verify=True)
    atomic_json(output / "artifact-complete.json", complete)
    atomic_json(output / "artifact-reuse.json", {
        "verified_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "reused_completion_manifest": reused_completion,
        "completion_sha256": file_hash(root / "complete.json"),
        "current_artifact_guard": check_space(root),
    })
    shutil.copyfile(model_dir / "model-provenance.json", output / "model-provenance.json")
    shutil.copyfile(backing_dir / "manifest.json", output / "preparation-manifest.json")
    identity_checkpoint(output, "preparation-after", paths)
    print(json.dumps({"persistent_artifacts_complete": True, "model_dir": str(model_dir),
                      "backing_dir": str(backing_dir), "artifact_guard": complete["artifact_guard"]}), flush=True)
    return complete


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "verify"])
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", default="configs/model.json")
    parser.add_argument("--output")
    parser.add_argument("--scratch")
    args = parser.parse_args()
    if args.action == "verify":
        model, backing, complete = resolve_artifacts(args.root)
        print(json.dumps({"model_dir": str(model), "backing_dir": str(backing), "complete": complete}))
    else:
        if not args.output or not args.scratch:
            parser.error("prepare requires --output and --scratch")
        prepare_store(args.root, args.config, args.output, args.scratch)


if __name__ == "__main__":
    main()
