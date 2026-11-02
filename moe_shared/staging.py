"""Verified disposable backing copies; canonical artifacts are never removed."""
from __future__ import annotations

import hashlib
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import time

from .audit import check_space
from .lifecycle import identity_checkpoint

OWNER_FILE = ".moe-staging-owner.json"


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_path(root, relative):
    relative = PurePosixPath(relative)
    if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative) or ":" in str(relative):
        raise ValueError("Unsafe backing manifest path")
    path = root.joinpath(*relative.parts)
    if path.is_symlink() or path.resolve() != path.absolute():
        raise ValueError(f"Backing path resolves through a symlink: {path}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Backing path escapes its root")
    return path


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def stage_backing(canonical_backing, run_owned_scratch, output_dir, scratch_root=None):
    """Reuse intact files and atomically repair missing/corrupt stage files.

    This method never deletes a directory or changes the canonical source. Its
    caller chooses a unique run-owned destination outside the canonical tree.
    """
    start = time.perf_counter()
    canonical, stage = Path(canonical_backing).absolute(), Path(run_owned_scratch).absolute()
    if canonical.resolve() != canonical or stage.resolve() != stage:
        raise ValueError("Canonical/stage directories must not resolve through symlinks")
    if stage == canonical or stage.is_relative_to(canonical) or canonical.is_relative_to(stage):
        raise ValueError("Canonical artifacts and disposable stage must not overlap")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    identity_checkpoint(output, "staging-before", {"canonical_backing": canonical, "stage": stage})
    raw_manifest = (canonical / "manifest.json").read_bytes()
    digest = hashlib.sha256(raw_manifest).hexdigest()
    manifest = json.loads(raw_manifest)
    entries = [*manifest["experts"].values(), *manifest["dense"].values()]
    files = {}
    for entry in entries:
        source = checked_path(canonical, entry["file"])
        if source.stat().st_size != entry["nbytes"] or file_hash(source) != entry["sha256"]:
            raise IOError(f"Canonical backing checksum mismatch: {source}")
        if entry["file"] in files:
            raise ValueError("Duplicate backing path")
        files[entry["file"]] = entry
    marker = stage / OWNER_FILE
    owner = {"kind": "cs736-disposable-backing-stage-v1", "canonical": str(canonical),
             "canonical_manifest_sha256": digest, "stage": str(stage)}
    if stage.exists() and not marker.is_file() and any(stage.iterdir()):
        raise ValueError("Refusing nonempty unmanaged stage directory")
    if marker.is_file() and json.loads(marker.read_text()) != owner:
        raise ValueError("Stage belongs to another canonical manifest")
    stage.mkdir(parents=True, exist_ok=True)
    _write_json(marker, owner)
    pending, reused = [], 0
    for relative, entry in files.items():
        target = checked_path(stage, relative)
        if target.is_file() and target.stat().st_size == entry["nbytes"] and file_hash(target) == entry["sha256"]:
            reused += 1
        else:
            pending.append((relative, entry))
    scope = Path(scratch_root) if scratch_root is not None else stage
    check_space(scope, additional_bytes=sum(entry["nbytes"] for _, entry in pending))
    copied_bytes = 0
    for relative, entry in pending:
        source, target = checked_path(canonical, relative), checked_path(stage, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".part")
        if temporary.is_symlink():
            raise ValueError("Stage temporary file is a symlink")
        check_space(scope, additional_bytes=entry["nbytes"])
        with source.open("rb") as reader, temporary.open("wb") as writer:
            for block in iter(lambda: reader.read(8 * 1024 * 1024), b""):
                writer.write(block)
            writer.flush()
            os.fsync(writer.fileno())
        if temporary.stat().st_size != entry["nbytes"] or file_hash(temporary) != entry["sha256"]:
            raise IOError(f"Stage copy verification failed: {temporary}")
        temporary.replace(target)
        copied_bytes += entry["nbytes"]
    for relative, entry in files.items():
        target = checked_path(stage, relative)
        if target.stat().st_size != entry["nbytes"] or file_hash(target) != entry["sha256"]:
            raise IOError(f"Final stage verification failed: {target}")
    temporary = stage / "manifest.json.part"
    with temporary.open("wb") as stream:
        stream.write(raw_manifest)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(stage / "manifest.json")
    report = {"complete": True, "canonical_manifest_sha256": digest,
              "canonical_backing": str(canonical), "staged_backing": str(stage),
              "verified_files": len(files), "reused_files": reused,
              "copied_files": len(pending), "copied_bytes": copied_bytes,
              "seconds": time.perf_counter()-start,
              "scope": "Preparation outside measured experiment timing",
              "cache_condition": "File hashing/copying warms filesystem cache; no cold-device claim",
              "space": check_space(scope)}
    _write_json(stage / "stage-complete.json", report)
    _write_json(output / "staging.json", report)
    identity_checkpoint(output, "staging-after", {"canonical_backing": canonical, "stage": stage})
    return stage


def remove_recovery_stage(stage, probe_root, canonical):
    """The only recursive deletion in production: an explicitly marked test copy."""
    stage, probe_root, canonical = (Path(value).absolute() for value in (stage, probe_root, canonical))
    if any(path.resolve() != path for path in (stage, probe_root, canonical)):
        raise ValueError("Recovery deletion paths must not contain symlinks")
    if stage != probe_root / "stage" or canonical == stage or canonical.is_relative_to(stage):
        raise ValueError("Refusing deletion outside the dedicated recovery stage")
    marker = json.loads((probe_root / ".moe-recovery-probe.json").read_text())
    if marker.get("kind") != "cs736-single-expert-recovery-probe" or marker.get("canonical") != str(canonical):
        raise ValueError("Recovery probe ownership mismatch")
    owner = json.loads((stage / OWNER_FILE).read_text())
    if owner.get("stage") != str(stage) or owner.get("canonical") != str(probe_root / "source-copy"):
        raise ValueError("Disposable stage ownership mismatch")
    if any(path.is_symlink() for path in stage.rglob("*")):
        raise ValueError("Refusing recovery deletion with unexpected symlinks")
    shutil.rmtree(stage)


def recovery_probe(artifacts, scratch, output):
    """Exercise loss/reconstruction with one actual pretrained expert (12 MiB)."""
    from .artifacts import resolve_artifacts
    _, canonical, completion = resolve_artifacts(artifacts, verify=False)
    canonical = canonical.absolute()
    scratch, output = Path(scratch).absolute(), Path(output).absolute()
    if scratch.resolve() != scratch:
        raise ValueError("Scratch root must not be a symlink")
    probe_root = scratch / "recovery" / output.name
    source_copy, stage = probe_root / "source-copy", probe_root / "stage"
    if canonical == probe_root or canonical.is_relative_to(probe_root) or probe_root.is_relative_to(canonical):
        raise ValueError("Recovery probe must not overlap canonical artifacts")
    if probe_root.exists():
        raise ValueError("Recovery probe requires a unique disposable directory")
    manifest_bytes = (canonical / "manifest.json").read_bytes()
    canonical_manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    key = sorted(manifest["experts"])[0]
    entry = manifest["experts"][key]
    canonical_file = checked_path(canonical, entry["file"])
    canonical_file_hash = file_hash(canonical_file)
    if canonical_file.stat().st_size != entry["nbytes"] or canonical_file_hash != entry["sha256"]:
        raise IOError("Canonical expert failed verification before recovery probe")
    check_space(scratch, additional_bytes=3 * entry["nbytes"] + len(manifest_bytes))
    probe_root.mkdir(parents=True)
    _write_json(probe_root / ".moe-recovery-probe.json", {
        "kind": "cs736-single-expert-recovery-probe", "canonical": str(canonical),
        "canonical_manifest_sha256": canonical_manifest_hash, "expert": key})
    output.mkdir(parents=True, exist_ok=True)
    report = {"complete": False, "passed": False,
              "scope": "Recovery of one real pretrained expert; not inference acceptance",
              "model_revision": completion["model_revision"] if "model_revision" in completion else completion.get("revision"),
              "expert": key, "expert_bytes": entry["nbytes"],
              "canonical_manifest_sha256": canonical_manifest_hash,
              "canonical_expert_sha256": canonical_file_hash, "probe_root": str(probe_root)}
    _write_json(output / "recovery.json", report)
    identity_checkpoint(output, "recovery-before", {"canonical": canonical, "probe": probe_root, "stage": stage})
    source_copy.mkdir()
    copied = checked_path(source_copy, entry["file"])
    copied.parent.mkdir(parents=True, exist_ok=True)
    with canonical_file.open("rb") as reader, copied.open("xb") as writer:
        for block in iter(lambda: reader.read(8 * 1024 * 1024), b""):
            writer.write(block)
        writer.flush()
        os.fsync(writer.fileno())
    if file_hash(copied) != canonical_file_hash:
        raise IOError("Recovery source-copy hash mismatch")
    reduced = {**manifest, "experts": {key: entry}, "dense": {},
               "expert_bytes": entry["nbytes"], "dense_bytes": 0,
               "total_bytes": entry["nbytes"], "max_expert_bytes": entry["nbytes"],
               "purpose": "single-expert disposable recovery fixture"}
    _write_json(source_copy / "manifest.json", reduced)
    stage_backing(source_copy, stage, output / "initial-stage", scratch)
    report["initial_stage_sha256"] = file_hash(checked_path(stage, entry["file"]))
    _write_json(output / "recovery.json", report)
    remove_recovery_stage(stage, probe_root, canonical)
    report["stage_exists_after_delete"] = stage.exists()
    report["canonical_unchanged_while_stage_missing"] = (
        file_hash(canonical_file) == canonical_file_hash
        and file_hash(canonical / "manifest.json") == canonical_manifest_hash)
    _write_json(output / "recovery.json", report)
    identity_checkpoint(output, "recovery-stage-deleted", {"canonical": canonical, "stage": stage})
    stage_backing(source_copy, stage, output / "reconstructed-stage", scratch)
    report["reconstructed_stage_sha256"] = file_hash(checked_path(stage, entry["file"]))
    report["canonical_unchanged_after_reconstruction"] = (
        file_hash(canonical_file) == canonical_file_hash
        and file_hash(canonical / "manifest.json") == canonical_manifest_hash)
    report["complete"] = True
    report["passed"] = (report["initial_stage_sha256"] == canonical_file_hash
                        and report["reconstructed_stage_sha256"] == canonical_file_hash
                        and not report["stage_exists_after_delete"]
                        and report["canonical_unchanged_while_stage_missing"]
                        and report["canonical_unchanged_after_reconstruction"])
    _write_json(output / "recovery.json", report)
    identity_checkpoint(output, "recovery-after", {"canonical": canonical, "probe": probe_root, "stage": stage})
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recovery", action="store_true", required=True)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--scratch", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = recovery_probe(args.artifacts, args.scratch, args.output)
    print(json.dumps(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
