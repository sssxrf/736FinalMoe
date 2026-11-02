"""Download and verify an immutable, minimal HF model snapshot on Linux only."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
from pathlib import Path

# Direct script execution does not otherwise include the release root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moe_shared.audit import check_space, directory_bytes
from moe_shared.artifacts import atomic_json, contained_file, no_symlinks, verify_model


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _upstream_hash(item):
    if isinstance(item.lfs, dict):
        return item.lfs.get("sha256")
    return getattr(item.lfs, "sha256", None) if item.lfs else None


def _verified_upstream(path, item):
    if not path.is_file() or path.stat().st_size != item.size:
        return False
    expected = _upstream_hash(item)
    if expected:
        return sha256_file(path) == expected
    blob_id = getattr(item, "blob_id", None)
    if blob_id:
        data = path.read_bytes()
        return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == blob_id
    return False


def resumable_download_bytes(target, item):
    """Existing exact-etag incomplete bytes occupy space but need not be duplicated.

    Layout is pinned to huggingface_hub 0.26.2 LocalDownloadFilePaths. These bytes
    are capacity credit only; final upstream size/hash verification is mandatory.
    """
    if (target / item.rfilename).exists():
        return 0  # Invalid existing targets force a new download, not append.
    etag = _upstream_hash(item) or getattr(item, "blob_id", None)
    if not etag or not re.fullmatch(r"[a-f0-9]{40}|[a-f0-9]{64}", etag):
        return 0
    partial = contained_file(target, f".cache/huggingface/download/{item.rfilename}.{etag}.incomplete")
    if partial.is_file():
        return min(partial.stat().st_size, item.size)
    return 0


def acquire(config_path, scratch=None, *, model_dir=None, artifact_root=None, backing_dir=None):
    from huggingface_hub import HfApi, hf_hub_download
    from concurrent.futures import ThreadPoolExecutor

    config = json.loads(Path(config_path).read_text())
    revision = config["revision"]
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("Model revision must be an immutable full Git commit")
    if model_dir is None:
        if scratch is None:
            raise ValueError("Specify a persistent model_dir or legacy scratch")
        target = no_symlinks(Path(scratch) / "model")
        storage_root = no_symlinks(scratch)
    else:
        target = no_symlinks(model_dir)
        storage_root = no_symlinks(artifact_root or target.parent)
    storage_root.mkdir(parents=True, exist_ok=True)
    provenance = target / "model-provenance.json"
    if provenance.exists():
        saved = json.loads(provenance.read_text())
        if saved["model_id"] != config["model_id"] or saved["revision"] != revision:
            raise RuntimeError("Existing model belongs to a different revision; select another task scratch directory")
        try:
            verify_model(target, saved)
        except (OSError, ValueError):
            print("Checkpoint is incomplete or changed; verifying individual upstream files before repair", flush=True)
        else:
            check_space(storage_root)
            print(json.dumps({"model": str(target), "revision": revision, "reuse": "verified"}), flush=True)
            return saved

    info = HfApi().model_info(config["model_id"], revision=revision, files_metadata=True)
    if info.sha != revision:
        raise RuntimeError(f"Hub revision mismatch: {info.sha} != {revision}")
    selected = [item for item in info.siblings if
                item.rfilename.endswith(".safetensors") or
                item.rfilename in {"config.json", "generation_config.json", "model.safetensors.index.json",
                                   "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}]
    if not selected or any(item.size is None for item in selected):
        raise RuntimeError("Hub metadata did not provide file sizes")
    download_bytes = sum(item.size for item in selected)
    conversion_bytes = sum(item.size for item in selected if item.rfilename.endswith(".safetensors"))
    # Reuse only content verified against upstream LFS SHA256 or Git blob ID.
    # Invalid/incomplete files are replaced by the Hub's temporary-download path.
    valid = {item.rfilename for item in selected
             if _verified_upstream(contained_file(target, item.rfilename), item)}
    present_bytes = sum(item.size for item in selected if item.rfilename in valid)
    resumable_bytes = sum(resumable_download_bytes(target, item) for item in selected
                          if item.rfilename not in valid)
    converted_valid = 0
    if backing_dir is not None:
        from moe_shared.prepare import verified_progress_bytes
        converted_valid = verified_progress_bytes(backing_dir)
    conversion_remaining = max(0, conversion_bytes - converted_valid)
    check_space(storage_root, additional_bytes=download_bytes - present_bytes - resumable_bytes
                + conversion_remaining + 256 * 1024 ** 2)
    target.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(storage_root / "hf-metadata"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    print(json.dumps({"download_bytes": download_bytes, "estimated_conversion_bytes": conversion_bytes,
                      "artifact_root": str(storage_root), "model_dir": str(target),
                      "reused_verified_files": len(valid), "resumable_download_bytes": resumable_bytes,
                      "revision": revision}), flush=True)
    # local_dir uses the target files directly and only stores small metadata under
    # .cache; there is no second model-sized hub blob cache.
    needed = [item.rfilename for item in selected if item.rfilename not in valid]
    if needed:
        def download(name):
            # Missing targets can resume existing .incomplete files. A present
            # but invalid target must bypass local Hub metadata and be replaced.
            result = hf_hub_download(repo_id=config["model_id"], revision=revision,
                                     filename=name, local_dir=str(target),
                                     force_download=(target / name).exists())
            check_space(storage_root)
            return result
        with ThreadPoolExecutor(max_workers=2) as workers:
            list(workers.map(download, needed))
    files = {}
    for item in selected:
        path = target / item.rfilename
        digest = sha256_file(path)
        if path.stat().st_size != item.size:
            raise RuntimeError(f"Downloaded size mismatch for {item.rfilename}")
        expected = _upstream_hash(item)
        if not _verified_upstream(path, item):
            raise RuntimeError(f"Downloaded upstream content hash mismatch for {item.rfilename}")
        files[item.rfilename] = {"bytes": item.size, "sha256": digest, "upstream_lfs_sha256": expected,
                                "upstream_git_blob": getattr(item, "blob_id", None)}
    saved = {"schema_version": 1, "model_id": config["model_id"], "revision": revision,
             "downloaded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
             "source": config["revision_source"], "files": files,
             "artifact_guard": check_space(storage_root), "model_directory_bytes": directory_bytes(target),
             "reused_verified_files": len(valid)}
    atomic_json(provenance, saved)
    print(json.dumps({"model": str(target), "revision": revision, "verified_files": len(files)}), flush=True)
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model.json")
    parser.add_argument("--scratch", help="Legacy fixture path; production setup uses persistent --model-dir")
    parser.add_argument("--model-dir")
    parser.add_argument("--artifact-root")
    args = parser.parse_args()
    if not args.scratch and not args.model_dir:
        parser.error("Specify --model-dir (persistent production storage) or --scratch")
    acquire(args.config, args.scratch, model_dir=args.model_dir, artifact_root=args.artifact_root)


if __name__ == "__main__":
    main()
