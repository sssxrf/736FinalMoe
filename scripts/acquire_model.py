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
from moe_shared.audit import GiB, check_space, directory_bytes


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def acquire(config_path, scratch):
    from huggingface_hub import HfApi, snapshot_download

    config = json.loads(Path(config_path).read_text())
    revision = config["revision"]
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("Model revision must be an immutable full Git commit")
    scratch = Path(scratch).resolve()
    target = scratch / "model"
    provenance = target / "model-provenance.json"
    if provenance.exists():
        saved = json.loads(provenance.read_text())
        if saved["model_id"] != config["model_id"] or saved["revision"] != revision:
            raise RuntimeError("Existing model belongs to a different revision; select another task scratch directory")
        for name, record in saved["files"].items():
            path = target / name
            if not path.is_file() or path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
                raise RuntimeError(f"Pinned model content changed or missing: {name}")
        check_space(scratch)
        print(json.dumps({"model": str(target), "revision": revision, "reuse": "verified"}), flush=True)
        return

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
    # Completed files can be reused by the Hub client. Count incomplete files
    # conservatively (their existing bytes remain in the task total).
    present_bytes = sum((target / item.rfilename).stat().st_size for item in selected
                        if (target / item.rfilename).is_file()
                        and (target / item.rfilename).stat().st_size == item.size)
    check_space(scratch, additional_bytes=download_bytes - present_bytes + conversion_bytes + 256 * 1024 ** 2)
    target.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(scratch / "hf-metadata"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    print(json.dumps({"download_bytes": download_bytes, "estimated_conversion_bytes": conversion_bytes,
                      "scratch": str(scratch), "revision": revision}), flush=True)
    # local_dir uses the target files directly and only stores small metadata under
    # .cache; there is no second model-sized hub blob cache.
    snapshot_download(repo_id=config["model_id"], revision=revision, local_dir=str(target),
                      allow_patterns=[item.rfilename for item in selected], max_workers=2)
    files = {}
    for item in selected:
        path = target / item.rfilename
        digest = sha256_file(path)
        if path.stat().st_size != item.size:
            raise RuntimeError(f"Downloaded size mismatch for {item.rfilename}")
        expected = getattr(item.lfs, "sha256", None) if item.lfs else None
        if isinstance(item.lfs, dict):
            expected = item.lfs.get("sha256")
        if expected is not None and digest != expected:
            raise RuntimeError(f"Downloaded LFS SHA256 mismatch for {item.rfilename}")
        files[item.rfilename] = {"bytes": item.size, "sha256": digest, "upstream_lfs_sha256": expected}
    saved = {"schema_version": 1, "model_id": config["model_id"], "revision": revision,
             "downloaded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
             "source": config["revision_source"], "files": files,
             "scratch_guard": check_space(scratch), "model_directory_bytes": directory_bytes(target)}
    temporary = provenance.with_suffix(".tmp")
    temporary.write_text(json.dumps(saved, indent=2) + "\n")
    temporary.replace(provenance)
    print(json.dumps({"model": str(target), "revision": revision, "verified_files": len(files)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model.json")
    parser.add_argument("--scratch", required=True)
    args = parser.parse_args()
    acquire(args.config, args.scratch)


if __name__ == "__main__":
    main()
