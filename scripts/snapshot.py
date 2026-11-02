"""Explicit working-tree snapshots and verified compact result extraction (stdlib only)."""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import uuid

SOURCE_DIRS = {"moe_shared", "scripts", "configs", "tests", "docs", "summaries"}
SOURCE_ROOTS = {"README.md", "LICENSE", ".gitignore", ".gitattributes", "requirements.lock", "pyproject.toml"}
SOURCE_SUFFIXES = {".py", ".ps1", ".sh", ".json", ".md", ".toml", ".yaml", ".yml", ".txt", ".csv", ".lock"}
DENIED_PARTS = {".git", ".controller", ".venv", "venv", "__pycache__", "node_modules", "results", "weights", "cache", ".cache", "credentials", "secrets"}
MAX_FILE = 8 * 1024 * 1024
MAX_SOURCE = 64 * 1024 * 1024


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def allowed_source(relative: str) -> bool:
    p = PurePosixPath(relative)
    if p.is_absolute() or ".." in p.parts or any(part in DENIED_PARTS for part in p.parts):
        return False
    name = p.name.lower()
    if name.startswith(".env") or ".local." in name or name.endswith(".local") or any(x in name for x in ("credential", "private_key", "id_rsa", "id_ed25519")):
        return False
    return relative in SOURCE_ROOTS or (len(p.parts) > 1 and p.parts[0] in SOURCE_DIRS and p.suffix in SOURCE_SUFFIXES)


def git_state(root: Path) -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root), *args], capture_output=True, text=True)
        except OSError:
            return subprocess.CompletedProcess(args, 127, stdout="", stderr="git unavailable")
    status = run("status", "--porcelain=v1", "--untracked-files=all")
    commit = run("rev-parse", "HEAD")
    return {"commit": commit.stdout.strip() if commit.returncode == 0 else None,
            "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
            "status": status.stdout.splitlines() if status.returncode == 0 else [],
            "git_available": status.returncode == 0}


def archive_bytes(files: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as tar:
            for name, data in sorted(files.items()):
                item = tarfile.TarInfo(name)
                item.size = len(data)
                item.mode = 0o644
                item.mtime = 0
                tar.addfile(item, io.BytesIO(data))
    return stream.getvalue()


def create_snapshot(root: Path, output: Path) -> dict:
    root = root.resolve()
    files = {}
    for base, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if d not in DENIED_PARTS and not (Path(base) / d).is_symlink()]
        for name in names:
            p = Path(base) / name
            rel = p.relative_to(root).as_posix()
            if not allowed_source(rel):
                continue
            if p.is_symlink():
                raise ValueError(f"Source symlinks are forbidden: {rel}")
            if p.stat().st_size > MAX_FILE:
                raise ValueError(f"Source file exceeds {MAX_FILE} bytes: {rel}")
            data = p.read_bytes()
            if p.suffix == ".sh" and b"\r" in data:
                raise ValueError(f"Shell source must have LF line endings: {rel}")
            files[rel] = data
    if sum(map(len, files.values())) > MAX_SOURCE:
        raise ValueError("Source snapshot exceeds 64 MiB; generated data must stay out of source")
    if not files:
        raise ValueError("No source files selected")
    identifier = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:10]
    manifest = {"schema": 1, "release_id": identifier, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "git": git_state(root), "files": {name: {"sha256": sha256(data), "bytes": len(data)} for name, data in sorted(files.items())},
                "selection": "Explicit source roots/suffixes; current working-tree bytes, including new and modified files"}
    files["source-manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    data = archive_bytes(files)
    destination = output / identifier
    destination.mkdir(parents=True, exist_ok=False)
    archive = destination / "source.tar.gz"
    archive.write_bytes(data)
    metadata = {"release_id": identifier, "archive": str(archive.resolve()), "archive_sha256": sha256(data),
                "file_count": len(manifest["files"]), "source_bytes": sum(x["bytes"] for x in manifest["files"].values()), "git": manifest["git"]}
    (destination / "deploy.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (destination / "source-manifest.json").write_bytes(files["source-manifest.json"])
    return metadata


def verified_members(archive: Path, expected_hash: str, limit: int = 128 * 1024 * 1024) -> dict[str, bytes]:
    raw = archive.read_bytes()
    if sha256(raw) != expected_hash:
        raise ValueError("Archive SHA256 mismatch")
    files = {}
    size = 0
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        for item in tar:
            p = PurePosixPath(item.name)
            if not item.isfile() or p.is_absolute() or ".." in p.parts or "\\" in item.name or ":" in item.name or item.name in files:
                raise ValueError(f"Unsafe archive member: {item.name}")
            size += item.size
            if size > limit:
                raise ValueError("Archive exceeds extraction bound")
            files[item.name] = tar.extractfile(item).read()
    return files


def extract_results(archive: Path, expected_hash: str, destination: Path) -> dict:
    files = verified_members(archive, expected_hash)
    manifest = json.loads(files.pop("fetch-manifest.json"))
    if set(files) != set(manifest["files"]):
        raise ValueError("Result manifest/member set mismatch")
    for name, data in files.items():
        expected = manifest["files"][name]
        if len(data) != expected["bytes"] or sha256(data) != expected["sha256"]:
            raise ValueError(f"Result hash mismatch: {name}")
    # Verification is complete before touching the destination. Refetches update only listed files.
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        p = destination / name
        if p.is_symlink() or any(x.is_symlink() for x in p.parents if x != destination.parent):
            raise ValueError(f"Destination symlink forbidden: {p}")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    (destination / "fetch-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    receipt = {"archive_sha256": expected_hash, "verified_files": len(files), "verified_bytes": sum(map(len, files.values())), "fetched_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    (destination / "transfer-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    create = sub.add_parser("create")
    create.add_argument("--root", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    fetch = sub.add_parser("extract")
    fetch.add_argument("--archive", type=Path, required=True)
    fetch.add_argument("--sha256", required=True)
    fetch.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = create_snapshot(args.root, args.output) if args.action == "create" else extract_results(args.archive, args.sha256, args.output)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
