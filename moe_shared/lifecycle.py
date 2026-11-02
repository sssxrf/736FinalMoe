"""Observed path/process/container identities; these records do not infer restarts."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import time
import uuid

SENTINEL = ".moe-lifetime-sentinel.json"


def _read(path):
    try:
        return Path(path).read_text().strip()
    except OSError as error:
        return {"unavailable": str(error)}


def _link(path):
    try:
        return {"link": os.readlink(path), "inode": os.stat(path).st_ino}
    except OSError as error:
        return {"unavailable": str(error)}


def process_identity(pid):
    raw = _read(f"/proc/{pid}/stat")
    start = None
    if isinstance(raw, str):
        try:
            start = raw.split(")", 1)[1].split()[19]
        except IndexError:
            pass
    return {"pid": os.getpid() if pid == "self" else pid, "start_ticks": start,
            "pid_namespace": _link(f"/proc/{pid}/ns/pid"),
            "mount_namespace": _link(f"/proc/{pid}/ns/mnt")}


def _mount_decode(value):
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)


def path_identity(path):
    path = Path(path).absolute()
    resolved = path.resolve()
    result = {"requested": str(path), "resolved": str(resolved),
              "exists": path.exists(), "is_symlink": path.is_symlink()}
    try:
        stat = path.stat()
        result["stat"] = {"device": stat.st_dev, "inode": stat.st_ino,
                          "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns}
    except OSError as error:
        result["stat"] = {"unavailable": str(error)}
    raw = _read("/proc/self/mountinfo")
    if isinstance(raw, str):
        matches = []
        for line in raw.splitlines():
            left, sep, right = line.partition(" - ")
            fields, tail = left.split(), right.split()
            if sep and len(fields) >= 6 and len(tail) >= 2:
                mountpoint = _mount_decode(fields[4])
                if str(resolved) == mountpoint or str(resolved).startswith(mountpoint.rstrip("/") + "/"):
                    matches.append({"mount_id": fields[0], "parent_mount_id": fields[1],
                                    "major_minor": fields[2], "root": _mount_decode(fields[3]),
                                    "mountpoint": mountpoint, "filesystem": tail[0],
                                    "source": _mount_decode(tail[1]), "raw": line})
        result["mount"] = max(matches, key=lambda item: len(item["mountpoint"])) if matches else None
        result["mountinfo_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    else:
        result["mount"] = raw
    sentinel = path / SENTINEL
    try:
        data = sentinel.read_bytes()
        result["sentinel"] = {"exists": True, "sha256": hashlib.sha256(data).hexdigest(),
                              "contents": json.loads(data)}
    except FileNotFoundError:
        result["sentinel"] = {"exists": False}
    except (OSError, ValueError) as error:
        result["sentinel"] = {"exists": sentinel.exists(), "error": str(error)}
    return result


def identity_checkpoint(output_dir, stage, paths, create_sentinels=False):
    """Record before any sentinel creation so disappearance remains observable."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", stage):
        raise ValueError("Unsafe lifecycle stage name")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = {"stage": stage, "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "monotonic_ns": time.monotonic_ns(), "hostname": socket.gethostname(),
            "process": process_identity("self"), "pid1": process_identity(1),
            "host_boot_id": _read("/proc/sys/kernel/random/boot_id"),
            "cgroup_membership": _read("/proc/self/cgroup"),
            "paths": {name: path_identity(path) for name, path in paths.items()},
            "interpretation": "Observed identities only. Host boot ID alone cannot establish a container restart."}
    if create_sentinels:
        for name, raw_path in paths.items():
            path = Path(raw_path).absolute()
            if path.is_symlink() or path.resolve() != path:
                raise ValueError(f"Refusing sentinel creation through symlink path: {path}")
            path.mkdir(parents=True, exist_ok=True)
            try:
                with (path / SENTINEL).open("x") as stream:
                    json.dump({"uuid": uuid.uuid4().hex, "created_utc": data["utc"],
                               "path": str(path), "hostname": data["hostname"]}, stream)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            except FileExistsError:
                pass
            data["paths"][name]["after_sentinel_creation"] = path_identity(path)
    destination = output / f"identity-{stage}.json"
    temporary = destination.with_suffix(".json.tmp")
    with temporary.open("w") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(destination)
    with (output / "lifecycle.jsonl").open("a") as stream:
        stream.write(json.dumps(data) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--path", action="append", default=[])
    parser.add_argument("--create-sentinels", action="store_true")
    args = parser.parse_args()
    paths = {}
    for value in args.path:
        name, separator, path = value.partition("=")
        if not separator or not name:
            parser.error("--path must be name=PATH")
        paths[name] = path
    identity_checkpoint(args.output, args.stage, paths, args.create_sentinels)


if __name__ == "__main__":
    main()
