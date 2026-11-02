"""Read-only Linux resource audit and conservative task scratch accounting.

No global cache manipulation or raw-device access is performed. Application
allocation accounting is intentionally separate from cgroup/page-cache totals.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import shutil
import socket
import subprocess
import time
from pathlib import Path

GiB = 1024 ** 3
CGROUP_FILES = (
    "memory.current", "memory.max", "memory.high", "memory.peak",
    "memory.stat", "memory.events", "memory.events.local",
    "memory.swap.current", "memory.swap.max", "memory.swap.peak",
    "cpu.max", "cpu.stat", "cpuset.cpus", "cpuset.cpus.effective",
    "io.stat", "io.max", "pids.current", "pids.max",
)


def read_text(path):
    try:
        return Path(path).read_text().strip()
    except (OSError, UnicodeError) as exc:
        return {"unavailable": str(exc)}


def parse_key_values(raw):
    """Parse proc/cgroup key-value integer counters, preserving unavailable data."""
    if not isinstance(raw, str):
        return raw
    result = {}
    for line in raw.splitlines():
        pieces = line.replace(":", " ", 1).split()
        if len(pieces) >= 2:
            try:
                result[pieces[0]] = int(pieces[1])
            except ValueError:
                result[pieces[0]] = " ".join(pieces[1:])
    return result


def parse_io_stat(raw):
    """Per-device counters; raw io.stat remains available for auditability."""
    if not isinstance(raw, str):
        return raw
    counters = {}
    for line in raw.splitlines():
        pieces = line.split()
        if not pieces:
            continue
        device = {}
        for field in pieces[1:]:
            name, separator, value = field.partition("=")
            if separator and value.isdigit():
                device[name] = int(value)
        counters[pieces[0]] = device
    return counters


def cgroup_paths(proc_path="/proc/self/cgroup", mount_path="/sys/fs/cgroup"):
    """Return visible cgroup-v2 group and ancestors; namespace root may hide limits."""
    data = read_text(proc_path)
    if not isinstance(data, str):
        return []
    mount = Path(mount_path).resolve()
    for line in data.splitlines():
        if line.startswith("0::"):
            rel = line[3:].lstrip("/")
            candidate = (mount / rel).resolve()
            # A namespace may report a host-relative group not mounted here.
            if not candidate.is_relative_to(mount) or not candidate.is_dir():
                candidate = mount
            paths = []
            while True:
                paths.append(candidate)
                if candidate == mount:
                    return paths
                candidate = candidate.parent
    return []


def cgroup_snapshot():
    groups = []
    for path in cgroup_paths():
        counters = {}
        for name in CGROUP_FILES:
            raw = read_text(path / name)
            if isinstance(raw, str) and "\n" in raw:
                counters[name] = parse_key_values(raw) if name != "io.stat" else raw
            elif isinstance(raw, str) and raw.isdigit():
                counters[name] = int(raw)
            else:
                counters[name] = raw
        groups.append({"path": str(path), **counters})
        groups[-1]["io_counters"] = parse_io_stat(read_text(path / "io.stat"))
    return {
        "groups": groups,
        "membership": read_text("/proc/self/cgroup"),
        "ancestor_visibility": "Only ancestors visible in this cgroup namespace are readable; hidden host ancestors cannot be ruled out.",
    }


def resource_snapshot():
    process_status = parse_key_values(read_text("/proc/self/status"))
    cgroup = cgroup_snapshot()
    # Numeric per-path dict additionally permits generic recursive deltas, unlike
    # the ordered groups list. Keep groups and raw io.stat for context.
    cgroup["io_counters"] = {group["path"]: group["io_counters"] for group in cgroup["groups"]}
    memory = {}
    for field in ("VmRSS", "VmHWM", "VmSize", "VmPeak", "RssAnon", "RssFile", "RssShmem"):
        value = process_status.get(field)
        if isinstance(value, int):
            memory[field + "_bytes"] = value * 1024
    return {
        "wall_time_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "monotonic_ns": time.monotonic_ns(),
        "process_io": parse_key_values(read_text("/proc/self/io")),
        "process_status": process_status,
        "process_status_memory_units": "KiB (values from /proc/self/status)",
        "process_memory_bytes": memory,
        "cgroup": cgroup,
    }


def directory_bytes(path):
    """Logical bytes in regular files, excluding symlinks and without following them."""
    root = Path(path)
    if not root.exists():
        return 0
    total = 0
    for folder, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if not (Path(folder) / d).is_symlink()]
        for name in files:
            candidate = Path(folder) / name
            if not candidate.is_symlink():
                total += candidate.stat().st_size
    return total


def check_space(scratch, additional_bytes=0, max_scratch_bytes=35 * GiB,
                headroom_bytes=20 * GiB):
    """Reject work that would exceed task bytes or currently available headroom.

    Free space is shared, not reserved; callers must recheck before each write.
    """
    scratch = Path(scratch).resolve()
    if additional_bytes < 0:
        raise ValueError("additional_bytes must be nonnegative")
    anchor = scratch
    while not anchor.exists():
        anchor = anchor.parent
    used = directory_bytes(scratch)
    free = shutil.disk_usage(anchor).free
    result = {"scratch": str(scratch), "task_logical_bytes": used,
              "filesystem_free_bytes": free, "additional_bytes": additional_bytes,
              "max_scratch_bytes": max_scratch_bytes, "headroom_bytes": headroom_bytes,
              "free_space_is_reserved": False}
    if used + additional_bytes > max_scratch_bytes:
        raise RuntimeError(f"Task scratch cap would be exceeded: {result}")
    if free - additional_bytes < headroom_bytes:
        raise RuntimeError(f"Filesystem headroom would be violated: {result}")
    return result


def command(args, timeout=15):
    try:
        completed = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return {"argv": args, "exit_code": completed.returncode,
                "stdout": completed.stdout.strip(), "stderr": completed.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": args, "unavailable": str(exc)}


def block_device_chain(major_minor, sysfs="/sys/dev/block", visited=None):
    """Trace visible device-mapper slaves without opening block devices."""
    visited = set() if visited is None else visited
    if major_minor in visited:
        return {"device": major_minor, "error": "cycle in sysfs block hierarchy"}
    visited.add(major_minor)
    link = Path(sysfs) / major_minor
    try:
        node = link.resolve(strict=True)
        slaves = list((node / "slaves").iterdir())
        children = []
        for slave in slaves:
            dev = read_text(slave / "dev")
            if isinstance(dev, str):
                children.append(block_device_chain(dev, sysfs, visited))
            else:
                children.append({"name": slave.name, "error": dev})
        return {"device": major_minor, "name": node.name, "sysfs_path": str(node),
                "rotational": read_text(node / "queue" / "rotational"),
                "model": read_text(node / "device" / "model"), "slaves": children,
                "leaf": not slaves}
    except OSError as exc:
        return {"device": major_minor, "unavailable": str(exc)}


def storage_audit(scratch):
    paths = dict.fromkeys([str(Path(scratch)), "/tmp", "/home", str(Path.home()), "/models"])
    mounts = {}
    for path in paths:
        if Path(path).exists():
            usage = shutil.disk_usage(path)
            mounts[path] = {"total_bytes": usage.total, "used_bytes": usage.used,
                            "free_bytes": usage.free,
                            "findmnt": command(["findmnt", "-J", "-T", path,
                                                "-o", "TARGET,SOURCE,FSTYPE,OPTIONS"])}
        else:
            mounts[path] = {"unavailable": "path does not exist"}
    tmp_type = command(["findmnt", "-n", "-T", "/tmp", "-o", "FSTYPE"])
    filesystem = tmp_type.get("stdout", "unknown filesystem").strip() or "unknown filesystem"
    tmp_device = command(["findmnt", "-n", "-T", "/tmp", "-o", "MAJ:MIN"])
    major_minor = tmp_device.get("stdout", "").strip()
    chain = block_device_chain(major_minor) if major_minor else {"unavailable": "findmnt did not identify /tmp device"}
    return {"paths": mounts,
            "lsblk": command(["lsblk", "-J", "-o", "NAME,KNAME,TYPE,FSTYPE,SIZE,MOUNTPOINTS,PKNAME,MODEL,ROTA"]),
            "tmp_physical_backing": f"{filesystem}; physical backing unverified",
            "tmp_sysfs_backing_trace": chain,
            "tmp_attribution_note": "An NVMe device elsewhere in lsblk does not establish the physical backing of this /tmp mount. No local-NVMe claim is made.",
            "mountinfo": read_text("/proc/self/mountinfo"),
            "diskstats": read_text("/proc/diskstats")}


def build_audit(scratch):
    result = {
        "schema_version": 1, "hostname": socket.gethostname(),
        "platform": platform.platform(), "python": platform.python_version(),
        "python_executable": __import__("sys").executable,
        "os_release": read_text("/etc/os-release"),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "lscpu": command(["lscpu", "-J"]),
        "memory_info": read_text("/proc/meminfo"),
        "swaps": read_text("/proc/swaps"),
        "resource_snapshot": resource_snapshot(),
        "storage": storage_audit(scratch),
        "nvidia_smi": command(["nvidia-smi"]),
        "nvidia_query": command(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,compute_cap", "--format=csv,noheader"]),
        "nvcc": command(["nvcc", "--version"]),
        "limitations": [
            "cgroup memory includes file/page cache and other processes in the same cgroup; it is separate from the accounted application pool.",
            "Process read_bytes and cgroup io.stat are observed kernel counters; neither assigns an application miss to a physical SSD request.",
            "Host kernel page cache is not dropped or controlled. Repeated runs are filesystem-cache-uncontrolled.",
        ],
    }
    try:
        result["scratch_guard"] = check_space(scratch)
    except (OSError, RuntimeError) as exc:
        result["scratch_guard"] = {"error": str(exc)}
    try:
        import torch
        result["torch"] = {"version": torch.__version__, "cuda_build": torch.version.cuda,
                           "cuda_available": torch.cuda.is_available()}
        if torch.cuda.is_available():
            result["torch"].update({"device": torch.cuda.get_device_name(0),
                                    "capability": list(torch.cuda.get_device_capability(0)),
                                    "bf16_supported": torch.cuda.is_bf16_supported()})
    except Exception as exc:
        result["torch"] = {"unavailable": str(exc)}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--scratch", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    audit = build_audit(args.scratch)
    output.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"audit": str(output), "hostname": audit["hostname"],
                      "torch": audit["torch"], "scratch_guard": audit["scratch_guard"]}))


if __name__ == "__main__":
    main()
