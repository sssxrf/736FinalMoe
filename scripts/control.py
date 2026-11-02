"""Task-owned immutable releases and durable Linux workers. No third-party dependencies."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid

from snapshot import MAX_SOURCE, archive_bytes, sha256, verified_members

OWNER = "cs736-moe-shared-memory-v1"
ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}$")


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path, data):
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def identifier(value):
    if not ID.fullmatch(value) or value in {".", ".."}:
        raise ValueError("Invalid release/run identifier")
    return value


def owned_root(root):
    root = Path(root)
    if not root.is_absolute() or str(root) in {"/", "/home", "/tmp"} or root.is_symlink():
        raise ValueError("Require an absolute task-specific root, not a symlink")
    root.mkdir(parents=True, exist_ok=True)
    marker = root / ".moe736-owner.json"
    if marker.exists():
        if json.loads(marker.read_text())["owner"] != OWNER:
            raise ValueError("Remote root belongs to another task")
    else:
        # Incoming may be created by the controller before first deployment.
        if any((root / name).exists() for name in ("releases", "runs", "exports")):
            raise ValueError("Refusing existing unmanaged releases/runs/exports directories")
        with marker.open("x") as f:
            json.dump({"owner": OWNER, "created_utc": utc()}, f)
    for name in ("incoming", "releases", "runs", "exports"):
        p = root / name
        if p.is_symlink():
            raise ValueError(f"Managed directory is a symlink: {p}")
        p.mkdir(exist_ok=True)
    return root


def verify_release(release):
    manifest = json.loads((release / "source-manifest.json").read_text())
    for name, expected in manifest["files"].items():
        p = release / name
        if p.is_symlink() or not p.is_file():
            raise ValueError(f"Missing or symlink source file: {name}")
        data = p.read_bytes()
        if len(data) != expected["bytes"] or sha256(data) != expected["sha256"]:
            raise ValueError(f"Deployed source hash mismatch: {name}")
    return manifest


def deploy(args, root):
    release_id = identifier(args.release)
    release = root / "releases" / release_id
    if release.exists():
        metadata = json.loads((release / "release.json").read_text())
        if metadata["archive_sha256"] != args.sha256:
            raise ValueError("Immutable release already exists with different content")
        verify_release(release)
        return metadata
    archive = root / "incoming" / (release_id + ".tar.gz")
    files = verified_members(archive, args.sha256, MAX_SOURCE + 2 * 1024 * 1024)
    manifest = json.loads(files.pop("source-manifest.json"))
    if manifest["release_id"] != release_id or set(files) != set(manifest["files"]):
        raise ValueError("Source manifest/archive file set mismatch")
    for name, data in files.items():
        expected = manifest["files"][name]
        if sha256(data) != expected["sha256"] or len(data) != expected["bytes"]:
            raise ValueError(f"Source hash mismatch: {name}")
    temporary = root / "releases" / ("." + release_id + ".preparing-" + uuid.uuid4().hex)
    temporary.mkdir()
    for name, data in files.items():
        p = temporary / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    atomic_json(temporary / "source-manifest.json", manifest)
    metadata = {"release_id": release_id, "archive_sha256": args.sha256, "file_count": len(files), "created_utc": utc(), "hostname": socket.gethostname()}
    atomic_json(temporary / "release.json", metadata)
    verify_release(temporary)
    for base, dirs, names in os.walk(temporary, topdown=False):
        for name in names:
            os.chmod(Path(base) / name, 0o444)
        os.chmod(base, 0o555)
    os.rename(temporary, release)
    archive.unlink()  # Only the exact verified task-owned uploaded archive.
    return metadata


def process_birth(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].split()[19]
    except (FileNotFoundError, PermissionError, IndexError):
        return None


@contextmanager
def execution_guard(root, kind):
    """Fail closed on competing setup/experiments in this task root.

    The setup lock also recognizes bootstrap processes started by older releases
    or directly from a Linux clone. Doctor is a read-only audit and needs no GPU.
    """
    if kind == "doctor":
        yield
        return
    import fcntl
    locks = root / "locks"
    locks.mkdir(exist_ok=True)
    handles = []
    try:
        names = ["execution.lock"]
        if kind in {"smoke", "pilot", "fixtures"}:
            names.append("setup.lock")
        for name in names:
            handle = (locks / name).open("a+")
            handles.append(handle)
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise BlockingIOError("Another setup/experiment holds this task's execution lock; retry after it completes") from exc
        yield
    finally:
        for handle in reversed(handles):
            handle.close()


def launch(args, root):
    release = root / "releases" / identifier(args.release)
    verify_release(release)
    lock = release / "requirements.lock"
    lock_hash = sha256(lock.read_bytes()) if lock.exists() else None
    env = root / "envs" / lock_hash if lock_hash else None
    python = str(env / "bin/python") if env and (env / ".complete").exists() else "python3"
    if args.kind in {"smoke", "pilot", "fixtures"} and python == "python3":
        raise ValueError("Pinned environment is not complete; run setup -Wait first")
    if not args.scratch or not Path(args.scratch).is_absolute() or not str(Path(args.scratch)).startswith("/tmp/"):
        raise ValueError("Scratch must be a task-owned directory under /tmp")
    run_id = identifier(args.run) if args.run else dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + args.kind + "-" + uuid.uuid4().hex[:8]
    run = root / "runs" / run_id
    run.mkdir()
    if args.kind == "setup":
        command = ["bash", "scripts/bootstrap.sh", str(root), args.scratch]
        timeout = 7200
    elif args.kind == "doctor":
        command = [python, "-m", "moe_shared.audit", "--output", str(run / "audit.json"), "--scratch", args.scratch]
        timeout = 180
    elif args.kind == "fixtures":
        command = [python, "-m", "moe_shared.fixtures", "--output", str(run)]
        timeout = 180
    else:
        config = release / "configs" / (args.kind + ".json")
        command = [python, "-m", "moe_shared.run", "--mode", args.kind, "--config", str(config), "--output", str(run), "--scratch", args.scratch]
        shutil.copyfile(config, run / "experiment-config.json")
        timeout = 1200
    request = {"run_id": run_id, "kind": args.kind, "release_id": args.release, "command": command,
               "cwd": str(release), "scratch_root": args.scratch, "dependency_lock_sha256": lock_hash,
               "created_utc": utc(), "hostname": socket.gethostname(), "timeout_seconds": timeout}
    atomic_json(run / "request.json", request)
    shutil.copyfile(release / "source-manifest.json", run / "source-manifest.json")
    shutil.copyfile(release / "release.json", run / "release.json")
    if env and (env / "environment.json").exists():
        shutil.copyfile(env / "environment.json", run / "environment.json")
    atomic_json(run / "status.json", {"run_id": run_id, "state": "launching", "exit_code": None, "created_utc": utc()})
    with (run / "supervisor.log").open("ab", buffering=0) as log:
        worker = subprocess.Popen([sys.executable, str(release / "scripts/control.py"), "--root", str(root), "worker", "--run", run_id],
                                  stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True, close_fds=True)
    return {"run_id": run_id, "release_id": args.release, "state": "launched", "worker_pid": worker.pid, "exit_code": None}


def worker(args, root):
    run = root / "runs" / identifier(args.run)
    request = json.loads((run / "request.json").read_text())
    state = {"run_id": args.run, "state": "running", "exit_code": None, "started_utc": utc(), "worker_pid": os.getpid(), "worker_birth": process_birth(os.getpid())}
    atomic_json(run / "status.json", state)
    start = time.monotonic()
    child = None
    try:
        environ = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false", MOE_RUN_DIR=str(run))
        with execution_guard(root, request["kind"]), (run / "worker.log").open("ab", buffering=0) as log:
            child = subprocess.Popen(request["command"], cwd=request["cwd"], env=environ,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
            state["child_pid"] = child.pid
            atomic_json(run / "status.json", state)
            try:
                code = child.wait(timeout=request["timeout_seconds"])
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                code = 124
                state["error"] = "Remote worker deadline exceeded"
        state.update(state="completed" if code == 0 else "failed", exit_code=code)
    except BaseException as exc:
        if child and child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        state.update(state="failed", exit_code=75 if isinstance(exc, BlockingIOError) else 125, error=repr(exc))
    # Capture setup provenance after it creates the environment/model, including failed setup audits.
    provenance = {
        root / "envs" / request["dependency_lock_sha256"] / "environment.json": "environment.json",
        Path(request["scratch_root"]) / "model" / "model-provenance.json": "model-provenance.json",
    } if request.get("dependency_lock_sha256") else {}
    if request["kind"] == "setup":
        for name in ("audit-before.json", "audit-after.json"):
            provenance[root / "setup" / name] = name
    for source, name in provenance.items():
        try:
            if source.is_file() and not (run / name).exists():
                shutil.copyfile(source, run / name)
        except OSError as exc:
            state.setdefault("provenance_errors", []).append(f"{name}: {exc}")
            state.update(state="failed", exit_code=126)
    state.update(finished_utc=utc(), elapsed_seconds=time.monotonic() - start)
    atomic_json(run / "status.json", state)
    return state


def status(args, root):
    run = root / "runs" / identifier(args.run)
    state = json.loads((run / "status.json").read_text())
    if state["state"] == "running":
        birth = process_birth(state.get("worker_pid"))
        if birth is None or birth != state.get("worker_birth"):
            state = dict(state, state="lost", exit_code=None, error="Worker no longer exists or cannot be identified; no completion receipt. This is not success.")
    if state["state"] == "launching":
        state["warning"] = "No worker startup receipt yet; this is not success"
    log = run / "worker.log"
    if log.exists():
        with log.open("rb") as f:
            f.seek(max(0, log.stat().st_size - 6000))
            state["log_tail"] = f.read().decode("utf-8", errors="replace")
    return state


def pack(args, root):
    run_id = identifier(args.run)
    run = root / "runs" / run_id
    files = {}
    for p in sorted(run.rglob("*")):
        if p.is_symlink():
            raise ValueError("Symlink in result directory")
        if not p.is_file() or p.suffix not in {".json", ".jsonl", ".csv", ".log", ".txt", ".md"}:
            continue
        if p.stat().st_size > 32 * 1024 * 1024:
            raise ValueError(f"Result too large for compact transfer: {p.name}")
        files[p.relative_to(run).as_posix()] = p.read_bytes()
    files["observed-status.json"] = (json.dumps(status(args, root), indent=2) + "\n").encode()
    if sum(map(len, files.values())) > 128 * 1024 * 1024:
        raise ValueError("Compact results exceed 128 MiB")
    manifest = {"schema": 1, "run_id": run_id, "created_utc": utc(),
                "files": {name: {"sha256": sha256(data), "bytes": len(data)} for name, data in sorted(files.items())}}
    files["fetch-manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    data = archive_bytes(files)
    archive = root / "exports" / (run_id + "-" + uuid.uuid4().hex[:8] + ".tar.gz")
    archive.write_bytes(data)
    return {"run_id": run_id, "archive": str(archive), "archive_sha256": sha256(data), "files": len(manifest["files"]), "bytes": len(data)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--expected-hostname")
    sub = parser.add_subparsers(dest="action", required=True)
    deploy_parser = sub.add_parser("deploy")
    deploy_parser.add_argument("--release", required=True)
    deploy_parser.add_argument("--sha256", required=True)
    launch_parser = sub.add_parser("launch")
    launch_parser.add_argument("--release", required=True)
    launch_parser.add_argument("--kind", choices=["doctor", "setup", "smoke", "pilot", "fixtures"], required=True)
    launch_parser.add_argument("--scratch", required=True)
    launch_parser.add_argument("--run", help="Controller-chosen unique ID, recorded locally before submitting the launch")
    for action in ("status", "pack", "worker"):
        sub.add_parser(action).add_argument("--run", required=True)
    args = parser.parse_args()
    if args.expected_hostname and socket.gethostname() != args.expected_hostname:
        raise ValueError(f"Wrong SSH target: {socket.gethostname()}")
    root = owned_root(args.root)
    print(json.dumps(globals()[args.action](args, root)))


if __name__ == "__main__":
    main()
