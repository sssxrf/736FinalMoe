"""Direct Linux controller for a fresh clone; uses the same immutable releases/workers."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import shutil
import sys
import time
import uuid

import control
import snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Task runtime root, separate from the source checkout")
    parser.add_argument("--scratch", type=Path, help="Task scratch under /tmp; needed for setup/doctor/smoke/pilot/fixtures")
    parser.add_argument("action", choices=["deploy", "setup", "doctor", "smoke", "pilot", "fixtures", "status", "fetch"])
    parser.add_argument("--run", help="Run ID for status/fetch; defaults to last run launched here")
    parser.add_argument("--wait", action="store_true", help="Wait for an explicit worker completion receipt")
    parser.add_argument("--wait-timeout", type=int, default=7500)
    parser.add_argument("--output", type=Path, help="Fetch destination; default CHECKOUT/results/RUN_ID")
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("This entry point runs on Linux; use scripts/remote.ps1 on the laptop")
    repo = Path(__file__).resolve().parents[1]
    if args.root.resolve() == repo:
        parser.error("Runtime --root must differ from the source checkout")
    root = control.owned_root(args.root.resolve())
    state_path = root / "linux-controller-state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if args.action in {"deploy", "setup", "doctor", "smoke", "pilot", "fixtures"}:
        if args.action != "deploy" and args.scratch is None:
            parser.error("--scratch is required for this action")
        source = snapshot.create_snapshot(repo, root / "incoming/snapshots")
        uploaded = root / "incoming" / (source["release_id"] + ".tar.gz")
        shutil.copyfile(source["archive"], uploaded)
        receipt = control.deploy(argparse.Namespace(release=source["release_id"], sha256=source["archive_sha256"]), root)
        state.update(release_id=receipt["release_id"], archive_sha256=receipt["archive_sha256"])
        control.atomic_json(state_path, state)
        if args.action == "deploy":
            print(json.dumps(receipt))
            return 0
        run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + args.action + "-" + uuid.uuid4().hex[:8]
        state["latest_run_id"] = run_id
        state[args.action + "_run_id"] = run_id
        control.atomic_json(state_path, state)
        receipt = control.launch(argparse.Namespace(release=receipt["release_id"], kind=args.action,
                                                   scratch=str(args.scratch), run=run_id), root)
        print(json.dumps(receipt), flush=True)
        if not args.wait:
            return 0
        began, update = time.monotonic(), 0
        while True:
            status = control.status(argparse.Namespace(run=run_id), root)
            if status["state"] in {"completed", "failed", "lost"}:
                print(json.dumps(status), flush=True)
                return 0 if status["state"] == "completed" and status["exit_code"] == 0 else 1
            elapsed = time.monotonic() - began
            if elapsed >= args.wait_timeout:
                raise TimeoutError(f"Local waiting deadline reached; remote worker may still run. Check status --run {run_id}")
            if elapsed >= update:
                print(f"Run {run_id}: {status['state']}, {elapsed:.0f}s; logs survive this terminal.", file=sys.stderr, flush=True)
                update += 30
            time.sleep(5)
    run_id = args.run or state.get("latest_run_id")
    if not run_id:
        parser.error("Specify --run RUN_ID; no run was launched by this Linux controller yet")
    if args.action == "status":
        print(json.dumps(control.status(argparse.Namespace(run=run_id), root)))
        return 0
    packed = control.pack(argparse.Namespace(run=run_id), root)
    destination = args.output or repo / "results" / run_id
    receipt = snapshot.extract_results(Path(packed["archive"]), packed["archive_sha256"], destination)
    print(json.dumps(dict(receipt, output=str(destination))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
