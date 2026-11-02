"""Select small, provenance-linked summaries for deliberate inclusion in Git."""
import argparse
import hashlib
import json
from pathlib import Path


def select(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "fetch-manifest.json").read_text())
    for name, expected in manifest["files"].items():
        data = (directory / name).read_bytes()
        if len(data) != expected["bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
            raise ValueError(f"Fetched result verification failed: {directory / name}")
    def read_optional(name, default=None):
        path = directory / name
        return json.loads(path.read_text()) if path.is_file() else default

    summary = read_optional("summary.json", {})
    environment = read_optional("environment.json", {})
    source = json.loads((directory / "source-manifest.json").read_text())
    release = json.loads((directory / "release.json").read_text())
    status = read_optional("observed-status.json", read_optional("status.json", {}))
    request = read_optional("request.json", {})
    selected = {"run_id": directory.name, "status": status["state"], "exit_code": status["exit_code"],
                "kind": request.get("kind"),
                "command_succeeded": status["state"] == "completed" and status["exit_code"] == 0,
                "experiment_complete": summary.get("complete"), "experiment_passed": summary.get("passed"),
                "wall_seconds": summary.get("wall_seconds", status.get("elapsed_seconds")), "source_release": release,
                "git": source["git"], "source_file_hashes": source["files"],
                "model": environment.get("model"), "torch": environment.get("torch", environment.get("packages", {}).get("torch")),
                "transformers": environment.get("transformers", environment.get("packages", {}).get("transformers")), "gpu": environment.get("gpu"),
                "trajectory_sha256": environment.get("trajectory_sha256"),
                "config": read_optional("experiment_config.json", read_optional("experiment-config.json")),
                "transfer": json.loads((directory / "transfer-receipt.json").read_text()),
                "fixtures": read_optional("fixture-tests.json", read_optional("fixture-summary.json")),
                "limitations": summary.get("limitations", []), "cases": []}
    if not summary.get("complete"):
        selected["log_tail"] = status.get("log_tail", "")[-2500:]
    provenance = read_optional("model-provenance.json")
    if provenance:
        selected["actual_model_revision"] = provenance["revision"]
        selected["actual_model_id"] = provenance["model_id"]
    audit = read_optional("audit-after.json", read_optional("audit.json"))
    if audit:
        selected["scratch_guard"] = audit.get("scratch_guard")
        group = audit.get("resource_snapshot", {}).get("cgroup", {}).get("groups", [{}])[0]
        selected["cgroup_memory_max"] = group.get("memory.max")
        selected["cgroup_historical_peak"] = group.get("memory.peak")
        selected["cpu_max"] = group.get("cpu.max")
        selected["storage_label"] = audit.get("storage", {}).get("tmp_physical_backing")
    for case in summary.get("cases", []):
        selected["cases"].append({key: value for key, value in case.items()
                                 if key not in ("resource_samples", "observed_counter_deltas")})
        deltas = case.get("observed_counter_deltas", {})
        selected["cases"][-1]["observed_process_io_delta"] = deltas.get("process_io")
        selected["cases"][-1]["observed_cgroup_io_delta"] = deltas.get("cgroup", {}).get("io_counters")
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"runs": [select(path) for path in args.runs]}, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
