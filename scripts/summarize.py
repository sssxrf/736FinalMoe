"""Select small, provenance-linked summaries for deliberate inclusion in Git."""
import argparse
import hashlib
import json
from pathlib import Path


def compact_fields(value, fields):
    return {key: value[key] for key in fields if key in value} if isinstance(value, dict) else None


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
    preflight = read_optional("preflight-complete.json", {})
    reference = read_optional("reference-outputs.json", {})
    reference_complete = read_optional("reference-complete.json", {})
    fixtures = read_optional("fixture-tests.json", read_optional("fixture-summary.json"))
    if isinstance(fixtures, dict):
        fixtures = {key: value for key, value in fixtures.items()
                    if not key.startswith("resources_") and key != "resource_samples"}
    source = json.loads((directory / "source-manifest.json").read_text())
    release = json.loads((directory / "release.json").read_text())
    status = read_optional("observed-status.json", read_optional("status.json", {}))
    request = read_optional("request.json", {})
    selected = {"run_id": directory.name, "status": status["state"], "exit_code": status["exit_code"],
                "kind": request.get("kind"),
                "command_succeeded": status["state"] == "completed" and status["exit_code"] == 0,
                "experiment_complete": summary.get("complete"), "experiment_passed": summary.get("passed"),
                "acceptance_scope": summary.get("acceptance_scope", reference_complete.get("scope")
                    if request.get("kind") == "reference" else summary.get("mode", request.get("kind"))),
                "model_kind": summary.get("model_kind", environment.get("model_kind", reference_complete.get("model_kind"))),
                "wall_seconds": summary.get("wall_seconds", status.get("elapsed_seconds")), "source_release": release,
                "preparation_seconds": summary.get("preparation_seconds", preflight.get("preparation_seconds")),
                "preparation_excluded_from_experiment_timing": preflight.get("excluded_from_experiment_timing"),
                "storage_mode": summary.get("storage_mode", environment.get("storage_mode", preflight.get("storage_mode"))),
                "storage_claim": environment.get("storage_claim"),
                "cache_conditions": environment.get("conditions"),
                "git": source["git"], "source_file_hashes": source["files"],
                "model": environment.get("model"), "torch": environment.get("torch", environment.get("packages", {}).get("torch")),
                "transformers": environment.get("transformers", environment.get("packages", {}).get("transformers")), "gpu": environment.get("gpu"),
                "trajectory_sha256": environment.get("trajectory_sha256"),
                "config": read_optional("experiment_config.json", read_optional("experiment-config.json")),
                "transfer": json.loads((directory / "transfer-receipt.json").read_text()),
                "fixtures": fixtures,
                "limitations": summary.get("limitations", []), "cases": []}
    completion = read_optional("artifact-completion.json", read_optional("artifact-complete.json", environment.get("artifact_completion")))
    if completion:
        selected["artifact_completion"] = compact_fields(completion, (
            "schema_version", "complete", "completed_at_utc", "model_id", "revision",
            "model_dir", "backing_dir", "model_provenance_sha256", "backing_manifest_sha256",
            "converter_version", "converter_format_version", "converter_source_sha256",
            "expert_count", "backing_bytes", "artifact_guard"))
        completion_name = next((name for name in ("artifact-completion.json", "artifact-complete.json")
                                if name in manifest["files"]), None)
        selected["artifact_completion_sha256"] = (manifest["files"][completion_name]["sha256"] if completion_name
                                                  else environment.get("artifact_completion_sha256"))
    reuse = read_optional("artifact-reuse.json")
    if reuse:
        selected["artifact_reuse"] = compact_fields(reuse, (
            "reused_completion_manifest", "completion_sha256", "verified_at_utc", "current_artifact_guard"))
    checkpoint = read_optional("checkpoint-stage.json")
    if checkpoint:
        selected["checkpoint_stage"] = compact_fields(checkpoint, (
            "complete", "model_id", "revision", "model_provenance_sha256", "verified_files"))
    storage = read_optional("artifact-storage-before.json")
    if storage:
        selected["artifact_storage_before"] = compact_fields(storage, (
            "utc", "account_home_bytes", "artifact_bytes", "runtime_bytes", "filesystem_total_bytes",
            "filesystem_used_bytes", "filesystem_free_bytes", "required_headroom_bytes", "quota_query", "quota_note"))
    recovery = read_optional("recovery.json")
    if recovery:
        selected["recovery"] = compact_fields(recovery, (
            "complete", "passed", "scope", "model_revision", "expert", "expert_bytes",
            "canonical_manifest_sha256", "canonical_expert_sha256", "initial_stage_sha256",
            "stage_exists_after_delete", "canonical_unchanged_while_stage_missing",
            "reconstructed_stage_sha256", "canonical_unchanged_after_reconstruction"))
    stage_fields = ("complete", "canonical_manifest_sha256", "verified_files", "reused_files",
                    "copied_files", "copied_bytes", "seconds", "scope", "cache_condition", "space")
    stages = {name: compact_fields(read_optional(name), stage_fields)
              for name in ("staging.json", "initial-stage/staging.json", "reconstructed-stage/staging.json")
              if name in manifest["files"]}
    if stages:
        selected["staging"] = stages
    if reference or reference_complete:
        selected["reference"] = compact_fields({**reference, **reference_complete}, (
            "complete", "scope", "model_kind", "checkpoint_count", "logits_file",
            "logits_file_bytes", "logits_file_sha256"))
        selected["reference"]["files"] = {name: manifest["files"][name]
            for name in ("reference-outputs.json", "reference-logits.npz", "trajectories.json")
            if name in manifest["files"]}
        selected["reference"]["token_histories"] = {
            name: {"token_count": len(tokens), "sha256": hashlib.sha256(
                json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()}
            for name, tokens in reference.get("token_histories", {}).items()}
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
