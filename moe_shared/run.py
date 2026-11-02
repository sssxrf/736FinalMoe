"""Single-GPU, synchronous token-step mechanism experiment.

The fully resident reference and every offloaded case run sequentially. All cases
consume the same stored IDs; logits/argmax agreement tests the mechanism, not text
quality. There is no CPU expert execution and no asynchronous overlap claim.
"""
import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import unittest
from datetime import datetime, timezone

from .workload import ArrivalQueue, distribution, make_trajectories


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


class Recorder:
    def __init__(self, output):
        self.origin = time.perf_counter()
        self.file = (output / "events.jsonl").open("w", buffering=1)
        self.case = "initialization"
        self.event_count = 0

    def __call__(self, name, **fields):
        self.event_count += 1
        self.file.write(json.dumps({"event": name, "case": self.case,
                                   "monotonic_seconds": time.perf_counter(),
                                   "elapsed_seconds": time.perf_counter()-self.origin,
                                   "utc": datetime.now(timezone.utc).isoformat(),
                                   **fields}, sort_keys=True) + "\n")

    def close(self):
        self.file.close()


def numeric_delta(before, after):
    """Only monotone numeric leaf counters; raw snapshots are retained too."""
    if isinstance(before, dict) and isinstance(after, dict):
        return {key: numeric_delta(before[key], value) for key, value in after.items()
                if key in before and isinstance(value, (dict, int, float))}
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        return after - before
    return None


def execute_schedule(model, trajectories, config, event, action, reference=None,
                     kv_manager=None, deadline=None):
    import torch
    from transformers import DynamicCache
    from .audit import resource_snapshot

    sessions = {entry["id"]: {"turns": entry["turns"], "history": [], "cache": None,
                              "handle": None, "turn_started": False, "index": 0,
                              "arrival": None, "first_token": None, "last_token": None,
                              "turn_start": None}
                for entry in trajectories["sessions"]}
    queue = ArrivalQueue()
    started = time.perf_counter()
    for sid in sessions:
        queue.schedule(sid, 0, started)
    saved_logits, comparisons, measurements = {}, [], []
    initial_ttft, returning_ttft, token_latencies = [], [], []
    turn_summaries = []
    active_steps_while_paused = 0
    paused = set()
    snapshots = [resource_snapshot()]
    while queue.pending():
        if deadline is not None and time.perf_counter() > deadline:
            raise TimeoutError("Experiment wall-clock cap reached at a token boundary")
        now = time.perf_counter()
        for sid, turn, arrival in queue.admit(now):
            sessions[sid]["arrival"] = arrival
            event("arrival", session=sid, turn=turn, scheduled_monotonic_seconds=arrival,
                  admission_delay_seconds=now-arrival)
            event("queue", session=sid, turn=turn, ready_depth=len(queue.ready))
        if not queue.ready:
            queue.idle_wait(now)
            continue
        sid, turn = queue.next()
        state = sessions[sid]
        spec = state["turns"][turn]
        first = not state["turn_started"]
        dispatch = time.perf_counter()
        resume_seconds = 0.0
        if first:
            state["turn_start"] = dispatch
            state["turn_started"] = True
            if turn:
                resume_start = time.perf_counter()
                event("resume_start", session=sid, turn=turn, action=action,
                      queue_seconds=dispatch-state["arrival"])
                if kv_manager is not None:
                    assert state["cache"] is None, "Stale GPU KV bypassed pause action"
                    state["cache"] = kv_manager.resume(state["handle"])
                    state["handle"] = None
                paused.discard(sid)
                resume_seconds = time.perf_counter() - resume_start
                event("resume", session=sid, turn=turn, action=action,
                      reload_seconds=resume_seconds,
                      recompute_required=state["cache"] is None)
            # Last emitted token has not yet entered the attention cache.
            if state["cache"] is None:
                input_ids = state["history"] + spec["prompt"]
                state["cache"] = DynamicCache()
            else:
                input_ids = state["history"][-1:] + spec["prompt"]
            state["history"].extend(spec["prompt"])
        else:
            input_ids = state["history"][-1:]
        cached_length = state["cache"].get_seq_length()
        total_length = cached_length + len(input_ids)
        assert total_length == len(state["history"]), "Cache/history/positions diverged"
        if total_length > config["max_context_tokens"]:
            raise ValueError("Context exceeds the configured active KV bound")
        ids = torch.tensor([input_ids], dtype=torch.long, device="cuda")
        mask = torch.ones((1, total_length), dtype=torch.long, device="cuda")
        positions = torch.arange(cached_length, total_length, device="cuda")
        torch.cuda.synchronize()
        compute_start = time.perf_counter()
        with torch.inference_mode():
            result = model(input_ids=ids, attention_mask=mask,
                           position_ids=positions.unsqueeze(0), cache_position=positions,
                           past_key_values=state["cache"], use_cache=True,
                           output_router_logits=False, return_dict=True)
        torch.cuda.synchronize()
        forward_end = time.perf_counter()
        state["cache"] = result.past_key_values
        logits = result.logits[0, -1].detach().float().cpu()
        argmax_token = logits.argmax().item()
        # Token availability includes device-to-host logits and token selection.
        # Correctness checks/logging follow this timestamp and affect later ITL.
        ready_time = time.perf_counter()
        result = ids = mask = positions = None
        key = f"{sid}/turn{turn}/token{state['index']}"
        if reference is None:
            saved_logits[key] = logits
        else:
            target = reference[key]
            difference = (logits-target).abs()
            tolerance = config["logit_atol"] + config["logit_rtol"] * target.abs()
            comparison = {"checkpoint": key, "returning_turn": turn > 0,
                          "max_absolute_error": difference.max().item(),
                          "mean_absolute_error": difference.mean().item(),
                          "within_tolerance": bool(torch.all(difference <= tolerance)),
                          "argmax_agreement": logits.argmax().item() == target.argmax().item(),
                          "reference_argmax": target.argmax().item(),
                          "actual_argmax": logits.argmax().item()}
            comparisons.append(comparison)
            difference = tolerance = target = None
        forced_token = spec["output"][state["index"]]
        state["history"].append(forced_token)
        latency = None if first else ready_time-state["last_token"]
        if first:
            state["first_token"] = ready_time
            ttft = ready_time-state["arrival"]
            (returning_ttft if turn else initial_ttft).append(ttft)
        elif latency is not None:
            token_latencies.append(latency)
        measurement = {"case": event.case, "session": sid, "turn": turn,
                       "token_index": state["index"], "forced_token_id": forced_token,
                       "argmax_token_id": argmax_token,
                       "arrival_monotonic_seconds": state["arrival"],
                       "dispatch_monotonic_seconds": dispatch,
                       "ready_monotonic_seconds": ready_time,
                       "queue_seconds": dispatch-state["arrival"] if first else None,
                       "ttft_seconds": ready_time-state["arrival"] if first else None,
                       "inter_token_seconds": latency,
                       "forward_seconds": forward_end-compute_start,
                       "resume_seconds": resume_seconds,
                       "context_tokens": total_length,
                       "active_gpu_kv_bytes": sum(
                           tensor.numel()*tensor.element_size()
                           for entry in sessions.values() if entry["cache"] is not None
                           for pair in zip(entry["cache"].key_cache, entry["cache"].value_cache)
                           for tensor in pair),
                       "gpu_allocated_bytes": torch.cuda.memory_allocated()}
        measurements.append(measurement)
        if torch.cuda.memory_allocated() > config["total_gpu_budget_bytes"]:
            raise MemoryError("Total PyTorch GPU allocation exceeded configured budget")
        event("token", **{k: v for k, v in measurement.items() if k != "case"})
        if paused and sid not in paused:
            active_steps_while_paused += 1
        logits = None
        state["last_token"] = ready_time
        state["index"] += 1
        if state["index"] == len(spec["output"]):
            completion = ready_time
            count = len(spec["output"])
            turn_summaries.append({"session": sid, "turn": turn, "tokens": count,
                                   "ttft_seconds": state["first_token"]-state["arrival"],
                                   "tpot_seconds": ((completion-state["first_token"])/(count-1)
                                                    if count > 1 else None)})
            event("completion", session=sid, turn=turn, tokens=count,
                  completion_monotonic_seconds=completion)
            if turn + 1 < len(state["turns"]):
                next_turn = turn+1
                arrival = completion + state["turns"][next_turn]["think_seconds"]
                queue.schedule(sid, next_turn, arrival)
                event("pause_start", session=sid, turn=turn, action=action,
                      cached_tokens=state["cache"].get_seq_length(),
                      next_arrival_monotonic_seconds=arrival)
                pause_start = time.perf_counter()
                if kv_manager is not None:
                    state["handle"] = kv_manager.pause(state["cache"], action, sid)
                    state["cache"] = None
                torch.cuda.synchronize()
                paused.add(sid)
                event("pause", session=sid, action=action,
                      seconds=time.perf_counter()-pause_start,
                      stale_gpu_cache=state["cache"] is not None and action != "resident")
            else:
                state["cache"] = None
            state.update(turn_started=False, index=0, first_token=None, last_token=None)
        else:
            queue.requeue(sid, turn)
        snapshots.append(resource_snapshot())
    assert active_steps_while_paused > 0, "Workload did not exercise overlapping active/paused sessions"
    all_close = all(c["within_tolerance"] for c in comparisons)
    agreement = (sum(c["argmax_agreement"] for c in comparisons)/len(comparisons)
                 if comparisons else 1.0)
    returning_comparisons = [c for c in comparisons if c["returning_turn"]]
    current_cgroup = [sample["cgroup"]["groups"][0] for sample in snapshots
                      if sample.get("cgroup", {}).get("groups")]
    summary = {"wall_seconds": time.perf_counter()-started,
               "initial_ttft_seconds": distribution(initial_ttft),
               "returning_ttft_seconds": distribution(returning_ttft),
               "inter_token_seconds": distribution(token_latencies),
               "per_session_inter_token_seconds": {
                   sid: distribution([m["inter_token_seconds"] for m in measurements
                                      if m["session"] == sid and m["inter_token_seconds"] is not None])
                   for sid in sessions},
               "turns": turn_summaries,
               "active_steps_while_paused": active_steps_while_paused,
               "correctness": {"all_logits_within_tolerance": all_close,
                               "argmax_agreement": agreement,
                               "returning_checkpoints": len(returning_comparisons),
                               "returning_argmax_agreement": (
                                   sum(c["argmax_agreement"] for c in returning_comparisons)/len(returning_comparisons)
                                   if returning_comparisons else None),
                               "checkpoints": len(comparisons),
                               "passed": all_close and agreement >= config["minimum_argmax_agreement"],
                               "atol": config["logit_atol"], "rtol": config["logit_rtol"]},
               "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
               "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
               "active_gpu_kv_peak_bytes": max(m["active_gpu_kv_bytes"] for m in measurements),
               "cgroup_sampled_peak_bytes": max((s.get("memory.current", 0) for s in current_cgroup), default=0),
               "cgroup_historical_peak_bytes": max((s.get("memory.peak", 0) for s in current_cgroup), default=0),
               "cgroup_peak_note": "Sampled current is run-local lower bound; kernel peak can include prior setup/other jobs and was not reset.",
               "resource_samples": snapshots,
               "observed_counter_deltas": numeric_delta(snapshots[0], snapshots[-1])}
    return saved_logits, comparisons, measurements, summary


def run(args):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .audit import resource_snapshot, check_space
    from .experts import load_offloaded
    from .pool import SharedPool
    from .kv import KVManager

    output, scratch = Path(args.output), Path(args.scratch)
    output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(args.config).read_text())
    model_spec = json.loads(Path("configs/model.json").read_text())
    # Acquisition/setup writes the real pinned model snapshot to this directory.
    model_dir, backing = scratch / "model", scratch / "backing"
    event = Recorder(output)
    summaries, all_correctness, all_measurements = [], [], []
    began = time.perf_counter()
    deadline = began + cfg["max_wall_seconds"]
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(cfg["torch_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA GPU with BF16 support is required")
    physical_gpu_bytes = torch.cuda.get_device_properties(0).total_memory
    if not 0 < cfg["total_gpu_budget_bytes"] < physical_gpu_bytes:
        raise ValueError("Configured total GPU budget must leave device headroom")
    torch.cuda.set_per_process_memory_fraction(cfg["total_gpu_budget_bytes"]/physical_gpu_bytes)
    check_space(scratch)
    missing = [str(path) for path in (model_dir / "model-provenance.json", backing / "manifest.json")
               if not path.is_file()]
    if missing:
        raise RuntimeError("Prepared scratch data is missing (possibly an execution-target restart). "
                           "Run setup on a stable target before smoke/pilot. Missing: " + ", ".join(missing))
    model_provenance = json.loads((model_dir / "model-provenance.json").read_text())
    if any(model_provenance[k] != model_spec[k] for k in ("model_id", "revision")):
        raise RuntimeError("Prepared model differs from configured immutable model revision")
    with (output / "fixture-tests.log").open("w") as test_log:
        suite = unittest.defaultTestLoader.discover("tests")
        fixture_result = unittest.TextTestRunner(stream=test_log, verbosity=2).run(suite)
    write_json(output / "fixture-tests.json", {"run": fixture_result.testsRun,
               "failures": len(fixture_result.failures), "errors": len(fixture_result.errors),
               "skipped": len(fixture_result.skipped), "passed": fixture_result.wasSuccessful()})
    if not fixture_result.wasSuccessful():
        raise RuntimeError("Fixture tests failed; see fixture-tests.log")
    del suite, fixture_result
    gc.collect()
    torch.cuda.empty_cache()
    preparation = json.loads((backing / "manifest.json").read_text())
    if any(preparation["model_provenance"][k] != model_spec[k] for k in ("model_id", "revision")):
        raise RuntimeError("Expert backing differs from configured immutable model revision")
    write_json(output / "preparation-manifest.json", preparation)
    expert_bytes = preparation["max_expert_bytes"]
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    trajectories = make_trajectories(tokenizer, cfg)
    hf_config = json.loads((model_dir / "config.json").read_text())
    kv_bytes_per_token = (hf_config["num_hidden_layers"] * hf_config["num_key_value_heads"]
                          * (hf_config["hidden_size"] // hf_config["num_attention_heads"]) * 2 * 2)
    active_kv_bound = len(trajectories["sessions"]) * cfg["max_context_tokens"] * kv_bytes_per_token
    gpu_payload_reservation = (preparation["dense_bytes"] + cfg["gpu_expert_slots"] * expert_bytes
                               + active_kv_bound + cfg["gpu_workspace_reservation_bytes"])
    if gpu_payload_reservation > cfg["total_gpu_budget_bytes"]:
        raise MemoryError("Dense, expert, active KV and workspace reservations exceed total GPU budget")
    trajectory_bytes = json.dumps(trajectories, sort_keys=True).encode()
    write_json(output / "trajectories.json", trajectories)
    write_json(output / "experiment_config.json", cfg)
    environment = {"python": sys.version, "platform": platform.platform(),
                   "torch": torch.__version__, "transformers": transformers.__version__,
                   "torch_cuda": torch.version.cuda,
                   "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
                   "gpu": torch.cuda.get_device_name(),
                   "gpu_budget": {"total_pytorch_budget_bytes": cfg["total_gpu_budget_bytes"],
                                  "dense_bytes": preparation["dense_bytes"],
                                  "active_kv_bound_bytes": active_kv_bound,
                                  "expert_budget_bytes": cfg["gpu_expert_slots"] * expert_bytes,
                                  "workspace_reservation_bytes": cfg["gpu_workspace_reservation_bytes"],
                                  "scope": "PyTorch allocator cap; CUDA context/driver memory measured outside it"},
                   "model": model_spec,
                   "trajectory_sha256": hashlib.sha256(trajectory_bytes).hexdigest(),
                   "preparation_manifest_sha256": hashlib.sha256((backing/"manifest.json").read_bytes()).hexdigest(),
                   "pid": os.getpid(), "initial_resources": resource_snapshot(),
                   "conditions": "New application caches per trial; kernel page cache uncontrolled, probably warm after conversion/reference. No global cache drop.",
                   "timing": "Synchronous pageable host I/O and CUDA synchronization; wall-clock token timings include serialized scheduling. Component timings are not summed.",
                   "scope": "OLMoE BF16 mechanism prototype; not evidence about Qwen3 FP8 performance."}
    write_json(output / "environment.json", environment)
    try:
        event.case = "resident-reference"
        event("model_load_start")
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16,
                    device_map={"": "cuda:0"}, low_cpu_mem_usage=True,
                    attn_implementation="eager", local_files_only=True, trust_remote_code=False).eval()
        assert all(p.device.type == "cuda" for p in model.parameters())
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        event("model_load_complete", resident_parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()))
        references, _, measures, summary = execute_schedule(model, trajectories, cfg, event,
                                                           "resident", deadline=deadline)
        summary.update(case=event.case, action="resident")
        summaries.append(summary)
        all_measurements.extend(measures)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        event("resident_model_released", gpu_allocated_bytes=torch.cuda.memory_allocated())
        # Count reference logits plus comparison scratch as an explicit reservation.
        # No checkpoint/expert tensors are retained on CPU by the runner.
        validation_bytes = sum(t.numel()*t.element_size() for t in references.values())
        validation_bytes += 6 * max(t.numel()*t.element_size() for t in references.values())
        for repeat in range(cfg["repeats"]):
            for slots in cfg["host_expert_slots"]:
                for action in cfg["actions"]:
                    if time.perf_counter() >= deadline:
                        raise TimeoutError("Pilot execution cap reached before next trial")
                    check_space(scratch)
                    event.case = f"{action}-host{slots}-repeat{repeat}"
                    pool_budget = validation_bytes + slots * expert_bytes + cfg["host_slack_bytes"]
                    gpu_budget = cfg["gpu_expert_slots"] * expert_bytes
                    pool = SharedPool(pool_budget)
                    pool.reserve("validation", validation_bytes, category="validation")
                    event("trial_start", host_budget_bytes=pool_budget,
                          gpu_expert_budget_bytes=gpu_budget, application_caches="cold",
                          filesystem_cache="uncontrolled; likely warm")
                    model, expert_cache = load_offloaded(model_dir, backing, pool, gpu_budget, event)
                    kv = KVManager(pool, backing / "kv" / output.name / event.case, event)
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    _, correctness, measures, summary = execute_schedule(
                        model, trajectories, cfg, event, action, references, kv, deadline)
                    stats = expert_cache.stats()
                    summary.update(case=event.case, action=action, host_expert_slots=slots,
                                   repeat=repeat, host_pool=pool.stats(), expert_cache=stats,
                                   gpu_expert_budget_bytes=gpu_budget,
                                   validation_reservation_bytes=validation_bytes,
                                   kv=kv.stats() if hasattr(kv, "stats") else {})
                    # These keys are adapter's explicit nonempty CUDA execution / miss counters.
                    summary["mechanism_passed"] = (stats["host_misses"] > 0 and stats["cuda_executions"] > 0
                                                   and summary["kv"].get(action, 0) == 1
                                                   and summary["kv"].get("live_handles", 1) == 0
                                                   and (action != "spill" or summary["kv"].get("reloads", 0) == 1))
                    summaries.append(summary)
                    all_correctness.extend(dict(case=event.case, **item) for item in correctness)
                    all_measurements.extend(measures)
                    event("trial_complete", correctness=summary["correctness"],
                          host_pool=summary["host_pool"], expert_cache=stats)
                    write_json(output / "summary.json", {"complete": False, "cases": summaries})
                    write_json(output / "correctness.json", all_correctness)
                    expert_cache.close()
                    if hasattr(kv, "close"):
                        kv.close()
                    pool.release("validation")
                    assert pool.stats()["used_bytes"] == 0, "Host reservation leaked"
                    del model, expert_cache, kv, pool
                    gc.collect()
                    torch.cuda.empty_cache()
        actions_exercised = {s["action"] for s in summaries if s["action"] != "resident"}
        passed = (all(s["correctness"]["passed"] and s.get("mechanism_passed", True) for s in summaries)
                  and actions_exercised == {"retain", "spill", "recompute"})
        report = {"complete": True, "passed": passed, "mode": args.mode,
                  "wall_seconds": time.perf_counter()-began, "cases": summaries,
                  "limitations": ["Tiny sample; no p99 claim.", "Application cache misses are logical reads, not proof of SSD reads.",
                                  "Synchronous I/O; no concurrent device I/O or optimized serving.",
                                  "Page cache and other cgroup activity can affect observed counters.",
                                  "Two sessions, short real-text forced trajectories, single GPU BF16 only."]}
        write_json(output / "summary.json", report)
        print(json.dumps({k: report[k] for k in ("complete", "passed", "mode", "wall_seconds")}), flush=True)
        return 0 if passed else 2
    except Exception as error:
        event("failure", error=repr(error), traceback=traceback.format_exc())
        write_json(output / "summary.json", {"complete": False, "passed": False,
                   "error": repr(error), "wall_seconds": time.perf_counter()-began, "cases": summaries})
        raise
    finally:
        write_json(output / "correctness.json", all_correctness)
        with (output / "measurements.jsonl").open("w") as handle:
            for item in all_measurements:
                handle.write(json.dumps(item, sort_keys=True) + "\n")
        if all_measurements:
            with (output / "measurements.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(all_measurements[0]))
                writer.writeheader()
                writer.writerows(all_measurements)
        event.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "pilot"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scratch", required=True)
    args = parser.parse_args()
    try:
        code = run(args)
    except Exception as error:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        if not (output / "summary.json").exists():
            write_json(output / "summary.json", {"complete": False, "passed": False,
                       "mode": args.mode, "phase": "preflight", "error": repr(error), "cases": []})
        raise
    raise SystemExit(code)


if __name__ == "__main__":
    main()
