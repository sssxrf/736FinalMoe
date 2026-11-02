"""Actual DynamicCache transitions through the same host pool as experts."""

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

import torch
from transformers.cache_utils import DynamicCache

from .prepare import tensor_bytes


@dataclass
class KVHandle:
    action: str
    nbytes: int
    seq_length: int
    metadata: dict
    tag: str
    tensors: list = field(default_factory=list, repr=False)
    path: Path | None = None
    consumed: bool = False


def cache_nbytes(cache):
    return sum(t.numel() * t.element_size()
               for pair in zip(cache.key_cache, cache.value_cache) for t in pair)


class KVManager:
    """Single-threaded, synchronous KV pause/resume operations.

    This adapter requires the uncompressed DynamicCache used by the pinned
    OLMoE model. Token IDs, attention masks, and scheduling state stay with the
    caller; cache length and seen-token metadata are retained here.
    """

    def __init__(self, pool, directory, event=None, device="cuda"):
        self.pool = pool
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.event = event
        self.device = device
        self.handles = {}
        self.counters = {key: 0 for key in ("retain", "spill", "recompute", "reloads",
            "logical_read_bytes", "logical_write_bytes", "h2d_bytes", "d2h_bytes",
            "paused_bytes", "resumed_bytes")}
        self.counters.update({key: 0.0 for key in
                             ("pause_seconds", "resume_seconds", "read_seconds", "write_seconds",
                              "h2d_seconds", "d2h_seconds")})

    def _emit(self, name, **fields):
        if self.event is not None:
            self.event(name, **fields)

    def _sync(self):
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

    def pause(self, cache, action, session_id):
        started = time.perf_counter()
        action = action.lower()
        if action not in ("retain", "spill", "recompute"):
            raise ValueError(f"unknown KV action: {action}")
        if type(cache) is not DynamicCache:
            raise TypeError("only the pinned Transformers DynamicCache is supported")
        if len(cache.key_cache) != len(cache.value_cache) or not cache.key_cache:
            raise ValueError("expected nonempty paired KV cache")
        layout = []
        offset = 0
        for layer, pair in enumerate(zip(cache.key_cache, cache.value_cache)):
            for role, tensor in zip(("key", "value"), pair):
                if tensor.device.type != torch.device(self.device).type:
                    raise AssertionError("paused cache was on an unexpected device")
                nbytes = tensor.numel() * tensor.element_size()
                layout.append({"layer": layer, "role": role, "shape": list(tensor.shape),
                               "dtype": str(tensor.dtype).removeprefix("torch."),
                               "offset_bytes": offset, "nbytes": nbytes})
                offset += nbytes
        metadata = {"format_version": 1, "cache_type": "DynamicCache", "layout": layout,
                    "seq_length": cache.get_seq_length(), "seen_tokens": cache._seen_tokens,
                    "session_id": str(session_id), "nbytes": offset}
        tag = "kv:" + uuid.uuid4().hex
        handle = KVHandle(action, offset, metadata["seq_length"], metadata, tag)
        if action in ("retain", "spill"):
            self.pool.reserve(tag, offset, "kv" if action == "retain" else "staging")
            try:
                transfer_start = time.perf_counter()
                handle.tensors = [torch.empty(tuple(item["shape"]),
                                   dtype=getattr(torch, item["dtype"]), device="cpu")
                                  for item in layout]
                for target, item in zip(handle.tensors, layout):
                    source = (cache.key_cache if item["role"] == "key"
                              else cache.value_cache)[item["layer"]]
                    target.copy_(source)
                self._sync()
                self.counters["d2h_seconds"] += time.perf_counter() - transfer_start
                self.counters["d2h_bytes"] += offset
                if action == "spill":
                    self._write(handle)
                    handle.tensors.clear()
                    # The loop-local tensor alias must not outlive accounting.
                    del target, source
                    self.pool.release(tag)
            except BaseException:
                handle.tensors.clear()
                self.pool.release(tag)
                raise
        # Mutate the original cache, so aliases (including a ModelOutput) cannot
        # silently keep usable GPU KV and bypass the chosen action.
        cache.key_cache.clear()
        cache.value_cache.clear()
        cache._seen_tokens = 0
        self.handles[tag] = handle
        self.counters[action] += 1
        self.counters["paused_bytes"] += offset
        elapsed = time.perf_counter() - started
        self.counters["pause_seconds"] += elapsed
        self._emit("kv_pause", action=action, session_id=str(session_id), nbytes=offset,
                   seq_length=handle.seq_length, seconds=elapsed, host_pool=self.pool.stats(),
                   stale_gpu_cache_entries=len(cache.key_cache))
        return handle

    def _write(self, handle):
        from .audit import check_space
        # This directory is normally scratch/backing/kv/<trial>. The original
        # scratch root is located from the prepared backing manifest.
        ancestor = self.directory
        while ancestor != ancestor.parent and not (ancestor / "manifest.json").is_file():
            ancestor = ancestor.parent
        if (ancestor / "manifest.json").is_file():
            check_space(ancestor.parent, additional_bytes=handle.nbytes)
        else:
            check_space(self.directory, additional_bytes=handle.nbytes)
        path = self.directory / (handle.tag.removeprefix("kv:") + ".bin")
        started = time.perf_counter()
        digest = hashlib.sha256()
        with path.open("xb", buffering=0) as stream:
            for tensor in handle.tensors:
                view = tensor_bytes(tensor)
                digest.update(view)
                offset = 0
                while offset < len(view):
                    count = stream.write(view[offset:])
                    if not count:
                        raise IOError("short KV spill write")
                    offset += count
            os.fsync(stream.fileno())
        handle.path = path
        handle.metadata["sha256"] = digest.hexdigest()
        with path.with_suffix(".json").open("x", encoding="utf-8") as stream:
            json.dump(handle.metadata, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.counters["logical_write_bytes"] += handle.nbytes
        self.counters["write_seconds"] += time.perf_counter() - started

    def resume(self, handle):
        if handle.consumed or self.handles.get(handle.tag) is not handle:
            raise ValueError("KV handle has already been consumed or belongs to another manager")
        started = time.perf_counter()
        if handle.action == "recompute":
            self.release(handle)
            self._emit("kv_resume", action="recompute", nbytes=handle.nbytes,
                       seq_length=handle.seq_length, seconds=0.0, cache_replay_required=True)
            return None
        if handle.action == "spill":
            self.pool.reserve(handle.tag, handle.nbytes, "staging")
            try:
                self._read(handle)
            except BaseException:
                handle.tensors.clear()
                self.pool.release(handle.tag)
                raise
        try:
            cache = DynamicCache()
            transfer_start = time.perf_counter()
            for item, cpu in zip(handle.metadata["layout"], handle.tensors):
                gpu = cpu.to(self.device, copy=True)
                (cache.key_cache if item["role"] == "key"
                 else cache.value_cache).append(gpu)
            cache._seen_tokens = handle.metadata["seen_tokens"]
            self._sync()
            self.counters["h2d_seconds"] += time.perf_counter() - transfer_start
            self.counters["h2d_bytes"] += handle.nbytes
            del cpu, gpu
            assert cache.get_seq_length() == handle.seq_length
        except BaseException:
            if handle.action == "spill":
                handle.tensors.clear()
                self.pool.release(handle.tag)
            raise
        handle.tensors.clear()
        self.pool.release(handle.tag)
        self.counters["resumed_bytes"] += handle.nbytes
        elapsed = time.perf_counter() - started
        self.counters["resume_seconds"] += elapsed
        self.release(handle)
        self._emit("kv_resume", action=handle.action, nbytes=handle.nbytes,
                   seq_length=handle.seq_length, seconds=elapsed, host_pool=self.pool.stats())
        return cache

    def _read(self, handle):
        started = time.perf_counter()
        metadata = json.loads(handle.path.with_suffix(".json").read_text())
        if metadata != handle.metadata:
            raise IOError("KV spill metadata changed")
        digest = hashlib.sha256()
        with handle.path.open("rb", buffering=0) as stream:
            for item in metadata["layout"]:
                tensor = torch.empty(tuple(item["shape"]),
                                     dtype=getattr(torch, item["dtype"]), device="cpu")
                handle.tensors.append(tensor)
                view = tensor_bytes(tensor)
                offset = 0
                while offset < len(view):
                    count = stream.readinto(view[offset:])
                    if not count:
                        raise IOError("truncated KV spill")
                    offset += count
                digest.update(view)
            if stream.read(1) or digest.hexdigest() != metadata["sha256"]:
                raise IOError("KV spill checksum/length mismatch")
        self.counters["logical_read_bytes"] += handle.nbytes
        self.counters["read_seconds"] += time.perf_counter() - started
        self.counters["reloads"] += 1

    def release(self, handle):
        """Release an abandoned handle or finalize a successfully resumed one."""
        if handle.consumed:
            return
        handle.tensors.clear()
        if handle.tag in self.pool.entries:
            self.pool.release(handle.tag)
        if handle.path is not None:
            handle.path.unlink(missing_ok=True)
            handle.path.with_suffix(".json").unlink(missing_ok=True)
        handle.consumed = True
        self.handles.pop(handle.tag, None)

    def close(self):
        for handle in list(self.handles.values()):
            self.release(handle)

    def stats(self):
        return {**self.counters, "live_handles": len(self.handles), "pinned_staging_bytes": 0,
                "transfer_mode": "synchronous pageable copies and fsync on spill",
                "timing_note": "action totals overlap component timings; do not add them"}
