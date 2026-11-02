"""Narrow expert-storage adapter; HF attention and routing remain unchanged."""

from collections import OrderedDict
import json
import time
from pathlib import Path

import torch
from torch import nn
from accelerate import init_empty_weights
from transformers import AutoConfig, OlmoeForCausalLM
from transformers.models.olmoe.modeling_olmoe import OlmoeMLP

from .pool import BudgetExceeded
from .prepare import read_tensor_file, tensor_bytes


def _event(callback, name, **fields):
    if callback is not None:
        callback(name, **fields)


class ExpertCache:
    """Global deterministic GPU LRU and pool-owned host LRU.

    Host tensors use pageable RAM. Raw readinto fills the already-accounted flat
    allocation directly. There is no mmap, checkpoint state dict, or separate
    pinned copy hidden behind the host budget. OS file cache is outside this pool.
    """

    def __init__(self, config, backing, pool, gpu_budget_bytes, event=None):
        self.config = config
        self.backing = Path(backing)
        self.manifest = json.loads((self.backing / "manifest.json").read_text())
        self.pool = pool
        self.gpu_budget_bytes = int(gpu_budget_bytes)
        if self.gpu_budget_bytes < self.manifest["max_expert_bytes"]:
            raise BudgetExceeded("GPU expert budget cannot hold one expert")
        self.event = event
        self.host = {}
        self.gpu = OrderedDict()
        self.gpu_bytes = 0
        self.gpu_peak_bytes = 0
        self.counters = {key: 0 for key in (
            "requests", "cuda_executions", "empty_skips", "gpu_hits", "gpu_misses",
            "host_hits", "host_misses", "gpu_evictions", "host_evictions",
            "logical_read_bytes", "h2d_bytes")}
        self.counters.update({"read_seconds": 0.0, "h2d_seconds": 0.0})

    def _drop_host(self, key):
        del self.host[key]
        self.counters["host_evictions"] += 1

    def _host_tensor(self, key):
        tag = "expert:" + key
        if key in self.host:
            self.counters["host_hits"] += 1
            self.pool.touch(tag)
            return self.host[key], 0.0
        self.counters["host_misses"] += 1
        entry = self.manifest["experts"][key]
        self.pool.reserve(tag, entry["nbytes"], "expert", evictable=True,
                          on_evict=lambda: self._drop_host(key))
        start = time.perf_counter()
        try:
            flat = torch.empty(entry["nbytes"] // 2, dtype=torch.bfloat16, device="cpu")
            self.host[key] = flat
            read_tensor_file(self.backing / entry["file"], flat)
        except BaseException:
            self.host.pop(key, None)
            self.pool.release(tag)
            raise
        elapsed = time.perf_counter() - start
        self.counters["read_seconds"] += elapsed
        self.counters["logical_read_bytes"] += entry["nbytes"]
        return flat, elapsed

    def get(self, key):
        self.counters["requests"] += 1
        if key in self.gpu:
            self.counters["gpu_hits"] += 1
            self.gpu.move_to_end(key)
            if key in self.host:
                self.pool.touch("expert:" + key)
            return self.gpu[key]
        self.counters["gpu_misses"] += 1
        entry = self.manifest["experts"][key]
        while self.gpu_bytes + entry["nbytes"] > self.gpu_budget_bytes:
            victim, module = self.gpu.popitem(last=False)
            self.gpu_bytes -= self.manifest["experts"][victim]["nbytes"]
            self.counters["gpu_evictions"] += 1
            del module
        flat, read_seconds = self._host_tensor(key)
        state = {}
        for layout in entry["layout"]:
            start = layout["offset_bytes"] // 2
            state[layout["name"]] = flat[start:start + layout["nbytes"] // 2].view(layout["shape"])
        started = time.perf_counter()
        with torch.device("meta"):
            module = OlmoeMLP(self.config)
        module.load_state_dict(state, strict=True, assign=True)
        module.requires_grad_(False)
        module.to(device="cuda", dtype=torch.bfloat16)
        module.eval()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        del state, flat
        assert all(p.device.type == "cuda" for p in module.parameters())
        self.gpu[key] = module
        self.gpu_bytes += entry["nbytes"]
        self.gpu_peak_bytes = max(self.gpu_peak_bytes, self.gpu_bytes)
        assert self.gpu_bytes <= self.gpu_budget_bytes
        self.counters["h2d_bytes"] += entry["nbytes"]
        self.counters["h2d_seconds"] += elapsed
        _event(self.event, "expert_gpu_miss", expert=key, nbytes=entry["nbytes"],
               read_seconds=read_seconds, h2d_seconds=elapsed,
               gpu_expert_bytes=self.gpu_bytes, host_pool_bytes=self.pool.used_bytes)
        return module

    def stats(self):
        return {**self.counters, "gpu_budget_bytes": self.gpu_budget_bytes,
                "gpu_bytes": self.gpu_bytes, "gpu_peak_bytes": self.gpu_peak_bytes,
                "gpu_entries": len(self.gpu), "host_entries": len(self.host),
                "expert_policy": "global LRU", "pinned_staging_bytes": 0,
                "transfer_mode": "synchronous pageable H2D; measured wall time includes module construction"}

    def close(self):
        self.gpu.clear()
        self.gpu_bytes = 0
        for key in list(self.host):
            del self.host[key]
            self.pool.release("expert:" + key)


class ExpertProxy(nn.Module):
    def __init__(self, cache, key):
        super().__init__()
        self.cache = cache  # plain Python object, not a registered model/state dict
        self.key = key

    def forward(self, inputs):
        if inputs.device.type != "cuda":
            raise AssertionError("every expert must execute on CUDA")
        # HF 4.46.3 calls all experts, including ones assigned zero tokens. The
        # empty matrix output equals this empty input without reading weights.
        if inputs.shape[0] == 0:
            self.cache.counters["empty_skips"] += 1
            return inputs
        if torch.is_grad_enabled():
            raise AssertionError("offloaded experts require inference_mode/no_grad")
        module = self.cache.get(self.key)
        assert all(p.device.type == "cuda" for p in module.parameters())
        output = module(inputs)
        assert output.device.type == "cuda"
        self.cache.counters["cuda_executions"] += 1
        return output


def _load_dense(model, manifest, backing, pool, chunk_bytes=8 * 1024 * 1024):
    """Read large dense tensors in bounded, accounted CPU staging chunks."""
    for name, entry in manifest["dense"].items():
        if len(entry["layout"]) != 1:
            raise ValueError("dense files must contain exactly one tensor")
        layout = entry["layout"][0]
        count = entry["nbytes"] // 2
        gpu = torch.empty(count, dtype=torch.bfloat16, device="cuda")
        stage_bytes = min(chunk_bytes, entry["nbytes"])
        tag = "dense_staging:" + name
        pool.reserve(tag, stage_bytes, "staging")
        stage = None
        try:
            stage = torch.empty(stage_bytes // 2, dtype=torch.bfloat16, device="cpu")
            with (Path(backing) / entry["file"]).open("rb", buffering=0) as stream:
                for offset in range(0, count, stage.numel()):
                    size = min(stage.numel(), count - offset)
                    view = tensor_bytes(stage[:size])
                    transferred = 0
                    while transferred < len(view):
                        got = stream.readinto(view[transferred:])
                        if not got:
                            raise IOError("truncated dense backing")
                        transferred += got
                    gpu[offset:offset + size].copy_(stage[:size])
                    del view
                if stream.read(1):
                    raise IOError("oversized dense backing")
            torch.cuda.synchronize()
        finally:
            del stage
            pool.release(tag)
        parent, attr = name.rsplit(".", 1)
        module = model.get_submodule(parent)
        if attr not in module._parameters:
            raise ValueError(f"unexpected model weight {name}")
        module._parameters[attr] = nn.Parameter(gpu.view(layout["shape"]), requires_grad=False)
        del gpu


def load_offloaded(model_dir, backing, pool, gpu_budget_bytes, event=None):
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA device is required for the offloaded model")
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    if config.model_type != "olmoe":
        raise ValueError("only OLMoE is supported by this narrow adapter")
    config._attn_implementation = "eager"
    with init_empty_weights(include_buffers=False):
        model = OlmoeForCausalLM(config)
    cache = ExpertCache(config, backing, pool, gpu_budget_bytes, event)
    for layer_id, layer in enumerate(model.model.layers):
        layer.mlp.experts = nn.ModuleList(
            ExpertProxy(cache, f"{layer_id}.{expert_id}")
            for expert_id in range(config.num_experts))
    _load_dense(model, cache.manifest, backing, pool)
    if config.tie_word_embeddings:
        model.tie_weights()
    remaining_meta = [name for name, value in model.named_parameters() if value.device.type == "meta"]
    if remaining_meta:
        raise AssertionError(f"missing dense checkpoint tensors: {remaining_meta}")
    model.to("cuda")  # moves small RoPE buffers; all model parameters are already CUDA
    for module in model.modules():
        # HF keeps this unregistered alias for dynamic-RoPE resets. Keep it on
        # the same device as its registered buffer as well.
        if hasattr(module, "original_inv_freq") and hasattr(module, "inv_freq"):
            module.original_inv_freq = module.inv_freq
    model.requires_grad_(False).eval()
    assert all(p.device.type == "cuda" for p in model.parameters())
    assert all(buffer.device.type == "cuda" for buffer in model.buffers())
    _event(event, "offloaded_model_ready", dense_bytes=cache.manifest["dense_bytes"],
           expert_count=len(cache.manifest["experts"]), gpu_expert_budget_bytes=gpu_budget_bytes,
           host_pool=pool.stats(), hidden_cpu_model_parameters=0)
    return model, cache
