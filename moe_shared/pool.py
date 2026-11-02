"""Explicit application-owned host-memory accounting, independent of PyTorch.

OS page cache and Python/runtime allocations are measured separately by audit.py.
Callers reserve BEFORE allocating tensors and release AFTER dropping references.
Only expert entries may be evicted; retained session state is never silently lost.
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable


class BudgetExceeded(MemoryError):
    pass


@dataclass
class Entry:
    nbytes: int
    category: str
    evictable: bool
    on_evict: Callable | None


class SharedPool:
    def __init__(self, budget_bytes: int):
        if not isinstance(budget_bytes, int) or budget_bytes <= 0:
            raise ValueError("budget_bytes must be a positive integer")
        self.budget_bytes = budget_bytes
        self.entries = OrderedDict()
        self.used_bytes = 0
        self.peak_bytes = 0
        self.evictions = 0

    def reserve(self, tag: str, nbytes: int, category="reservation", *,
                evictable=False, on_evict=None):
        if tag in self.entries:
            raise ValueError(f"duplicate pool reservation: {tag}")
        if not isinstance(nbytes, int) or nbytes < 0:
            raise ValueError("nbytes must be a nonnegative integer")
        if evictable and (category != "expert" or on_evict is None):
            raise ValueError("only expert entries with eviction callbacks are evictable")
        available = self.budget_bytes - self.used_bytes
        reclaimable = sum(e.nbytes for e in self.entries.values() if e.evictable)
        if nbytes > available + reclaimable:
            raise BudgetExceeded(
                f"host pool cannot reserve {tag}={nbytes}: used={self.used_bytes}, "
                f"budget={self.budget_bytes}, reclaimable_experts={reclaimable}"
            )
        while self.used_bytes + nbytes > self.budget_bytes:
            victim = next(k for k, e in self.entries.items() if e.evictable)
            entry = self.entries[victim]
            # Drop actual tensor references before removing their accounting.
            entry.on_evict()
            self.release(victim)
            self.evictions += 1
        self.entries[tag] = Entry(nbytes, category, evictable, on_evict)
        self.used_bytes += nbytes
        self.peak_bytes = max(self.peak_bytes, self.used_bytes)
        assert self.used_bytes <= self.budget_bytes

    def release(self, tag: str):
        entry = self.entries.pop(tag)
        self.used_bytes -= entry.nbytes
        assert self.used_bytes >= 0

    def touch(self, tag: str):
        self.entries.move_to_end(tag)

    def stats(self):
        categories = {}
        for entry in self.entries.values():
            categories[entry.category] = categories.get(entry.category, 0) + entry.nbytes
        return {"budget_bytes": self.budget_bytes, "used_bytes": self.used_bytes,
                "peak_bytes": self.peak_bytes, "by_category": categories,
                "entries": len(self.entries), "evictions": self.evictions,
                "pinned_bytes": 0,
                "accounting_scope": "tensor payloads and explicit reservations; runtime and OS page cache separate"}
