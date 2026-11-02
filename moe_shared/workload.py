"""Deterministic, real-text teacher-forced session trajectories (no torch dependency)."""
from collections import deque
import time


TEXT_A = """An operating system shares memory among applications while keeping track of
which pages are useful. A server can keep a conversation in memory when a person
pauses to think. Keeping that state makes the next response faster, but leaves
less room for other data. When memory becomes scarce, the server may write old
state to storage or rebuild it later. The best choice depends on the cost of
reading data, the amount of computation, and the time before the conversation
returns. Careful experiments hold the requests constant and measure each delay.
An application cache and the filesystem page cache are different mechanisms.
Reading a file does not always cause a physical device read. A useful report
therefore records both logical bytes and the available operating system counters.
The experiment also records the resources used by each concurrent session.
"""
TEXT_B = """A student is writing a small service that answers questions about systems.
The service processes one token at a time and alternates between conversations.
While one person is thinking, another person continues receiving an answer.
Experts are selected by a learned router and run on the graphics processor.
Some expert weights remain nearby, while other weights must be read from files.
A scheduler records when a request arrives and when its first token is ready.
These timestamps distinguish waiting in a queue from time spent computing.
"""


def make_trajectories(tokenizer, config):
    """Tokenize ONCE, then reuse the exact saved IDs in every reference/trial."""
    a = tokenizer.encode(TEXT_A * 8, add_special_tokens=True)
    b = tokenizer.encode(TEXT_B * 8, add_special_tokens=True)
    ap, bp = config["paused_prefix_tokens"], config["active_prefix_tokens"]
    n, m = config["first_turn_tokens"], config["active_output_tokens"]
    rp, rn = config["return_prompt_tokens"], config["return_output_tokens"]
    assert ap + n + rp + rn <= len(a) and bp + m <= len(b)
    return {
        "format": 1,
        "description": "Recorded real-text IDs; every output token is teacher-forced, not freely generated.",
        "sessions": [
            {"id": "paused", "turns": [
                {"prompt": a[:ap], "output": a[ap:ap+n], "think_seconds": 0.0},
                {"prompt": a[ap+n:ap+n+rp], "output": a[ap+n+rp:ap+n+rp+rn],
                 "think_seconds": config["think_seconds"]},
            ]},
            {"id": "active", "turns": [
                {"prompt": b[:bp], "output": b[bp:bp+m], "think_seconds": 0.0},
            ]},
        ],
    }


class ArrivalQueue:
    """Round-robin ready queue; later arrivals are relative to prior completion."""
    def __init__(self):
        self.ready = deque()
        self.future = []

    def schedule(self, session, turn, arrival):
        self.future.append((arrival, session, turn))
        self.future.sort()

    def admit(self, now):
        arrivals = []
        while self.future and self.future[0][0] <= now:
            arrival, session, turn = self.future.pop(0)
            self.ready.append((session, turn))
            arrivals.append((session, turn, arrival))
        return arrivals

    def next(self):
        return self.ready.popleft()

    def requeue(self, session, turn):
        self.ready.append((session, turn))

    def pending(self):
        return bool(self.ready or self.future)

    def idle_wait(self, now):
        if not self.ready and self.future:
            time.sleep(min(max(0, self.future[0][0] - now), 0.01))


def distribution(values):
    """Small samples: show count, mean, median, max; intentionally no p99."""
    if not values:
        return {"count": 0, "mean": None, "median": None, "max": None}
    ordered = sorted(values)
    n = len(ordered)
    return {"count": n, "mean": sum(ordered) / n,
            "median": (ordered[(n-1)//2] + ordered[n//2]) / 2,
            "max": ordered[-1]}
