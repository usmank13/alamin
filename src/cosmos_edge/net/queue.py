"""Virtual-time network: latency is injected as a scheduled delay, never measured.

A run is exactly reproducible from (seed, latency trace)."""
import heapq
from dataclasses import dataclass

import numpy as np

from ..messages import Envelope


@dataclass
class LatencyTrace:
    """One-way delay model in seconds.

    kind: ``fixed`` (base), ``jitter`` (base + |N(0, scale)|), ``heavy`` (base +
    Pareto(shape)·scale), ``replay`` (delays read cyclically from ``values``).
    ``drop`` is a per-message loss probability; ``outages`` are (start, end) windows
    in virtual seconds during which every message is lost.
    """
    kind: str = "fixed"
    base: float = 0.05
    scale: float = 0.0
    shape: float = 2.5
    drop: float = 0.0
    outages: tuple = ()
    values: tuple = ()

    def sample(self, rng, now, index):
        if any(start <= now < end for start, end in self.outages) or rng.random() < self.drop:
            return None
        if self.kind == "fixed":
            return self.base
        if self.kind == "jitter":
            return self.base + abs(rng.normal(0.0, self.scale))
        if self.kind == "heavy":
            return self.base + rng.pareto(self.shape) * self.scale
        if self.kind == "replay":
            return self.values[index % len(self.values)]
        raise ValueError(f"unknown latency kind {self.kind!r}")


class VirtualNet:
    """Two directional channels over a single event queue, logged for analysis."""

    def __init__(self, down: LatencyTrace, up: LatencyTrace, seed=0):
        self.traces = {"down": down, "up": up}
        self.rng = np.random.default_rng(seed)
        self.queue = []
        self.seq = 0
        self.log = []

    def schedule(self, channel, payload, now, extra_delay=0.0):
        delay = self.traces[channel].sample(self.rng, now, self.seq)
        entry = {"channel": channel, "type": type(payload).__name__, "sent": now,
                 "bytes": payload.nbytes() if hasattr(payload, "nbytes") else 0}
        if delay is None:
            entry["dropped"] = True
        else:
            entry["received"] = now + extra_delay + delay
            heapq.heappush(self.queue, Envelope(now + extra_delay + delay, self.seq, now, channel, payload))
        self.log.append(entry)
        self.seq += 1

    def deliver(self, now):
        """Pop every message due by ``now``, in arrival order."""
        out = []
        while self.queue and self.queue[0].deliver_at <= now + 1e-12:
            out.append(heapq.heappop(self.queue))
        return out
