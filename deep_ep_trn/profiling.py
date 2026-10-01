"""Opt-in per-stage wall-clock timers for Buffer ops (benchmarking only).

Disabled by default. When enabled, ``sync`` (e.g. ``torch.neuron.synchronize``)
runs at every stage boundary so device work is charged to the stage that issued
it; that perturbs end-to-end latency, so measure breakdown and e2e separately.

Stages: in_d2h (inputs to host), plan, pack, h2d, kernel, d2h, unpack,
out_h2d (outputs back to the caller's device).
"""
from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Callable, Optional


class StageTimer:
    def __init__(self):
        self.enabled = False
        self.sync: Optional[Callable[[], None]] = None
        self.totals: dict[str, float] = defaultdict(float)

    def reset(self) -> None:
        self.totals.clear()

    @contextmanager
    def stage(self, name: str):
        if not self.enabled:
            yield
            return
        if self.sync:
            self.sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            if self.sync:
                self.sync()
            self.totals[name] += time.perf_counter() - start

    def per_call_us(self, calls: int) -> dict[str, float]:
        return {k: v * 1e6 / calls for k, v in self.totals.items()}


timer = StageTimer()
