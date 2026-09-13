"""Latency / FPS helpers."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class Timer:
    """Simple stage timer; values are milliseconds."""

    stages: dict[str, float] = field(default_factory=dict)
    _marks: dict[str, float] = field(default_factory=dict, repr=False)

    def start(self, name: str) -> None:
        self._marks[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        began = self._marks.pop(name, None)
        if began is None:
            return 0.0
        ms = (time.perf_counter() - began) * 1000.0
        self.stages[name] = round(ms, 3)
        return self.stages[name]

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    @property
    def total_ms(self) -> float:
        return float(self.stages.get("total", sum(self.stages.values())))

    def as_dict(self) -> dict[str, float]:
        out = dict(self.stages)
        if "total" not in out and out:
            out["total"] = round(sum(v for k, v in out.items() if k != "total"), 3)
        return out


@dataclass
class EmaFps:
    """Exponential moving average of frames-per-second."""

    alpha: float = 0.2
    value: float = 0.0
    _last: float | None = field(default=None, repr=False)

    def tick(self, now: float | None = None) -> float:
        now = time.perf_counter() if now is None else now
        if self._last is None:
            self._last = now
            return self.value
        dt = now - self._last
        self._last = now
        if dt <= 0:
            return self.value
        inst = 1.0 / dt
        if self.value <= 0:
            self.value = inst
        else:
            self.value = self.alpha * inst + (1.0 - self.alpha) * self.value
        return self.value
