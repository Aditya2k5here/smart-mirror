"""In-memory countdown timers (also used for focus sessions)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import itertools
import time

from core.timeparse import format_duration


@dataclass
class Timer:
    id: int
    label: str
    total: int
    ends_at: float
    kind: str = "timer"  # timer | focus | break
    fired: bool = False


class TimerService:
    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._timers: list[Timer] = []
        self._ids = itertools.count(1)

    def start(self, seconds: int, label: str = "", kind: str = "timer") -> Timer:
        seconds = max(1, int(seconds))
        timer = Timer(next(self._ids), label.strip() or format_duration(seconds), seconds, self._clock() + seconds, kind)
        self._timers.append(timer)
        return timer

    def cancel(self, label: str = "") -> list[Timer]:
        label = label.lower().strip()
        keep, removed = [], []
        for t in self._timers:
            (removed if not label or label in t.label.lower() else keep).append(t)
        self._timers = keep
        return removed

    def active(self) -> list[Timer]:
        return sorted((t for t in self._timers if not t.fired), key=lambda t: t.ends_at)

    def remaining(self, timer: Timer) -> int:
        return max(0, int(round(timer.ends_at - self._clock())))

    def ringing(self) -> list[Timer]:
        return [t for t in self._timers if t.fired]

    def dismiss_ringing(self) -> None:
        self._timers = [t for t in self._timers if not t.fired]

    def pop_finished(self) -> list[Timer]:
        now = self._clock()
        done = [t for t in self._timers if not t.fired and t.ends_at <= now]
        for t in done:
            t.fired = True
        # Keep fired timers briefly so the UI can flash them, then drop.
        self._timers = [t for t in self._timers if not t.fired or now - t.ends_at < 8]
        return done

    def snapshot(self) -> list[dict]:
        return [
            {"id": t.id, "label": t.label, "total": t.total, "kind": t.kind,
             "ends_at_ms": int(t.ends_at * 1000), "remaining": self.remaining(t), "fired": t.fired}
            for t in sorted(self._timers, key=lambda t: t.ends_at)
        ]

    def status_text(self) -> str:
        active = self.active()
        if not active:
            return "You have no timers running."
        parts = [f"{t.label}: {format_duration(self.remaining(t))} left" for t in active[:3]]
        return "; ".join(parts) + "."
