"""Guided routines (brushing, skincare, stretching...) and breathing sessions.

Both are tick-driven state machines: the runtime calls `tick(now)` and speaks
whatever events come back, so they behave identically on the camera loop and in demo mode.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
import time

ROUTINES: dict[str, dict[str, Any]] = {
    "brushing": {
        "title": "Brushing teeth",
        "aliases": ("brush", "brushing", "teeth", "tooth"),
        "steps": [
            ("Upper left. Small circles.", 30), ("Upper right.", 30),
            ("Lower left.", 30), ("Lower right.", 30),
        ],
        "outro": "Two minutes, done. Lovely.",
    },
    "morning": {
        "title": "Morning routine",
        "aliases": ("morning", "wake up", "start my day"),
        "steps": [
            ("Drink a full glass of water.", None),
            ("Stretch up tall, then fold forward and hang.", 45),
            ("Three slow breaths. In through the nose.", 20),
            ("Say your one priority for today out loud.", None),
        ],
        "outro": "Morning routine complete. Go and have a good one.",
    },
    "skincare": {
        "title": "Skincare",
        "aliases": ("skincare", "skin care", "face", "skin"),
        "steps": [
            ("Cleanse. Gentle circles, one minute.", 60), ("Serum or toner. Press it in.", 30),
            ("Moisturizer.", 30), ("Sunscreen if it is daytime.", None),
        ],
        "outro": "Skincare done.",
    },
    "stretch": {
        "title": "Quick stretch",
        "aliases": ("stretch", "stretching", "mobility", "warm up", "warmup"),
        "steps": [
            ("Neck rolls. Slow, both directions.", 30), ("Shoulder rolls backward.", 30),
            ("Reach overhead and lean side to side.", 40), ("Forward fold. Soft knees.", 40),
            ("Wrists and fingers.", 20),
        ],
        "outro": "Nice. Body feels better already.",
    },
    "winddown": {
        "title": "Wind-down",
        "aliases": ("wind down", "winddown", "evening", "night", "bedtime", "sleep", "unwind"),
        "steps": [
            ("Put your phone down and dim the lights.", None),
            ("Think of one thing that went well today.", 20),
            ("Name the first thing you will do tomorrow.", None),
            ("Four slow breaths. Longer out than in.", 30),
        ],
        "outro": "Sleep well.",
    },
}


def find_routine(text: str) -> str | None:
    text = text.lower()
    best, best_len = None, 0
    for key, spec in ROUTINES.items():
        for alias in (key, *spec["aliases"]):
            if alias in text and len(alias) > best_len:
                best, best_len = key, len(alias)
    return best


@dataclass
class RoutineEvent:
    kind: str  # step | finished
    text: str


class RoutineRunner:
    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self.key: str | None = None
        self.index = 0
        self.step_started = 0.0

    @property
    def active(self) -> bool:
        return self.key is not None

    def start(self, key: str) -> RoutineEvent:
        self.key, self.index, self.step_started = key, 0, self._clock()
        return RoutineEvent("step", self._step_text())

    def stop(self) -> None:
        self.key = None

    def _steps(self) -> list[tuple[str, int | None]]:
        return ROUTINES[self.key]["steps"] if self.key else []

    def _step_text(self) -> str:
        return self._steps()[self.index][0]

    def advance(self) -> RoutineEvent:
        if not self.key:
            return RoutineEvent("finished", "")
        self.index += 1
        self.step_started = self._clock()
        if self.index >= len(self._steps()):
            outro = ROUTINES[self.key]["outro"]
            self.stop()
            return RoutineEvent("finished", outro)
        return RoutineEvent("step", self._step_text())

    def tick(self) -> RoutineEvent | None:
        if not self.key:
            return None
        duration = self._steps()[self.index][1]
        if duration and self._clock() - self.step_started >= duration:
            return self.advance()
        return None

    def snapshot(self) -> dict[str, Any]:
        if not self.key:
            return {"active": False}
        steps = self._steps()
        text, duration = steps[self.index]
        return {
            "active": True, "key": self.key, "title": ROUTINES[self.key]["title"], "step": self.index + 1,
            "steps": len(steps), "text": text, "duration": duration,
            "ends_at_ms": int((self.step_started + duration) * 1000) if duration else None,
        }


BREATHING: dict[str, dict[str, Any]] = {
    "box": {"title": "Box breathing", "phases": [("Inhale", 4), ("Hold", 4), ("Exhale", 4), ("Hold", 4)]},
    "478": {"title": "4-7-8 breathing", "phases": [("Inhale", 4), ("Hold", 7), ("Exhale", 8)]},
    "calm": {"title": "Calming breath", "phases": [("Inhale", 4), ("Exhale", 6)]},
}


def find_breathing(text: str) -> str:
    text = text.lower().replace("-", " ").replace(" ", "")
    if "478" in text or "4,7,8" in text:
        return "478"
    if "box" in text or "square" in text:
        return "box"
    return "calm"


class BreathingSession:
    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self.pattern: str | None = None
        self.started = 0.0
        self.ends = 0.0

    @property
    def active(self) -> bool:
        return self.pattern is not None

    def start(self, pattern: str = "calm", minutes: float = 1.5) -> None:
        self.pattern = pattern if pattern in BREATHING else "calm"
        self.started = self._clock()
        self.ends = self.started + minutes * 60

    def stop(self) -> None:
        self.pattern = None

    def tick(self) -> bool:
        """True exactly once, when the session has run its course."""
        if self.pattern and self._clock() >= self.ends:
            self.stop()
            return True
        return False

    def snapshot(self) -> dict[str, Any]:
        if not self.pattern:
            return {"active": False}
        spec = BREATHING[self.pattern]
        return {
            "active": True, "pattern": self.pattern, "title": spec["title"],
            "phases": [{"label": l, "seconds": s} for l, s in spec["phases"]],
            "started_at_ms": int(self.started * 1000), "ends_at_ms": int(self.ends * 1000),
        }
