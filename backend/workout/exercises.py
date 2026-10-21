"""Exercise catalogue and a pure-math rep counter.

The counter only needs a dict of landmark name -> (x, y, visibility) in normalised image
coordinates, so it is unit-testable without MediaPipe or a camera.
"""
from __future__ import annotations

import math
import re
import time
from typing import Callable

Landmarks = dict[str, tuple[float, float, float]]

EXERCISES: dict[str, dict[str, object]] = {
    "curls": {"title": "Bicep curls", "aliases": ("curl", "curls", "bicep", "biceps")},
    "squats": {"title": "Squats", "aliases": ("squat", "squats")},
    "pushups": {"title": "Push-ups", "aliases": ("push up", "push ups", "pushup", "pushups", "press up")},
    "lunges": {"title": "Lunges", "aliases": ("lunge", "lunges")},
    "shoulder_press": {"title": "Shoulder press", "aliases": ("shoulder press", "overhead press", "military press")},
    "jumping_jacks": {"title": "Jumping jacks", "aliases": ("jumping jack", "jumping jacks", "star jump")},
    "plank": {"title": "Plank hold", "aliases": ("plank", "planks")},
    "posture": {"title": "Posture check", "aliases": ("posture", "mobility")},
}


def find_exercise(text: str) -> str:
    text = (text or "").lower()
    best, best_len = "curls", 0
    for key, spec in EXERCISES.items():
        for alias in spec["aliases"]:  # type: ignore[union-attr]
            if re.search(rf"\b{re.escape(alias)}\b", text) and len(alias) > best_len:
                best, best_len = key, len(alias)
    return best


def normalize_exercise(raw: str) -> str:
    return find_exercise(raw)


def exercise_title(key: str) -> str:
    return str(EXERCISES.get(key, EXERCISES["curls"])["title"])


def angle(a, b, c) -> float:
    """Angle ABC in degrees (0-180) from three (x, y, ...) points."""
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    deg = abs(radians * 180.0 / math.pi)
    return 360 - deg if deg > 180 else deg


def _vis(lm: Landmarks, *names: str) -> float:
    return min((lm[n][2] for n in names if n in lm), default=0.0)


def _best_side(lm: Landmarks, joints: tuple[str, str, str]) -> tuple[str, float] | None:
    best = None
    for side in ("left", "right"):
        names = tuple(f"{side}_{j}" for j in joints)
        if all(n in lm for n in names):
            vis = _vis(lm, *names)
            if best is None or vis > best[1]:
                best = (side, vis)
    return best if best and best[1] >= 0.4 else None


class RepCounter:
    def __init__(self, exercise: str, clock: Callable[[], float] = time.monotonic) -> None:
        self.exercise = exercise if exercise in EXERCISES else "curls"
        self._clock = clock
        self.reps = 0
        self.stage = "ready"
        self.hold_seconds = 0.0
        self._hold_since: float | None = None
        self.feedback = ""

    def update(self, lm: Landmarks) -> bool:
        """Feed one frame of landmarks. Returns True when a rep was just counted."""
        handler = getattr(self, f"_{self.exercise}", None)
        return bool(handler(lm)) if handler else False

    # -- exercises ----------------------------------------------------------
    def _arm_angle(self, lm: Landmarks) -> float | None:
        side = _best_side(lm, ("shoulder", "elbow", "wrist"))
        if not side:
            return None
        s = side[0]
        return angle(lm[f"{s}_shoulder"], lm[f"{s}_elbow"], lm[f"{s}_wrist"])

    def _leg_angles(self, lm: Landmarks) -> list[float]:
        out = []
        for side in ("left", "right"):
            names = (f"{side}_hip", f"{side}_knee", f"{side}_ankle")
            if all(n in lm for n in names) and _vis(lm, *names) >= 0.4:
                out.append(angle(lm[names[0]], lm[names[1]], lm[names[2]]))
        return out

    def _curls(self, lm: Landmarks) -> bool:
        a = self._arm_angle(lm)
        if a is None:
            return False
        if a > 155:
            self.stage = "down"
        elif a < 35 and self.stage == "down":
            self.stage = "up"
            self.reps += 1
            return True
        return False

    def _pushups(self, lm: Landmarks) -> bool:
        a = self._arm_angle(lm)
        if a is None:
            return False
        if a > 155:
            self.stage = "up"
        elif a < 85 and self.stage == "up":
            self.stage = "down"
            self.reps += 1
            return True
        return False

    def _squats(self, lm: Landmarks) -> bool:
        angles = self._leg_angles(lm)
        if not angles:
            return False
        a = min(angles)
        if a > 155:
            self.stage = "up"
            return False
        if a < 95 and self.stage == "up":
            self.stage = "down"
            self.reps += 1
            return True
        self.feedback = "Go a little lower." if self.stage == "up" and 95 <= a < 125 else ""
        return False

    def _lunges(self, lm: Landmarks) -> bool:
        angles = self._leg_angles(lm)
        if len(angles) < 2:
            return False
        if min(angles) > 150:
            self.stage = "up"
        elif min(angles) < 100 and max(angles) > 100 and self.stage == "up":
            self.stage = "down"
            self.reps += 1
            return True
        return False

    def _shoulder_press(self, lm: Landmarks) -> bool:
        side = _best_side(lm, ("shoulder", "elbow", "wrist"))
        if not side:
            return False
        s = side[0]
        a = angle(lm[f"{s}_shoulder"], lm[f"{s}_elbow"], lm[f"{s}_wrist"])
        wrist_above = lm[f"{s}_wrist"][1] < lm[f"{s}_shoulder"][1] - 0.05
        if a < 95:
            self.stage = "down"
        elif a > 160 and wrist_above and self.stage == "down":
            self.stage = "up"
            self.reps += 1
            return True
        return False

    def _jumping_jacks(self, lm: Landmarks) -> bool:
        needed = ("left_wrist", "right_wrist", "left_ankle", "right_ankle", "left_shoulder", "right_shoulder", "nose")
        if not all(n in lm for n in needed):
            return False
        shoulder_w = abs(lm["left_shoulder"][0] - lm["right_shoulder"][0]) or 0.1
        ankle_w = abs(lm["left_ankle"][0] - lm["right_ankle"][0])
        hands_up = lm["left_wrist"][1] < lm["nose"][1] and lm["right_wrist"][1] < lm["nose"][1]
        hands_down = lm["left_wrist"][1] > lm["left_shoulder"][1] and lm["right_wrist"][1] > lm["right_shoulder"][1]
        if hands_up and ankle_w > shoulder_w * 1.6:
            self.stage = "open"
        elif hands_down and ankle_w < shoulder_w * 1.2 and self.stage == "open":
            self.stage = "closed"
            self.reps += 1
            return True
        return False

    def _plank(self, lm: Landmarks) -> bool:
        side = _best_side(lm, ("shoulder", "hip", "ankle"))
        now = self._clock()
        if side and angle(lm[f"{side[0]}_shoulder"], lm[f"{side[0]}_hip"], lm[f"{side[0]}_ankle"]) > 160:
            if self._hold_since is None:
                self._hold_since = now
            self.hold_seconds += 0.0
            self.stage = "holding"
            self.feedback = ""
        else:
            if self._hold_since is not None:
                self.hold_seconds += now - self._hold_since
                self._hold_since = None
            self.stage = "rest"
            self.feedback = "Hips level with your shoulders."
        return False

    def _posture(self, lm: Landmarks) -> bool:
        self.stage = "monitoring"
        return False

    @property
    def held(self) -> float:
        """Total plank seconds, including the hold in progress."""
        running = (self._clock() - self._hold_since) if self._hold_since is not None else 0.0
        return self.hold_seconds + running
