from __future__ import annotations

import logging
import time

import cv2

from workout.exercises import (
    RepCounter, angle as calculate_angle, exercise_title, find_exercise, normalize_exercise,
)

try:
    import mediapipe as mp

    MEDIAPIPE_AVAILABLE = hasattr(mp, "solutions") and hasattr(mp.solutions, "pose")
except ImportError:
    mp = None
    MEDIAPIPE_AVAILABLE = False

__all__ = ["WorkoutService", "workout_service", "normalize_exercise", "calculate_angle"]


class WorkoutService:
    def __init__(self) -> None:
        self.active = False
        self.exercise = "curls"
        self.counter = RepCounter("curls")
        self.started_at: float | None = None
        self.pose = None
        self.mp_pose = None

        if MEDIAPIPE_AVAILABLE:
            try:
                self.mp_pose = mp.solutions.pose
                self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            except Exception:
                logging.warning("MediaPipe pose init failed; workout tracking is disabled.")
        else:
            logging.warning("MediaPipe is not installed; workout tracking is disabled.")

    @property
    def reps(self) -> int:
        return self.counter.reps

    @property
    def stage(self) -> str:
        return self.counter.stage

    @property
    def tracking_available(self) -> bool:
        return self.pose is not None

    def start(self, exercise: str = "curls") -> None:
        self.active = True
        self.exercise = normalize_exercise(exercise)
        self.counter = RepCounter(self.exercise)
        self.started_at = time.monotonic()

    def stop(self) -> dict[str, object]:
        """End the session and return what happened (for logging)."""
        summary = {
            "exercise": self.exercise,
            "reps": self.counter.reps,
            "duration_seconds": int(time.monotonic() - self.started_at) if self.started_at else 0,
            "hold_seconds": int(self.counter.held),
        }
        self.active = False
        self.started_at = None
        return summary

    def process_frame(self, frame) -> int:
        """Returns the number of reps newly counted in this frame (0 or 1)."""
        if not self.active or not self.pose:
            return 0
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb)
        except Exception:
            logging.exception("Workout pose processing failed.")
            return 0
        if not results.pose_landmarks:
            return 0
        try:
            return int(self.counter.update(self._landmarks(results.pose_landmarks.landmark)))
        except Exception:
            logging.exception("Workout rep calculation failed.")
            return 0

    def _landmarks(self, landmarks) -> dict[str, tuple[float, float, float]]:
        out = {}
        for member in self.mp_pose.PoseLandmark:
            lm = landmarks[member.value]
            out[member.name.lower()] = (lm.x, lm.y, lm.visibility)
        return out

    def snapshot(self, timer: str) -> dict[str, object]:
        return {
            "active": self.active,
            "exercise": self.exercise,
            "title": exercise_title(self.exercise),
            "reps": self.counter.reps,
            "timer": timer,
            "stage": self.counter.stage,
            "hold": int(self.counter.held) if self.exercise == "plank" else None,
            "feedback": self.counter.feedback,
            "tracking": self.tracking_available,
        }


workout_service = WorkoutService()
