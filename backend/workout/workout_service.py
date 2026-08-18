from __future__ import annotations

import logging
import math

import cv2

try:
    import mediapipe as mp

    MEDIAPIPE_AVAILABLE = hasattr(mp, "solutions") and hasattr(mp.solutions, "pose")
except ImportError:
    mp = None
    MEDIAPIPE_AVAILABLE = False


class WorkoutService:
    def __init__(self) -> None:
        self.active = False
        self.exercise = "curls"
        self.reps = 0
        self.stage = "down"
        self.pose = None
        self.mp_pose = None

        if MEDIAPIPE_AVAILABLE:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            logging.warning("MediaPipe is not installed; workout tracking is disabled.")

    def start(self, exercise: str = "curls") -> None:
        self.active = True
        self.exercise = normalize_exercise(exercise)
        self.reps = 0
        self.stage = "down"

    def stop(self) -> None:
        self.active = False

    def process_frame(self, frame) -> int:
        if not self.active or not self.pose:
            return self.reps

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb)
        except Exception:
            logging.exception("Workout pose processing failed.")
            return self.reps

        if not results.pose_landmarks:
            return self.reps

        landmarks = results.pose_landmarks.landmark
        try:
            if self.exercise == "squats":
                self._process_squat(landmarks)
            elif self.exercise == "pushups":
                self._process_pushup(landmarks)
            elif self.exercise == "posture":
                self.stage = "monitoring"
            else:
                self._process_curl(landmarks)
        except Exception:
            logging.exception("Workout rep calculation failed.")

        return self.reps

    def snapshot(self, timer: str) -> dict[str, object]:
        return {
            "active": self.active,
            "exercise": self.exercise,
            "reps": self.reps,
            "timer": timer,
            "stage": self.stage,
        }

    def _process_curl(self, landmarks) -> None:
        shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        angle = calculate_angle(shoulder, elbow, wrist)

        if angle > 155:
            self.stage = "down"
        elif angle < 35 and self.stage == "down":
            self.stage = "up"
            self.reps += 1

    def _process_squat(self, landmarks) -> None:
        hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        angle = calculate_angle(hip, knee, ankle)

        if angle > 155:
            self.stage = "up"
        elif angle < 95 and self.stage == "up":
            self.stage = "down"
            self.reps += 1

    def _process_pushup(self, landmarks) -> None:
        shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        angle = calculate_angle(shoulder, elbow, wrist)

        if angle > 155:
            self.stage = "up"
        elif angle < 85 and self.stage == "up":
            self.stage = "down"
            self.reps += 1


def normalize_exercise(raw: str) -> str:
    text = (raw or "").lower()
    if "squat" in text:
        return "squats"
    if "push" in text:
        return "pushups"
    if "posture" in text or "mobility" in text:
        return "posture"
    return "curls"


def calculate_angle(a, b, c) -> float:
    radians = math.atan2(c.y - b.y, c.x - b.x) - math.atan2(
        a.y - b.y, a.x - b.x
    )
    angle = abs(radians * 180.0 / math.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


workout_service = WorkoutService()
