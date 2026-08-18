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


class PostureService:
    def __init__(self) -> None:
        self.pose = None
        self.mp_pose = None
        if MEDIAPIPE_AVAILABLE:
            try:
                self.mp_pose = mp.solutions.pose
                self.pose = self.mp_pose.Pose(
                    min_detection_confidence=0.55,
                    min_tracking_confidence=0.55,
                )
            except Exception:
                logging.warning("MediaPipe pose init failed; using face-height posture.")
        else:
            logging.warning("MediaPipe is not installed; using face-height posture.")

    def analyze(self, frame, faces: list[tuple[int, int, int, int]]) -> str:
        if self.pose:
            pose_result = self._analyze_pose(frame)
            if pose_result:
                return pose_result
        return self._analyze_face_height(frame, faces)

    def _analyze_pose(self, frame) -> str | None:
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb)
        except Exception:
            logging.exception("Posture pose processing failed.")
            return None

        if not results.pose_landmarks:
            return None

        landmarks = results.pose_landmarks.landmark
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

        visible = min(nose.visibility, left_shoulder.visibility, right_shoulder.visibility)
        if visible < 0.45:
            return None

        shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2
        shoulder_slope = abs(left_shoulder.y - right_shoulder.y)
        head_above_shoulders = shoulder_mid_y - nose.y

        if shoulder_slope > 0.09 or head_above_shoulders < 0.09:
            return "slouching"
        return "good"

    def _analyze_face_height(self, frame, faces: list[tuple[int, int, int, int]]) -> str:
        if not faces:
            return "good"

        frame_height = frame.shape[0]
        _, y, _, h = faces[0]
        face_center_y = y + h / 2
        if face_center_y > frame_height * 0.62:
            return "slouching"
        return "good"


def calculate_angle(a, b, c) -> float:
    radians = math.atan2(c.y - b.y, c.x - b.x) - math.atan2(
        a.y - b.y, a.x - b.x
    )
    angle = abs(radians * 180.0 / math.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle
