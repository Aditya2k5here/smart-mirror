from __future__ import annotations

import logging

import cv2

try:
    from deepface import DeepFace

    DEEPFACE_AVAILABLE = True
except ImportError:
    DeepFace = None
    DEEPFACE_AVAILABLE = False


class EmotionService:
    def __init__(self) -> None:
        if not DEEPFACE_AVAILABLE:
            logging.warning("DeepFace is not installed; emotion defaults to neutral.")

    def analyze(self, frame) -> str:
        if not DEEPFACE_AVAILABLE:
            return "neutral"

        try:
            small = cv2.resize(frame, (480, 270))
            result = DeepFace.analyze(
                small,
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
            )
            first = result[0] if isinstance(result, list) else result
            return first.get("dominant_emotion", "neutral")
        except Exception:
            logging.exception("Emotion analysis failed.")
            return "neutral"

