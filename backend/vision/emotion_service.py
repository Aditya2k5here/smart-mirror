from __future__ import annotations

import logging
import threading

import cv2


class EmotionService:
    """DeepFace emotion estimate, loaded lazily.

    DeepFace pulls in TensorFlow, which is slow to import and can fail in many ways
    (e.g. TF >= 2.16 without `tf-keras`). Loading on first use keeps startup fast, and any
    failure just disables emotion (it reports "neutral") instead of crashing the mirror.
    """

    def __init__(self) -> None:
        self._deepface = None
        self._state = "unloaded"  # unloaded | ready | unavailable
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self._state == "ready"

    def _load(self) -> None:
        with self._lock:
            if self._state != "unloaded":
                return
            try:
                from deepface import DeepFace

                self._deepface = DeepFace
                self._state = "ready"
                logging.info("DeepFace emotion model loaded.")
            except Exception as exc:  # ImportError, ValueError (tf-keras), DLL errors...
                self._state = "unavailable"
                logging.warning("Emotion detection disabled: %s", exc)

    def analyze(self, frame) -> str:
        if self._state == "unloaded":
            self._load()
        if self._state != "ready":
            return "neutral"
        try:
            small = cv2.resize(frame, (480, 270))
            result = self._deepface.analyze(small, actions=["emotion"], enforce_detection=False, silent=True)
            first = result[0] if isinstance(result, list) else result
            return first.get("dominant_emotion", "neutral")
        except Exception:
            logging.exception("Emotion analysis failed.")
            return "neutral"
