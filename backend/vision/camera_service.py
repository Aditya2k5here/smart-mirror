from __future__ import annotations

import cv2


class CameraService:
    def __init__(self, camera_index: int, width: int, height: int) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        self.capture = cv2.VideoCapture(self.camera_index)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self.capture.isOpened():
            raise RuntimeError(
                f"Could not open webcam index {self.camera_index}. "
                "Check camera permissions or set MIRROR_CAMERA_INDEX."
            )

    def read(self):
        if not self.capture:
            raise RuntimeError("Camera is not open.")
        return self.capture.read()

    def release(self) -> None:
        if self.capture:
            self.capture.release()
            self.capture = None

