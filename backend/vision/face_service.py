from __future__ import annotations

from pathlib import Path
import logging
import os

import cv2

try:
    import face_recognition

    FACE_RECOGNITION_AVAILABLE = all(
        hasattr(face_recognition, name)
        for name in (
            "load_image_file",
            "face_encodings",
            "face_locations",
            "face_distance",
        )
    )
except ImportError:
    face_recognition = None
    FACE_RECOGNITION_AVAILABLE = False


class FaceService:
    def __init__(self, known_faces_dir: Path, tolerance: float = 0.48) -> None:
        self.known_faces_dir = known_faces_dir
        self.tolerance = tolerance
        self.known_face_encodings = []
        self.known_face_names: list[str] = []
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.load_known_faces()

    def load_known_faces(self) -> None:
        self.known_faces_dir.mkdir(parents=True, exist_ok=True)
        self.known_face_encodings.clear()
        self.known_face_names.clear()

        if not FACE_RECOGNITION_AVAILABLE:
            logging.warning("face_recognition is not installed; identity is disabled.")
            return

        for filename in os.listdir(self.known_faces_dir):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image_path = self.known_faces_dir / filename
            try:
                image = face_recognition.load_image_file(str(image_path))
                encodings = face_recognition.face_encodings(image)
            except Exception:
                logging.exception("Could not load known face: %s", image_path)
                continue

            if encodings:
                self.known_face_encodings.append(encodings[0])
                self.known_face_names.append(image_path.stem)

        logging.info(
            "Loaded %s known face(s): %s",
            len(self.known_face_names),
            ", ".join(self.known_face_names) or "none",
        )

    def detect_faces(self, frame) -> list[tuple[int, int, int, int]]:
        half = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        gray = cv2.cvtColor(half, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(30, 30),
        )
        return [(x * 2, y * 2, w * 2, h * 2) for (x, y, w, h) in faces]

    def recognize(self, frame) -> str:
        if not FACE_RECOGNITION_AVAILABLE or not self.known_face_encodings:
            return "Guest"

        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        try:
            locations = face_recognition.face_locations(rgb, model="hog")
            encodings = face_recognition.face_encodings(rgb, locations)
        except Exception:
            logging.exception("Face recognition failed.")
            return "Guest"

        for encoding in encodings:
            distances = face_recognition.face_distance(
                self.known_face_encodings, encoding
            )
            if len(distances) == 0:
                continue

            best_index = int(distances.argmin())
            if distances[best_index] <= self.tolerance:
                return self.known_face_names[best_index]

        return "Guest"
