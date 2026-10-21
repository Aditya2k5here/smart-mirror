"""Face detection, recognition and enrollment.

Backends (chosen automatically, best first):
  * opencv  - YuNet detector + SFace recogniser (ONNX, auto-downloaded once, no dlib needed)
  * dlib    - the `face_recognition` package, if installed
  * haar    - detection only (no identity)

People are stored as `known_faces/<Name>/<n>.jpg` (several samples each) or legacy flat `Name.jpg`.
"""
from __future__ import annotations

from pathlib import Path
import logging
import re
import shutil
import threading
import time
import urllib.request

import cv2
import numpy as np

from core.config import CONFIG

try:
    import face_recognition

    FACE_RECOGNITION_AVAILABLE = all(
        hasattr(face_recognition, n) for n in ("load_image_file", "face_encodings", "face_locations", "face_distance")
    )
except ImportError:
    face_recognition = None
    FACE_RECOGNITION_AVAILABLE = False

MODEL_URLS = {
    "yunet.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "sface.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}
IMAGE_EXT = (".jpg", ".jpeg", ".png")
MAX_SAMPLES_PER_PERSON = 8
COSINE_FLAG = getattr(cv2, "FaceRecognizerSF_FR_COSINE", 0)
COSINE_MATCH = 0.40  # SFace's documented operating point is 0.363; slightly stricter avoids look-alikes
COSINE_CONFIDENT = 0.58  # only this sure may an automatic extra sample be saved
AUTO_LEARN_EVERY_SECONDS = 3 * 24 * 3600


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9 _-]+", "", name.strip())
    return re.sub(r"\s+", "_", cleaned).strip("_") or "User"


def display_name(stem: str) -> str:
    return stem.replace("_", " ")


class FaceService:
    def __init__(self, known_faces_dir: Path, tolerance: float = 0.48, models_dir: Path | None = None,
                 backend: str | None = None) -> None:
        self.known_faces_dir = Path(known_faces_dir)
        self.tolerance = tolerance
        self.models_dir = Path(models_dir or CONFIG.models_dir)
        self.requested_backend = (backend or CONFIG.face_backend or "auto").lower()
        self.backend = "haar"
        self.names: list[str] = []
        self.encodings: list = []
        self._lock = threading.RLock()
        self._detector = None
        self._recognizer = None
        self._detector_size = (0, 0)
        self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self._choose_backend()
        self.load_known_faces()

    # ---------------------------------------------------------------- setup
    def _download_models(self) -> bool:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        for filename, url in MODEL_URLS.items():
            target = self.models_dir / filename
            if target.exists() and target.stat().st_size > 100_000:
                continue
            logging.info("Downloading face model %s (one time)...", filename)
            try:
                tmp = target.with_suffix(".part")
                with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as fh:
                    shutil.copyfileobj(resp, fh)
                tmp.replace(target)
            except Exception as exc:
                logging.warning("Could not download %s: %s", filename, exc)
                return False
        return True

    def _choose_backend(self) -> None:
        want = self.requested_backend
        if want in ("auto", "opencv") and hasattr(cv2, "FaceDetectorYN") and hasattr(cv2, "FaceRecognizerSF"):
            if self._download_models():
                try:
                    self._detector = cv2.FaceDetectorYN.create(
                        str(self.models_dir / "yunet.onnx"), "", (320, 320), 0.85, 0.3, 500)
                    self._recognizer = cv2.FaceRecognizerSF.create(str(self.models_dir / "sface.onnx"), "")
                    self.backend = "opencv"
                    return
                except Exception:
                    logging.exception("OpenCV face models failed to load.")
        if want in ("auto", "dlib") and FACE_RECOGNITION_AVAILABLE:
            self.backend = "dlib"
            return
        self.backend = "haar"
        logging.warning("No face recognition backend available; identity is disabled (detection only).")

    # ------------------------------------------------------------- known faces
    def _iter_samples(self):
        """Yield (name, image_path) for every stored sample."""
        self.known_faces_dir.mkdir(parents=True, exist_ok=True)
        for entry in sorted(self.known_faces_dir.iterdir()):
            if entry.is_file() and entry.suffix.lower() in IMAGE_EXT:
                yield display_name(entry.stem), entry
            elif entry.is_dir():
                for img in sorted(entry.iterdir()):
                    if img.suffix.lower() in IMAGE_EXT:
                        yield display_name(entry.name), img

    def load_known_faces(self) -> None:
        with self._lock:
            self.names, self.encodings = [], []
            if self.backend == "haar":
                return
            for name, path in self._iter_samples():
                image = cv2.imread(str(path))
                if image is None:
                    continue
                enc = self._encode_largest(image)
                if enc is not None:
                    self.names.append(name)
                    self.encodings.append(enc)
            people = sorted(set(self.names))
            logging.info("Loaded %s face sample(s) for %s person(s): %s [%s]", len(self.names), len(people),
                         ", ".join(people) or "none", self.backend)

    @property
    def known_people(self) -> list[str]:
        return sorted(set(self.names))

    def sample_count(self, name: str) -> int:
        return sum(1 for n in self.names if n == name)

    # -------------------------------------------------------------- detection
    def _yunet(self, frame) -> np.ndarray:
        h, w = frame.shape[:2]
        if (w, h) != self._detector_size:
            self._detector.setInputSize((w, h))
            self._detector_size = (w, h)
        _, faces = self._detector.detect(frame)
        return faces if faces is not None else np.empty((0, 15), dtype=np.float32)

    def detect_faces(self, frame) -> list[tuple[int, int, int, int]]:
        if self.backend == "opencv":
            with self._lock:
                rows = self._yunet(frame)
            return [(max(0, int(r[0])), max(0, int(r[1])), int(r[2]), int(r[3])) for r in rows]
        half = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        gray = cv2.cvtColor(half, cv2.COLOR_BGR2GRAY)
        faces = self.haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        return [(x * 2, y * 2, w * 2, h * 2) for (x, y, w, h) in faces]

    # ------------------------------------------------------------ recognition
    def _encode_largest(self, frame):
        """Embedding of the largest face in the frame, or None."""
        if self.backend == "opencv":
            with self._lock:
                rows = self._yunet(frame)
                if len(rows) == 0:
                    return None
                best = max(rows, key=lambda r: r[2] * r[3])
                aligned = self._recognizer.alignCrop(frame, best)
                return self._recognizer.feature(aligned).copy()
        if self.backend == "dlib":
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            encs = face_recognition.face_encodings(rgb)
            return encs[0] if encs else None
        return None

    def recognize_detailed(self, frame) -> tuple[str, float]:
        """(name or 'Guest', confidence 0-1). Confidence is similarity for opencv, 1-distance for dlib."""
        if self.backend == "haar" or not self.encodings:
            return "Guest", 0.0
        try:
            if self.backend == "opencv":
                with self._lock:
                    rows = self._yunet(frame)
                    if len(rows) == 0:
                        return "Guest", 0.0
                    best_row = max(rows, key=lambda r: r[2] * r[3])
                    feat = self._recognizer.feature(self._recognizer.alignCrop(frame, best_row))
                    scores: dict[str, float] = {}
                    for name, known in zip(self.names, self.encodings):
                        sim = float(self._recognizer.match(feat, known, COSINE_FLAG))
                        scores[name] = max(scores.get(name, -1.0), sim)
                name, sim = max(scores.items(), key=lambda kv: kv[1])
                return (name, sim) if sim >= COSINE_MATCH else ("Guest", sim)
            small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb, model="hog")
            for enc in face_recognition.face_encodings(rgb, locations):
                dist = face_recognition.face_distance(self.encodings, enc)
                idx = int(dist.argmin())
                if dist[idx] <= self.tolerance:
                    return self.names[idx], float(1 - dist[idx])
        except Exception:
            logging.exception("Face recognition failed.")
        return "Guest", 0.0

    def recognize(self, frame) -> str:
        name, confidence = self.recognize_detailed(frame)
        if name != "Guest" and CONFIG.auto_learn_faces and confidence >= COSINE_CONFIDENT:
            self._maybe_learn(name, frame)
        return name

    # ----------------------------------------------------------- enrollment
    def _person_dir(self, name: str) -> Path:
        return self.known_faces_dir / safe_name(name)

    def enroll(self, name: str, frame) -> tuple[bool, str]:
        """Save a new sample of the largest face. Returns (ok, reason)."""
        if self.backend == "haar":
            return False, "face recognition isn't available on this machine"
        faces = self.detect_faces(frame)
        if not faces:
            return False, "I couldn't see your face clearly"
        if len(faces) > 1:
            return False, "I can see more than one face"
        _, _, w, h = faces[0]
        if w < frame.shape[1] * 0.08:
            return False, "you're a little far away"
        return self._save_sample(name, frame)

    def _save_sample(self, name: str, frame) -> tuple[bool, str]:
        enc = self._encode_largest(frame)
        if enc is None:
            return False, "I couldn't read your face clearly"
        folder = self._person_dir(name)
        folder.mkdir(parents=True, exist_ok=True)
        # Cap samples: drop the oldest once full so the model tracks how you look now.
        samples = sorted(folder.glob("*"), key=lambda p: p.stat().st_mtime)
        while len(samples) >= MAX_SAMPLES_PER_PERSON:
            samples.pop(0).unlink(missing_ok=True)
        height, width = frame.shape[:2]
        scale = min(1.0, 640 / max(width, 1))
        small = cv2.resize(frame, (0, 0), fx=scale, fy=scale) if scale < 1 else frame
        cv2.imwrite(str(folder / f"{int(time.time())}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 90])
        with self._lock:
            self.load_known_faces()
        return True, "ok"

    def _maybe_learn(self, name: str, frame) -> None:
        folder = self._person_dir(name)
        if folder.exists():
            samples = list(folder.glob("*.jpg"))
            if samples and time.time() - max(p.stat().st_mtime for p in samples) < AUTO_LEARN_EVERY_SECONDS:
                return
        elif self.sample_count(name) == 0:
            return
        ok, _ = self._save_sample(name, frame)
        if ok:
            logging.info("Learned a fresh face sample for %s", name)

    def delete_person(self, name: str) -> None:
        shutil.rmtree(self._person_dir(name), ignore_errors=True)
        for ext in IMAGE_EXT:
            (self.known_faces_dir / f"{safe_name(name)}{ext}").unlink(missing_ok=True)
        self.load_known_faces()
