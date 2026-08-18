from __future__ import annotations

from queue import Empty, Queue
import logging
import threading
import time

from core.config import CONFIG

try:
    import pyttsx3

    PYTTSX3_AVAILABLE = True
except ImportError:
    pyttsx3 = None
    PYTTSX3_AVAILABLE = False


class TTSService:
    def __init__(self, min_gap_seconds: float = CONFIG.tts_min_gap_seconds) -> None:
        self.min_gap_seconds = min_gap_seconds
        self._queue: Queue[str] = Queue()
        self._thread: threading.Thread | None = None
        self._started = False
        self._last_started_at = 0.0
        self._last_text = ""

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def speak(self, text: str, priority: bool = False) -> None:
        text = " ".join((text or "").split())
        if not text:
            return
        if text == self._last_text and time.monotonic() - self._last_started_at < 30:
            return
        if priority:
            self._drain()
        self._queue.put(text)

    def _drain(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                return

    def _run(self) -> None:
        if not PYTTSX3_AVAILABLE:
            logging.warning("pyttsx3 is not installed; speech output is disabled.")
            return

        try:
            engine = pyttsx3.init()
            voices = engine.getProperty("voices")
            if voices:
                preferred = voices[1].id if len(voices) > 1 else voices[0].id
                engine.setProperty("voice", preferred)
            engine.setProperty("rate", 132)
            engine.setProperty("volume", 0.92)
        except Exception:
            logging.exception("Could not initialize text-to-speech.")
            return

        while True:
            text = self._queue.get()
            wait_for = self.min_gap_seconds - (time.monotonic() - self._last_started_at)
            if wait_for > 0:
                time.sleep(wait_for)

            self._last_text = text
            self._last_started_at = time.monotonic()
            logging.info("Mirror says: %s", text)
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception:
                logging.exception("Text-to-speech failed.")


tts_service = TTSService()

