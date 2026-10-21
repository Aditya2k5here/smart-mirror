from __future__ import annotations

from typing import Callable
import logging
import os
import threading
import time

try:
    import speech_recognition as sr

    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    sr = None
    SPEECH_RECOGNITION_AVAILABLE = False

LANGUAGE = os.environ.get("MIRROR_SPEECH_LANG", "en-US")


class SpeechService:
    """Always-on microphone listener.

    `should_pause` lets the runtime mute listening while the mirror is talking, so it never
    transcribes its own voice. Final transcripts are passed to `on_command`.
    """

    def __init__(self, on_command: Callable[[str], None], should_pause: Callable[[], bool] | None = None) -> None:
        self.on_command = on_command
        self.should_pause = should_pause or (lambda: False)
        self._thread: threading.Thread | None = None
        self.muted = False  # privacy mode

    def start(self) -> bool:
        if self._thread:
            return True
        if not SPEECH_RECOGNITION_AVAILABLE:
            logging.warning("SpeechRecognition is not installed. Run: pip install SpeechRecognition pyaudio")
            return False
        self._thread = threading.Thread(target=self._listen_loop, daemon=True, name="speech")
        self._thread.start()
        return True

    def _listen_loop(self) -> None:
        recognizer = sr.Recognizer()
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.9
        recognizer.non_speaking_duration = 0.5

        try:
            mic = sr.Microphone()
        except Exception:
            logging.exception("No microphone found.")
            return

        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.8)
            logging.info("Microphone ready. Say 'Hey mirror'.")

        while True:
            if self.muted or self.should_pause():
                time.sleep(0.15)
                continue
            try:
                with mic as source:
                    audio = recognizer.listen(source, timeout=1, phrase_time_limit=9)
                if self.muted or self.should_pause():
                    continue  # heard ourselves, or privacy mode switched on mid-phrase
                text = recognizer.recognize_google(audio, language=LANGUAGE).strip()
                if text:
                    logging.info("Voice detected: %s", text)
                    self.on_command(text)
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                continue
            except Exception:
                logging.exception("Speech recognition failed.")
                time.sleep(1.0)
