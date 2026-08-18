from __future__ import annotations

from typing import Callable
import logging
import threading

try:
    import speech_recognition as sr

    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    sr = None
    SPEECH_RECOGNITION_AVAILABLE = False


class SpeechService:
    def __init__(self, on_command: Callable[[str], None]) -> None:
        self.on_command = on_command
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        if self._thread:
            return True
        if not SPEECH_RECOGNITION_AVAILABLE:
            logging.warning(
                "SpeechRecognition is not installed. Run: pip install SpeechRecognition pyaudio"
            )
            return False
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        return True

    def _listen_loop(self) -> None:
        recognizer = sr.Recognizer()
        recognizer.dynamic_energy_threshold = True

        try:
            mic = sr.Microphone()
        except Exception:
            logging.exception("No microphone found.")
            return

        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.8)
            logging.info("Microphone ready. Say 'Hey mirror'.")

        while True:
            try:
                with mic as source:
                    audio = recognizer.listen(source, timeout=1, phrase_time_limit=6)
                text = recognizer.recognize_google(audio).lower().strip()
                if text:
                    logging.info("Voice detected: %s", text)
                    self.on_command(text)
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                continue
            except Exception:
                logging.exception("Speech recognition failed.")
