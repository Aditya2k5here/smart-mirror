from __future__ import annotations

from queue import Empty, Queue
from typing import Callable
import asyncio
import logging
import os
import sys
import tempfile
import threading
import time

from core.config import CONFIG

try:
    import pyttsx3

    PYTTSX3_AVAILABLE = True
except ImportError:
    pyttsx3 = None
    PYTTSX3_AVAILABLE = False

try:
    import edge_tts

    EDGE_TTS_AVAILABLE = sys.platform == "win32"  # playback below uses the Windows MCI player
except ImportError:
    edge_tts = None
    EDGE_TTS_AVAILABLE = False

try:
    import winsound
except ImportError:  # non-Windows
    winsound = None


def _mci(command: str) -> None:
    import ctypes

    ctypes.windll.winmm.mciSendStringW(command, None, 0, 0)


class TTSService:
    """Queued speech with start/end callbacks, interruption and graceful degradation.

    Engines, in order: edge-tts (neural, needs internet), pyttsx3 (offline SAPI), or silent
    (the UI still shows captions and the speaking orb for a realistic duration).
    """

    def __init__(self, min_gap_seconds: float = CONFIG.tts_min_gap_seconds) -> None:
        self.min_gap_seconds = min_gap_seconds
        self._queue: Queue[str] = Queue()
        self._thread: threading.Thread | None = None
        self._started = False
        self._last_started_at = 0.0
        self._last_text = ""
        self.speaking = threading.Event()
        self.on_start: Callable[[str], None] | None = None
        self.on_end: Callable[[], None] | None = None
        self._stop_flag = threading.Event()
        self._engine = None
        self._edge_failed = False
        self.engine_name = "none"

    # ---------------------------------------------------------------- public
    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="tts")
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

    def stop(self) -> None:
        """Interrupt the current sentence and clear the queue."""
        self._drain()
        self._stop_flag.set()
        if self.engine_name == "edge" and sys.platform == "win32":
            try:
                _mci("stop mirrorvoice")
            except Exception:
                pass
        elif self._engine is not None:
            try:
                self._engine.stop()
            except Exception:
                pass

    def alarm(self, beeps: int = 4) -> None:
        """Short non-blocking alarm for timers."""
        def run() -> None:
            for _ in range(beeps):
                if winsound:
                    try:
                        winsound.Beep(1175, 180)
                        winsound.Beep(1568, 260)
                    except RuntimeError:
                        break
                else:
                    print("\a", end="", flush=True)
                time.sleep(0.25)

        threading.Thread(target=run, daemon=True, name="alarm").start()

    @property
    def is_speaking(self) -> bool:
        return self.speaking.is_set()

    # --------------------------------------------------------------- internals
    def _drain(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                return

    def _init_engines(self) -> None:
        provider = CONFIG.tts_provider
        if provider in ("auto", "edge") and EDGE_TTS_AVAILABLE:
            self.engine_name = "edge"
        if PYTTSX3_AVAILABLE and (provider in ("auto", "pyttsx3") or self.engine_name == "none"):
            try:
                engine = pyttsx3.init()
                voices = engine.getProperty("voices")
                if voices:
                    # Prefer a female/natural-sounding SAPI voice when present.
                    preferred = next((v for v in voices if any(k in (v.name or "").lower() for k in ("zira", "aria", "hazel", "heera"))), voices[0])
                    engine.setProperty("voice", preferred.id)
                engine.setProperty("rate", CONFIG.tts_rate)
                engine.setProperty("volume", 0.95)
                self._engine = engine
                if self.engine_name == "none":
                    self.engine_name = "pyttsx3"
            except Exception:
                logging.exception("Could not initialize pyttsx3.")
        if self.engine_name == "none":
            logging.warning("No speech engine available (pip install pyttsx3 edge-tts). Running with captions only.")

    def _speak_edge(self, text: str) -> bool:
        path = os.path.join(tempfile.gettempdir(), f"mirror_tts_{os.getpid()}.mp3")
        try:
            asyncio.run(edge_tts.Communicate(text, CONFIG.edge_voice).save(path))
            if self._stop_flag.is_set():
                return True
            _mci("close mirrorvoice")
            _mci(f'open "{path}" type mpegvideo alias mirrorvoice')
            _mci("play mirrorvoice wait")
            _mci("close mirrorvoice")
            return True
        except Exception as exc:
            logging.warning("edge-tts failed (%s); falling back to the offline voice.", exc)
            self._edge_failed = True
            return False

    def _run(self) -> None:
        self._init_engines()
        while True:
            text = self._queue.get()
            wait_for = self.min_gap_seconds - (time.monotonic() - self._last_started_at)
            if wait_for > 0:
                time.sleep(wait_for)

            self._stop_flag.clear()
            self._last_text = text
            self._last_started_at = time.monotonic()
            self.speaking.set()
            logging.info("Mirror says: %s", text)
            if self.on_start:
                try:
                    self.on_start(text)
                except Exception:
                    logging.exception("TTS on_start failed")
            try:
                done = False
                if self.engine_name == "edge" and not self._edge_failed:
                    done = self._speak_edge(text)
                if not done and self._engine is not None:
                    self._engine.say(text)
                    self._engine.runAndWait()
                elif not done:
                    # captions-only: hold the "speaking" state for a natural duration
                    end = time.monotonic() + max(1.0, len(text) / 14.0)
                    while time.monotonic() < end and not self._stop_flag.is_set():
                        time.sleep(0.05)
            except Exception:
                logging.exception("Text-to-speech failed.")
            finally:
                self.speaking.clear()
                if self.on_end:
                    try:
                        self.on_end()
                    except Exception:
                        logging.exception("TTS on_end failed")


tts_service = TTSService()
