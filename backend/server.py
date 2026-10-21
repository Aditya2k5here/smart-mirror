"""Smart Mirror entrypoint.

    python -m backend.server            # real mode: webcam + microphone
    python -m backend.server --demo     # no hardware or keys needed; seeded persona, typed commands

Then open http://localhost:8000
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Keep TensorFlow's C++ start-up chatter (oneDNN / CPU-feature notices) out of the console.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# The demo flag must be applied before core.config is imported (it chooses the database).
if "--demo" in sys.argv:
    os.environ["MIRROR_DEMO"] = "1"
    os.environ.setdefault("MIRROR_DB_PATH", str(Path(__file__).resolve().parent / "data" / "demo.db"))

import asyncio
import logging

# Configure logging before the backend modules are imported: some of them log warnings at import
# time, and the first warning would otherwise install Python's default WARNING-only handler,
# hiding every INFO line ("Mirror UI", "Voice detected", "Mirror says"). Logs go to stdout so
# PowerShell (run-mirror.bat) does not paint each line as an error.
try:
    sys.stdout.reconfigure(errors="backslashreplace")  # an odd character must never break a log line
except (AttributeError, ValueError):
    pass
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout, force=True)
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import websockets

from core.config import CONFIG, PROJECT_ROOT
from core.health_service import health_service
from demo import DEMO_USER, demo_environment, seed_demo
from http_server import start_http_server
from runtime import MirrorRuntime
from voice.speech_service import SpeechService
from voice.tts_service import tts_service


def allowed_origins() -> list[str | None]:
    origins: list[str | None] = [None]  # non-browser clients
    for host in {"localhost", "127.0.0.1", CONFIG.http_host}:
        origins.append(f"http://{host}:{CONFIG.http_port}")
    origins.extend(CONFIG.allowed_origins)
    return origins


async def main() -> None:
    demo = CONFIG.demo
    runtime = MirrorRuntime(demo=demo)
    await runtime.start()
    loop = asyncio.get_running_loop()

    if demo:
        seeded = await asyncio.to_thread(seed_demo, runtime.services)
        runtime.state.update(**demo_environment())
        logging.info("Demo mode: persona %s%s. Hardware is not used.", DEMO_USER, " (seeded)" if seeded else "")
    else:
        speech = SpeechService(
            lambda text: asyncio.run_coroutine_threadsafe(runtime.handle_text(text, "voice"), loop),
            should_pause=lambda: tts_service.is_speaking,
        )
        if speech.start():
            health_service.set_runtime("microphone", "Microphone", "ready", "listening for 'Hey mirror'")
        else:
            health_service.set_runtime("microphone", "Microphone", "offline", "speech package missing; use typed commands")
        runtime.speech = speech

    api = {
        "/api/state": runtime.state.snapshot,
        "/api/health": health_service.snapshot,
        "/api/export": lambda: runtime.services.profile.export_user(runtime.state.state.name),
    }
    http = start_http_server(PROJECT_ROOT / "frontend", CONFIG.http_host, CONFIG.http_port, api)
    if http:
        logging.info("Mirror UI: http://%s:%s", CONFIG.http_host, CONFIG.http_port)

    logging.info("Websocket: ws://%s:%s", CONFIG.websocket_host, CONFIG.websocket_port)
    logging.info("Memory: %s", CONFIG.db_path)

    perception = runtime.static_presence_loop(DEMO_USER) if demo else runtime.camera_loop()
    async with websockets.serve(
        runtime.ws.handler, CONFIG.websocket_host, CONFIG.websocket_port,
        ping_interval=20, ping_timeout=20, origins=allowed_origins(),
    ):
        await asyncio.gather(runtime.assistant_loop(), runtime.integrations_loop(), perception)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Smart Mirror stopped.")
