from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import importlib.util
import os

from core.config import CONFIG, SETTINGS_PATH


@dataclass
class HealthItem:
    id: str
    label: str
    status: str
    detail: str
    required: bool = False


class HealthService:
    def __init__(self) -> None:
        self._runtime: dict[str, HealthItem] = {}

    def set_runtime(self, item_id: str, label: str, status: str, detail: str, required: bool = False) -> None:
        self._runtime[item_id] = HealthItem(item_id, label, status, detail, required)

    def snapshot(self) -> dict[str, Any]:
        items = self._static_items()
        by_id = {item.id: item for item in items}
        by_id.update(self._runtime)
        final = list(by_id.values())

        critical = any(item.required and item.status != "ready" for item in final)
        warning = any(item.status in {"warning", "offline"} for item in final)
        overall = "blocked" if critical else "warning" if warning else "ready"

        return {
            "overall": overall,
            "summary": self._summary(overall),
            "items": [item.__dict__ for item in final],
            "checked_at": datetime.now().isoformat(timespec="seconds"),
        }

    def settings_snapshot(self) -> dict[str, Any]:
        return {
            "camera": CONFIG.camera_index,
            "city": CONFIG.weather_city,
            "ai": CONFIG.ai_router_mode,
            "fps": CONFIG.target_fps,
            "preview": CONFIG.camera_preview,
            "settings_file": str(SETTINGS_PATH),
        }

    def _static_items(self) -> list[HealthItem]:
        known_faces = list(CONFIG.known_faces_dir.glob("*.jpg")) + list(CONFIG.known_faces_dir.glob("*.png"))
        return [
            HealthItem(
                "settings",
                "Settings",
                "ready" if SETTINGS_PATH.exists() else "warning",
                "settings.json loaded" if SETTINGS_PATH.exists() else "using defaults",
            ),
            HealthItem(
                "faces",
                "Known faces",
                "ready" if known_faces else "warning",
                f"{len(known_faces)} registered",
            ),
            HealthItem(
                "ai",
                "AI router",
                "ready" if CONFIG.groq_api_key or CONFIG.gemini_api_key or os.environ.get("OPENAI_API_KEY") else "offline",
                CONFIG.ai_router_mode,
            ),
            HealthItem(
                "weather",
                "Weather",
                "ready" if CONFIG.weather_api_key else "offline",
                CONFIG.weather_city,
            ),
            HealthItem(
                "calendar",
                "Calendar",
                "ready" if (Path(CONFIG.data_dir.parent) / "token.json").exists() else "warning",
                "token found" if (Path(CONFIG.data_dir.parent) / "token.json").exists() else "not connected",
            ),
            HealthItem(
                "spotify",
                "Spotify",
                "ready" if os.environ.get("SPOTIFY_CLIENT_ID") and os.environ.get("SPOTIFY_CLIENT_SECRET") else "offline",
                "configured" if os.environ.get("SPOTIFY_CLIENT_ID") else "not configured",
            ),
            HealthItem(
                "speech",
                "Speech package",
                "ready" if importlib.util.find_spec("speech_recognition") else "offline",
                "available" if importlib.util.find_spec("speech_recognition") else "missing",
            ),
            HealthItem(
                "tts",
                "Voice output",
                "ready" if importlib.util.find_spec("pyttsx3") else "offline",
                "available" if importlib.util.find_spec("pyttsx3") else "missing",
            ),
        ]

    def _summary(self, overall: str) -> str:
        if overall == "ready":
            return "System ready."
        if overall == "blocked":
            return "A required service needs attention."
        return "Mirror is usable with a few optional services offline."


health_service = HealthService()
