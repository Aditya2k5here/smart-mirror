"""Smart-home control through a Home Assistant instance (optional).

Set HOME_ASSISTANT_URL and HOME_ASSISTANT_TOKEN (a long-lived access token). Rooms map to light
entities in `backend/home_assistant.json`, e.g.
  {"lights": {"": "light.living_room", "bedroom": "light.bedroom"}, "scenes": {"movie": "scene.movie_night"}}
Without that file, "the lights" targets every light.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import logging

import requests

from core.config import BACKEND_DIR, CONFIG


class HomeAssistantService:
    def __init__(self, url: str = CONFIG.home_assistant_url, token: str = CONFIG.home_assistant_token,
                 mapping_path: Path = BACKEND_DIR / "home_assistant.json", post=None) -> None:
        self.url, self.token = url, token
        self._post = post or requests.post
        self.mapping: dict[str, Any] = {}
        if mapping_path.exists():
            try:
                self.mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                logging.warning("Could not read %s", mapping_path)

    @property
    def configured(self) -> bool:
        return bool(self.url and self.token)

    def _call(self, domain: str, service: str, data: dict[str, Any]) -> None:
        resp = self._post(f"{self.url}/api/services/{domain}/{service}",
                          headers={"Authorization": f"Bearer {self.token}"}, json=data, timeout=5)
        resp.raise_for_status()

    def lights(self, command: str, room: str = "") -> str:
        if not self.configured:
            return "Smart home isn't connected yet. Add your Home Assistant address and token in the settings."
        room = (room or "").lower().strip()
        entity = (self.mapping.get("lights") or {}).get(room) or (self.mapping.get("lights") or {}).get("") or "all"
        data: dict[str, Any] = {"entity_id": entity}
        try:
            if command in ("on", "off"):
                self._call("light", f"turn_{command}", data)
                return f"Lights {command}."
            if command in ("dim", "brighten"):
                self._call("light", "turn_on", {**data, "brightness_step_pct": -30 if command == "dim" else 30})
                return "Dimmed." if command == "dim" else "Brighter."
            if command == "scene":
                scene = (self.mapping.get("scenes") or {}).get(room)
                if not scene:
                    return f"I don't know a scene called {room}."
                self._call("scene", "turn_on", {"entity_id": scene})
                return f"{room.title()} scene on."
        except Exception as exc:
            logging.warning("Home Assistant call failed: %s", exc)
            return "I couldn't reach your smart home."
        return "I'm not sure how to do that with the lights."


home_assistant = HomeAssistantService()
