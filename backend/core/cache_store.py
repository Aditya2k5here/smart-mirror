from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import time

from core.config import CONFIG


class CacheStore:
    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or (CONFIG.data_dir / "cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str, max_age_seconds: float | None = None) -> Any | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

        if max_age_seconds is not None:
            age = time.time() - float(record.get("saved_at", 0))
            if age > max_age_seconds:
                return None
        return record.get("value")

    def set(self, key: str, value: Any) -> None:
        path = self._path(key)
        record = {"saved_at": time.time(), "value": value}
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    def with_stale_flag(self, key: str, max_age_seconds: float | None = None) -> Any | None:
        value = self.get(key, max_age_seconds=max_age_seconds)
        if isinstance(value, dict):
            return {**value, "stale": True}
        return value

    def _path(self, key: str) -> Path:
        safe = "".join(ch for ch in key if ch.isalnum() or ch in {"_", "-"}).strip()
        return self.cache_dir / f"{safe or 'cache'}.json"


cache_store = CacheStore()
