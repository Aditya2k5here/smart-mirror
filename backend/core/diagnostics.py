from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Any
import time


class DiagnosticsService:
    def __init__(self) -> None:
        self.frame_times: deque[float] = deque(maxlen=90)
        self.loop_ms: deque[float] = deque(maxlen=90)
        self.last_ai_provider = ""
        self.last_ai_latency_ms = 0
        self.background_tasks = 0
        self.websocket_clients = 0

    def record_frame(self, loop_started_at: float, active_tasks: int, clients: int) -> None:
        now = time.monotonic()
        self.frame_times.append(now)
        self.loop_ms.append((now - loop_started_at) * 1000)
        self.background_tasks = active_tasks
        self.websocket_clients = clients

    def record_ai(self, provider: str, latency_ms: int) -> None:
        self.last_ai_provider = provider
        self.last_ai_latency_ms = latency_ms

    def snapshot(self) -> dict[str, Any]:
        fps = 0.0
        if len(self.frame_times) >= 2:
            elapsed = self.frame_times[-1] - self.frame_times[0]
            if elapsed > 0:
                fps = (len(self.frame_times) - 1) / elapsed

        avg_loop = sum(self.loop_ms) / len(self.loop_ms) if self.loop_ms else 0
        pressure = "low"
        if avg_loop > 45:
            pressure = "high"
        elif avg_loop > 28:
            pressure = "medium"

        return {
            "fps": round(fps, 1),
            "loop_ms": round(avg_loop, 1),
            "cpu_pressure": pressure,
            "background_tasks": self.background_tasks,
            "websocket_clients": self.websocket_clients,
            "ai_provider": self.last_ai_provider,
            "ai_latency_ms": self.last_ai_latency_ms,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }


diagnostics_service = DiagnosticsService()
