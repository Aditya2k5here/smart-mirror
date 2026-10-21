from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
import json

from core.config import CONFIG
from memory.db import Store, known_user


NEGATIVE_EMOTION_WEIGHT = {
    "angry": 1.0,
    "fear": 0.9,
    "sad": 0.75,
    "disgust": 0.55,
    "neutral": 0.2,
    "surprise": 0.25,
    "happy": 0.0,
}


class MemoryService(Store):
    """Sessions, tasks/reminders and the raw event log (the original memory layer)."""

    def __init__(self, db_path: Path = CONFIG.db_path) -> None:
        super().__init__(db_path)

    def init_db(self) -> None:
        self.ensure_schema()

    # ---- sessions --------------------------------------------------------
    def log_session(
        self,
        user: str,
        start_time: datetime | None,
        end_time: datetime,
        emotions_list: list[str],
        slouch_count: int,
    ) -> None:
        if not self._known_user(user) or not start_time:
            return

        duration = int((end_time - start_time).total_seconds())
        dominant_emotion = "neutral"
        if emotions_list:
            dominant_emotion = Counter(emotions_list).most_common(1)[0][0]

        self.execute(
            """
            INSERT INTO sessions (
                user, start_time, end_time, dominant_emotion,
                slouch_count, duration_seconds
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user, start_time.isoformat(), end_time.isoformat(), dominant_emotion, slouch_count, duration),
        )
        self.log_event(
            user,
            "SESSION_ENDED",
            {
                "dominant_emotion": dominant_emotion,
                "slouch_count": slouch_count,
                "duration_seconds": duration,
            },
        )

    def get_last_session(self, user: str) -> dict[str, Any] | None:
        if not self._known_user(user):
            return None
        row = self.one(
            """
            SELECT dominant_emotion, start_time, end_time, slouch_count, duration_seconds
            FROM sessions
            WHERE user = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (user,),
        )
        if not row:
            return None
        return {
            "emotion": row["dominant_emotion"],
            "time": row["start_time"],
            "end_time": row["end_time"],
            "slouch_count": row["slouch_count"],
            "duration_seconds": row["duration_seconds"],
        }

    # ---- tasks & reminders ----------------------------------------------
    def add_task(self, user: str, task_text: str, due_at: datetime | None = None) -> int | None:
        if not self._known_user(user) or not task_text.strip():
            return None
        task_id = self.execute(
            "INSERT INTO tasks (user, task, status, created_at, due_at) VALUES (?, ?, 'pending', ?, ?)",
            (
                user,
                task_text.strip(),
                datetime.now().isoformat(timespec="seconds"),
                due_at.isoformat(timespec="seconds") if due_at else None,
            ),
        )
        self.log_event(
            user, "TASK_CREATED", {"task": task_text.strip(), "due_at": due_at.isoformat() if due_at else None}
        )
        return task_id

    def complete_latest_task(self, user: str) -> str | None:
        if not self._known_user(user):
            return None
        row = self.one(
            "SELECT id FROM tasks WHERE user = ? AND status = 'pending' ORDER BY id DESC LIMIT 1", (user,)
        )
        return self.complete_task(user, row["id"]) if row else None

    def complete_task(self, user: str, task_id: int) -> str | None:
        row = self.one(
            "SELECT task FROM tasks WHERE id = ? AND user = ? AND status = 'pending'", (task_id, user)
        )
        if not row:
            return None
        self.execute(
            "UPDATE tasks SET status = 'done', completed_at = ? WHERE id = ?",
            (datetime.now().isoformat(timespec="seconds"), task_id),
        )
        self.log_event(user, "TASK_COMPLETED", {"task": row["task"]})
        return row["task"]

    def get_tasks(self, user: str) -> list[dict[str, Any]]:
        if not self._known_user(user):
            return []
        rows = self.query(
            """
            SELECT id, task, created_at, due_at
            FROM tasks
            WHERE user = ? AND status = 'pending'
            ORDER BY CASE WHEN due_at IS NULL THEN 1 ELSE 0 END, due_at ASC, created_at ASC
            """,
            (user,),
        )
        return [
            {"id": r["id"], "text": r["task"], "created_at": r["created_at"], "due_at": r["due_at"]}
            for r in rows
        ]

    def due_reminders(self, user: str, now: datetime | None = None) -> list[dict[str, Any]]:
        """Pending reminders whose time has come and that haven't been announced yet."""
        if not self._known_user(user):
            return []
        now = now or datetime.now()
        return self.query(
            "SELECT id, task AS text, due_at FROM tasks WHERE user = ? AND status = 'pending' "
            "AND due_at IS NOT NULL AND notified_at IS NULL AND due_at <= ? ORDER BY due_at",
            (user, now.isoformat(timespec="seconds")),
        )

    def mark_notified(self, task_id: int) -> None:
        self.execute(
            "UPDATE tasks SET notified_at = ? WHERE id = ?",
            (datetime.now().isoformat(timespec="seconds"), task_id),
        )

    # ---- events ----------------------------------------------------------
    def log_event(
        self, user: str, event_type: str, payload: dict[str, Any] | None = None
    ) -> None:
        self.execute(
            "INSERT INTO memory_events (user, event_type, payload, created_at) VALUES (?, ?, ?, ?)",
            (
                user or "Unknown",
                event_type,
                json.dumps(payload or {}, default=str),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )

    def get_recent_sessions(self, user: str, days: int = 7) -> list[dict[str, Any]]:
        if not self._known_user(user):
            return []
        since = (datetime.now() - timedelta(days=days)).isoformat()
        return self.query(
            """
            SELECT dominant_emotion, start_time, slouch_count, duration_seconds
            FROM sessions
            WHERE user = ? AND start_time >= ?
            ORDER BY start_time DESC
            """,
            (user, since),
        )

    def get_stress_score(self, user: str, days: int = 7) -> float:
        sessions = self.get_recent_sessions(user, days=days)
        if not sessions:
            return 0.0
        weighted = [
            NEGATIVE_EMOTION_WEIGHT.get(row["dominant_emotion"], 0.2)
            for row in sessions
        ]
        return min(1.0, sum(weighted) / len(weighted))

    def get_posture_score(self, user: str, days: int = 7) -> float:
        sessions = self.get_recent_sessions(user, days=days)
        if not sessions:
            return 1.0
        slouches = sum(int(row["slouch_count"] or 0) for row in sessions)
        minutes = sum(int(row["duration_seconds"] or 0) for row in sessions) / 60
        if minutes <= 0:
            return 1.0
        return max(0.0, min(1.0, 1 - (slouches / max(4, minutes / 6))))

    def get_personal_context(self, user: str) -> dict[str, Any]:
        tasks = self.get_tasks(user)
        sessions = self.get_recent_sessions(user)
        last_session = self.get_last_session(user)
        return {
            "last_session": last_session,
            "pending_tasks": tasks,
            "recent_sessions": sessions,
            "stress_score": self.get_stress_score(user),
            "posture_score": self.get_posture_score(user),
        }

    def _known_user(self, user: str) -> bool:
        return known_user(user)


memory_service = MemoryService()


def init_db() -> None:
    memory_service.init_db()


def log_session(user, start_time, end_time, emotions_list, slouch_count) -> None:
    memory_service.log_session(user, start_time, end_time, emotions_list, slouch_count)


def get_last_session(user):
    return memory_service.get_last_session(user)


def add_task(user, task_text) -> None:
    memory_service.add_task(user, task_text)


def complete_latest_task(user):
    return memory_service.complete_latest_task(user)


def get_tasks(user):
    return memory_service.get_tasks(user)


def log_event(user, event_type, payload=None) -> None:
    memory_service.log_event(user, event_type, payload)
