from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
import json
import sqlite3

from core.config import CONFIG


NEGATIVE_EMOTION_WEIGHT = {
    "angry": 1.0,
    "fear": 0.9,
    "sad": 0.75,
    "disgust": 0.55,
    "neutral": 0.2,
    "surprise": 0.25,
    "happy": 0.0,
}


class MemoryService:
    def __init__(self, db_path: Path = CONFIG.db_path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY,
                    user TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    dominant_emotion TEXT,
                    slouch_count INTEGER,
                    duration_seconds INTEGER
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY,
                    user TEXT,
                    task TEXT,
                    status TEXT,
                    created_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_events (
                    id INTEGER PRIMARY KEY,
                    user TEXT,
                    event_type TEXT,
                    payload TEXT,
                    created_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_notes (
                    id INTEGER PRIMARY KEY,
                    user TEXT,
                    note_key TEXT,
                    note_value TEXT,
                    updated_at TEXT,
                    UNIQUE(user, note_key)
                )
                """
            )
            self._ensure_column(conn, "tasks", "completed_at", "TEXT")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sessions_user_time "
                "ON sessions(user, start_time)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_user_status "
                "ON tasks(user, status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_user_type "
                "ON memory_events(user, event_type)"
            )

    def _ensure_column(
        self, conn: sqlite3.Connection, table: str, column: str, column_type: str
    ) -> None:
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")

    def log_session(
        self,
        user: str,
        start_time: datetime | None,
        end_time: datetime,
        emotions_list: list[str],
        slouch_count: int,
    ) -> None:
        if not user or user == "Unknown" or not start_time:
            return

        duration = int((end_time - start_time).total_seconds())
        dominant_emotion = "neutral"
        if emotions_list:
            dominant_emotion = Counter(emotions_list).most_common(1)[0][0]

        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (
                    user, start_time, end_time, dominant_emotion,
                    slouch_count, duration_seconds
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    user,
                    start_time.isoformat(),
                    end_time.isoformat(),
                    dominant_emotion,
                    slouch_count,
                    duration,
                ),
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
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT dominant_emotion, start_time, slouch_count, duration_seconds
                FROM sessions
                WHERE user = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (user,),
            ).fetchone()
        if not row:
            return None
        return {
            "emotion": row["dominant_emotion"],
            "time": row["start_time"],
            "slouch_count": row["slouch_count"],
            "duration_seconds": row["duration_seconds"],
        }

    def add_task(self, user: str, task_text: str) -> None:
        if not self._known_user(user) or not task_text.strip():
            return
        now = datetime.now().isoformat(timespec="seconds")
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO tasks (user, task, status, created_at)
                VALUES (?, ?, 'pending', ?)
                """,
                (user, task_text.strip(), now),
            )
        self.log_event(user, "TASK_CREATED", {"task": task_text.strip()})

    def complete_latest_task(self, user: str) -> str | None:
        if not self._known_user(user):
            return None

        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT id, task
                FROM tasks
                WHERE user = ? AND status = 'pending'
                ORDER BY id DESC
                LIMIT 1
                """,
                (user,),
            ).fetchone()
            if not row:
                return None

            conn.execute(
                """
                UPDATE tasks
                SET status = 'done', completed_at = ?
                WHERE id = ?
                """,
                (datetime.now().isoformat(timespec="seconds"), row["id"]),
            )

        self.log_event(user, "TASK_COMPLETED", {"task": row["task"]})
        return row["task"]

    def get_tasks(self, user: str) -> list[dict[str, Any]]:
        if not self._known_user(user):
            return []
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, task, created_at
                FROM tasks
                WHERE user = ? AND status = 'pending'
                ORDER BY created_at ASC
                """,
                (user,),
            ).fetchall()
        return [
            {"id": row["id"], "text": row["task"], "created_at": row["created_at"]}
            for row in rows
        ]

    def log_event(
        self, user: str, event_type: str, payload: dict[str, Any] | None = None
    ) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO memory_events (user, event_type, payload, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    user or "Unknown",
                    event_type,
                    json.dumps(payload or {}),
                    now,
                ),
            )

    def get_recent_sessions(self, user: str, days: int = 7) -> list[dict[str, Any]]:
        if not self._known_user(user):
            return []
        since = (datetime.now() - timedelta(days=days)).isoformat()
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT dominant_emotion, start_time, slouch_count, duration_seconds
                FROM sessions
                WHERE user = ? AND start_time >= ?
                ORDER BY start_time DESC
                """,
                (user, since),
            ).fetchall()
        return [dict(row) for row in rows]

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
        return bool(user and user != "Unknown")


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

