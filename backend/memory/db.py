"""Shared SQLite access and schema for every memory service."""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable
import json
import sqlite3
import threading

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY, user TEXT, start_time TEXT, end_time TEXT,
    dominant_emotion TEXT, slouch_count INTEGER, duration_seconds INTEGER
);
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY, user TEXT, task TEXT, status TEXT, created_at TEXT,
    completed_at TEXT, due_at TEXT, notified_at TEXT
);
CREATE TABLE IF NOT EXISTS memory_events (
    id INTEGER PRIMARY KEY, user TEXT, event_type TEXT, payload TEXT, created_at TEXT
);
CREATE TABLE IF NOT EXISTS user_notes (
    id INTEGER PRIMARY KEY, user TEXT, note_key TEXT, note_value TEXT, updated_at TEXT,
    UNIQUE(user, note_key)
);
CREATE TABLE IF NOT EXISTS profiles (
    user TEXT PRIMARY KEY, age INTEGER, sex TEXT, height_cm REAL,
    activity_level TEXT, wake_time TEXT, bed_time TEXT, timezone TEXT,
    units TEXT DEFAULT 'metric', created_at TEXT, updated_at TEXT, last_seen_at TEXT,
    visit_count INTEGER DEFAULT 0, extra TEXT DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY, user TEXT, category TEXT, text TEXT, source TEXT,
    importance REAL DEFAULT 0.5, created_at TEXT, last_seen_at TEXT, active INTEGER DEFAULT 1
);
CREATE TABLE IF NOT EXISTS goals (
    id INTEGER PRIMARY KEY, user TEXT, kind TEXT, title TEXT, metric TEXT, unit TEXT,
    start_value REAL, target_value REAL, direction TEXT, start_date TEXT, target_date TEXT,
    status TEXT DEFAULT 'active', created_at TEXT, completed_at TEXT, milestone INTEGER DEFAULT 0,
    delta_value REAL
);
CREATE TABLE IF NOT EXISTS measurements (
    id INTEGER PRIMARY KEY, user TEXT, metric TEXT, value REAL, unit TEXT,
    at TEXT, day TEXT, note TEXT, source TEXT
);
CREATE TABLE IF NOT EXISTS meals (
    id INTEGER PRIMARY KEY, user TEXT, description TEXT, meal_type TEXT,
    calories REAL, protein_g REAL, carbs_g REAL, fat_g REAL, at TEXT, day TEXT, estimated INTEGER DEFAULT 1
);
CREATE TABLE IF NOT EXISTS habits (
    id INTEGER PRIMARY KEY, user TEXT, name TEXT, unit TEXT, daily_target REAL DEFAULT 1,
    remind_times TEXT DEFAULT '', active INTEGER DEFAULT 1, created_at TEXT
);
CREATE TABLE IF NOT EXISTS habit_logs (
    id INTEGER PRIMARY KEY, habit_id INTEGER, user TEXT, amount REAL, at TEXT, day TEXT
);
CREATE TABLE IF NOT EXISTS workouts (
    id INTEGER PRIMARY KEY, user TEXT, exercise TEXT, reps INTEGER, duration_seconds INTEGER,
    at TEXT, day TEXT
);
CREATE TABLE IF NOT EXISTS journal (
    id INTEGER PRIMARY KEY, user TEXT, day TEXT, kind TEXT, text TEXT, mood TEXT, created_at TEXT
);
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY, user TEXT, role TEXT, text TEXT, at TEXT, day TEXT
);
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY, user TEXT, list_name TEXT, text TEXT, done INTEGER DEFAULT 0, created_at TEXT
);
CREATE TABLE IF NOT EXISTS nudge_log (
    id INTEGER PRIMARY KEY, user TEXT, key TEXT, day TEXT, at TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_user_time ON sessions(user, start_time);
CREATE INDEX IF NOT EXISTS idx_tasks_user_status ON tasks(user, status);
CREATE INDEX IF NOT EXISTS idx_events_user_type ON memory_events(user, event_type);
CREATE INDEX IF NOT EXISTS idx_facts_user ON facts(user, active);
CREATE INDEX IF NOT EXISTS idx_goals_user ON goals(user, status);
CREATE INDEX IF NOT EXISTS idx_meas_user_metric ON measurements(user, metric, at);
CREATE INDEX IF NOT EXISTS idx_meals_user_day ON meals(user, day);
CREATE INDEX IF NOT EXISTS idx_habit_logs ON habit_logs(user, habit_id, day);
CREATE INDEX IF NOT EXISTS idx_workouts_user_day ON workouts(user, day);
CREATE INDEX IF NOT EXISTS idx_journal_user_day ON journal(user, day, kind);
CREATE INDEX IF NOT EXISTS idx_conv_user_at ON conversations(user, at);
CREATE INDEX IF NOT EXISTS idx_nudge_user_key ON nudge_log(user, key, day);
"""

# Columns added after the first release; applied to databases created earlier.
MIGRATIONS: dict[str, dict[str, str]] = {
    "tasks": {"completed_at": "TEXT", "due_at": "TEXT", "notified_at": "TEXT"},
    "goals": {"delta_value": "REAL"},
}

_schema_lock = threading.Lock()
_migrated: set[str] = set()


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def today_str(now: datetime | None = None) -> str:
    return (now or datetime.now()).date().isoformat()


class Store:
    """Base class: one short-lived connection per call keeps threads simple."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.ensure_schema()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def ensure_schema(self) -> None:
        key = str(self.db_path)
        with _schema_lock:
            if key in _migrated and self.db_path.exists():
                return
            conn = self.connect()
            try:
                with conn:
                    conn.executescript(SCHEMA)
                    for table, columns in MIGRATIONS.items():
                        existing = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
                        for column, ctype in columns.items():
                            if column not in existing:
                                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ctype}")
            finally:
                conn.close()
            _migrated.add(key)

    # -- tiny query helpers -------------------------------------------------
    def query(self, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        conn = self.connect()
        try:
            return [dict(r) for r in conn.execute(sql, tuple(params)).fetchall()]
        finally:
            conn.close()

    def one(self, sql: str, params: Iterable[Any] = ()) -> dict[str, Any] | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def execute(self, sql: str, params: Iterable[Any] = ()) -> int:
        conn = self.connect()
        try:
            with conn:
                cur = conn.execute(sql, tuple(params))
                return cur.lastrowid or cur.rowcount
        finally:
            conn.close()


def known_user(user: str | None) -> bool:
    return bool(user) and user not in ("Unknown", "Guest")


def dumps(value: Any) -> str:
    return json.dumps(value, default=lambda o: o.isoformat() if isinstance(o, (date, datetime)) else str(o))
