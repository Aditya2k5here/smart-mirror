"""Conversation log, journal entries, daily summaries, notes/lists and nudge bookkeeping."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from core.config import CONFIG
from memory.db import Store, known_user, now_iso, today_str


class JournalService(Store):
    # ---- conversation memory --------------------------------------------
    def add_turn(self, user: str, role: str, text: str, at: datetime | None = None) -> None:
        if not known_user(user) or not text.strip():
            return
        at = at or datetime.now()
        self.execute(
            "INSERT INTO conversations (user, role, text, at, day) VALUES (?,?,?,?,?)",
            (user, role, text.strip()[:1000], at.isoformat(timespec="seconds"), at.date().isoformat()),
        )

    def recent_turns(self, user: str, minutes: int = 45, limit: int = 8) -> list[dict[str, Any]]:
        if not known_user(user):
            return []
        since = (datetime.now() - timedelta(minutes=minutes)).isoformat(timespec="seconds")
        rows = self.query(
            "SELECT role, text, at FROM conversations WHERE user = ? AND at >= ? ORDER BY id DESC LIMIT ?",
            (user, since, limit),
        )
        return list(reversed(rows))

    def turns_on(self, user: str, day: str | None = None) -> list[dict[str, Any]]:
        return self.query("SELECT role, text, at FROM conversations WHERE user = ? AND day = ? ORDER BY id",
                          (user, day or today_str()))

    # ---- journal ---------------------------------------------------------
    def add_entry(self, user: str, text: str, mood: str = "", kind: str = "entry", day: str | None = None) -> None:
        if not known_user(user) or not text.strip():
            return
        self.execute(
            "INSERT INTO journal (user, day, kind, text, mood, created_at) VALUES (?,?,?,?,?,?)",
            (user, day or today_str(), kind, text.strip()[:2000], mood, now_iso()),
        )

    def entries(self, user: str, days: int = 14, kind: str | None = None) -> list[dict[str, Any]]:
        since = (datetime.now() - timedelta(days=days)).date().isoformat()
        sql = "SELECT * FROM journal WHERE user = ? AND day >= ?"
        params: list[Any] = [user, since]
        if kind:
            sql += " AND kind = ?"
            params.append(kind)
        return self.query(sql + " ORDER BY id DESC", params)

    def summary_for(self, user: str, day: str) -> dict[str, Any] | None:
        return self.one("SELECT * FROM journal WHERE user = ? AND day = ? AND kind = 'summary' ORDER BY id DESC LIMIT 1",
                        (user, day))

    def save_summary(self, user: str, day: str, text: str, mood: str = "") -> None:
        self.execute("DELETE FROM journal WHERE user = ? AND day = ? AND kind = 'summary'", (user, day))
        self.add_entry(user, text, mood=mood, kind="summary", day=day)

    def mood_by_day(self, user: str, days: int = 7) -> list[dict[str, Any]]:
        since = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
        return self.query(
            "SELECT substr(start_time,1,10) AS day, dominant_emotion AS emotion FROM sessions "
            "WHERE user = ? AND start_time >= ? ORDER BY start_time", (user, since),
        )

    # ---- notes & lists ---------------------------------------------------
    def add_note(self, user: str, text: str, list_name: str = "notes") -> int | None:
        if not known_user(user) or not text.strip():
            return None
        return self.execute("INSERT INTO notes (user, list_name, text, created_at) VALUES (?,?,?,?)",
                            (user, list_name, text.strip()[:300], now_iso()))

    def list_notes(self, user: str, list_name: str | None = None, include_done: bool = False) -> list[dict[str, Any]]:
        sql = "SELECT * FROM notes WHERE user = ?"
        params: list[Any] = [user]
        if list_name:
            sql += " AND list_name = ?"
            params.append(list_name)
        if not include_done:
            sql += " AND done = 0"
        return self.query(sql + " ORDER BY id", params)

    def clear_list(self, user: str, list_name: str) -> int:
        return self.execute("UPDATE notes SET done = 1 WHERE user = ? AND list_name = ? AND done = 0", (user, list_name))

    # ---- nudge bookkeeping ----------------------------------------------
    def nudge_fired(self, user: str, key: str, day: str | None = None) -> bool:
        row = self.one("SELECT 1 AS x FROM nudge_log WHERE user = ? AND key = ? AND day = ? LIMIT 1",
                       (user, key, day or today_str()))
        return bool(row)

    def mark_nudge(self, user: str, key: str) -> None:
        self.execute("INSERT INTO nudge_log (user, key, day, at) VALUES (?,?,?,?)", (user, key, today_str(), now_iso()))

    def last_nudge_at(self, user: str, key: str) -> datetime | None:
        row = self.one("SELECT at FROM nudge_log WHERE user = ? AND key = ? ORDER BY id DESC LIMIT 1", (user, key))
        return datetime.fromisoformat(row["at"]) if row else None


journal_service = JournalService(CONFIG.db_path)
