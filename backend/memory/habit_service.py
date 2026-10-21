"""Habits (including water), streaks and workout history."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

from core.config import CONFIG
from memory.db import Store, known_user, now_iso, today_str

WATER = "Water"


class HabitService(Store):
    # ---- habits ----------------------------------------------------------
    def ensure_habit(self, user: str, name: str, unit: str = "", daily_target: float = 1,
                     remind_times: str = "") -> dict[str, Any] | None:
        if not known_user(user) or not name.strip():
            return None
        name = name.strip()
        row = self.one("SELECT * FROM habits WHERE user = ? AND lower(name) = lower(?) AND active = 1", (user, name))
        if row:
            return row
        hid = self.execute(
            "INSERT INTO habits (user, name, unit, daily_target, remind_times, created_at) VALUES (?,?,?,?,?,?)",
            (user, name, unit, daily_target, remind_times, now_iso()),
        )
        return self.one("SELECT * FROM habits WHERE id = ?", (hid,))

    def find_habit(self, user: str, text: str) -> dict[str, Any] | None:
        text = (text or "").lower()
        best = None
        for h in self.habits(user):
            name = h["name"].lower()
            if name in text or text in name:
                if best is None or len(name) > len(best["name"]):
                    best = h
        return best

    def habits(self, user: str) -> list[dict[str, Any]]:
        if not known_user(user):
            return []
        return self.query("SELECT * FROM habits WHERE user = ? AND active = 1 ORDER BY id", (user,))

    def remove_habit(self, user: str, name: str) -> bool:
        h = self.find_habit(user, name)
        if h:
            self.execute("UPDATE habits SET active = 0 WHERE id = ?", (h["id"],))
        return bool(h)

    def log(self, user: str, habit_id: int, amount: float = 1, at: datetime | None = None) -> None:
        at = at or datetime.now()
        self.execute(
            "INSERT INTO habit_logs (habit_id, user, amount, at, day) VALUES (?,?,?,?,?)",
            (habit_id, user, amount, at.isoformat(timespec="seconds"), at.date().isoformat()),
        )

    def done_on(self, habit_id: int, day: str) -> float:
        row = self.one("SELECT COALESCE(SUM(amount),0) AS s FROM habit_logs WHERE habit_id = ? AND day = ?", (habit_id, day))
        return float(row["s"]) if row else 0.0

    def streak(self, habit: dict[str, Any]) -> int:
        rows = self.query(
            "SELECT day, SUM(amount) AS s FROM habit_logs WHERE habit_id = ? GROUP BY day", (habit["id"],)
        )
        ok = {r["day"] for r in rows if r["s"] >= (habit["daily_target"] or 1)}
        cursor, streak = date.today(), 0
        if cursor.isoformat() not in ok:
            cursor -= timedelta(days=1)
        while cursor.isoformat() in ok:
            streak += 1
            cursor -= timedelta(days=1)
        return streak

    def today(self, user: str) -> list[dict[str, Any]]:
        day = today_str()
        out = []
        for h in self.habits(user):
            done = self.done_on(h["id"], day)
            out.append({
                "id": h["id"], "name": h["name"], "unit": h["unit"], "target": h["daily_target"], "done": done,
                "complete": done >= (h["daily_target"] or 1), "streak": self.streak(h),
                "remind_times": h["remind_times"],
            })
        return out

    # ---- water -----------------------------------------------------------
    def log_water(self, user: str, ml: float, target_ml: float | None = None) -> dict[str, Any] | None:
        if not known_user(user):
            return None
        habit = self.ensure_habit(user, WATER, "ml", target_ml or 2000)
        if target_ml and habit and abs((habit["daily_target"] or 0) - target_ml) > 1:
            self.execute("UPDATE habits SET daily_target = ? WHERE id = ?", (target_ml, habit["id"]))
        self.log(user, habit["id"], ml)
        return {"total_ml": self.water_today(user), "habit_id": habit["id"]}

    def water_today(self, user: str, day: str | None = None) -> float:
        habit = self.one("SELECT id FROM habits WHERE user = ? AND name = ? AND active = 1", (user, WATER))
        return self.done_on(habit["id"], day or today_str()) if habit else 0.0

    def last_water_at(self, user: str) -> datetime | None:
        habit = self.one("SELECT id FROM habits WHERE user = ? AND name = ? AND active = 1", (user, WATER))
        if not habit:
            return None
        row = self.one("SELECT at FROM habit_logs WHERE habit_id = ? ORDER BY id DESC LIMIT 1", (habit["id"],))
        return datetime.fromisoformat(row["at"]) if row else None

    # ---- workouts --------------------------------------------------------
    def log_workout(self, user: str, exercise: str, reps: int = 0, duration_seconds: int = 0,
                    at: datetime | None = None) -> bool:
        if not known_user(user) or (reps <= 0 and duration_seconds < 20):
            return False
        at = at or datetime.now()
        self.execute(
            "INSERT INTO workouts (user, exercise, reps, duration_seconds, at, day) VALUES (?,?,?,?,?,?)",
            (user, exercise, reps, duration_seconds, at.isoformat(timespec="seconds"), at.date().isoformat()),
        )
        return True

    def workouts_since(self, user: str, days: int = 7) -> list[dict[str, Any]]:
        since = (date.today() - timedelta(days=days - 1)).isoformat()
        return self.query("SELECT * FROM workouts WHERE user = ? AND day >= ? ORDER BY at DESC", (user, since))

    def days_since_workout(self, user: str) -> int | None:
        row = self.one("SELECT day FROM workouts WHERE user = ? ORDER BY at DESC LIMIT 1", (user,))
        return (date.today() - date.fromisoformat(row["day"])).days if row else None

    def workout_summary(self, user: str, days: int = 7) -> dict[str, Any]:
        rows = self.workouts_since(user, days)
        by_exercise: dict[str, int] = {}
        for r in rows:
            by_exercise[r["exercise"]] = by_exercise.get(r["exercise"], 0) + int(r["reps"] or 0)
        return {
            "sessions": len(rows),
            "active_days": len({r["day"] for r in rows}),
            "minutes": round(sum(int(r["duration_seconds"] or 0) for r in rows) / 60),
            "reps": by_exercise,
        }


habit_service = HabitService(CONFIG.db_path)
