"""Goals, body measurements and progress maths (trend, pace, ETA, milestones)."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

from core.config import CONFIG
from memory.db import Store, known_user, now_iso, today_str

KG_PER_LB = 0.45359237
MILESTONES = (25, 50, 75, 100)


def to_kg(value: float, unit: str) -> float:
    return value * KG_PER_LB if unit.lower().startswith(("lb", "pound")) else value


def kg_to_display(value_kg: float, units: str) -> tuple[float, str]:
    if units == "imperial":
        return round(value_kg / KG_PER_LB, 1), "lb"
    return round(value_kg, 1), "kg"


def ema(values: list[float], alpha: float = 0.3) -> list[float]:
    out: list[float] = []
    for v in values:
        out.append(v if not out else alpha * v + (1 - alpha) * out[-1])
    return out


def linear_slope_per_week(points: list[tuple[date, float]]) -> float | None:
    """Least-squares slope in units/week. None when there isn't enough signal."""
    if len(points) < 3:
        return None
    span = (points[-1][0] - points[0][0]).days
    if span < 5:
        return None
    xs = [(d - points[0][0]).days for d, _ in points]
    ys = [v for _, v in points]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return None
    slope_per_day = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    return slope_per_day * 7


class GoalService(Store):
    # ---- measurements ----------------------------------------------------
    def add_measurement(self, user: str, metric: str, value: float, unit: str = "", at: datetime | None = None,
                        note: str = "", source: str = "voice") -> int | None:
        if not known_user(user):
            return None
        at = at or datetime.now()
        mid = self.execute(
            "INSERT INTO measurements (user, metric, value, unit, at, day, note, source) VALUES (?,?,?,?,?,?,?,?)",
            (user, metric, float(value), unit, at.isoformat(timespec="seconds"), at.date().isoformat(), note, source),
        )
        if metric == "weight":
            self._backfill_start(user, float(value))
        return mid

    def _backfill_start(self, user: str, value: float) -> None:
        """First weigh-in after a relative goal ('lose 5 kg'): fix the start and resolve the target."""
        for goal in self.query(
            "SELECT id, delta_value, direction, target_value FROM goals "
            "WHERE user = ? AND metric = 'weight' AND status = 'active' AND start_value IS NULL", (user,)
        ):
            target = goal["target_value"]
            title = None
            if target is None and goal["delta_value"] is not None:
                sign = 1 if goal["direction"] == "up" else -1
                target = value + sign * abs(goal["delta_value"])
                title = f"{'Gain' if sign > 0 else 'Lose'} {abs(goal['delta_value']):.1f} kg"
            self.execute(
                "UPDATE goals SET start_value = ?, target_value = ?, title = COALESCE(?, title) WHERE id = ?",
                (value, target, title, goal["id"]),
            )

    def latest(self, user: str, metric: str) -> dict[str, Any] | None:
        if not known_user(user):
            return None
        return self.one(
            "SELECT * FROM measurements WHERE user = ? AND metric = ? ORDER BY at DESC, id DESC LIMIT 1", (user, metric)
        )

    def series(self, user: str, metric: str, days: int = 90) -> list[tuple[date, float]]:
        """One value per day (the last reading of that day), oldest first."""
        if not known_user(user):
            return []
        since = (date.today() - timedelta(days=days)).isoformat()
        rows = self.query(
            "SELECT day, value FROM measurements WHERE user = ? AND metric = ? AND day >= ? ORDER BY at ASC, id ASC",
            (user, metric, since),
        )
        by_day: dict[str, float] = {}
        for r in rows:
            by_day[r["day"]] = r["value"]
        return [(date.fromisoformat(d), v) for d, v in sorted(by_day.items())]

    def has_today(self, user: str, metric: str) -> bool:
        row = self.one("SELECT 1 AS x FROM measurements WHERE user = ? AND metric = ? AND day = ? LIMIT 1",
                       (user, metric, today_str()))
        return bool(row)

    def logging_streak(self, user: str, metric: str = "weight") -> int:
        days = {d for d, _ in self.series(user, metric, days=400)}
        streak, cursor = 0, date.today()
        if cursor not in days:
            cursor -= timedelta(days=1)
        while cursor in days:
            streak += 1
            cursor -= timedelta(days=1)
        return streak

    def metric_average(self, user: str, metric: str, days: int = 7) -> float | None:
        pts = self.series(user, metric, days=days)
        return sum(v for _, v in pts) / len(pts) if pts else None

    # ---- goals -----------------------------------------------------------
    def create_goal(self, user: str, title: str, kind: str = "custom", metric: str = "", unit: str = "",
                    start_value: float | None = None, target_value: float | None = None,
                    direction: str = "", target_date: date | None = None,
                    delta_value: float | None = None) -> dict[str, Any] | None:
        if not known_user(user) or not title.strip():
            return None
        if metric and start_value is None:
            latest = self.latest(user, metric)
            start_value = latest["value"] if latest else None
        if not direction and start_value is not None and target_value is not None:
            direction = "down" if target_value < start_value else "up" if target_value > start_value else "maintain"
        if kind == "weight":
            # Only one active weight goal at a time.
            self.execute("UPDATE goals SET status = 'replaced' WHERE user = ? AND kind = 'weight' AND status = 'active'", (user,))
        gid = self.execute(
            "INSERT INTO goals (user, kind, title, metric, unit, start_value, target_value, direction, start_date, "
            "target_date, status, created_at, delta_value) VALUES (?,?,?,?,?,?,?,?,?,?,'active',?,?)",
            (user, kind, title.strip(), metric, unit, start_value, target_value, direction, today_str(),
             target_date.isoformat() if target_date else None, now_iso(), delta_value),
        )
        return self.get_goal(gid)

    def create_weight_goal(self, user: str, target_kg: float | None = None, delta_kg: float | None = None,
                           by: date | None = None, direction: str = "") -> dict[str, Any] | None:
        latest = self.latest(user, "weight")
        start = latest["value"] if latest else None
        if target_kg is None and delta_kg is not None and start is not None:
            sign = -1 if direction == "down" or (not direction) else 1
            target_kg = start + sign * abs(delta_kg)
        if target_kg is None and delta_kg is not None and start is None:
            # No weigh-in yet: remember the relative target and resolve it on the first measurement.
            return self.create_goal(user, "Lose weight" if direction != "up" else "Gain weight", kind="weight",
                                    metric="weight", unit="kg", start_value=None, target_value=None,
                                    direction=direction or "down", target_date=by, delta_value=abs(delta_kg))
        if target_kg is None:
            return None
        if start is not None:
            diff = target_kg - start
            title = (f"Lose {abs(diff):.1f} kg" if diff < 0 else f"Gain {abs(diff):.1f} kg" if diff > 0
                     else "Maintain weight")
        else:
            title = f"Reach {target_kg:.1f} kg"
        return self.create_goal(user, title, kind="weight", metric="weight", unit="kg", start_value=start,
                                target_value=target_kg, direction="", target_date=by)

    def get_goal(self, goal_id: int) -> dict[str, Any] | None:
        return self.one("SELECT * FROM goals WHERE id = ?", (goal_id,))

    def active_goals(self, user: str) -> list[dict[str, Any]]:
        if not known_user(user):
            return []
        return self.query("SELECT * FROM goals WHERE user = ? AND status = 'active' ORDER BY id ASC", (user,))

    def weight_goal(self, user: str) -> dict[str, Any] | None:
        for g in self.active_goals(user):
            if g["kind"] == "weight":
                return g
        return None

    def set_status(self, goal_id: int, status: str) -> None:
        self.execute("UPDATE goals SET status = ?, completed_at = ? WHERE id = ?", (status, now_iso(), goal_id))

    def abandon_goal(self, user: str, match: str = "") -> str | None:
        for g in self.active_goals(user):
            if not match or match.lower() in g["title"].lower() or (match == "weight" and g["kind"] == "weight"):
                self.set_status(g["id"], "abandoned")
                return g["title"]
        return None

    # ---- progress --------------------------------------------------------
    def progress(self, goal: dict[str, Any], units: str | None = None) -> dict[str, Any]:
        units = units or CONFIG.units
        metric = goal.get("metric") or ""
        base: dict[str, Any] = {
            "id": goal["id"], "title": goal["title"], "kind": goal["kind"], "metric": metric,
            "target": goal.get("target_value"), "start": goal.get("start_value"),
            "target_date": goal.get("target_date"), "direction": goal.get("direction") or "",
            "percent": None, "status": "no_data", "headline": "", "series": [], "current": None,
            "trend": None, "rate_per_week": None, "eta": None, "unit": goal.get("unit") or "",
        }
        if not metric:
            base["status"] = "manual"
            base["headline"] = goal["title"]
            return base

        points = self.series(goal["user"], metric, days=180)
        points = [(d, v) for d, v in points if d >= date.fromisoformat(goal["start_date"]) - timedelta(days=14)]
        if not points:
            base["headline"] = "Waiting for your first weigh-in." if metric == "weight" else "No readings yet."
            return base

        smoothed = ema([v for _, v in points])
        current = points[-1][1]
        trend = smoothed[-1]
        base["current"], base["trend"] = current, trend
        base["series"] = [{"day": d.isoformat(), "value": round(v, 2), "trend": round(t, 2)}
                          for (d, v), t in zip(points[-60:], smoothed[-60:])]

        # Regress on raw readings: least squares already averages out day-to-day noise,
        # whereas the EMA's warm-up lag would bias the slope toward zero.
        recent = [(d, v) for d, v in points if d >= date.today() - timedelta(days=28)]
        rate = linear_slope_per_week(recent)
        base["rate_per_week"] = rate

        start, target = goal.get("start_value"), goal.get("target_value")
        if start is None:
            start = points[0][1]
        if target is None:
            base["status"] = "just_started"
            base["headline"] = "Tell me your target and I'll map the pace."
            return base

        total = target - start
        direction = goal.get("direction") or ("down" if total < 0 else "up")
        reached = (current <= target if direction == "down" else current >= target if direction == "up"
                   else abs(current - target) <= abs(target) * 0.015)
        if abs(total) < 1e-9:
            percent = 100.0
        else:
            percent = max(0.0, min(100.0, (trend - start) / total * 100.0))
        if reached:
            percent = 100.0
        base["percent"] = round(percent, 1)

        remaining = target - current
        weeks_left = None
        if goal.get("target_date"):
            weeks_left = (date.fromisoformat(goal["target_date"]) - date.today()).days / 7
        eta = None
        heading_right = rate is not None and rate * remaining > 0 and abs(rate) > 0.02
        if heading_right and not reached:
            eta_date = date.today() + timedelta(weeks=abs(remaining / rate))
            if eta_date < date.today() + timedelta(days=1500):
                eta = eta_date
        base["eta"] = eta.isoformat() if eta else None
        required = (remaining / weeks_left) if weeks_left and weeks_left > 0 else None
        base["required_rate_per_week"] = required

        span_days = (points[-1][0] - points[0][0]).days
        if reached:
            status = "done"
        elif len(points) < 3 or span_days < 5:
            status = "just_started"
        elif not heading_right:
            status = "stalled"
        elif goal.get("target_date") and eta and eta > date.fromisoformat(goal["target_date"]) + timedelta(days=3):
            status = "behind"
        elif goal.get("target_date") and eta and eta < date.fromisoformat(goal["target_date"]) - timedelta(days=7):
            status = "ahead"
        else:
            status = "on_track"
        base["status"] = status
        base["headline"] = self._headline(goal, base, remaining, units)
        return base

    def _headline(self, goal: dict[str, Any], p: dict[str, Any], remaining: float, units: str) -> str:
        weight = goal["metric"] == "weight"
        fmt = (lambda v: "%.1f %s" % kg_to_display(abs(v), units)) if weight else (lambda v: f"{abs(v):.1f} {p['unit']}".strip())
        status = p["status"]
        if status == "done":
            return "Goal reached. Remarkable work."
        done = abs((p["trend"] or 0) - (p["start"] or 0))
        parts = [f"{fmt(done)} down" if goal.get("direction") == "down" else f"{fmt(done)} so far", f"{fmt(remaining)} to go"]
        if status == "on_track" and p["eta"]:
            parts.append(f"on pace for {date.fromisoformat(p['eta']).strftime('%b %d').replace(' 0', ' ')}")
        elif status == "ahead":
            parts.append("ahead of schedule")
        elif status == "behind":
            parts.append("a little behind pace")
        elif status == "stalled":
            parts.append("progress has plateaued")
        return ", ".join(parts)

    def pending_milestone(self, goal: dict[str, Any], percent: float | None) -> int | None:
        """Return a newly crossed 25/50/75/100% milestone (and record it)."""
        if percent is None:
            return None
        reached = max((m for m in MILESTONES if percent >= m), default=0)
        if reached > int(goal.get("milestone") or 0):
            self.execute("UPDATE goals SET milestone = ? WHERE id = ?", (reached, goal["id"]))
            if reached == 100:
                self.set_status(goal["id"], "achieved")
            return reached
        return None

    def safe_pace_warning(self, user: str, rate_per_week_kg: float | None) -> str:
        latest = self.latest(user, "weight")
        if not latest or rate_per_week_kg is None:
            return ""
        if rate_per_week_kg < 0 and abs(rate_per_week_kg) > latest["value"] * 0.01:
            return "That is faster than the usual healthy pace of about one percent of body weight a week."
        return ""


goal_service = GoalService(CONFIG.db_path)
