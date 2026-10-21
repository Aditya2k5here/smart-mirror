"""Proactive behaviour: what the mirror says and shows without being asked.

Pure decision logic. The runtime feeds it perception (`Env`, `Signals`) and speaks whatever
`Nudge`s come back. Everything is rate-limited and quiet-hours aware so it never nags.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable
import time

from assistant.brain import Brain
from assistant.models import Env
from assistant.reports import Reporter, money
from assistant.services import Services
from core.config import CONFIG
from core.timeparse import humanize_when
from memory.db import known_user

HABIT_STREAK_MILESTONES = (7, 14, 30, 60, 100, 200, 365)


@dataclass
class Nudge:
    key: str
    text: str
    kind: str
    priority: int = 1  # 0 ambient, 1 normal, 2 important, 3 critical (reminders, alarms)
    speak: bool = True
    pending: tuple[str, dict[str, Any]] | None = None
    ui: dict[str, Any] = field(default_factory=dict)
    celebrate: str = ""


@dataclass
class Signals:
    present_seconds: float = 0.0
    slouch_events_recent: int = 0  # slouching detections in the last few minutes
    negative_streak: int = 0  # consecutive negative emotion readings
    workout_active: bool = False


def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def in_quiet_hours(now: datetime, wake: str, bed: str) -> bool:
    n, w, b = now.hour * 60 + now.minute, _minutes(wake), _minutes(bed)
    return (n >= b or n < w) if b > w else (b <= n < w)


class ProactiveEngine:
    def __init__(self, services: Services, reporter: Reporter, brain: Brain,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.s, self.r, self.brain, self._clock = services, reporter, brain, clock
        self.last_spoken_at = -1e9
        self._cooldowns: dict[str, float] = {}
        self.enabled = CONFIG.proactive

    # ----------------------------------------------------------------- gating
    def _cooldown_ok(self, key: str, seconds: float) -> bool:
        return self._clock() - self._cooldowns.get(key, -1e9) >= seconds

    def _touch(self, key: str) -> None:
        self._cooldowns[key] = self._clock()

    def note_spoken(self) -> None:
        """Call whenever the mirror speaks for any reason (including replies) to space nudges out."""
        self.last_spoken_at = self._clock()

    def _gate(self, nudges: list[Nudge], env: Env, busy: bool) -> list[Nudge]:
        """Apply quiet hours, busy state and the global spacing rule; record what fires."""
        if not self.enabled or env.privacy:
            return [n for n in nudges if n.priority >= 3]
        wake, bed = self.s.profile.wake_bed(env.user) if known_user(env.user) else (CONFIG.default_wake_time, CONFIG.default_bed_time)
        quiet = in_quiet_hours(env.now, wake, bed)
        out: list[Nudge] = []
        spoke_in_batch = False
        for n in sorted(nudges, key=lambda x: -x.priority):
            if n.priority < 3:
                if busy or env.workout_active or self.s.routines.active or self.s.breathing.active:
                    continue
                if quiet and n.priority < 2:
                    continue
                if n.speak and (spoke_in_batch or self._clock() - self.last_spoken_at < CONFIG.quiet_nudge_gap_seconds):
                    if n.priority < 2:
                        continue
            out.append(n)
            if n.speak:
                spoke_in_batch = True
                self.last_spoken_at = self._clock()
            if known_user(env.user) and n.key:
                self.s.journal.mark_nudge(env.user, n.key)
        return out

    def _fired(self, env: Env, key: str) -> bool:
        return self.s.journal.nudge_fired(env.user, key, env.now.date().isoformat())

    # ---------------------------------------------------------------- arrival
    def guest_prompt(self, env: Env, busy: bool = False) -> list[Nudge]:
        """Someone unrecognised has been in front of the mirror for a moment."""
        if not env.is_guest or not self.enabled or env.privacy or busy:
            return []
        if not self._cooldown_ok("guest", 20 * 60):
            return []
        self._touch("guest")
        self.last_spoken_at = self._clock()
        return [Nudge("", "Hi there. I don't think we've met. What's your name?", "guest", 2,
                      pending=("enroll_name", {}))]

    def arrival(self, env: Env, away_seconds: float | None) -> list[Nudge]:
        if not known_user(env.user):
            return []
        user, now = env.user, env.now
        self.s.profile.touch_visit(user)
        first = user.split()[0]
        out: list[Nudge] = []
        wake, bed = self.s.profile.wake_bed(user)

        # reminders that came due while they were away
        due = self.s.memory.due_reminders(user, now)
        if due:
            out.append(Nudge(f"missed:{due[0]['id']}", "While you were away: " + "; ".join(d["text"] for d in due[:3]) + ".", "reminder", 3))
            for d in due:
                self.s.memory.mark_notified(d["id"])

        if in_quiet_hours(now, wake, bed):
            return self._gate(out, env, busy=False)

        if not self._fired(env, "briefing") and env.day_part != "evening":
            text = self.r.briefing(env)
            pending = None
            goal = self.s.goals.weight_goal(user)
            if goal and env.day_part == "morning" and not self.s.goals.has_today(user, "weight"):
                text += " When you're ready, step on the scale and tell me your weight."
                pending = ("weigh_in", {})
            days = self.s.habits.days_since_workout(user)
            if goal and days is not None and days >= 3 and env.day_part == "morning":
                text += f" It's been {days} days since your last workout. Even ten minutes counts."
            out.append(Nudge("briefing", text, "briefing", 2, pending=pending, ui={"focus": "none"}))
        elif away_seconds is not None and away_seconds >= 30 * 60 and not due:
            out.append(Nudge("", f"Welcome back, {first}.", "welcome", 1))

        if env.day_part in ("evening", "night") and now.hour >= 19 and not self._fired(env, "recap"):
            recap = self.r.evening_recap(env)
            text = (recap + " " if recap else "") + "How was your day?"
            out.append(Nudge("recap", text, "recap", 2, pending=("journal", {})))

        week_key = f"weekly:{now.isocalendar()[1]}"
        if now.weekday() == 6 and now.hour >= 17 and not self.s.journal.nudge_fired(user, week_key, now.date().isoformat()):
            speech, data = self.r.week_review(user)
            out.append(Nudge(week_key, speech, "weekly", 1, ui={"focus": "week", "seconds": 25, "week": data}))

        for habit in self.s.habits.today(user):
            if habit["complete"] and habit["streak"] in HABIT_STREAK_MILESTONES and habit["name"] != "Water":
                key = f"streak:{habit['id']}:{habit['streak']}"
                if not self.s.journal.nudge_fired(user, key, now.date().isoformat()):
                    out.append(Nudge(key, f"{habit['streak']} days in a row of {habit['name'].lower()}. That's a real habit now.",
                                     "streak", 1, celebrate=f"{habit['streak']}-day streak"))
        return self._gate(out, env, busy=False)

    # ------------------------------------------------------------------- tick
    def tick(self, env: Env, signals: Signals, busy: bool = False) -> list[Nudge]:
        if not known_user(env.user):
            return []
        user, now = env.user, env.now
        out: list[Nudge] = []

        # reminders: critical, always delivered
        for due in self.s.memory.due_reminders(user, now):
            out.append(Nudge(f"reminder:{due['id']}", f"Reminder: {due['text']}.", "reminder", 3, ui={"focus": "none"}))
            self.s.memory.mark_notified(due["id"])

        if not env.present:
            return self._gate(out, env, busy)

        # calendar heads-up
        for event in env.events[:4]:
            start = event.get("start")
            if not start:
                continue
            try:
                starts = datetime.fromisoformat(start).replace(tzinfo=None)
            except ValueError:
                continue
            mins = (starts - now).total_seconds() / 60
            key = f"cal:{event['title'][:30]}:{start}"
            if 0 < mins <= 12 and not self._fired(env, key):
                out.append(Nudge(key, f"Heads up: {event['title']} starts in {max(1, int(round(mins)))} minutes.", "calendar", 2))

        # habit reminders (HH:MM list)
        for habit in self.s.habits.today(user):
            if habit["complete"] or not habit["remind_times"]:
                continue
            for hhmm in habit["remind_times"].split(","):
                try:
                    due_min = _minutes(hhmm.strip())
                except ValueError:
                    continue
                n = now.hour * 60 + now.minute
                key = f"habit:{habit['id']}:{hhmm.strip()}"
                if due_min <= n < due_min + 90 and not self._fired(env, key):
                    out.append(Nudge(key, f"Time to {habit['name'].lower()}.", "habit", 2))

        goal = self.s.goals.weight_goal(user)
        has_water = bool(self.s.habits.last_water_at(user))

        # hydration
        if (goal or has_water) and 8 <= now.hour <= 21 and self._cooldown_ok("hydration", 90 * 60):
            t = self.r.targets(user)
            ml = self.s.habits.water_today(user)
            expected = max(0.0, min(1.0, (now.hour + now.minute / 60 - 8) / 12)) * t["water_ml"]
            last = self.s.habits.last_water_at(user)
            since_water = (now - last).total_seconds() / 60 if last else 999
            if ml < expected * 0.6 and since_water > 75 and signals.present_seconds > 120:
                self._touch("hydration")
                out.append(Nudge("hydration", f"Water check. You're at {money(ml)} of {money(t['water_ml'])} millilitres.", "hydration", 1))

        # sedentary break
        if signals.present_seconds >= 50 * 60 and self._cooldown_ok("break", 60 * 60):
            self._touch("break")
            out.append(Nudge("", "You've been here a while. Stand up, roll your shoulders, look at something far away.", "break", 1))

        # posture (only after sustained slouching)
        if signals.slouch_events_recent >= 3 and self._cooldown_ok("posture", 10 * 60):
            self._touch("posture")
            out.append(Nudge("", "Shoulders back. Easy.", "posture", 0))

        # mood check-in: needs a sustained negative read, and is rare
        if signals.negative_streak >= 4 and self._cooldown_ok("mood", 45 * 60) and not signals.workout_active:
            self._touch("mood")
            out.append(Nudge("", "Hey. Rough moment? Want to take a minute and breathe together?", "mood", 1,
                             pending=("offer_breathing", {})))

        # wind-down near bedtime
        wake, bed = self.s.profile.wake_bed(user)
        mins_to_bed = _minutes(bed) - (now.hour * 60 + now.minute)
        if 0 <= mins_to_bed <= 30 and not self._fired(env, "winddown"):
            out.append(Nudge("winddown", "It's nearly bedtime. Want a two minute wind-down?", "winddown", 1, pending=("offer_routine", {"name": "winddown"})))

        return self._gate(out, env, busy)
