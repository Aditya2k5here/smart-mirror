"""Read-side intelligence: turns raw rows into things worth saying (and showing).

Used by the brain (answers), the proactive engine (briefings, recaps) and the UI snapshot.
"""
from __future__ import annotations

from collections import Counter
from datetime import date, datetime, timedelta
from typing import Any

from assistant.services import Services
from assistant.models import Env
from core.config import CONFIG
from core.timeparse import humanize_when
from memory.goal_service import KG_PER_LB, ema, kg_to_display
from memory.db import known_user

RAIN_WORDS = ("rain", "drizzle", "shower", "storm", "thunder")


def money(n: float) -> str:
    return f"{int(round(n)):,}"


class Reporter:
    def __init__(self, services: Services) -> None:
        self.s = services

    # ---- basics ----------------------------------------------------------
    def units(self, user: str) -> str:
        return (self.s.profile.get_profile(user).get("units") or CONFIG.units) if known_user(user) else CONFIG.units

    def fmt_weight(self, kg: float, user: str, spoken: bool = True) -> str:
        value, unit = kg_to_display(kg, self.units(user))
        if spoken:
            return f"{value:g} {'pounds' if unit == 'lb' else 'kilos'}"
        return f"{value:g} {unit}"

    def weight_kg(self, user: str) -> float | None:
        latest = self.s.goals.latest(user, "weight")
        return latest["value"] if latest else None

    def targets(self, user: str) -> dict[str, Any]:
        profile = self.s.profile.get_profile(user)
        weight = self.weight_kg(user)
        return self.s.nutrition.targets(user, profile, weight, self._goal_rate_kg(user, weight))

    def _goal_rate_kg(self, user: str, weight: float | None) -> float | None:
        goal = self.s.goals.weight_goal(user)
        if not goal or weight is None or goal.get("target_value") is None:
            return None
        remaining = goal["target_value"] - weight
        if abs(remaining) < 0.2:
            return 0.0
        if goal.get("target_date"):
            weeks = max(1.0, (date.fromisoformat(goal["target_date"]) - date.today()).days / 7)
            rate = remaining / weeks
        else:
            rate = -0.5 if remaining < 0 else 0.25
        return max(-weight * 0.01, min(rate, weight * 0.005))

    # ---- daily numbers ---------------------------------------------------
    def day_stats(self, user: str, day: str | None = None) -> dict[str, Any]:
        day = day or date.today().isoformat()
        totals = self.s.nutrition.totals(user, day)
        weight = self.s.goals.one(
            "SELECT value FROM measurements WHERE user = ? AND metric = 'weight' AND day = ? ORDER BY at DESC LIMIT 1",
            (user, day))
        sleep = self.s.goals.one(
            "SELECT value FROM measurements WHERE user = ? AND metric = 'sleep_hours' AND day = ? ORDER BY at DESC LIMIT 1",
            (user, day))
        workouts = self.s.habits.query("SELECT COUNT(*) AS n, COALESCE(SUM(duration_seconds),0) AS s FROM workouts WHERE user = ? AND day = ?", (user, day))[0]
        done_tasks = self.s.memory.query("SELECT COUNT(*) AS n FROM tasks WHERE user = ? AND status = 'done' AND substr(completed_at,1,10) = ?", (user, day))[0]["n"]
        return {
            "day": day, "calories": totals["calories"], "protein_g": totals["protein_g"], "meals": totals["meals"],
            "water_ml": self.s.habits.water_today(user, day), "weight_kg": weight["value"] if weight else None,
            "sleep_hours": sleep["value"] if sleep else None, "workouts": workouts["n"],
            "workout_minutes": round(workouts["s"] / 60), "tasks_done": done_tasks,
        }

    # ---- spoken answers --------------------------------------------------
    def progress_speech(self, user: str) -> str:
        goals = self.s.goals.active_goals(user)
        weight = self.weight_kg(user)
        if not goals:
            if weight is None:
                return "I don't have any weigh-ins or goals yet. Tell me your weight, or say: I want to lose five kilos."
            return (f"You're at {self.fmt_weight(weight, user)}. You haven't set a goal yet. "
                    "Say, I want to lose five kilos by December, and I'll build a plan.")
        parts = []
        for goal in goals[:2]:
            p = self.s.goals.progress(goal, self.units(user))
            line = f"{goal['title']}: {p['headline']}" if p["headline"] else goal["title"]
            if p["percent"] is not None:
                line += f", {int(p['percent'])} percent of the way"
            parts.append(line + ".")
        extra = ""
        goal = self.s.goals.weight_goal(user)
        if goal:
            p = self.s.goals.progress(goal, self.units(user))
            if p["status"] == "stalled":
                extra = " Your trend has flattened. A small tweak usually restarts it: a walk after dinner, or one fewer snack."
            elif p["status"] == "behind":
                extra = " A little extra effort this week gets you back on pace."
            elif p["status"] in {"on_track", "ahead"}:
                extra = " Keep doing exactly what you're doing."
            warn = self.s.goals.safe_pace_warning(user, p["rate_per_week"])
            extra += f" {warn}" if warn else ""
        return " ".join(parts) + extra

    def calories_speech(self, user: str) -> str:
        t = self.targets(user)
        totals = self.s.nutrition.totals(user)
        eaten, target = totals["calories"], t["calories"]
        if totals["meals"] == 0:
            line = f"You haven't logged any food today. Your target is about {money(target)} calories."
        else:
            left = target - eaten
            line = (f"You've had {money(eaten)} calories, {money(left)} left of {money(target)}." if left >= 0
                    else f"You've had {money(eaten)} calories, {money(-left)} over your {money(target)} target.")
            line += f" Protein is {int(totals['protein_g'])} of {t['protein_g']} grams."
        if t["missing"]:
            names = {"weight": "your weight", "height_cm": "your height", "age": "your age", "sex": "whether you're male or female"}
            line += " Tell me " + ", ".join(names[m] for m in t["missing"][:3]) + " and I'll personalise that."
        return line

    def water_speech(self, user: str) -> str:
        t = self.targets(user)
        ml = self.s.habits.water_today(user)
        if ml <= 0:
            return f"No water logged yet. Aim for about {money(t['water_ml'])} millilitres today."
        pct = int(ml / t["water_ml"] * 100)
        return f"{money(ml)} of {money(t['water_ml'])} millilitres so far, {pct} percent."

    def meals_speech(self, user: str) -> str:
        meals = self.s.nutrition.meals_for(user)
        if not meals:
            return "You haven't logged any meals today."
        parts = [f"{m['description'][:40]}, {int(m['calories'])}" for m in meals[-4:]]
        return "Today: " + "; ".join(parts) + f". Total {money(sum(m['calories'] for m in meals))} calories."

    def facts_speech(self, user: str) -> str:
        facts = self.s.profile.list_facts(user, limit=6)
        prof = self.s.profile.get_profile(user)
        bits = []
        if prof.get("height_cm"):
            bits.append(f"you're {int(prof['height_cm'])} centimeters tall")
        if prof.get("age"):
            bits.append(f"{prof['age']} years old")
        bits += [f["text"] for f in facts[:4]]
        if not bits:
            return "Not much yet. Tell me about yourself and I'll remember it."
        n = len(self.s.profile.list_facts(user, limit=500))
        return f"I know {n} thing{'s' if n != 1 else ''} about you. For example: " + "; ".join(bits[:5]) + ". Say forget that, and I'll remove any of it."

    def agenda_speech(self, env: Env) -> str:
        parts = []
        if env.events:
            parts.append("; ".join(f"{e['title']} at {e['time']}" for e in env.events[:3]))
        due = self.s.memory.get_tasks(env.user)
        dated = [t for t in due if t.get("due_at")]
        if dated:
            parts.append("Reminders: " + "; ".join(f"{t['text']} {humanize_when(datetime.fromisoformat(t['due_at']), env.now)}" for t in dated[:3]))
        return ". ".join(parts) + "." if parts else "Nothing scheduled. Your day is open."

    def weather_speech(self, env: Env) -> str:
        w = env.weather or {}
        if w.get("temperature") is None:
            return "I can't see the weather right now. Add a weather key in your settings to enable it."
        line = f"It's {w['temperature']} degrees and {w.get('description', 'clear')}"
        if w.get("high") is not None:
            line += f", with a high of {w['high']}"
        line += "."
        if any(word in str(w.get("description", "")).lower() for word in RAIN_WORDS):
            line += " Take an umbrella."
        elif w["temperature"] >= 32:
            line += " Dress light and drink plenty of water."
        elif w["temperature"] <= 12:
            line += " Layer up."
        return line

    def habits_speech(self, user: str) -> str:
        habits = [h for h in self.s.habits.today(user) if h["name"] != "Water"]
        if not habits:
            return "You aren't tracking any habits yet. Say, track vitamins daily."
        return "; ".join(
            f"{h['name']}: {'done' if h['complete'] else 'not yet'}, streak {h['streak']}" for h in habits[:5]) + "."

    def workouts_speech(self, user: str) -> str:
        w = self.s.habits.workout_summary(user, 7)
        if not w["sessions"]:
            return "No workouts in the last seven days. Say, start a workout, whenever you're ready."
        reps = ", ".join(f"{n} {k.replace('_', ' ')}" for k, n in w["reps"].items() if n)
        return f"{w['sessions']} workout{'s' if w['sessions'] != 1 else ''} on {w['active_days']} days this week" + (f": {reps}." if reps else ".")

    def sleep_speech(self, user: str) -> str:
        avg = self.s.goals.metric_average(user, "sleep_hours", 7)
        if avg is None:
            return "I haven't logged any sleep. Tell me, I slept seven hours, and I'll track it."
        verdict = "that's a healthy range" if avg >= 7 else "a bit short; protect your bedtime tonight" if avg >= 6 else "that's quite low"
        return f"You've averaged {avg:.1f} hours over the last week; {verdict}."

    def mood_speech(self, user: str) -> str:
        sessions = self.s.journal.mood_by_day(user, 7)
        moods = [e["mood"] for e in self.s.journal.entries(user, 7, kind="mood") if e.get("mood")]
        counts = Counter([s["emotion"] for s in sessions] + moods)
        if not counts:
            return "Not enough mood data yet."
        top = ", ".join(f"{k} {v}" for k, v in counts.most_common(3))
        return f"Over the last week, your mood readings were mostly: {top}."

    # ---- weekly review ---------------------------------------------------
    def week_review(self, user: str) -> tuple[str, dict[str, Any]]:
        today = date.today()
        days = [(today - timedelta(days=i)).isoformat() for i in range(6, -1, -1)]
        stats = [self.day_stats(user, d) for d in days]
        t = self.targets(user)
        logged = [s for s in stats if s["meals"]]
        avg_cal = sum(s["calories"] for s in logged) / len(logged) if logged else None
        water_days = [s for s in stats if s["water_ml"]]
        water_pct = (sum(min(1.0, s["water_ml"] / t["water_ml"]) for s in water_days) / len(water_days) * 100) if water_days else None
        series = self.s.goals.series(user, "weight", days=21)
        change = None
        if len(series) >= 2:
            sm = ema([v for _, v in series])
            week_ago = [i for i, (d, _) in enumerate(series) if d <= today - timedelta(days=7)]
            base = sm[week_ago[-1]] if week_ago else sm[0]
            change = sm[-1] - base
        workouts = sum(s["workouts"] for s in stats)
        data = {"weight_change_kg": change, "avg_calories": avg_cal, "days_logged_food": len(logged),
                "workouts": workouts, "water_pct": water_pct, "target_calories": t["calories"]}
        bits = []
        if change is not None:
            word = "down" if change < -0.05 else "up" if change > 0.05 else "flat"
            bits.append("weight is flat" if word == "flat" else f"weight is {word} {self.fmt_weight(abs(change), user)}")
        if avg_cal is not None:
            bits.append(f"you logged food on {len(logged)} days, averaging {money(avg_cal)} against {money(t['calories'])}")
        bits.append(f"{workouts} workout{'s' if workouts != 1 else ''}")
        if water_pct is not None:
            bits.append(f"water averaged {int(water_pct)} percent of target")
        wins = []
        if change is not None and change < -0.1:
            wins.append("the trend is moving the right way")
        if workouts >= 3:
            wins.append("you stayed active")
        if len(logged) >= 5:
            wins.append("you tracked consistently")
        focus = ("log your meals more often" if len(logged) < 4 else "move a little more" if workouts < 2
                 else "drink more water" if (water_pct or 100) < 70 else "keep the rhythm")
        speech = "This week, " + ", ".join(bits) + "."
        speech += f" Best thing: {wins[0]}." if wins else ""
        speech += f" Next week, {focus}."
        return speech, data

    # ---- briefing / recap ------------------------------------------------
    def yesterday_line(self, user: str, now: datetime) -> str:
        day = (now.date() - timedelta(days=1)).isoformat()
        summary = self.s.journal.summary_for(user, day)
        if summary:
            return summary["text"]
        st = self.day_stats(user, day)
        t = self.targets(user)
        bits = []
        if st["calories"]:
            bits.append(f"{money(st['calories'])} calories")
        if st["water_ml"] and st["water_ml"] >= t["water_ml"] * 0.9:
            bits.append("hit your water target")
        if st["workouts"]:
            bits.append("worked out")
        return "Yesterday you " + ", ".join(bits) + "." if bits else ""

    def briefing(self, env: Env, short: bool = False) -> str:
        user, now = env.user, env.now
        name = "" if env.is_guest else f", {user.split()[0]}"
        parts = [f"Good {env.day_part}{name}."]
        if env.weather and env.weather.get("temperature") is not None:
            parts.append(self.weather_speech(env))
        if env.events:
            e = env.events[0]
            line = f"First up: {e['title']} at {e['time']}."
            if len(env.events) > 1 and not short:
                line += f" Then {env.events[1]['title']} at {env.events[1]['time']}."
            parts.append(line)
        else:
            parts.append("Your calendar is clear.")
        if not env.is_guest:
            tasks = self.s.memory.get_tasks(user)
            if tasks:
                parts.append(f"You have {len(tasks)} open reminder{'s' if len(tasks) != 1 else ''}.")
            goal = self.s.goals.weight_goal(user)
            if goal and not short:
                p = self.s.goals.progress(goal, self.units(user))
                if p["headline"] and p["status"] != "no_data":
                    parts.append(p["headline"].capitalize() + ".")
            if not short:
                yline = self.yesterday_line(user, now)
                if yline:
                    parts.append(yline)
                t = self.targets(user)
                if self.weight_kg(user) is not None and t["basis"] == "estimated":
                    parts.append(f"Today's target is about {money(t['calories'])} calories.")
        return " ".join(parts)

    def evening_recap(self, env: Env) -> str:
        user = env.user
        st = self.day_stats(user)
        t = self.targets(user)
        bits = []
        if st["calories"]:
            diff = t["calories"] - st["calories"]
            bits.append(f"{money(st['calories'])} calories, {money(abs(diff))} {'under' if diff >= 0 else 'over'} target")
        if st["water_ml"]:
            bits.append(f"{int(st['water_ml'] / t['water_ml'] * 100)} percent of your water")
        if st["workouts"]:
            bits.append(f"{st['workouts']} workout")
        if st["tasks_done"]:
            bits.append(f"{st['tasks_done']} task{'s' if st['tasks_done'] != 1 else ''} finished")
        return ("Today you logged " + ", ".join(bits) + ".") if bits else ""

    # ---- LLM dossier -----------------------------------------------------
    def dossier(self, env: Env, query: str = "") -> str:
        user, now = env.user, env.now
        lines = [f"Now: {now.strftime('%A %d %B %Y, %H:%M')} ({env.day_part})."]
        if env.weather and env.weather.get("temperature") is not None:
            lines.append(f"Weather: {self.weather_speech(env)}")
        if env.events:
            lines.append("Calendar next: " + "; ".join(f"{e['title']} at {e['time']}" for e in env.events[:3]))
        if env.track and env.track.get("title"):
            lines.append(f"Music playing: {env.track['title']} by {env.track.get('artist', '')}.")
        if env.is_guest:
            lines.append("The person in front of the mirror is not recognised yet (guest).")
            return "\n".join(lines)

        prof = self.s.profile.get_profile(user)
        who = [f"Name: {user}"]
        for key, label in (("age", "age"), ("sex", "sex"), ("height_cm", "height cm"), ("activity_level", "activity")):
            if prof.get(key):
                who.append(f"{label} {prof[key]}")
        if prof.get("visit_count"):
            who.append(f"seen {prof['visit_count']} times")
        lines.append("Person: " + ", ".join(who) + ".")
        lines.append(f"Looks {env.emotion} on camera; posture {env.posture}.")

        weight = self.weight_kg(user)
        if weight is not None:
            lines.append(f"Latest weight: {self.fmt_weight(weight, user, spoken=False)}.")
        for goal in self.s.goals.active_goals(user)[:3]:
            p = self.s.goals.progress(goal, self.units(user))
            lines.append(f"Goal: {goal['title']} - {p['headline'] or p['status']} ({p['status']})"
                         + (f", by {goal['target_date']}" if goal.get("target_date") else "") + ".")
        st = self.day_stats(user)
        t = self.targets(user)
        lines.append(f"Today so far: {money(st['calories'])}/{money(t['calories'])} kcal, protein {int(st['protein_g'])}/{t['protein_g']} g, "
                     f"water {int(st['water_ml'])}/{t['water_ml']} ml, workouts {st['workouts']}.")
        if t["missing"]:
            lines.append("Missing profile data for calorie planning: " + ", ".join(t["missing"]) + ".")
        days_since = self.s.habits.days_since_workout(user)
        if days_since is not None:
            lines.append(f"Last workout {days_since} day(s) ago.")
        sleep = self.s.goals.metric_average(user, "sleep_hours", 7)
        if sleep:
            lines.append(f"Average sleep this week: {sleep:.1f} h.")
        tasks = self.s.memory.get_tasks(user)
        if tasks:
            lines.append("Open reminders: " + "; ".join(t_["text"] for t_ in tasks[:4]) + ".")
        yline = self.yesterday_line(user, now)
        if yline:
            lines.append("Yesterday: " + yline)
        recalled = self.s.profile.recall(user, query, limit=7)
        if recalled:
            lines.append("Things you know about them: " + " | ".join(recalled))
        return "\n".join(lines)

    # ---- UI snapshot -----------------------------------------------------
    def ui_snapshot(self, user: str) -> dict[str, Any]:
        if not known_user(user):
            return {}
        units = self.units(user)
        goals_ui = []
        for goal in self.s.goals.active_goals(user)[:3]:
            p = self.s.goals.progress(goal, units)

            def disp(v):
                return None if v is None else (kg_to_display(v, units)[0] if goal["metric"] == "weight" else round(v, 1))

            goals_ui.append({
                "id": goal["id"], "title": goal["title"], "metric": goal["metric"], "status": p["status"],
                "percent": p["percent"], "headline": p["headline"], "eta": p["eta"], "target_date": goal.get("target_date"),
                "current": disp(p["current"]), "target": disp(p["target"]), "start": disp(p["start"]),
                "unit": kg_to_display(1, units)[1] if goal["metric"] == "weight" else p["unit"],
                "rate": None if p["rate_per_week"] is None else round(disp(abs(p["rate_per_week"])) * (1 if p["rate_per_week"] >= 0 else -1), 2),
                "series": [{"day": pt["day"], "value": disp(pt["value"]), "trend": disp(pt["trend"])} for pt in p["series"]],
            })
        t = self.targets(user)
        st = self.day_stats(user)
        habits = [h for h in self.s.habits.today(user) if h["name"] != "Water"]
        return {
            "goals": goals_ui,
            "today": {
                "calories": {"eaten": round(st["calories"]), "target": t["calories"], "protein": round(st["protein_g"]),
                             "protein_target": t["protein_g"], "estimated": t["basis"] == "estimated"},
                "water": {"ml": round(st["water_ml"]), "target": t["water_ml"]},
                "workouts": st["workouts"], "workout_minutes": st["workout_minutes"],
                "meals": [{"text": m["description"], "calories": round(m["calories"]), "type": m["meal_type"]}
                          for m in self.s.nutrition.meals_for(user)][-5:],
            },
            "habits": habits[:5],
            "streaks": {"weigh_in": self.s.goals.logging_streak(user)},
            "facts": [f["text"] for f in self.s.profile.list_facts(user, limit=8)],
            "units": units,
        }
