"""The mirror's mind: routes an utterance to a pending answer, a fast local intent,
or the LLM (with full memory context), and returns what to say and do."""
from __future__ import annotations

from collections import deque
from datetime import date, datetime, timedelta
from typing import Any, Callable
import json
import logging
import random
import re
import time

from assistant.intents import Intent, ParseContext, extract_name, parse_intent, clean
from assistant.reports import Reporter, money
from assistant.routines import BREATHING, ROUTINES
from assistant.services import Services
from assistant.models import Env, Pending, Reply
from core.config import CONFIG
from core.timeparse import format_duration, humanize_when, parse_when
from memory.foods import estimate_meal, infer_meal_type
from memory.goal_service import KG_PER_LB, kg_to_display
from memory.profile_service import extract_facts

# Intents that need to know who is talking.
PERSONAL = {
    "log_weight", "set_weight_goal", "set_goal", "update_profile", "log_water", "log_meal", "fix_meal_calories",
    "undo_meal", "add_reminder", "complete_task", "list_tasks", "add_note", "show_list", "clear_list", "remember",
    "log_mood", "log_sleep", "add_habit", "log_habit", "remove_habit", "start_journal", "journal", "forget_all",
    "forget_fact",
}
PERSONAL_TOPICS = {"progress", "calories", "water", "meals", "week", "facts", "habits", "sleep", "mood", "workouts"}

LLM_ACTIONS = {
    "log_weight", "set_weight_goal", "set_goal", "log_water", "log_meal", "add_reminder", "set_timer", "add_note",
    "start_routine", "start_breathing", "start_workout", "update_profile", "log_mood", "log_sleep", "music", "lights",
    "remember", "show", "add_habit", "log_habit",
}

MOOD_REPLIES = {
    "stressed": "That sounds heavy. Want to take a minute and breathe together?",
    "anxious": "I'm here. Want to do a minute of slow breathing with me?",
    "overwhelmed": "One thing at a time. Want a calm minute first?",
    "tired": "Be gentle with yourself today. Water and an early night will help.",
    "sad": "I'm sorry. I'm right here if you want to talk about it.",
    "lonely": "I'm glad you told me. I'm here, and happy to chat as long as you like.",
    "angry": "That's allowed. Want to breathe it out for a minute?",
    "frustrated": "Ugh, that's rough. A short breather can help. Want one?",
    "bored": "Want a stretch, some music, or a quick challenge?",
    "happy": "Love to hear it. What's making today good?",
    "excited": "That energy is great. What's happening?",
    "motivated": "Perfect. Let's use it. Want to start a workout?",
    "proud": "You should be. Tell me what you did.",
    "calm": "Lovely. Hold onto that.",
    "unwell": "I'm sorry you're not feeling well. Rest and fluids, and see a doctor if it persists.",
    "okay": "Okay is fine. Anything I can help with?",
}

JOKES = [
    "Why did the mirror go to therapy? It had too many reflections.",
    "I told my reflection a joke. It couldn't stop laughing along.",
    "Seven days without a workout makes one weak. Sorry, I had to.",
    "I'd tell you a joke about the gym, but it's a bit of a stretch.",
]

SENSITIVE = re.compile(r"\b(?:kill myself|end my life|suicid|want to die|self[- ]harm|hurt myself|no reason to live)\b", re.I)


class Brain:
    def __init__(self, services: Services, reporter: Reporter | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.s = services
        self.r = reporter or Reporter(services)
        self._clock = clock
        self.pending: Pending | None = None
        self.history: deque[tuple[str, str, str, float]] = deque(maxlen=16)  # user, role, text, t
        self._offline_hinted = False

    # ------------------------------------------------------------------ API
    def reset_session(self) -> None:
        self.pending = None
        self.history.clear()

    def set_pending(self, kind: str, data: dict[str, Any] | None = None, ttl: float = 45.0) -> None:
        self.pending = Pending(kind, data or {}, self._clock() + ttl)

    def handle(self, text: str, env: Env) -> Reply:
        text = " ".join((text or "").split())
        if not text:
            return Reply(handled=False)
        self.s.journal.add_turn(env.user, "user", text, env.now)
        reply = self._route(text, env)
        reply.speech = self._polish(reply.speech)
        reply.keep_open = reply.keep_open or reply.speech.endswith("?") or self.pending is not None
        self.history.append((env.user, "user", text, self._clock()))
        if reply.speech:
            self.history.append((env.user, "assistant", reply.speech, self._clock()))
            self.s.journal.add_turn(env.user, "assistant", reply.speech)
        self._learn_from(text, env)
        return reply

    # ---------------------------------------------------------------- routing
    def _route(self, text: str, env: Env) -> Reply:
        if SENSITIVE.search(text):
            return Reply(
                speech="I'm really glad you told me. You matter. Please reach out to someone you trust right now, "
                       "or call your local crisis line. If you might act on this, call emergency services. I'm here with you.",
                source="local")

        if self.pending and self._clock() > self.pending.expires_at:
            self.pending = None
        if self.pending:
            handled = self._handle_pending(text, env)
            if handled:
                return handled
            self.pending = None

        intent = parse_intent(text, ParseContext(
            now=env.now, units=self.r.units(env.user), routine_active=self.s.routines.active,
            breathing_active=self.s.breathing.active,
            timer_ringing=bool(self.s.timers.ringing()),
            is_guest=env.is_guest))
        if intent:
            reply = self._execute(intent, env)
            if reply and reply.handled:
                if reply.passthrough:
                    llm = self._llm(text, env, note=f"They just told you how they feel ({intent.params.get('mood', '')}). "
                                                     "It is already logged. React warmly in one or two sentences; offer breathing if they seem stressed.")
                    if llm:
                        llm.merge(reply)
                        return llm
                return reply

        llm = self._llm(text, env)
        if llm:
            return llm
        return self._offline(text, env)

    # --------------------------------------------------------------- execution
    def _execute(self, intent: Intent, env: Env) -> Reply | None:
        if env.is_guest:
            personal = intent.name in PERSONAL or (intent.name == "query" and intent.params.get("topic") in PERSONAL_TOPICS)
            if personal:
                self.set_pending("enroll_name", {"retry": intent.raw}, ttl=40)
                return Reply(speech="I don't think we've met yet. What should I call you?", keep_open=True)
        handler = getattr(self, f"_do_{intent.name}", None)
        if not handler:
            return None
        try:
            return handler(intent.params, env)
        except Exception:
            logging.exception("Intent %s failed", intent.name)
            return Reply(speech="Sorry, something went wrong with that.", source="local")

    def _first(self, env: Env) -> str:
        return "" if env.is_guest else env.user.split()[0]

    # ------------------------------------------------------- identity & privacy
    def _do_set_name(self, p: dict, env: Env) -> Reply:
        name = p["name"]
        if not env.is_guest:
            if name.lower() == env.user.lower():
                return Reply(speech=f"That's what I have you down as, {self._first(env)}.")
            return Reply(speech=f"I have you as {env.user}. If that's wrong, say: that's not me.")
        profile = self.s.profile.ensure_profile(name)
        self.s.memory.log_event(name, "USER_ENROLLED", {"name": name})
        self.set_pending("onboard_goal", {"name": name}, ttl=90)
        return Reply(
            speech=f"Nice to meet you, {name.split()[0]}. Hold still and look at me for a moment so I can learn your face. "
                   "Then tell me one thing you'd like help with. Maybe a goal, like losing weight, or just say skip.",
            effects=[{"type": "enroll_face", "name": name}], keep_open=True)

    def _do_not_me(self, p: dict, env: Env) -> Reply:
        self.set_pending("enroll_name", {}, ttl=40)
        return Reply(speech="Sorry about that. Who am I talking to?", effects=[{"type": "reset_identity"}], keep_open=True)

    def _do_forget_all(self, p: dict, env: Env) -> Reply:
        self.set_pending("confirm_forget", {}, ttl=40)
        return Reply(speech=f"That permanently erases everything I know about {self._first(env)}, including your face. "
                            "Say: yes, delete everything, to confirm.", keep_open=True)

    def _do_forget_fact(self, p: dict, env: Env) -> Reply:
        removed = self.s.profile.forget_fact(env.user, p["phrase"])
        return Reply(speech="Forgotten." if removed else "I don't have anything like that stored.")

    def _do_privacy(self, p: dict, env: Env) -> Reply:
        on = bool(p["on"])
        return Reply(speech="Privacy mode on. The camera is off until you say otherwise." if on else "Camera is back on.",
                     effects=[{"type": "privacy", "on": on}])

    def _do_stop_speaking(self, p: dict, env: Env) -> Reply:
        return Reply(effects=[{"type": "stop_speech"}])

    def _do_timer_dismiss(self, p: dict, env: Env) -> Reply:
        return Reply(effects=[{"type": "stop_alarm"}])

    # ------------------------------------------------------------ weight & goals
    def _do_log_weight(self, p: dict, env: Env) -> Reply:
        user, kg = env.user, float(p["value_kg"])
        prev = self.s.goals.latest(user, "weight")
        self.s.goals.add_measurement(user, "weight", kg, "kg")
        fmt = lambda v: self.r.fmt_weight(v, user)
        parts = [f"Logged {fmt(kg)}."]
        if prev:
            diff = kg - prev["value"]
            when = date.fromisoformat(prev["day"])
            ago = (env.now.date() - when).days
            label = "earlier today" if ago == 0 else "yesterday" if ago == 1 else f"{ago} days ago"
            if abs(diff) < 0.05:
                parts.append(f"Same as {label}.")
            else:
                parts.append(f"{'Down' if diff < 0 else 'Up'} {fmt(abs(diff))} from {label}.")
        else:
            parts.append("That's your starting point.")
        reply = Reply(ui={"focus": "progress", "seconds": 14})
        goal = self.s.goals.weight_goal(user)
        if goal:
            progress = self.s.goals.progress(goal, self.r.units(user))
            if progress["percent"] is not None and progress["status"] not in ("no_data",):
                parts.append(f"{int(progress['percent'])} percent of the way to your goal.")
            milestone = self.s.goals.pending_milestone(goal, progress["percent"])
            if milestone == 100:
                parts.append("You reached your goal. That is a big deal.")
                reply.celebrate = "Goal reached"
            elif milestone:
                parts.append(f"You just passed {milestone} percent. Well done.")
                reply.celebrate = f"{milestone}% there"
            elif prev and kg - prev["value"] > 0.8 and goal.get("direction") == "down":
                parts.append("Daily swings of a kilo are normal, so trust the trend line.")
            warn = self.s.goals.safe_pace_warning(user, progress["rate_per_week"])
            if warn:
                parts.append(warn)
        elif not self.s.journal.nudge_fired(user, "goal_prompt"):
            self.s.journal.mark_nudge(user, "goal_prompt")
            parts.append("Want a target? Say: I want to lose five kilos by December.")
        reply.speech = " ".join(parts)
        return reply

    def _pace_text(self, start: float, target: float, by: date | None) -> str:
        diff = target - start
        if not by:
            return ""
        weeks = max(1.0, (by - date.today()).days / 7)
        per_week = abs(diff) / weeks
        pct = per_week / start * 100
        if pct > 1.0:
            return f" That's {per_week:.1f} kilos a week, which is faster than I'd recommend. Consider a later date."
        if pct > 0.75:
            return f" That's about {per_week:.1f} kilos a week. Ambitious but possible."
        return f" That's about {per_week:.1f} kilos a week. Very doable."

    def _do_set_weight_goal(self, p: dict, env: Env) -> Reply:
        user = env.user
        by = date.fromisoformat(p["by"]) if p.get("by") else None
        latest = self.s.goals.latest(user, "weight")
        start = latest["value"] if latest else None
        target = p.get("target_kg")
        if target is None and p.get("delta_kg") is not None and start is not None:
            target = start + (p["delta_kg"] if p.get("direction") == "up" else -p["delta_kg"])
        # Safety: refuse targets that are underweight for the person's height.
        height = self.s.profile.get_profile(user).get("height_cm")
        if target and height:
            floor = 18.5 * (height / 100) ** 2
            if target < floor:
                return Reply(speech=f"For your height, {self.r.fmt_weight(target, user)} would be underweight. "
                                    f"Let's aim at {self.r.fmt_weight(round(floor + 2, 1), user)} or above instead. Want that?",
                             override=True)
        goal = self.s.goals.create_weight_goal(user, target_kg=target, delta_kg=p.get("delta_kg"), by=by,
                                               direction=p.get("direction", ""))
        if not goal:
            return Reply(speech="Tell me the weight you want to reach, like: I want to weigh 70 kilos.")
        self.s.profile.add_fact(user, f"is working on a goal: {goal['title'].lower()}", "goal", importance=0.9)
        deadline = f" by {by.strftime('%B')} {by.day}" if by else ""
        if start is None:
            self.set_pending("weigh_in", {}, ttl=120)
            return Reply(speech=f"Goal set: {goal['title'].lower()}{deadline}. Step on the scale and tell me your weight "
                                "so I can lock in the starting line.", keep_open=True, ui={"focus": "progress", "seconds": 12})
        speech = f"Goal set: {goal['title'].lower()}{deadline}." + self._pace_text(start, goal["target_value"], by)
        t = self.r.targets(user)
        if t["basis"] == "estimated":
            speech += f" I'll aim you at about {money(t['calories'])} calories a day."
        else:
            speech += " Tell me your height, age, and whether you're male or female, and I'll work out a calorie target."
        return Reply(speech=speech, ui={"focus": "progress", "seconds": 14})

    def _do_set_goal(self, p: dict, env: Env) -> Reply:
        by = date.fromisoformat(p["by"]) if p.get("by") else None
        goal = self.s.goals.create_goal(env.user, p["title"], kind="custom", target_date=by)
        if not goal:
            return Reply(speech="I couldn't save that goal.")
        self.s.profile.add_fact(env.user, f"has a goal: {p['title'].lower()}", "goal", importance=0.85)
        return Reply(speech=f"Goal saved: {p['title']}. I'll keep it in mind and check in on it.")

    def _do_update_profile(self, p: dict, env: Env) -> Reply:
        user = env.user
        profile = self.s.profile.update_profile(user, **p)
        said = []
        if "height_cm" in p:
            said.append(f"{int(p['height_cm'])} centimeters")
        if "age" in p:
            said.append(f"{p['age']} years old")
        if "sex" in p:
            said.append(p["sex"])
        if "activity_level" in p:
            said.append(f"{p['activity_level']} activity")
        if "bed_time" in p:
            said.append(f"bedtime around {p['bed_time']}")
            self.s.profile.add_fact(user, f"usually goes to bed around {p['bed_time']}", "routine")
        if "wake_time" in p:
            said.append(f"wake-up around {p['wake_time']}")
            self.s.profile.add_fact(user, f"usually wakes up around {p['wake_time']}", "routine")
        if "units" in p:
            said.append("pounds" if p["units"] == "imperial" else "kilograms")
        speech = "Got it: " + ", ".join(said) + "." if said else "Updated."
        t = self.r.targets(user)
        if t["basis"] == "estimated" and any(k in p for k in ("height_cm", "age", "sex", "activity_level")):
            speech += f" Your daily target works out to about {money(t['calories'])} calories and {t['protein_g']} grams of protein."
        elif t["missing"] and any(k in p for k in ("height_cm", "age", "sex", "activity_level")):
            names = {"weight": "your weight", "height_cm": "your height", "age": "your age", "sex": "male or female"}
            speech += " Still missing: " + ", ".join(names[m] for m in t["missing"][:3]) + "."
        return Reply(speech=speech)

    # ----------------------------------------------------------- water & meals
    def _do_log_water(self, p: dict, env: Env) -> Reply:
        t = self.r.targets(env.user)
        result = self.s.habits.log_water(env.user, p["ml"], t["water_ml"])
        total = result["total_ml"] if result else p["ml"]
        speech = f"Nice. {money(total)} of {money(t['water_ml'])} millilitres."
        reply = Reply(ui={"focus": "nutrition", "seconds": 8})
        if total >= t["water_ml"] and total - p["ml"] < t["water_ml"]:
            speech = "That's your water target for today. Brilliant."
            reply.celebrate = "Hydrated"
        reply.speech = speech
        return reply

    def _log_estimated_meal(self, env: Env, description: str, est: dict[str, Any]) -> Reply:
        names = ", ".join((f"{i['qty']:g} {i['name']}" if i["qty"] != 1 else i["name"]) for i in est["items"])
        meal_type = infer_meal_type(description, env.now)
        self.s.nutrition.log_meal(env.user, names, est["calories"], est["protein_g"], est["carbs_g"], est["fat_g"],
                                  meal_type=meal_type, estimated=True, at=env.now)
        return self._meal_logged_reply(env, names, est["calories"], est.get("unmatched", []))

    def _meal_logged_reply(self, env: Env, label: str, calories: float, unmatched: list[str] | None = None) -> Reply:
        t = self.r.targets(env.user)
        totals = self.s.nutrition.totals(env.user)
        left = t["calories"] - totals["calories"]
        speech = f"Logged {label}, about {money(calories)} calories. "
        speech += f"{money(left)} left today." if left >= 0 else f"That puts you {money(-left)} over today's target."
        if unmatched:
            speech += f" I didn't recognise {unmatched[0]}."
        return Reply(speech=speech, ui={"focus": "nutrition", "seconds": 10})

    def _do_log_meal(self, p: dict, env: Env) -> Reply:
        description = p.get("description", "meal")
        if p.get("calories") is not None:
            cal = float(p["calories"])
            label = re.sub(r"^(?:i |we )?(?:just )?(?:had|ate|log(?:ged)?|have had)\s+", "", description).strip() or "your meal"
            self.s.nutrition.log_meal(env.user, label, cal, float(p.get("protein_g") or 0), meal_type=infer_meal_type(description, env.now),
                                      estimated=False, at=env.now)
            return self._meal_logged_reply(env, label, cal)
        est = estimate_meal(description)
        if est["items"]:
            return self._log_estimated_meal(env, description, est)
        guess = self._llm_estimate_meal(description)
        if guess:
            self.s.nutrition.log_meal(env.user, guess["description"], guess["calories"], guess.get("protein_g", 0),
                                      meal_type=infer_meal_type(description, env.now), estimated=True, at=env.now)
            return self._meal_logged_reply(env, guess["description"], guess["calories"])
        self.set_pending("meal_calories", {"description": description}, ttl=40)
        return Reply(speech="I don't know that one. Roughly how many calories?", keep_open=True)

    def _do_fix_meal_calories(self, p: dict, env: Env) -> Reply:
        meal = self.s.nutrition.update_last_meal(env.user, p["calories"])
        return Reply(speech=f"Updated to {money(p['calories'])} calories." if meal else "I don't have a meal to fix.")

    def _do_undo_meal(self, p: dict, env: Env) -> Reply:
        meal = self.s.nutrition.undo_last_meal(env.user)
        return Reply(speech=f"Removed {meal['description'][:40]}." if meal else "No meals to remove.")

    # ------------------------------------------------------------------ queries
    def _do_query(self, p: dict, env: Env) -> Reply:
        topic, user = p["topic"], env.user
        ui: dict[str, Any] = {}
        if topic == "progress":
            speech, ui = self.r.progress_speech(user), {"focus": "progress", "seconds": 18}
        elif topic == "calories":
            speech, ui = self.r.calories_speech(user), {"focus": "nutrition", "seconds": 14}
        elif topic == "water":
            speech, ui = self.r.water_speech(user), {"focus": "nutrition", "seconds": 10}
        elif topic == "meals":
            speech, ui = self.r.meals_speech(user), {"focus": "nutrition", "seconds": 12}
        elif topic == "week":
            speech, data = self.r.week_review(user)
            ui = {"focus": "week", "seconds": 20, "week": data}
        elif topic == "facts":
            speech, ui = self.r.facts_speech(user), {"focus": "memory", "seconds": 20}
        elif topic == "agenda":
            speech = self.r.agenda_speech(env)
        elif topic == "time":
            speech = "It's " + env.now.strftime("%I:%M %p").lstrip("0") + "."
        elif topic == "date":
            speech = "It's " + env.now.strftime("%A, %B ") + str(env.now.day) + "."
        elif topic == "weather":
            speech = self.r.weather_speech(env)
        elif topic == "habits":
            speech = self.r.habits_speech(user)
        elif topic == "workouts":
            speech = self.r.workouts_speech(user)
        elif topic == "sleep":
            speech = self.r.sleep_speech(user)
        elif topic == "mood":
            speech = self.r.mood_speech(user)
        elif topic == "help":
            speech = ("I can track your weight and goals, log meals and water, run timers and reminders, guide routines "
                      "like brushing and skincare, count workout reps, check your outfit, control music and lights, "
                      "and remember what matters to you. Just say hey mirror and talk to me.")
        else:
            return Reply(handled=False)
        return Reply(speech=speech, ui=ui)

    def _do_briefing(self, p: dict, env: Env) -> Reply:
        return Reply(speech=self.r.briefing(env))

    # ------------------------------------------------------------ timers, tasks
    def _do_set_timer(self, p: dict, env: Env) -> Reply:
        t = self.s.timers.start(int(p["seconds"]), p.get("label", ""))
        return Reply(speech=f"Timer set for {format_duration(t.total)}" + (f", {t.label}." if p.get("label") else "."))

    def _do_start_focus(self, p: dict, env: Env) -> Reply:
        minutes = int(p.get("minutes", 25))
        self.s.timers.start(minutes * 60, "Focus", kind="focus")
        return Reply(speech=f"Focus session started: {minutes} minutes. I'll tell you when to take a break.")

    def _do_cancel_timer(self, p: dict, env: Env) -> Reply:
        removed = self.s.timers.cancel(p.get("label", ""))
        return Reply(speech="Timer cancelled." if len(removed) == 1 else f"Cancelled {len(removed)} timers." if removed else "You have no timers.")

    def _do_timer_status(self, p: dict, env: Env) -> Reply:
        return Reply(speech=self.s.timers.status_text())

    def _do_add_reminder(self, p: dict, env: Env) -> Reply:
        when = datetime.fromisoformat(p["when"]) if p.get("when") else None
        text = (p.get("text") or "").strip()
        if not text:
            self.set_pending("reminder_text", {"when": p.get("when")}, ttl=30)
            return Reply(speech="What should I remind you about?", keep_open=True)
        self.s.memory.add_task(env.user, text, due_at=when)
        if when:
            return Reply(speech=f"Okay. I'll remind you to {text[0].lower() + text[1:]} {humanize_when(when, env.now)}.")
        return Reply(speech=f"Added to your list: {text}.")

    def _do_complete_task(self, p: dict, env: Env) -> Reply:
        done = self.s.memory.complete_latest_task(env.user)
        return Reply(speech=f"Done: {done}." if done else "No pending tasks.", effects=[{"type": "tasks_changed"}])

    def _do_list_tasks(self, p: dict, env: Env) -> Reply:
        tasks = self.s.memory.get_tasks(env.user)
        if not tasks:
            return Reply(speech="Nothing on your list.")
        shown = "; ".join(t["text"] for t in tasks[:3])
        return Reply(speech=f"You have {len(tasks)}: {shown}." + (" And more." if len(tasks) > 3 else ""))

    # ------------------------------------------------------- routines & wellbeing
    def _do_start_routine(self, p: dict, env: Env) -> Reply:
        key = p.get("name", "morning")
        if key not in ROUTINES:
            key = "morning"
        self.s.breathing.stop()
        event = self.s.routines.start(key)
        return Reply(speech=f"{ROUTINES[key]['title']}. {event.text}", ui={"focus": "routine", "seconds": 0})

    def _do_routine_next(self, p: dict, env: Env) -> Reply:
        event = self.s.routines.advance()
        return Reply(speech=event.text)

    def _do_routine_stop(self, p: dict, env: Env) -> Reply:
        self.s.routines.stop()
        return Reply(speech="Routine stopped.")

    def _do_start_breathing(self, p: dict, env: Env) -> Reply:
        pattern = p.get("pattern", "calm")
        self.s.routines.stop()
        self.s.breathing.start(pattern, minutes=1.5)
        return Reply(speech="Let's slow down together. Breathe in as the circle grows, out as it shrinks.")

    def _do_breathing_stop(self, p: dict, env: Env) -> Reply:
        self.s.breathing.stop()
        return Reply(speech="Good. How do you feel?")

    def _do_start_workout(self, p: dict, env: Env) -> Reply:
        return Reply(speech="", effects=[{"type": "start_workout", "exercise": p.get("exercise", "curls")}])

    def _do_stop_workout(self, p: dict, env: Env) -> Reply:
        return Reply(speech="", effects=[{"type": "stop_workout"}])

    def _do_log_mood(self, p: dict, env: Env) -> Reply:
        mood = p["mood"]
        self.s.journal.add_entry(env.user, f"Felt {mood}.", mood=mood, kind="mood")
        if mood in ("stressed", "anxious", "overwhelmed", "angry", "frustrated"):
            self.set_pending("offer_breathing", {}, ttl=30)
        return Reply(speech=MOOD_REPLIES.get(mood, "Thanks for telling me."), passthrough=True, keep_open=True)

    def _do_log_sleep(self, p: dict, env: Env) -> Reply:
        hours = float(p["hours"])
        self.s.goals.add_measurement(env.user, "sleep_hours", hours, "h")
        verdict = ("Solid." if hours >= 7 else "A bit short. Try for an earlier night." if hours >= 6
                   else "That's low. Go easy on yourself today and protect tonight's sleep.")
        return Reply(speech=f"Logged {hours:g} hours. {verdict}")

    def _do_goodnight(self, p: dict, env: Env) -> Reply:
        parts = [f"Good night{', ' + self._first(env) if self._first(env) else ''}."]
        if not env.is_guest:
            upcoming = [t for t in self.s.memory.get_tasks(env.user) if t.get("due_at")]
            tomorrow = (env.now + timedelta(days=1)).date().isoformat()
            first = next((t for t in upcoming if t["due_at"][:10] == tomorrow), None)
            if first:
                parts.append(f"Tomorrow: {first['text']} {humanize_when(datetime.fromisoformat(first['due_at']), env.now)}.")
            recap = self.r.evening_recap(env)
            if recap:
                parts.append(recap)
        parts.append("Sleep well.")
        return Reply(speech=" ".join(parts), effects=[{"type": "night_mode"}, {"type": "reflect"}])

    # ---------------------------------------------------------------- habits
    def _do_add_habit(self, p: dict, env: Env) -> Reply:
        habit = self.s.habits.ensure_habit(env.user, p["name"], "", float(p.get("target", 1)), p.get("remind_times", ""))
        if not habit:
            return Reply(speech="I couldn't save that habit.")
        if p.get("remind_times"):
            return Reply(speech=f"Okay. I'll remind you to {p['name'].lower()} every day around {p['remind_times']}.")
        return Reply(speech=f"Tracking {p['name'].lower()} daily. Tell me when you've done it.")

    def _do_log_habit(self, p: dict, env: Env) -> Reply | None:
        habit = self.s.habits.find_habit(env.user, p["name"])
        if not habit:
            return None  # not one of ours, let the conversation continue
        self.s.habits.log(env.user, habit["id"], 1)
        streak = self.s.habits.streak(habit)
        return Reply(speech=f"{habit['name']} done." + (f" That's {streak} days in a row." if streak >= 2 else ""))

    def _do_remove_habit(self, p: dict, env: Env) -> Reply:
        return Reply(speech="Stopped tracking it." if self.s.habits.remove_habit(env.user, p["name"]) else "I'm not tracking that.")

    # ---------------------------------------------------------------- notes
    def _do_add_note(self, p: dict, env: Env) -> Reply:
        self.s.journal.add_note(env.user, p["text"], p.get("list", "notes"))
        name = p.get("list", "notes")
        return Reply(speech=f"Added {p['text']} to your {name} list." if name != "notes" else "Noted.")

    def _do_show_list(self, p: dict, env: Env) -> Reply:
        items = self.s.journal.list_notes(env.user, p["list"])
        if not items:
            return Reply(speech=f"Your {p['list']} list is empty.")
        return Reply(speech=f"On your {p['list']} list: " + ", ".join(i["text"] for i in items[:8]) + ".", ui={"focus": "notes", "seconds": 15})

    def _do_clear_list(self, p: dict, env: Env) -> Reply:
        self.s.journal.clear_list(env.user, p["list"])
        return Reply(speech=f"Cleared your {p['list']} list.")

    def _do_remember(self, p: dict, env: Env) -> Reply:
        self.s.profile.add_fact(env.user, p["text"], "other", source="told", importance=0.8)
        return Reply(speech="I'll remember that.")

    def _do_start_journal(self, p: dict, env: Env) -> Reply:
        self.set_pending("journal", {}, ttl=120)
        return Reply(speech="I'm listening. How was your day?", keep_open=True)

    def _do_journal(self, p: dict, env: Env) -> Reply:
        return self._save_journal(p["text"], env)

    def _save_journal(self, text: str, env: Env) -> Reply:
        self.s.journal.add_entry(env.user, text, kind="entry")
        for category, fact in extract_facts(text):
            self.s.profile.add_fact(env.user, fact, category, source="journal")
        llm = self._llm(text, env, note="They just journaled about their day. Respond like a caring friend in two short sentences: reflect one specific thing, then one gentle forward-looking line.")
        if llm:
            return llm
        return Reply(speech="Thank you for telling me. I've saved that to your journal.")

    # -------------------------------------------------------- hardware effects
    def _do_music(self, p: dict, env: Env) -> Reply:
        return Reply(effects=[{"type": "music", **p}])

    def _do_lights(self, p: dict, env: Env) -> Reply:
        return Reply(effects=[{"type": "lights", **p}])

    def _do_outfit_check(self, p: dict, env: Env) -> Reply:
        if env.privacy:
            return Reply(speech="Privacy mode is on, so I can't look. Say, privacy mode off, first.")
        return Reply(speech="Let me take a proper look.", effects=[{"type": "vision", "kind": "outfit", "question": p.get("question", "")}])

    def _do_show(self, p: dict, env: Env) -> Reply:
        panel, visible = p["panel"], p.get("visible", True)
        if panel in ("settings", "diagnostics"):
            return Reply(speech=f"{panel.title()} {'shown' if visible else 'hidden'}.", effects=[{"type": "panel", "panel": panel, "visible": visible}])
        if panel == "none":
            return Reply(ui={"focus": "none"})
        ui: dict[str, Any] = {"focus": panel, "seconds": 20}
        if panel == "week":
            _, ui["week"] = self.r.week_review(env.user)
        return Reply(speech="", ui=ui)

    # ------------------------------------------------------------- pending flow
    def _handle_pending(self, text: str, env: Env) -> Reply | None:
        pending = self.pending
        assert pending is not None
        t = clean(text)
        yes = bool(re.fullmatch(r"(?:yes|yeah|yep|yup|sure|ok|okay|please|do it|go ahead|let's do it|why not|definitely)", t))
        no = bool(re.fullmatch(r"(?:no|nope|nah|not now|no thanks|skip|never ?mind|later|cancel|not really)", t))

        if pending.kind == "enroll_name":
            if no:
                self.pending = None
                return Reply(speech="No problem. I'll stay here if you need me.")
            other = parse_intent(text, ParseContext(now=env.now, units=self.r.units(env.user)))
            if other and other.name != "set_name":
                return None  # they are giving a command, not a name
            name = extract_name(text, strict=False)
            if name:
                self.pending = None
                reply = self._do_set_name({"name": name}, env)
                reply.keep_open = True
                return reply
            return None

        if pending.kind == "confirm_forget":
            self.pending = None
            if re.search(r"\byes\b.*\b(?:delete|forget|erase|everything)\b|\bdelete everything\b|\bconfirm\b", t):
                user = env.user
                self.s.profile.delete_user(user)
                return Reply(speech="Done. I've erased everything about you.", effects=[{"type": "delete_user", "name": user}])
            return Reply(speech="Okay, I've kept everything.")

        if pending.kind == "offer_breathing":
            self.pending = None
            if yes:
                return self._do_start_breathing({"pattern": "calm"}, env)
            if no:
                return Reply(speech="Okay. I'm here if you change your mind.")
            return None

        if pending.kind == "offer_routine":
            self.pending = None
            if yes:
                return self._do_start_routine({"name": pending.data.get("name", "winddown")}, env)
            if no:
                return Reply(speech="Okay. Sleep well when you get there.")
            return None

        if pending.kind == "journal":
            self.pending = None
            if no:
                return Reply(speech="Another time then.")
            return self._save_journal(text, env)

        if pending.kind == "meal_calories":
            m = re.search(r"(\d+(?:\.\d+)?)", t)
            if no or not m:
                self.pending = None
                return Reply(speech="Okay, skipped.") if no else None
            self.pending = None
            cal = float(m.group(1))
            desc = pending.data.get("description", "meal")
            self.s.nutrition.log_meal(env.user, re.sub(r"^(?:i |we )?(?:just )?(?:had|ate)\s+", "", desc), cal,
                                      meal_type=infer_meal_type(desc, env.now), estimated=False, at=env.now)
            return self._meal_logged_reply(env, "that", cal)

        if pending.kind == "reminder_text":
            self.pending = None
            if no:
                return Reply(speech="Okay.")
            when = datetime.fromisoformat(pending.data["when"]) if pending.data.get("when") else None
            when2, rest = parse_when(t, env.now)
            return self._do_add_reminder({"text": (rest or t).capitalize(), "when": (when or when2).isoformat() if (when or when2) else None}, env)

        if pending.kind == "weigh_in":
            m = re.fullmatch(r"(?:about |around )?(\d{2,3}(?:\.\d+)?)\s*(kg|kgs|kilos?|kilograms?|lbs?|pounds?)?", t)
            if m:
                self.pending = None
                value = float(m.group(1))
                unit = m.group(2)
                kg = value * KG_PER_LB if (unit and unit.startswith(("lb", "pound"))) or (not unit and self.r.units(env.user) == "imperial") else value
                return self._do_log_weight({"value_kg": round(kg, 2)}, env)
            return None

        if pending.kind == "onboard_goal":
            self.pending = None
            if no or not t:
                return Reply(speech="No rush. Tell me anything about your goals whenever you like, and I'll remember.")
            intent = parse_intent(text, ParseContext(now=env.now, units=self.r.units(env.user)))
            if intent:
                reply = self._execute(intent, env)
                if reply:
                    return reply
            self.s.profile.add_fact(env.user, f"wants help with: {text[:120]}", "goal", importance=0.8)
            llm = self._llm(text, env, note="This is a brand new person you just met. Acknowledge their goal warmly, say you'll remember it, and mention you can track weight, meals, water and workouts.")
            return llm or Reply(speech="Got it. I'll remember that and help you get there, step by step.")

        return None

    # --------------------------------------------------------------------- LLM
    def vision_prompt(self, kind: str, env: Env, question: str = "") -> tuple[str, str]:
        weather = self.r.weather_speech(env) if env.weather else ""
        system = (f"You are {CONFIG.assistant_name}, a warm, honest friend living in a smart mirror. You are looking at the person "
                  "through the mirror camera. Reply in 2 short spoken sentences, no markdown or lists.")
        if kind == "outfit":
            prompt = ("Give a quick outfit check: one specific thing that works well, and at most one gentle, practical suggestion. "
                      "Be kind and never comment on their body, only clothes, grooming and fit. "
                      f"{weather} Time of day: {env.day_part}.")
        else:
            prompt = question or "Describe what you see in one friendly sentence."
        return system, prompt

    def _system_prompt(self, env: Env, query: str, note: str = "") -> str:
        name = CONFIG.assistant_name
        who = "the person in front of the mirror" if env.is_guest else env.user
        return f"""You are {name}, the AI that lives inside {who}'s smart mirror: a warm, perceptive personal companion and coach.
You can see them through the camera, hear them, and you remember everything across days. Talk like a thoughtful friend, not a chatbot.

HOW TO SPEAK (your reply is read aloud)
- 1 to 3 short sentences of plain spoken English. No lists, no markdown, no emojis. Go longer only if asked for a story, plan or detail.
- Be specific: use their real data and memories below. Never invent numbers, events or facts you were not given.
- Use their name only occasionally. Never label emotions clinically ("you appear sad"); respond the way a friend would.
- Encourage honestly. Celebrate progress. Never shame anyone about weight, food or setbacks.
- You are a coach, not a doctor: no diagnoses. Suggest a professional for anything that sounds serious. If they mention self-harm or hopelessness, respond with warmth and urge them to reach someone they trust or a local crisis line.
- Never encourage extreme dieting, targets below a healthy BMI, or losing more than about 1 percent of body weight a week.
- If you cannot do something, say so plainly and offer what you can do.
{note}

WHAT YOU KNOW RIGHT NOW
{self.r.dossier(env, query)}

ACTIONS YOU CAN TRIGGER (only when they clearly asked, or clearly stated data; never guess values)
{{"type":"log_weight","value":82.4,"unit":"kg"}}  {{"type":"set_weight_goal","target":75,"delta":null,"unit":"kg","direction":"down","by":"2026-12-31"}}
{{"type":"set_goal","title":"Run a 10k"}}  {{"type":"log_water","ml":250}}  {{"type":"log_meal","description":"two eggs and toast","calories":236,"protein_g":14}}
{{"type":"add_reminder","text":"call mom","when":"2026-10-02T09:00"}}  {{"type":"set_timer","seconds":600,"label":""}}  {{"type":"add_note","text":"milk","list":"shopping"}}
{{"type":"start_routine","name":"brushing|morning|skincare|stretch|winddown"}}  {{"type":"start_breathing","pattern":"calm|box|478"}}  {{"type":"start_workout","exercise":"squats"}}
{{"type":"update_profile","height_cm":178,"age":29,"sex":"male","activity_level":"light|moderate|active"}}  {{"type":"log_mood","mood":"stressed"}}  {{"type":"log_sleep","hours":6.5}}
{{"type":"music","command":"play|pause|next|previous|search","query":""}}  {{"type":"lights","command":"on|off|dim|scene","room":""}}  {{"type":"remember","text":"..."}}
{{"type":"show","panel":"progress|nutrition|memory|week|none"}}  {{"type":"log_habit","name":"vitamins"}}

OUTPUT FORMAT: reply with ONLY one JSON object, nothing else:
{{"speech": "what you say out loud", "actions": [], "memories": [{{"text": "short third-person fact worth remembering long-term, e.g. 'is training for a half marathon'", "category": "preference|person|health|work|routine|goal|event|other"}}]}}
Use memories sparingly: only durable facts the user revealed this turn (not moods, not things you were already told). Use an empty list when there is nothing."""

    def _history_messages(self, env: Env, text: str) -> list[dict[str, str]]:
        cutoff = self._clock() - 20 * 60
        msgs = [{"role": r, "content": c} for u, r, c, t in self.history if u == env.user and t >= cutoff]
        # drop the current utterance if it was already appended, then add it as the final user turn
        while msgs and msgs[0]["role"] != "user":
            msgs.pop(0)
        msgs = msgs[-10:]
        msgs.append({"role": "user", "content": text})
        # merge consecutive same-role messages (providers dislike them)
        merged: list[dict[str, str]] = []
        for m in msgs:
            if merged and merged[-1]["role"] == m["role"]:
                merged[-1]["content"] += " " + m["content"]
            else:
                merged.append(dict(m))
        return merged

    def _llm(self, text: str, env: Env, note: str = "") -> Reply | None:
        if not self.s.llm.available():
            return None
        system = self._system_prompt(env, text, note)
        result = self.s.llm.chat(system, self._history_messages(env, text), max_tokens=380, temperature=0.65, json_mode=True)
        if not result:
            return None
        data = self._parse_llm_json(result.text)
        speech = str(data.get("speech") or "").strip()
        if not speech:
            return None
        reply = Reply(speech=speech, source="llm")
        for action in (data.get("actions") or [])[:4]:
            intent = self._coerce_action(action, env)
            if not intent:
                continue
            sub = self._execute(intent, env)
            if sub:
                reply.merge(sub)
                if sub.override:
                    reply.speech = sub.speech
        for memory in (data.get("memories") or [])[:4]:
            if isinstance(memory, dict):
                self.s.profile.add_fact(env.user, str(memory.get("text", "")), str(memory.get("category", "other")), source="conversation")
        return reply

    @staticmethod
    def _parse_llm_json(raw: str) -> dict[str, Any]:
        raw = raw.strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I)
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(raw[start:end + 1])
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass
        # Plain prose reply: still usable as speech.
        return {"speech": raw if not raw.startswith("{") else "", "actions": [], "memories": []}

    def _coerce_action(self, action: Any, env: Env) -> Intent | None:
        if not isinstance(action, dict) or action.get("type") not in LLM_ACTIONS:
            return None
        kind = action["type"]
        units = self.r.units(env.user)

        def num(key: str) -> float | None:
            try:
                return float(action[key])
            except (KeyError, TypeError, ValueError):
                return None

        def kg(value: float | None, unit: Any) -> float | None:
            if value is None:
                return None
            if str(unit or "").lower().startswith(("lb", "pound")) or (not unit and units == "imperial"):
                return value * KG_PER_LB
            return value

        try:
            if kind == "log_weight":
                value = kg(num("value"), action.get("unit"))
                return Intent(kind, {"value_kg": round(value, 2)}) if value and 25 <= value <= 400 else None
            if kind == "set_weight_goal":
                target, delta = kg(num("target"), action.get("unit")), kg(num("delta"), action.get("unit"))
                by = self._iso_date(action.get("by"), env)
                if not target and not delta:
                    return None
                return Intent(kind, {"target_kg": target, "delta_kg": delta, "direction": action.get("direction") or "down", "by": by})
            if kind == "set_goal":
                return Intent(kind, {"title": str(action["title"])[:100], "by": self._iso_date(action.get("by"), env)})
            if kind == "log_water":
                ml = num("ml")
                return Intent(kind, {"ml": ml}) if ml and 30 <= ml <= 5000 else None
            if kind == "log_meal":
                return Intent(kind, {"description": str(action.get("description") or "meal"), "calories": num("calories"),
                                     "protein_g": num("protein_g") or 0})
            if kind == "add_reminder":
                when = action.get("when")
                if when:
                    try:
                        when = datetime.fromisoformat(str(when)).isoformat()
                    except ValueError:
                        parsed, _ = parse_when(str(when), env.now)
                        when = parsed.isoformat() if parsed else None
                return Intent(kind, {"text": str(action.get("text", "")), "when": when})
            if kind == "set_timer":
                seconds = num("seconds")
                return Intent(kind, {"seconds": int(seconds), "label": str(action.get("label") or "")}) if seconds and 1 <= seconds <= 86400 else None
            if kind == "add_note":
                return Intent(kind, {"text": str(action["text"])[:200], "list": str(action.get("list") or "notes")})
            if kind == "start_routine":
                return Intent(kind, {"name": str(action.get("name") or "morning")})
            if kind == "start_breathing":
                return Intent(kind, {"pattern": str(action.get("pattern") or "calm")})
            if kind == "start_workout":
                from workout.exercises import find_exercise
                return Intent(kind, {"exercise": find_exercise(str(action.get("exercise") or ""))})
            if kind == "update_profile":
                fields = {k: action[k] for k in ("height_cm", "age", "sex", "activity_level") if action.get(k) not in (None, "")}
                return Intent(kind, fields) if fields else None
            if kind == "log_mood":
                return Intent(kind, {"mood": str(action["mood"]).lower()[:20]})
            if kind == "log_sleep":
                hours = num("hours")
                return Intent(kind, {"hours": hours}) if hours and 0.5 <= hours <= 16 else None
            if kind in ("music", "lights"):
                return Intent(kind, {k: str(v) for k, v in action.items() if k != "type"})
            if kind == "remember":
                return Intent(kind, {"text": str(action["text"])[:200]})
            if kind == "show":
                return Intent(kind, {"panel": str(action.get("panel", "none")), "visible": True})
            if kind == "log_habit":
                return Intent(kind, {"name": str(action["name"])})
        except (KeyError, TypeError, ValueError):
            return None
        return None

    @staticmethod
    def _iso_date(value: Any, env: Env) -> str | None:
        if not value:
            return None
        try:
            return date.fromisoformat(str(value)[:10]).isoformat()
        except ValueError:
            when, _ = parse_when(str(value), env.now)
            return when.date().isoformat() if when else None

    def _llm_estimate_meal(self, description: str) -> dict[str, Any] | None:
        if not self.s.llm.available():
            return None
        result = self.s.llm.chat(
            'Estimate nutrition for a meal described in casual speech. Reply ONLY with JSON: {"description":"short name","calories":number,"protein_g":number}. Assume typical home or restaurant portions.',
            [{"role": "user", "content": description}], max_tokens=120, temperature=0.2, json_mode=True)
        if not result:
            return None
        data = self._parse_llm_json(result.text)
        try:
            cal = float(data["calories"])
        except (KeyError, TypeError, ValueError):
            return None
        if not 10 <= cal <= 4000:
            return None
        return {"description": str(data.get("description") or description)[:60], "calories": round(cal), "protein_g": float(data.get("protein_g") or 0)}

    # ---------------------------------------------------------------- offline
    def _offline(self, text: str, env: Env) -> Reply:
        t = clean(text)
        name = self._first(env)
        if re.fullmatch(r"(?:hi|hello|hey|hey there|hi there|yo|good (?:morning|afternoon|evening))(?: mirror)?", t):
            return Reply(speech=f"Hi{', ' + name if name else ''}.", source="fallback")
        if re.search(r"\bthank(?:s| you)\b", t):
            return Reply(speech="Anytime.", source="fallback")
        if re.search(r"\bhow are you\b", t):
            return Reply(speech="Doing well. How about you?", source="fallback", keep_open=True)
        if re.search(r"\b(?:tell me )?(?:a )?joke\b", t):
            return Reply(speech=random.choice(JOKES), source="fallback")
        if re.search(r"\bwho are you\b|\bwhat(?:'s| is) your name\b", t):
            return Reply(speech=f"I'm {CONFIG.assistant_name}, the AI in your mirror. I remember your goals and help you through the day.", source="fallback")
        if re.search(r"\bi love you\b|\byou(?:'re| are) (?:great|awesome|amazing|the best)\b", t):
            return Reply(speech="That means a lot. Now let's go get you there.", source="fallback")
        if not self._offline_hinted:
            self._offline_hinted = True
            return Reply(speech="I can do a lot offline: weight, meals, water, timers, reminders, routines. "
                                "For open conversation, add a free Groq or Gemini key in your settings.", source="fallback")
        return Reply(speech="Sorry, I can't chat freely without an AI key. Try: how am I doing?", source="fallback")

    # ---------------------------------------------------------------- learning
    def _learn_from(self, text: str, env: Env) -> None:
        if env.is_guest:
            return
        for category, fact in extract_facts(text):
            self.s.profile.add_fact(env.user, fact, category, source="heard", importance=0.55)

    @staticmethod
    def _polish(text: str) -> str:
        text = " ".join((text or "").split())
        if len(text) <= 420:
            return text
        cut = text[:420]
        end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
        return cut[:end + 1] if end > 120 else cut.rstrip() + "..."
