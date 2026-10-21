from datetime import datetime, timedelta

import pytest

from assistant.brain import Brain
from assistant.models import Env
from assistant.proactive import ProactiveEngine, Signals, in_quiet_hours
from assistant.reflection import Reflector
from assistant.reports import Reporter
from assistant.services import Services

from test_brain import FakeLLM, make_brain  # noqa: F401  (shared fakes)

BASE = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def make(tmp_path, with_llm=False, replies=None):
    brain, s, fake = make_brain(tmp_path, with_llm=with_llm, replies=replies)
    clock = Clock()
    engine = ProactiveEngine(s, brain.r, brain, clock=clock)
    engine.enabled = True
    return engine, brain, s, clock, fake


def env(now=BASE, user="Ada", **kw):
    return Env(user=user, now=now, **kw)


def test_quiet_hours():
    assert in_quiet_hours(BASE.replace(hour=23, minute=30), "07:00", "23:00")
    assert in_quiet_hours(BASE.replace(hour=3), "07:00", "23:00")
    assert not in_quiet_hours(BASE.replace(hour=12), "07:00", "23:00")
    assert in_quiet_hours(BASE.replace(hour=2), "09:00", "01:00")  # late sleeper
    assert not in_quiet_hours(BASE.replace(hour=0, minute=30), "09:00", "01:00")


def test_first_arrival_gives_briefing_once_per_day(tmp_path):
    engine, brain, s, clock, _ = make(tmp_path)
    s.profile.ensure_profile("Ada")
    # Last weigh-in was yesterday: the briefing only asks for the scale when today's is missing.
    s.goals.add_measurement("Ada", "weight", 85, "kg", at=BASE - timedelta(days=1))
    s.goals.create_weight_goal("Ada", target_kg=80)
    out = engine.arrival(env(events=[{"title": "Standup", "time": "10:00 AM"}]), None)
    assert [n.kind for n in out] == ["briefing"]
    assert "Good morning, Ada" in out[0].text and "Standup" in out[0].text
    assert "step on the scale" in out[0].text and out[0].pending[0] == "weigh_in"
    assert engine.arrival(env(), 120) == []  # briefed already; short absence is silent
    clock.t += 3600  # an hour away
    out = engine.arrival(env(), 3600)
    assert [n.kind for n in out] == ["welcome"]


def test_no_briefing_in_quiet_hours_but_missed_reminders_delivered(tmp_path):
    engine, brain, s, clock, _ = make(tmp_path)
    night = BASE.replace(hour=2)
    s.memory.add_task("Ada", "take medicine", due_at=night - timedelta(minutes=5))
    out = engine.arrival(env(night), None)
    assert [n.kind for n in out] == ["reminder"] and "take medicine" in out[0].text


def test_evening_arrival_recap_and_journal_prompt(tmp_path):
    engine, brain, s, clock, _ = make(tmp_path)
    s.nutrition.log_meal("Ada", "dinner", 600, 30)
    evening = BASE.replace(hour=20)
    out = engine.arrival(env(evening), None)
    kinds = [n.kind for n in out]
    assert "recap" in kinds and "briefing" not in kinds
    recap = next(n for n in out if n.kind == "recap")
    assert "600 calories" in recap.text and recap.pending[0] == "journal"


def test_due_reminder_is_critical_and_fires_once(tmp_path):
    engine, brain, s, clock, _ = make(tmp_path)
    s.memory.add_task("Ada", "call mom", due_at=BASE - timedelta(minutes=1))
    out = engine.tick(env(present=True), Signals())
    assert out and out[0].kind == "reminder" and out[0].priority == 3
    assert engine.tick(env(present=True), Signals()) == []


def test_calendar_heads_up(tmp_path):
    engine, brain, s, clock, _ = make(tmp_path)
    start = (BASE + timedelta(minutes=10)).isoformat()
    e = env(events=[{"title": "Design review", "time": "08:10 AM", "start": start}])
    out = engine.tick(e, Signals())
    assert out and "Design review starts in 10 minutes" in out[0].text
    assert engine.tick(e, Signals()) == []


def test_global_spacing_blocks_low_priority_nudges(tmp_path):
    engine, brain, s, clock, _ = make(tmp_path)
    sig = Signals(present_seconds=3600)
    first = engine.tick(env(), sig)
    assert any(n.kind == "break" for n in first)
    clock.t += 30
    sig2 = Signals(slouch_events_recent=5)
    assert engine.tick(env(), sig2) == []  # spoke 30s ago, posture nudge is low priority
    clock.t += 600
    assert [n.kind for n in engine.tick(env(), sig2)] == ["posture"]


def test_mood_nudge_needs_sustained_streak(tmp_path):
    engine, brain, s, clock, _ = make(tmp_path)
    assert engine.tick(env(), Signals(negative_streak=2)) == []
    out = engine.tick(env(), Signals(negative_streak=4))
    assert out and out[0].kind == "mood" and out[0].pending[0] == "offer_breathing"
    clock.t += 20 * 60
    assert engine.tick(env(), Signals(negative_streak=6)) == []  # inside the 45 minute cooldown
    clock.t += 30 * 60
    assert [n.kind for n in engine.tick(env(), Signals(negative_streak=6))] == ["mood"]  # cooldown over


def test_busy_and_workout_suppress_nudges(tmp_path):
    engine, brain, s, clock, _ = make(tmp_path)
    assert engine.tick(env(), Signals(present_seconds=3600), busy=True) == []
    assert engine.tick(env(workout_active=True), Signals(present_seconds=3600)) == []


def test_hydration_nudge_only_when_behind(tmp_path):
    engine, brain, s, clock, _ = make(tmp_path)
    s.goals.add_measurement("Ada", "weight", 70, "kg")
    s.goals.create_weight_goal("Ada", target_kg=65)
    noon = BASE.replace(hour=14)
    out = engine.tick(env(noon), Signals(present_seconds=600))
    assert out and out[0].kind == "hydration" and "Water check" in out[0].text
    engine2, _, s2, _, _ = make(tmp_path / "b")
    s2.goals.add_measurement("Ada", "weight", 70, "kg")
    s2.goals.create_weight_goal("Ada", target_kg=65)
    s2.habits.log_water("Ada", 2000, 2000)
    assert engine2.tick(env(noon), Signals(present_seconds=600)) == []


def test_habit_reminder_and_streak_celebration(tmp_path):
    engine, brain, s, clock, _ = make(tmp_path)
    habit = s.habits.ensure_habit("Ada", "Take vitamins", "", 1, "08:00")
    out = engine.tick(env(BASE.replace(minute=15)), Signals())
    assert any("take vitamins" in n.text.lower() for n in out)
    for i in range(7):
        s.habits.log("Ada", habit["id"], 1, at=BASE - timedelta(days=i))
    first = engine.arrival(env(), None)
    assert [n.kind for n in first] == ["briefing"]  # one spoken nudge per batch: the briefing wins
    clock.t += 10 * 60
    arrival = engine.arrival(env(), None)
    assert any(n.kind == "streak" and "7 days in a row" in n.text for n in arrival)


def test_bedtime_winddown_offer(tmp_path):
    engine, brain, s, clock, _ = make(tmp_path)
    s.profile.update_profile("Ada", bed_time="23:00")
    out = engine.tick(env(BASE.replace(hour=22, minute=40)), Signals())
    assert out and out[0].kind == "winddown"
    brain.set_pending(*out[0].pending)
    r = brain.handle("yes", env())
    assert s.routines.active and "Wind-down" in r.speech


def test_guest_prompt_has_cooldown(tmp_path):
    engine, brain, s, clock, _ = make(tmp_path)
    g = env(user="Guest")
    assert engine.guest_prompt(g)[0].pending[0] == "enroll_name"
    assert engine.guest_prompt(g) == []
    clock.t += 21 * 60
    assert engine.guest_prompt(g)


def test_reflection_without_llm_summarises_stats(tmp_path):
    engine, brain, s, clock, _ = make(tmp_path)
    s.nutrition.log_meal("Ada", "lunch", 700, 30)
    s.habits.log_water("Ada", 1000, 2000)
    text = Reflector(s, brain.r).reflect("Ada")
    # Water target is derived from body weight (70 kg default -> 2,500 ml), so 1,000 ml is 40 percent.
    assert "700 calories" in text and "40 percent" in text
    assert s.journal.summary_for("Ada", BASE.date().isoformat())["text"] == text
    # next morning's briefing can quote it
    assert "700 calories" in brain.r.yesterday_line("Ada", datetime.now() + timedelta(days=1))


def test_reflection_with_llm_learns_facts(tmp_path):
    reply = {"summary": "You pushed through a tough day and still cooked dinner.", "mood": "tired",
             "facts": [{"text": "cooks dinner for the family on weekdays", "category": "routine"}]}
    engine, brain, s, clock, fake = make(tmp_path, with_llm=True, replies=[reply])
    s.journal.add_turn("Ada", "user", "long day at work")
    s.journal.add_turn("Ada", "user", "cooked dinner for everyone")
    text = Reflector(s, brain.r).reflect("Ada")
    assert text.startswith("You pushed through")
    assert any("cooks dinner" in f["text"] for f in s.profile.list_facts("Ada"))


def test_reflection_skips_empty_days(tmp_path):
    engine, brain, s, clock, _ = make(tmp_path)
    assert Reflector(s, brain.r).reflect("Ada") is None
