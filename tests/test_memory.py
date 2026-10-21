from datetime import date, datetime, timedelta

import pytest

from memory.db import Store
from memory.goal_service import GoalService, ema, linear_slope_per_week
from memory.habit_service import HabitService
from memory.journal_service import JournalService
from memory.memory_service import MemoryService
from memory.nutrition_service import NutritionService, estimate_meal
from memory.profile_service import ProfileService, extract_facts


@pytest.fixture()
def svc(db_path):
    return {
        "mem": MemoryService(db_path), "profile": ProfileService(db_path), "goals": GoalService(db_path),
        "nutrition": NutritionService(db_path), "habits": HabitService(db_path), "journal": JournalService(db_path),
    }


def test_tasks_with_due_dates(svc):
    mem = svc["mem"]
    now = datetime(2026, 10, 1, 12, 0)
    mem.add_task("Ada", "call mom", due_at=now + timedelta(minutes=5))
    mem.add_task("Ada", "buy milk")
    assert [t["text"] for t in mem.get_tasks("Ada")] == ["call mom", "buy milk"]  # dated first
    assert mem.due_reminders("Ada", now) == []
    due = mem.due_reminders("Ada", now + timedelta(minutes=6))
    assert len(due) == 1
    mem.mark_notified(due[0]["id"])
    assert mem.due_reminders("Ada", now + timedelta(minutes=6)) == []
    assert mem.complete_latest_task("Ada") == "buy milk"
    assert mem.add_task("Guest", "x") is None


def test_profile_and_facts(svc):
    p = svc["profile"]
    assert p.ensure_profile("Ada")["is_new"] is True
    assert p.ensure_profile("Ada")["is_new"] is False
    p.update_profile("Ada", age=29, height_cm=170, sex="female", activity_level="moderate")
    assert p.get_profile("Ada")["height_cm"] == 170
    p.add_fact("Ada", "likes spicy food", "preference")
    p.add_fact("Ada", "likes spicy food a lot", "preference")  # near-duplicate merges
    p.add_fact("Ada", "works as a data engineer at Acme", "work")
    p.add_fact("Ada", "dog is named Biscuit", "person")
    assert len(p.list_facts("Ada")) == 3
    assert "dog is named Biscuit" in p.recall("Ada", "what is my dog called")[0:2]
    assert p.forget_fact("Ada", "likes spicy food")
    assert all("spicy" not in f["text"] for f in p.list_facts("Ada"))
    assert p.add_fact("Unknown", "nope fact") is None


def test_extract_facts_offline():
    facts = dict((t, c) for c, t in extract_facts("I love jazz music and my dog is called Biscuit. I'm vegetarian."))
    assert any("jazz" in t for t in facts)
    assert any("Biscuit" in t for t in facts)
    assert any("vegetarian" in t for t in facts)


def test_weight_goal_progress(svc):
    g = svc["goals"]
    start = date.today() - timedelta(days=28)
    at = lambda i: datetime.combine(start + timedelta(days=i), datetime.min.time())
    g.add_measurement("Ada", "weight", 90, "kg", at=at(0))
    goal = g.create_weight_goal("Ada", target_kg=80, by=date.today() + timedelta(days=100))
    assert goal["start_value"] == 90
    for i in range(2, 29, 2):  # lose ~0.1 kg/day
        g.add_measurement("Ada", "weight", 90 - i * 0.1, "kg", at=at(i))
    p = g.progress(g.get_goal(goal["id"]))
    assert p["rate_per_week"] == pytest.approx(-0.7, abs=0.05)
    assert p["status"] in {"on_track", "ahead"}
    assert 15 < p["percent"] < 40
    assert p["eta"]
    assert "to go" in p["headline"]


def test_relative_goal_resolves_on_first_weigh_in(svc):
    g = svc["goals"]
    goal = g.create_weight_goal("Ada", delta_kg=5, direction="down")
    assert goal["target_value"] is None
    g.add_measurement("Ada", "weight", 80, "kg")
    resolved = g.weight_goal("Ada")
    assert resolved["start_value"] == 80 and resolved["target_value"] == 75
    assert resolved["title"] == "Lose 5.0 kg"


def test_milestones_fire_once(svc):
    g = svc["goals"]
    g.add_measurement("Ada", "weight", 90, "kg", at=datetime.now() - timedelta(days=20))
    goal = g.create_weight_goal("Ada", target_kg=80)
    for days, w in ((15, 89), (10, 87.5), (5, 86.5), (0, 85)):
        g.add_measurement("Ada", "weight", w, "kg", at=datetime.now() - timedelta(days=days))
    p = g.progress(g.get_goal(goal["id"]))
    first = g.pending_milestone(g.get_goal(goal["id"]), p["percent"])
    assert first in (25, 50)
    assert g.pending_milestone(g.get_goal(goal["id"]), p["percent"]) is None


def test_math_helpers():
    assert ema([10, 10, 10]) == [10, 10, 10]
    pts = [(date(2026, 1, 1) + timedelta(days=i), 100 - i * 0.2) for i in range(14)]
    assert linear_slope_per_week(pts) == pytest.approx(-1.4, abs=0.01)
    assert linear_slope_per_week(pts[:2]) is None


def test_meal_estimates():
    r = estimate_meal("I had two eggs and a toast for breakfast")
    assert r["calories"] == 2 * 78 + 80
    r = estimate_meal("had 2 rotis with dal and a bowl of rice")
    names = {i["name"] for i in r["items"]}
    assert {"roti", "dal", "rice"} <= names
    r = estimate_meal("a quantum pizza of unobtainium")
    assert r["items"] and r["items"][0]["name"] == "pizza"
    assert estimate_meal("blorp")["unmatched"] == ["blorp"]


def test_targets_and_meals(svc):
    n = svc["nutrition"]
    t = n.targets("Ada", {"height_cm": 175, "age": 30, "sex": "male", "activity_level": "moderate"}, 90, -0.5)
    assert t["basis"] == "estimated"
    assert t["calories"] < t["tdee"] and t["calories"] >= 1500
    t2 = n.targets("Ada", {}, None)
    assert "weight" in t2["missing"] and t2["calories"] == 2000
    n.log_meal("Ada", "eggs", 156, 12)
    n.log_meal("Ada", "rice", 205, 4)
    assert n.totals("Ada")["calories"] == 361
    assert n.undo_last_meal("Ada")["description"] == "rice"
    assert n.totals("Ada")["calories"] == 156


def test_habits_and_water(svc):
    h = svc["habits"]
    h.log_water("Ada", 500, target_ml=2000)
    h.log_water("Ada", 250)
    assert h.water_today("Ada") == 750
    meds = h.ensure_habit("Ada", "Vitamins", "", 1)
    h.log("Ada", meds["id"], 1)
    today = {x["name"]: x for x in h.today("Ada")}
    assert today["Vitamins"]["complete"] and today["Vitamins"]["streak"] == 1
    assert not today["Water"]["complete"]
    assert h.log_workout("Ada", "squats", reps=20, duration_seconds=120)
    assert not h.log_workout("Ada", "squats", reps=0, duration_seconds=3)
    assert h.workout_summary("Ada")["reps"] == {"squats": 20}


def test_journal_and_nudges(svc):
    j = svc["journal"]
    j.add_turn("Ada", "user", "hello")
    j.add_turn("Ada", "assistant", "hi Ada")
    assert [t["role"] for t in j.recent_turns("Ada")] == ["user", "assistant"]
    j.save_summary("Ada", "2026-10-01", "A calm day.")
    j.save_summary("Ada", "2026-10-01", "A calmer day.")
    assert j.summary_for("Ada", "2026-10-01")["text"] == "A calmer day."
    j.add_note("Ada", "eggs", "shopping")
    assert [n["text"] for n in j.list_notes("Ada", "shopping")] == ["eggs"]
    assert not j.nudge_fired("Ada", "briefing")
    j.mark_nudge("Ada", "briefing")
    assert j.nudge_fired("Ada", "briefing")


def test_delete_user(svc):
    svc["profile"].ensure_profile("Ada")
    svc["goals"].add_measurement("Ada", "weight", 80, "kg")
    svc["profile"].delete_user("Ada")
    assert svc["goals"].latest("Ada", "weight") is None
    assert svc["profile"].get_profile("Ada") == {}
