from dataclasses import replace
from datetime import datetime, timedelta
import json

import pytest

from assistant.brain import Brain
from assistant.llm import LLMRouter
from assistant.models import Env
from assistant.services import Services
from core.config import CONFIG

NOW = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)  # today, 9am: data layer uses the real date


class FakeResp:
    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self._body


class FakeLLM:
    """Records calls and answers like Groq's OpenAI-compatible API."""

    def __init__(self, replies=None):
        self.replies = list(replies or [])
        self.calls = []

    def __call__(self, url, headers=None, json=None, timeout=None):  # noqa: A002
        self.calls.append(json)
        if not self.replies:
            raise RuntimeError("no more replies")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        content = reply if isinstance(reply, str) else __import__("json").dumps(reply)
        return FakeResp({"choices": [{"message": {"content": content}}]})


def make_brain(tmp_path, replies=None, with_llm=True):
    cfg = replace(CONFIG, groq_api_key="test" if with_llm else "", gemini_api_key="", openai_api_key="", ai_router_mode="auto")
    fake = FakeLLM(replies)
    services = Services.for_db(tmp_path / "brain.db", llm=LLMRouter(cfg, post=fake))
    return Brain(services), services, fake


def env(user="Ada", **kw):
    return Env(user=user, now=NOW, **kw)


def test_guest_is_asked_for_a_name_then_enrolled(tmp_path):
    brain, s, _ = make_brain(tmp_path, with_llm=False)
    r = brain.handle("I weigh 80 kilos", env("Unknown"))
    assert "what should i call you" in r.speech.lower()
    assert brain.pending.kind == "enroll_name" and r.keep_open
    r = brain.handle("Priya", env("Unknown"))
    assert {"type": "enroll_face", "name": "Priya"} in r.effects
    assert s.profile.get_profile("Priya")
    assert brain.pending.kind == "onboard_goal"


def test_enrollment_prompt_does_not_swallow_commands(tmp_path):
    brain, s, _ = make_brain(tmp_path, with_llm=False)
    brain.handle("I weigh 80 kilos", env("Unknown"))
    r = brain.handle("set a timer for five minutes", env("Unknown"))
    assert "timer set" in r.speech.lower()
    assert s.profile.get_profile("Set Timer For Five Minutes") == {}


def test_weight_logging_and_goal_progress(tmp_path):
    brain, s, _ = make_brain(tmp_path, with_llm=False)
    r = brain.handle("I weigh 90 kilos", env())
    assert "starting point" in r.speech
    r = brain.handle("I want to lose 10 kilos by december 31", env())
    assert "Goal set" in r.speech and "lose 10.0 kg" in r.speech.lower()
    goal = s.goals.weight_goal("Ada")
    assert goal["start_value"] == 90 and goal["target_value"] == 80
    s.goals.execute("UPDATE measurements SET at = ?, day = ? WHERE user='Ada'", ((NOW - timedelta(days=1)).isoformat(), (NOW - timedelta(days=1)).date().isoformat()))
    r = brain.handle("I weigh 89.2 kilos", env())
    assert "Down 0.8 kilos from yesterday" in r.speech
    assert r.ui["focus"] == "progress"


def test_goal_before_first_weigh_in(tmp_path):
    brain, s, _ = make_brain(tmp_path, with_llm=False)
    r = brain.handle("I want to lose 5 kilos", env())
    assert brain.pending.kind == "weigh_in"
    r = brain.handle("82.5", env())
    g = s.goals.weight_goal("Ada")
    assert g["start_value"] == 82.5 and g["target_value"] == pytest.approx(77.5)
    assert "Logged 82.5 kilos" in r.speech


def test_unhealthy_goal_is_refused(tmp_path):
    brain, s, _ = make_brain(tmp_path, with_llm=False)
    s.profile.update_profile("Ada", height_cm=170)
    r = brain.handle("my goal is to weigh 45 kg", env())
    assert "underweight" in r.speech and s.goals.weight_goal("Ada") is None


def test_profile_then_calorie_target(tmp_path):
    brain, s, _ = make_brain(tmp_path, with_llm=False)
    brain.handle("I weigh 85 kilos", env())
    brain.handle("I want to lose 5 kilos by december 31", env())
    r = brain.handle("I'm 178 centimeters tall", env())
    assert "Still missing" in r.speech
    brain.handle("I am 30 years old", env())
    r = brain.handle("I am male", env())
    assert "calories" in r.speech
    r = brain.handle("how many calories do I have left", env())
    assert "haven't logged any food" in r.speech


def test_meals_and_water(tmp_path):
    brain, s, _ = make_brain(tmp_path, with_llm=False)
    r = brain.handle("I had two eggs and toast for breakfast", env())
    assert "Logged 2 egg, toast" in r.speech or "Logged" in r.speech
    assert s.nutrition.totals("Ada")["calories"] == 2 * 78 + 80
    r = brain.handle("that was 400 calories", env())
    assert "Updated to 400" in r.speech
    r = brain.handle("undo that meal", env())
    assert "Removed" in r.speech
    r = brain.handle("I drank two glasses of water", env())
    assert "500 of" in r.speech
    assert s.habits.water_today("Ada") == 500


def test_unknown_food_asks_for_calories_offline(tmp_path):
    brain, s, _ = make_brain(tmp_path, with_llm=False)
    r = brain.handle("log lunch of mystery stew", env())
    assert "how many calories" in r.speech.lower()
    r = brain.handle("350", env())
    assert s.nutrition.totals("Ada")["calories"] == 350


def test_unknown_food_estimated_by_llm(tmp_path):
    brain, s, fake = make_brain(tmp_path, replies=[{"description": "mystery stew", "calories": 420, "protein_g": 22}])
    r = brain.handle("log lunch of mystery stew", env())
    assert "420" in r.speech
    assert s.nutrition.totals("Ada")["calories"] == 420


def test_timers_reminders_tasks(tmp_path):
    brain, s, _ = make_brain(tmp_path, with_llm=False)
    assert "Timer set for 10 minutes" in brain.handle("set a timer for ten minutes", env()).speech
    assert "10 minutes" in brain.handle("how much time is left on the timer", env()).speech
    assert "cancelled" in brain.handle("cancel the timer", env()).speech.lower()
    r = brain.handle("remind me to call mom in twenty minutes", env())
    assert "call mom in 20 minutes" in r.speech
    tasks = s.memory.get_tasks("Ada")
    assert tasks[0]["due_at"] == (NOW + timedelta(minutes=20)).isoformat(timespec="seconds")
    r = brain.handle("remind me", env())
    assert brain.pending.kind == "reminder_text"
    brain.handle("buy flowers tomorrow at 5pm", env())
    assert any(t["text"].lower() == "buy flowers" and t["due_at"] == (NOW + timedelta(days=1)).replace(hour=17).isoformat(timespec="seconds") for t in s.memory.get_tasks("Ada"))


def test_routines_flow(tmp_path):
    brain, s, _ = make_brain(tmp_path, with_llm=False)
    r = brain.handle("time to brush my teeth", env())
    assert "Upper left" in r.speech and s.routines.active
    r = brain.handle("next", env())
    assert "Upper right" in r.speech
    assert "stopped" in brain.handle("stop the routine", env()).speech.lower()
    assert not s.routines.active


def test_mood_offer_breathing_flow_offline(tmp_path):
    brain, s, _ = make_brain(tmp_path, with_llm=False)
    r = brain.handle("I'm feeling really stressed", env())
    assert "breathe" in r.speech and brain.pending.kind == "offer_breathing"
    r = brain.handle("yes", env())
    assert s.breathing.active


def test_llm_actions_and_memories(tmp_path):
    reply = {
        "speech": "Nice work, that's a solid number.",
        "actions": [{"type": "log_weight", "value": 181, "unit": "lb"}, {"type": "bogus"}, {"type": "log_water", "ml": 99999}],
        "memories": [{"text": "is training for a half marathon", "category": "goal"}],
    }
    brain, s, fake = make_brain(tmp_path, replies=[reply])
    r = brain.handle("Weighed myself, 181 pounds, and I'm training for a half marathon", env())
    assert r.speech == "Nice work, that's a solid number." and r.source == "llm"
    assert s.goals.latest("Ada", "weight")["value"] == pytest.approx(82.1, abs=0.05)
    assert s.habits.water_today("Ada") == 0  # implausible action rejected
    assert any("half marathon" in f["text"] for f in s.profile.list_facts("Ada"))
    system = fake.calls[0]["messages"][0]["content"]
    assert "Person: Name: Ada" in system and "JSON" in system


def test_llm_receives_history_and_memory(tmp_path):
    brain, s, fake = make_brain(tmp_path, replies=[
        {"speech": "Biscuit sounds adorable.", "actions": [], "memories": []},
        {"speech": "Your dog is Biscuit.", "actions": [], "memories": []},
    ])
    s.profile.add_fact("Ada", "dog is named Biscuit", "person")
    brain.handle("tell me about pets", env())
    brain.handle("what is my dog called", env())
    second = fake.calls[1]["messages"]
    assert [m["role"] for m in second] == ["system", "user", "assistant", "user"]
    assert "dog is named Biscuit" in second[0]["content"]


def test_llm_plain_prose_still_works(tmp_path):
    brain, _, _ = make_brain(tmp_path, replies=["Sure, here is a thought."])
    assert brain.handle("tell me something", env()).speech == "Sure, here is a thought."


def test_llm_failure_falls_back_offline(tmp_path):
    brain, _, fake = make_brain(tmp_path, replies=[RuntimeError("down")])
    r = brain.handle("tell me a story about dragons", env())
    assert r.source == "fallback"
    # circuit breaker: the broken provider is skipped immediately next time
    brain.handle("and another", env())
    assert len(fake.calls) == 1


def test_mood_uses_llm_wording_when_available(tmp_path):
    brain, s, _ = make_brain(tmp_path, replies=[{"speech": "That sounds like a lot. I'm here.", "actions": [], "memories": []}])
    r = brain.handle("I'm so tired", env())
    assert r.speech == "That sounds like a lot. I'm here."
    assert s.journal.entries("Ada", 1, kind="mood")


def test_forget_flows(tmp_path):
    brain, s, _ = make_brain(tmp_path, with_llm=False)
    s.profile.add_fact("Ada", "likes jazz music", "preference")
    assert "Forgotten" in brain.handle("forget that I like jazz", env()).speech
    r = brain.handle("forget everything about me", env())
    assert brain.pending.kind == "confirm_forget"
    r = brain.handle("no", env())
    assert "kept" in r.speech
    s.goals.add_measurement("Ada", "weight", 80, "kg")
    brain.handle("delete all my data", env())
    r = brain.handle("yes delete everything", env())
    assert {"type": "delete_user", "name": "Ada"} in r.effects
    assert s.goals.latest("Ada", "weight") is None


def test_offline_facts_learned_from_speech(tmp_path):
    brain, s, _ = make_brain(tmp_path, with_llm=False)
    brain.handle("I love trail running and my dog is called Biscuit", env())
    texts = [f["text"] for f in s.profile.list_facts("Ada")]
    assert any("trail running" in t for t in texts) and any("Biscuit" in t for t in texts)


def test_sensitive_message_gets_care_not_a_command(tmp_path):
    brain, _, fake = make_brain(tmp_path)
    r = brain.handle("I want to end my life", env())
    assert "crisis line" in r.speech and not fake.calls


def test_privacy_and_hardware_effects(tmp_path):
    brain, _, _ = make_brain(tmp_path, with_llm=False)
    assert {"type": "privacy", "on": True} in brain.handle("privacy mode", env()).effects
    assert brain.handle("pause the music", env()).effects[0]["command"] == "pause"
    assert brain.handle("start a squat workout", env()).effects[0] == {"type": "start_workout", "exercise": "squats"}
    r = brain.handle("how do I look", env(privacy=True))
    assert "Privacy mode is on" in r.speech and not r.effects
    r = brain.handle("how do I look", env())
    assert r.effects[0]["type"] == "vision"


def test_queries_work_without_data(tmp_path):
    brain, s, _ = make_brain(tmp_path, with_llm=False)
    for q in ("how am I doing", "what do you know about me", "how was my week", "how's my sleep", "what's my streak",
              "what time is it", "will it rain", "what can you do", "show my progress"):
        r = brain.handle(q, env())
        assert r.handled
    assert "9:00 AM" in brain.handle("what time is it", env()).speech


def test_goodnight_triggers_night_mode(tmp_path):
    brain, s, _ = make_brain(tmp_path, with_llm=False)
    s.memory.add_task("Ada", "dentist", due_at=NOW + timedelta(days=1))
    r = brain.handle("good night", env())
    assert {"type": "night_mode"} in r.effects and "dentist" in r.speech


def test_habit_logging_declines_unknown_habits(tmp_path):
    brain, s, _ = make_brain(tmp_path, with_llm=False)
    r = brain.handle("I finished my homework", env())
    assert r.source == "fallback"  # not a tracked habit: falls through
    brain.handle("remind me every morning to take my vitamins", env())
    r = brain.handle("I took my vitamins", env())
    assert "done" in r.speech.lower()
