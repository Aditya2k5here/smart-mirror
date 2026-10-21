from datetime import datetime

import pytest

from assistant.intents import ParseContext, extract_name, parse_intent

NOW = datetime(2026, 10, 1, 14, 30)  # Thursday afternoon


def ctx(**kw):
    return ParseContext(now=NOW, **kw)


def intent(text, **kw):
    return parse_intent(text, ctx(**kw))


@pytest.mark.parametrize("text,kg", [
    ("I weigh 82.5 kilos", 82.5),
    ("my weight is eighty two", 82),
    ("I weigh 180 pounds", 81.65),
    ("weighed in at 79 point 4 kg", 79.4),
    ("log my weight 77", 77),
    ("I'm 91 kilos today", 91),
    ("Hey mirror, my weight today is 68.2", 68.2),
])
def test_weight_logging(text, kg):
    i = intent(text)
    assert i and i.name == "log_weight"
    assert i.params["value_kg"] == pytest.approx(kg, abs=0.05)


@pytest.mark.parametrize("text", ["I am 28 years old", "I'm 28", "my weight is a mystery", "I weigh 5 kilos", "I am 5 minutes away"])
def test_weight_false_positives(text):
    i = intent(text)
    assert not (i and i.name == "log_weight")


def test_weight_goals():
    i = intent("I want to lose 8 kilos by December 31")
    assert i.name == "set_weight_goal" and i.params["delta_kg"] == 8 and i.params["direction"] == "down"
    assert i.params["by"] == "2026-12-31"
    i = intent("my goal is to weigh 70 kg")
    assert i.name == "set_weight_goal" and i.params["target_kg"] == 70
    i = intent("I want to gain 4 kg")
    assert i.params["direction"] == "up"
    i = intent("help me lose 10 pounds in 3 months")
    assert i.params["delta_kg"] == pytest.approx(4.54, abs=0.01) and i.params["by"] == "2027-01-01"


def test_generic_goal():
    i = intent("my goal is to run a 10k by march")
    assert i.name == "set_goal" and "10k" in i.params["title"].lower()


def test_profile():
    i = intent("I'm 178 centimeters tall")
    assert i.name == "update_profile" and i.params["height_cm"] == 178
    i = intent("I am 5 feet 10")
    assert i.params["height_cm"] == pytest.approx(177.8)
    i = intent("I'm 29 years old")
    assert i.params == {"age": 29}
    i = intent("I am male")
    assert i.params == {"sex": "male"}
    i = intent("I work out 4 times a week")
    assert i.params["activity_level"] == "moderate"
    i = intent("I usually go to bed at 11pm")
    assert i.params["bed_time"] == "23:00"
    i = intent("I usually wake up at 6:30")
    assert i.params["wake_time"] == "06:30"


@pytest.mark.parametrize("text,ml", [
    ("I drank a glass of water", 250), ("had two glasses of water", 500), ("log 500 ml of water", 500),
    ("I finished a bottle of water", 500), ("I just drank some water", 250), ("drank a litre of water", 1000),
])
def test_water(text, ml):
    i = intent(text)
    assert i and i.name == "log_water" and i.params["ml"] == ml


def test_water_questions_are_not_logs():
    assert intent("how much water did I drink").params == {"topic": "water"}


def test_meals():
    i = intent("I had two eggs and toast for breakfast")
    assert i.name == "log_meal"
    i = intent("I ate 450 calories of pasta")
    assert i.name == "log_meal" and i.params["calories"] == 450
    assert intent("that was 600 calories").name == "fix_meal_calories"
    assert intent("undo that meal").name == "undo_meal"
    assert intent("I had a glass of water").name == "log_water"


def test_timers():
    i = intent("set a timer for ten minutes")
    assert i.name == "set_timer" and i.params["seconds"] == 600
    i = intent("timer for 1 hour 30 minutes")
    assert i.params["seconds"] == 5400
    assert intent("cancel the timer").name == "cancel_timer"
    assert intent("how much time is left on the timer").name == "timer_status"
    i = intent("start a pomodoro")
    assert i.name == "start_focus" and i.params["minutes"] == 25
    i = intent("set a 20 minute timer for laundry")
    assert i.params["seconds"] == 1200 and i.params["label"] == "laundry"


def test_reminders():
    i = intent("remind me to call mom in twenty minutes")
    assert i.name == "add_reminder" and i.params["text"] == "Call mom"
    assert i.params["when"] == "2026-10-01T14:50:00"
    i = intent("remind me tomorrow at 9am to send the report")
    assert i.params["text"] == "Send the report" and i.params["when"] == "2026-10-02T09:00:00"
    i = intent("remind me to buy milk")
    assert i.params["text"] == "Buy milk" and i.params["when"] is None
    i = intent("remind me every morning to take my vitamins")
    assert i.name == "add_habit" and i.params["remind_times"] == "08:00"
    assert intent("what are my reminders").name == "list_tasks"
    assert intent("mark task done").name == "complete_task"


def test_routines_and_breathing():
    assert intent("start my morning routine").params == {"name": "morning"}
    assert intent("let's do skincare").params == {"name": "skincare"}
    assert intent("time to brush my teeth").params == {"name": "brushing"}
    assert intent("I need to calm down").name == "start_breathing"
    assert intent("box breathing").params["pattern"] == "box"
    # control phrases only apply while a routine runs
    assert intent("next", routine_active=True).name == "routine_next"
    assert intent("next") is None
    assert intent("stop the routine", routine_active=True).name == "routine_stop"


def test_workout():
    i = intent("start a squat workout")
    assert i.name == "start_workout" and i.params["exercise"] == "squats"
    assert intent("stop workout").name == "stop_workout"
    assert intent("training mode").name == "start_workout"
    assert intent("how many workouts did I do").params == {"topic": "workouts"}


def test_music_and_home():
    assert intent("pause the music").params == {"command": "pause"}
    assert intent("next song").params == {"command": "next"}
    assert intent("turn the volume down").params == {"command": "volume_down"}
    i = intent("play some lo-fi beats")
    assert i.params == {"command": "search", "query": "lo-fi beats"}
    i = intent("turn off the bedroom lights")
    assert i.name == "lights" and i.params == {"command": "off", "room": "bedroom"}
    assert intent("turn on the lights").params["command"] == "on"
    assert intent("dim the lights").params["command"] == "dim"


def test_notes():
    i = intent("add milk to my shopping list")
    assert i.name == "add_note" and i.params == {"text": "milk", "list": "shopping"}
    assert intent("what's on my shopping list").params == {"list": "shopping"}
    assert intent("remember that my locker code is 4412").name == "remember"
    assert intent("take a note call the plumber").params["text"] == "call the plumber"


def test_mood_sleep_misc():
    assert intent("I'm feeling really stressed").params == {"mood": "stressed"}
    assert intent("I am so tired").params == {"mood": "tired"}
    assert intent("I slept 6 hours last night").params == {"hours": 6}
    assert intent("good night").name == "goodnight"
    assert intent("how do I look").name == "outfit_check"
    assert intent("privacy mode").params == {"on": True}
    assert intent("turn privacy mode off").params == {"on": False}
    assert intent("forget everything about me").name == "forget_all"
    assert intent("forget that I like jazz").params == {"phrase": "i like jazz"}


def test_queries():
    for text, topic in [
        ("how am I doing", "progress"), ("how many calories do I have left", "calories"),
        ("what do you know about me", "facts"), ("what's on my calendar", "agenda"),
        ("what time is it", "time"), ("will it rain today", "weather"), ("how was my week", "week"),
        ("what can you do", "help"),
    ]:
        i = intent(text)
        assert i and i.name == "query" and i.params["topic"] == topic, text


def test_names():
    assert extract_name("my name is aditya") == "Aditya"
    assert extract_name("call me adi") == "Adi"
    assert extract_name("my name is priya sharma") == "Priya Sharma"
    assert extract_name("it's Sam", strict=True) == "Sam"
    assert extract_name("Sam", strict=False) == "Sam"
    assert extract_name("yes", strict=False) is None
    assert extract_name("I'm tired") is None
    assert intent("my name is aditya").name == "set_name"


def test_stop_and_nothing():
    assert intent("stop").name == "stop_speaking"
    assert intent("tell me a story about dragons") is None
    assert intent("what's the capital of France") is None
    assert intent("I had a great day at work") is None
