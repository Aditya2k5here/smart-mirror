"""Demo persona: seeds a believable history so the UI and assistant have something real to show.

`python -m backend.server --demo` runs with no camera, mic or API keys needed.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
import math
import random

from assistant.services import Services

DEMO_USER = "Alex"


def demo_environment(now: datetime | None = None) -> dict:
    now = now or datetime.now()
    nxt = (now + timedelta(minutes=70)).replace(second=0, microsecond=0)
    later = (now + timedelta(hours=4)).replace(second=0, microsecond=0)
    return {
        "weather": {"city": "Bangalore", "temperature": 24, "description": "partly cloudy", "high": 28, "low": 19, "stale": False},
        "events": [
            {"time": nxt.strftime("%I:%M %p"), "title": "Design review", "start": nxt.isoformat()},
            {"time": later.strftime("%I:%M %p"), "title": "Gym with Sam", "start": later.isoformat()},
        ],
        "news": [
            "Sleep researchers link consistent bedtimes to better weight control",
            "Morning walks beat evening ones for blood sugar, new study finds",
            "City adds 40 km of protected cycle lanes",
        ],
    }


def seed_demo(services: Services, user: str = DEMO_USER) -> bool:
    """Idempotent: only seeds a fresh database."""
    if services.profile.get_profile(user):
        return False
    rng = random.Random(7)
    today = date.today()
    services.profile.ensure_profile(user)
    services.profile.update_profile(user, age=29, sex="male", height_cm=178, activity_level="light",
                                    wake_time="07:00", bed_time="23:30")

    # Eight weeks of weigh-ins: ~0.55 kg/week down with believable daily noise.
    weeks = 8
    for i in range(weeks * 7, -1, -1):
        day = today - timedelta(days=i)
        if i % 7 == 3 and rng.random() < 0.5:
            continue  # the odd missed weigh-in
        value = 88.6 - (weeks * 7 - i) * 0.078 + math.sin(i / 2.3) * 0.35 + rng.uniform(-0.25, 0.25)
        services.goals.add_measurement(user, "weight", round(value, 1), "kg",
                                       at=datetime.combine(day, datetime.min.time()).replace(hour=7, minute=15), source="demo")
    start = today - timedelta(days=weeks * 7)
    goal = services.goals.create_goal(user, "Lose 10.0 kg", kind="weight", metric="weight", unit="kg",
                                      start_value=88.6, target_value=78.6, direction="down",
                                      target_date=today + timedelta(days=88))
    services.goals.execute("UPDATE goals SET start_date = ? WHERE id = ?", (start.isoformat(), goal["id"]))
    services.goals.execute("UPDATE goals SET milestone = 25 WHERE id = ?", (goal["id"],))

    for hours, offset in ((6.5, 3), (7.2, 2), (6.8, 1)):
        services.goals.add_measurement(user, "sleep_hours", hours, "h", at=datetime.now() - timedelta(days=offset))

    # Habits, with a streak worth celebrating.
    vitamins = services.habits.ensure_habit(user, "Vitamins", "", 1, "08:00")
    for i in range(1, 7):
        services.habits.log(user, vitamins["id"], 1, at=datetime.now() - timedelta(days=i))
    meditate = services.habits.ensure_habit(user, "Meditate", "min", 10)
    for i in range(1, 4):
        services.habits.log(user, meditate["id"], 10, at=datetime.now() - timedelta(days=i))
    services.habits.log_water(user, 750, 2500)

    # Today's food so far.
    services.nutrition.log_meal(user, "oatmeal, banana", 265, 7.3, 54, 3.4, meal_type="breakfast", at=datetime.now().replace(hour=8, minute=5))
    services.nutrition.log_meal(user, "2 egg, toast", 236, 14.6, 15, 11.6, meal_type="breakfast", at=datetime.now().replace(hour=8, minute=10))

    # Workouts this week.
    for offset, exercise, reps in ((1, "squats", 40), (3, "pushups", 30), (5, "curls", 36)):
        services.habits.log_workout(user, exercise, reps, 420, at=datetime.now() - timedelta(days=offset))

    # What the mirror has learned over time.
    for text, category in (
        ("loves trail running on weekends", "preference"),
        ("dog is named Biscuit", "person"),
        ("works as a product designer", "work"),
        ("prefers a quiet, calm tone in the morning", "preference"),
        ("is vegetarian on weekdays", "health"),
        ("is training for a 10k in December", "goal"),
    ):
        services.profile.add_fact(user, text, category, source="demo", importance=0.7)

    services.memory.add_task(user, "Send the invoice", due_at=datetime.now() + timedelta(hours=2))
    services.memory.add_task(user, "Book dentist")
    services.memory.add_task(user, "Pick up dry cleaning", due_at=datetime.now() + timedelta(hours=6))
    services.journal.save_summary(
        user, (today - timedelta(days=1)).isoformat(),
        "You hit your water target, squeezed in a squat session and cooked dinner at home.")
    return True
