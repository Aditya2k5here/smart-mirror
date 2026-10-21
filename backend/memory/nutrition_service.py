"""Meals, calorie/protein targets and an offline food estimator."""
from __future__ import annotations

from datetime import date, datetime
from typing import Any
import re

from core.config import CONFIG
from memory.db import Store, known_user, now_iso, today_str
from memory.foods import estimate_meal, infer_meal_type  # noqa: F401  (re-exported)
from memory.profile_service import ACTIVITY_LEVELS

KCAL_PER_KG_FAT = 7700

def bmr_mifflin(weight_kg: float, height_cm: float, age: int, sex: str) -> float:
    base = 10 * weight_kg + 6.25 * height_cm - 5 * age
    return base + 5 if sex == "male" else base - 161 if sex == "female" else base - 78


class NutritionService(Store):
    def log_meal(self, user: str, description: str, calories: float | None, protein_g: float = 0,
                 carbs_g: float = 0, fat_g: float = 0, meal_type: str = "", estimated: bool = True,
                 at: datetime | None = None) -> dict[str, Any] | None:
        if not known_user(user) or calories is None:
            return None
        at = at or datetime.now()
        meal_type = meal_type or infer_meal_type(description, at)
        mid = self.execute(
            "INSERT INTO meals (user, description, meal_type, calories, protein_g, carbs_g, fat_g, at, day, estimated) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (user, description[:200], meal_type, calories, protein_g, carbs_g, fat_g,
             at.isoformat(timespec="seconds"), at.date().isoformat(), int(estimated)),
        )
        return {"id": mid, "description": description, "calories": calories, "protein_g": protein_g, "meal_type": meal_type}

    def undo_last_meal(self, user: str) -> dict[str, Any] | None:
        row = self.one("SELECT * FROM meals WHERE user = ? ORDER BY id DESC LIMIT 1", (user,))
        if row:
            self.execute("DELETE FROM meals WHERE id = ?", (row["id"],))
        return row

    def update_last_meal(self, user: str, calories: float) -> dict[str, Any] | None:
        row = self.one("SELECT * FROM meals WHERE user = ? ORDER BY id DESC LIMIT 1", (user,))
        if not row:
            return None
        self.execute("UPDATE meals SET calories = ?, estimated = 0 WHERE id = ?", (calories, row["id"]))
        return {**row, "calories": calories}

    def meals_for(self, user: str, day: str | None = None) -> list[dict[str, Any]]:
        return self.query("SELECT * FROM meals WHERE user = ? AND day = ? ORDER BY at ASC", (user, day or today_str()))

    def totals(self, user: str, day: str | None = None) -> dict[str, float]:
        row = self.one(
            "SELECT COALESCE(SUM(calories),0) c, COALESCE(SUM(protein_g),0) p, COALESCE(SUM(carbs_g),0) cb, "
            "COALESCE(SUM(fat_g),0) f, COUNT(*) n FROM meals WHERE user = ? AND day = ?",
            (user, day or today_str()),
        ) or {}
        return {"calories": row.get("c", 0), "protein_g": row.get("p", 0), "carbs_g": row.get("cb", 0),
                "fat_g": row.get("f", 0), "meals": row.get("n", 0)}

    def history(self, user: str, days: int = 7) -> list[dict[str, Any]]:
        return self.query(
            "SELECT day, SUM(calories) AS calories, SUM(protein_g) AS protein_g, COUNT(*) AS meals FROM meals "
            "WHERE user = ? AND day >= date('now', ?) GROUP BY day ORDER BY day",
            (user, f"-{days} day"),
        )

    # ---- targets ---------------------------------------------------------
    def targets(self, user: str, profile: dict[str, Any], weight_kg: float | None,
                weekly_goal_rate_kg: float | None = None) -> dict[str, Any]:
        """Daily calorie/protein/water targets with the information still missing."""
        missing = [k for k in ("height_cm", "age", "sex") if not profile.get(k)]
        if weight_kg is None:
            missing.insert(0, "weight")
        level = ACTIVITY_LEVELS.get(profile.get("activity_level") or "light", 1.375)
        result: dict[str, Any] = {"missing": missing, "basis": "default"}
        if not missing:
            bmr = bmr_mifflin(weight_kg, profile["height_cm"], int(profile["age"]), profile["sex"])
            tdee = bmr * level
            target = tdee
            if weekly_goal_rate_kg:
                target = tdee + weekly_goal_rate_kg * KCAL_PER_KG_FAT / 7  # negative rate => deficit
            floor = 1500 if profile["sex"] == "male" else 1200
            target = max(floor, min(target, tdee + 700))
            result.update(bmr=round(bmr), tdee=round(tdee), calories=int(round(target / 10) * 10), basis="estimated")
        else:
            result.update(calories=2000 if weight_kg is None else int(round((weight_kg * 28) / 10) * 10))
        weight = weight_kg or 70
        result["protein_g"] = int(round(weight * (1.6 if weekly_goal_rate_kg and weekly_goal_rate_kg < 0 else 1.2)))
        result["water_ml"] = int(round(weight * 35 / 250) * 250)
        return result


nutrition_service = NutritionService(CONFIG.db_path)
