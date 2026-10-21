"""Offline food database and meal-text estimator (pure functions, no I/O)."""
from __future__ import annotations

from datetime import datetime
from typing import Any
import re

from core.timeparse import normalize_numbers

# name(s) -> (kcal, protein_g, carbs_g, fat_g) for one typical serving
_FOODS: list[tuple[tuple[str, ...], tuple[float, float, float, float]]] = [
    (("egg", "eggs", "boiled egg", "boiled eggs", "fried egg"), (78, 6.3, 0.6, 5.3)),
    (("omelette", "omelet"), (154, 10, 1, 12)),
    (("toast", "bread", "slice of bread", "slices of bread"), (80, 3, 14, 1)),
    (("sandwich",), (300, 12, 34, 12)),
    (("avocado toast",), (250, 6, 25, 15)),
    (("peanut butter",), (95, 4, 3, 8)),
    (("banana",), (105, 1.3, 27, 0.4)),
    (("apple",), (95, 0.5, 25, 0.3)),
    (("orange",), (62, 1.2, 15, 0.2)),
    (("mango",), (200, 2.8, 50, 1.3)),
    (("grapes",), (100, 1, 27, 0.2)),
    (("watermelon",), (46, 1, 11, 0.2)),
    (("papaya",), (62, 0.7, 16, 0.4)),
    (("berries", "strawberries", "blueberries"), (70, 1, 17, 0.5)),
    (("oatmeal", "oats", "porridge"), (160, 6, 27, 3)),
    (("cereal", "cornflakes", "muesli"), (150, 3, 33, 1)),
    (("poha",), (250, 5, 45, 6)),
    (("upma",), (250, 6, 38, 8)),
    (("idli", "idlis", "idly"), (58, 2, 12, 0.3)),
    (("masala dosa",), (250, 5, 38, 9)),
    (("dosa", "dosas"), (135, 3, 22, 4)),
    (("vada", "medu vada"), (150, 4, 15, 8)),
    (("paratha", "parathas"), (260, 6, 36, 10)),
    (("roti", "rotis", "chapati", "chapatis", "phulka"), (100, 3, 18, 2)),
    (("naan",), (260, 8, 45, 5)),
    (("puri", "poori"), (100, 2, 12, 5)),
    (("fried rice",), (350, 8, 58, 10)),
    (("biryani", "chicken biryani", "veg biryani"), (450, 18, 60, 16)),
    (("pulao", "pilaf"), (300, 6, 52, 7)),
    (("rice", "steamed rice", "white rice"), (205, 4, 45, 0.4)),
    (("dal", "daal", "lentils", "lentil soup"), (180, 10, 28, 3)),
    (("sambar",), (130, 6, 20, 3)),
    (("rajma",), (220, 12, 35, 3)),
    (("chole", "chana masala", "chickpeas"), (270, 13, 40, 7)),
    (("paneer curry", "paneer butter masala", "palak paneer", "paneer"), (300, 14, 10, 23)),
    (("chicken curry",), (280, 25, 8, 16)),
    (("grilled chicken", "chicken breast", "chicken"), (165, 31, 0, 3.6)),
    (("fish curry",), (250, 22, 8, 14)),
    (("fish",), (200, 22, 0, 12)),
    (("egg curry",), (220, 13, 8, 15)),
    (("curd", "yogurt", "yoghurt", "raita"), (100, 6, 8, 4)),
    (("milk",), (120, 6, 10, 5)),
    (("lassi",), (150, 5, 22, 4)),
    (("chai", "masala chai", "tea with milk"), (80, 2, 11, 3)),
    (("black coffee",), (5, 0.3, 0, 0)),
    (("coffee", "latte", "cappuccino"), (90, 4, 8, 4)),
    (("green tea", "tea"), (3, 0, 0.5, 0)),
    (("juice",), (110, 1, 26, 0.3)),
    (("soda", "coke", "cola", "soft drink"), (140, 0, 39, 0)),
    (("samosa", "samosas"), (260, 4, 28, 15)),
    (("pakora", "pakoda", "bhajji", "bhajia"), (250, 5, 25, 15)),
    (("pizza", "slice of pizza", "slices of pizza"), (285, 12, 36, 10)),
    (("burger", "cheeseburger"), (500, 25, 40, 26)),
    (("fries", "french fries"), (365, 4, 48, 17)),
    (("pasta", "spaghetti", "macaroni"), (400, 14, 70, 7)),
    (("noodles", "maggi", "ramen"), (380, 9, 56, 14)),
    (("salad",), (80, 3, 10, 3)),
    (("soup",), (100, 4, 13, 3)),
    (("sushi",), (250, 9, 38, 5)),
    (("wrap", "roll", "kathi roll"), (350, 15, 40, 14)),
    (("protein shake", "whey", "protein powder"), (150, 25, 5, 2)),
    (("protein bar",), (200, 20, 22, 7)),
    (("smoothie",), (200, 5, 40, 3)),
    (("almonds", "nuts", "cashews", "walnuts", "peanuts"), (165, 5.5, 6, 14)),
    (("cookie", "cookies", "biscuit", "biscuits"), (70, 1, 10, 3)),
    (("chocolate", "chocolate bar"), (230, 3, 26, 13)),
    (("ice cream",), (140, 2, 16, 7)),
    (("cake", "pastry", "brownie"), (350, 4, 48, 16)),
    (("chips", "crisps"), (150, 2, 15, 10)),
    (("gulab jamun",), (150, 2, 25, 5)),
    (("jalebi",), (150, 1, 30, 4)),
]
_ALIAS_INDEX: list[tuple[str, tuple[float, float, float, float], str]] = sorted(
    ((alias, vals, names[0]) for names, vals in _FOODS for alias in names),
    key=lambda row: -len(row[0]),
)

_MEAL_WORDS = ("breakfast", "lunch", "dinner", "snack", "brunch", "supper")
_SIZE = {"large": 1.3, "big": 1.3, "huge": 1.6, "small": 0.7, "little": 0.6, "tiny": 0.5, "medium": 1.0, "half": 0.5}


def infer_meal_type(text: str, now: datetime | None = None) -> str:
    for word in _MEAL_WORDS:
        if word in text:
            return "dinner" if word == "supper" else word
    hour = (now or datetime.now()).hour
    if 5 <= hour < 11:
        return "breakfast"
    if 11 <= hour < 15:
        return "lunch"
    if 15 <= hour < 18:
        return "snack"
    if 18 <= hour < 23:
        return "dinner"
    return "snack"


def estimate_meal(text: str) -> dict[str, Any]:
    """Estimate calories from a spoken description. Unknown foods are reported back."""
    t = normalize_numbers(text)
    t = re.sub(r"\b(?:i|we)?\s*(?:just\s+)?(?:had|ate|eaten|have|having|eating|log|logged|for|breakfast|lunch|dinner|snack|brunch)\b", " ", t)
    t = re.sub(r"\b(?:today|this morning|tonight|earlier|a while ago|some|of|my|the|with)\b", lambda m: " " if m.group(0) != "with" else ",", t)
    chunks = [c.strip() for c in re.split(r",|\band\b|\bplus\b|\+|\bwith\b", t) if c.strip()]
    items: list[dict[str, Any]] = []
    unmatched: list[str] = []
    for chunk in chunks:
        match = next(((alias, vals, name) for alias, vals, name in _ALIAS_INDEX
                      if re.search(rf"\b{re.escape(alias)}\b", chunk)), None)
        if not match:
            if re.search(r"[a-z]{3,}", chunk) and chunk not in ("a", "an"):
                unmatched.append(chunk)
            continue
        alias, (kcal, protein, carbs, fat), name = match
        qty = 1.0
        number = re.search(r"\b(\d+(?:\.\d+)?)\b", chunk)
        if number:
            qty = float(number.group(1))
        for word, mult in _SIZE.items():
            if re.search(rf"\b{word}\b", chunk):
                qty = qty * mult if number else mult
                break
        qty = max(0.25, min(qty, 20.0))
        items.append({
            "name": name, "qty": qty, "calories": round(kcal * qty), "protein_g": round(protein * qty, 1),
            "carbs_g": round(carbs * qty, 1), "fat_g": round(fat * qty, 1),
        })
    return {
        "items": items,
        "calories": sum(i["calories"] for i in items),
        "protein_g": round(sum(i["protein_g"] for i in items), 1),
        "carbs_g": round(sum(i["carbs_g"] for i in items), 1),
        "fat_g": round(sum(i["fat_g"] for i in items), 1),
        "unmatched": unmatched,
    }
