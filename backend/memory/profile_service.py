"""Who the user is: profile numbers, durable facts, and relevance-ranked recall."""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from math import log
from pathlib import Path
from typing import Any
import json
import re

from core.config import CONFIG
from memory.db import Store, known_user, now_iso

ACTIVITY_LEVELS = {
    "sedentary": 1.2,
    "light": 1.375,
    "moderate": 1.55,
    "active": 1.725,
    "athlete": 1.9,
}

FACT_CATEGORIES = {"preference", "person", "health", "work", "routine", "goal", "event", "other"}

_STOP = set(
    "a an and are as at be but by do for from had has have he her his i if in is it its me my of on or our she so that the "
    "their them they this to was we were what when where which who will with you your about can how just not now than then "
    "there these those too very would could should did does done been being into out up".split()
)


def _stem(word: str) -> str:
    """Tiny plural/verb-s stemmer: enough to match 'likes'/'like' and 'dogs'/'dog'."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def tokens(text: str) -> list[str]:
    words = (w.strip("'") for w in re.findall(r"[a-z0-9']+", text.lower()))
    return [_stem(w) for w in words if w and w not in _STOP and len(w) > 1]


class ProfileService(Store):
    # ---- profile ---------------------------------------------------------
    def get_profile(self, user: str) -> dict[str, Any]:
        if not known_user(user):
            return {}
        row = self.one("SELECT * FROM profiles WHERE user = ?", (user,))
        if not row:
            return {}
        row["extra"] = json.loads(row.get("extra") or "{}")
        return row

    def ensure_profile(self, user: str) -> dict[str, Any]:
        """Create the profile on first sight. Returns the profile (with `is_new`)."""
        if not known_user(user):
            return {}
        existing = self.get_profile(user)
        if existing:
            return {**existing, "is_new": False}
        now = now_iso()
        self.execute(
            "INSERT INTO profiles (user, created_at, updated_at, last_seen_at, visit_count, units) VALUES (?, ?, ?, ?, 0, ?)",
            (user, now, now, now, "metric"),
        )
        return {**self.get_profile(user), "is_new": True}

    def update_profile(self, user: str, **fields: Any) -> dict[str, Any]:
        if not known_user(user):
            return {}
        self.ensure_profile(user)
        allowed = {"age", "sex", "height_cm", "activity_level", "wake_time", "bed_time", "timezone", "units"}
        sets, params = [], []
        for key, value in fields.items():
            if key in allowed and value is not None:
                sets.append(f"{key} = ?")
                params.append(value)
        if "extra" in fields and isinstance(fields["extra"], dict):
            merged = {**self.get_profile(user).get("extra", {}), **fields["extra"]}
            sets.append("extra = ?")
            params.append(json.dumps(merged))
        if sets:
            sets.append("updated_at = ?")
            params.extend([now_iso(), user])
            self.execute(f"UPDATE profiles SET {', '.join(sets)} WHERE user = ?", params)
        return self.get_profile(user)

    def touch_visit(self, user: str) -> None:
        if known_user(user):
            self.ensure_profile(user)
            self.execute(
                "UPDATE profiles SET last_seen_at = ?, visit_count = COALESCE(visit_count, 0) + 1 WHERE user = ?",
                (now_iso(), user),
            )

    def wake_bed(self, user: str) -> tuple[str, str]:
        profile = self.get_profile(user)
        return (profile.get("wake_time") or CONFIG.default_wake_time, profile.get("bed_time") or CONFIG.default_bed_time)

    def delete_user(self, user: str) -> None:
        if not known_user(user):
            return
        for table in (
            "profiles", "facts", "goals", "measurements", "meals", "habits", "habit_logs", "workouts",
            "journal", "conversations", "notes", "nudge_log", "sessions", "tasks", "memory_events", "user_notes",
        ):
            self.execute(f"DELETE FROM {table} WHERE user = ?", (user,))

    def export_user(self, user: str) -> dict[str, Any]:
        """Everything stored about one person, for data portability."""
        if not known_user(user):
            return {}
        tables = ("profiles", "facts", "goals", "measurements", "meals", "habits", "habit_logs", "workouts",
                  "journal", "conversations", "notes", "tasks", "sessions")
        return {"user": user, "exported_at": now_iso(),
                **{t: self.query(f"SELECT * FROM {t} WHERE user = ?", (user,)) for t in tables}}

    # ---- facts -----------------------------------------------------------
    def add_fact(self, user: str, text: str, category: str = "other", source: str = "conversation",
                 importance: float = 0.5) -> int | None:
        text = " ".join((text or "").split())
        if not known_user(user) or len(text) < 4:
            return None
        category = category if category in FACT_CATEGORIES else "other"
        duplicate = self._find_similar(user, text)
        now = now_iso()
        if duplicate:
            self.execute(
                "UPDATE facts SET last_seen_at = ?, importance = MIN(1.0, importance + 0.05) WHERE id = ?",
                (now, duplicate["id"]),
            )
            return duplicate["id"]
        return self.execute(
            "INSERT INTO facts (user, category, text, source, importance, created_at, last_seen_at) VALUES (?,?,?,?,?,?,?)",
            (user, category, text[:300], source, importance, now, now),
        )

    def _find_similar(self, user: str, text: str) -> dict[str, Any] | None:
        mine = set(tokens(text))
        if not mine:
            return None
        for row in self.query("SELECT id, text FROM facts WHERE user = ? AND active = 1", (user,)):
            theirs = set(tokens(row["text"]))
            if theirs and len(mine & theirs) / len(mine | theirs) >= 0.72:
                return row
        return None

    def list_facts(self, user: str, limit: int = 50, category: str | None = None) -> list[dict[str, Any]]:
        if not known_user(user):
            return []
        sql = "SELECT * FROM facts WHERE user = ? AND active = 1"
        params: list[Any] = [user]
        if category:
            sql += " AND category = ?"
            params.append(category)
        sql += " ORDER BY importance DESC, last_seen_at DESC LIMIT ?"
        params.append(limit)
        return self.query(sql, params)

    def forget_fact(self, user: str, phrase: str) -> list[str]:
        """Deactivate facts matching a phrase ('forget that I like jazz')."""
        want = set(tokens(phrase))
        if not known_user(user) or not want:
            return []
        removed = []
        for row in self.query("SELECT id, text FROM facts WHERE user = ? AND active = 1", (user,)):
            have = set(tokens(row["text"]))
            if have and len(want & have) / len(want) >= 0.6:
                self.execute("UPDATE facts SET active = 0 WHERE id = ?", (row["id"],))
                removed.append(row["text"])
        return removed

    def recall(self, user: str, query: str, limit: int = 6) -> list[str]:
        """BM25-flavoured ranking of facts and past journal lines against a query."""
        if not known_user(user):
            return []
        q = tokens(query)
        if not q:
            return [f["text"] for f in self.list_facts(user, limit=limit)]
        docs: list[tuple[str, float]] = [
            (f["text"], float(f["importance"] or 0.5)) for f in self.list_facts(user, limit=400)
        ]
        docs += [
            (r["text"], 0.35)
            for r in self.query(
                "SELECT text FROM journal WHERE user = ? ORDER BY id DESC LIMIT 120", (user,)
            )
        ]
        if not docs:
            return []
        df: Counter[str] = Counter()
        doc_tokens = []
        for text, _ in docs:
            toks = set(tokens(text))
            doc_tokens.append(toks)
            df.update(toks)
        n = len(docs)
        scored = []
        for (text, weight), toks in zip(docs, doc_tokens):
            overlap = [t for t in q if t in toks]
            if not overlap:
                continue
            score = sum(log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in set(overlap)) * (0.7 + weight)
            scored.append((score, text))
        scored.sort(reverse=True)
        seen, out = set(), []
        for _, text in scored:
            if text not in seen:
                seen.add(text)
                out.append(text)
            if len(out) >= limit:
                break
        return out

    # ---- derived profile numbers ----------------------------------------
    def age_label(self, user: str) -> str:
        profile = self.get_profile(user)
        return str(profile["age"]) if profile.get("age") else ""


_FACT_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"\bi (?:really )?(?:like|love|enjoy|prefer|adore) (?P<x>[^.,;!?]{3,60})", re.I), "preference", "likes {x}"),
    (re.compile(r"\bi (?:really )?(?:hate|dislike|can't stand|cannot stand) (?P<x>[^.,;!?]{3,60})", re.I), "preference", "dislikes {x}"),
    (re.compile(r"\bi(?:'m| am) (?:allergic|intolerant) to (?P<x>[^.,;!?]{3,40})", re.I), "health", "is allergic or intolerant to {x}"),
    (re.compile(r"\bi(?:'m| am) (?P<x>vegetarian|vegan|pescatarian|vegetarian|lactose[- ]intolerant|diabetic|gluten[- ]free)\b", re.I), "health", "is {x}"),
    (re.compile(r"\bi (?:work|am working) (?:at|for|as) (?P<x>[^.,;!?]{3,50})", re.I), "work", "works {x}"),
    (re.compile(r"\bi(?:'m| am) (?:a|an) (?P<x>[a-z ]{4,40}?(?:engineer|developer|designer|student|teacher|doctor|nurse|manager|founder|writer|artist|analyst|lawyer|chef))\b", re.I), "work", "is a {x}"),
    (re.compile(r"\bmy (?P<rel>wife|husband|partner|girlfriend|boyfriend|mom|mother|dad|father|brother|sister|son|daughter|dog|cat|friend|boss)(?:'s name)? (?:is|called|named) (?P<x>[a-z][a-z]+)", re.I), "person", "{rel} is named {x}"),
    (re.compile(r"\bi (?:usually |always |normally )?(?:wake up|get up) (?:at )?(?P<x>\d{1,2}(?::\d{2})?\s*(?:am|pm)?)", re.I), "routine", "usually wakes up around {x}"),
    (re.compile(r"\bi (?:usually |always |normally )?(?:go to bed|sleep) (?:at )?(?P<x>\d{1,2}(?::\d{2})?\s*(?:am|pm)?)", re.I), "routine", "usually goes to bed around {x}"),
    (re.compile(r"\bi(?:'m| am) training for (?P<x>[^.,;!?]{3,50})", re.I), "goal", "is training for {x}"),
    (re.compile(r"\b(?:remember|note) that (?P<x>[^.!?]{4,140})", re.I), "other", "{x}"),
]


def extract_facts(text: str) -> list[tuple[str, str]]:
    """Cheap offline fact extraction used when no LLM is reachable."""
    found: list[tuple[str, str]] = []
    for pattern, category, template in _FACT_PATTERNS:
        for m in pattern.finditer(text):
            groups = {k: (v or "").strip() for k, v in m.groupdict().items()}
            if category == "person" and "x" in groups:
                groups["x"] = groups["x"].title()
            found.append((category, template.format(**groups)))
    return found


profile_service = ProfileService(CONFIG.db_path)
