"""End-of-day reflection: distils the day into a diary line and durable facts.

This is how the mirror 'keeps learning': conversations become memories, and yesterday's
summary feeds tomorrow's briefing and the LLM's context.
"""
from __future__ import annotations

from datetime import date
from typing import Any
import logging

from assistant.brain import Brain
from assistant.reports import Reporter, money
from assistant.services import Services
from memory.db import known_user


class Reflector:
    def __init__(self, services: Services, reporter: Reporter) -> None:
        self.s, self.r = services, reporter

    def reflect(self, user: str, day: str | None = None) -> str | None:
        if not known_user(user):
            return None
        day = day or date.today().isoformat()
        turns = self.s.journal.turns_on(user, day)
        user_turns = [t["text"] for t in turns if t["role"] == "user"]
        stats = self.r.day_stats(user, day)
        has_data = bool(stats["calories"] or stats["water_ml"] or stats["workouts"] or stats["weight_kg"])
        if len(user_turns) < 2 and not has_data:
            return None

        summary = self._llm_reflect(user, day, turns, stats) or self._plain_summary(user, stats)
        if summary:
            self.s.journal.save_summary(user, day, summary)
        return summary

    def _plain_summary(self, user: str, stats: dict[str, Any]) -> str:
        t = self.r.targets(user)
        bits = []
        if stats["weight_kg"]:
            bits.append(f"weighed in at {self.r.fmt_weight(stats['weight_kg'], user)}")
        if stats["calories"]:
            bits.append(f"ate about {money(stats['calories'])} calories")
        if stats["water_ml"]:
            bits.append(f"drank {int(stats['water_ml'] / t['water_ml'] * 100)} percent of your water target")
        if stats["workouts"]:
            bits.append(f"did {stats['workouts']} workout{'s' if stats['workouts'] != 1 else ''}")
        if stats["tasks_done"]:
            bits.append(f"finished {stats['tasks_done']} task{'s' if stats['tasks_done'] != 1 else ''}")
        return ("You " + ", ".join(bits) + ".") if bits else ""

    def _llm_reflect(self, user: str, day: str, turns: list[dict[str, Any]], stats: dict[str, Any]) -> str | None:
        if not self.s.llm.available():
            return None
        transcript = "\n".join(f"{'Them' if t['role'] == 'user' else 'Mirror'}: {t['text']}" for t in turns[-40:])
        entries = "\n".join(e["text"] for e in self.s.journal.entries(user, 1, kind="entry"))
        facts_known = "; ".join(f["text"] for f in self.s.profile.list_facts(user, limit=15))
        prompt = (
            f"Date: {day}\nStats: {stats}\nJournal entries: {entries or 'none'}\nConversation:\n{transcript or 'none'}\n"
            f"Already known facts: {facts_known or 'none'}"
        )
        system = (
            "You maintain a smart mirror's memory of one person. From their day, write (1) summary: at most two warm, specific "
            "sentences addressed to them as 'you', mentioning concrete things they did or felt; (2) mood: one word; "
            "(3) facts: NEW durable facts worth remembering long-term (preferences, people, health notes, plans, routines), "
            "as short third-person phrases. Skip trivia and anything already known. "
            'Reply ONLY with JSON: {"summary": "...", "mood": "...", "facts": [{"text": "...", "category": "preference|person|health|work|routine|goal|event|other"}]}'
        )
        result = self.s.llm.chat(system, [{"role": "user", "content": prompt}], max_tokens=420, temperature=0.4, json_mode=True)
        if not result:
            return None
        data = Brain._parse_llm_json(result.text)
        summary = str(data.get("summary") or "").strip()
        if not summary:
            return None
        for fact in (data.get("facts") or [])[:6]:
            if isinstance(fact, dict):
                self.s.profile.add_fact(user, str(fact.get("text", "")), str(fact.get("category", "other")), source="reflection", importance=0.6)
        if data.get("mood"):
            self.s.journal.add_entry(user, f"Overall mood: {data['mood']}", mood=str(data["mood"]), kind="mood", day=day)
        logging.info("Reflection saved for %s (%s)", user, day)
        return summary
