from __future__ import annotations

from collections import Counter
from typing import Any

from memory.memory_service import MemoryService, memory_service


class AnalyticsService:
    def __init__(self, memory: MemoryService = memory_service) -> None:
        self.memory = memory

    def build_user_context(self, user: str) -> dict[str, Any]:
        if not user or user == "Unknown":
            return {
                "stress_score": 0.0,
                "posture_score": 1.0,
                "insight": "",
                "conversation_context": "The person has not been recognized yet.",
            }

        context = self.memory.get_personal_context(user)
        sessions = context["recent_sessions"]
        emotion_counts = Counter(row["dominant_emotion"] for row in sessions)
        dominant_recent = emotion_counts.most_common(1)[0][0] if emotion_counts else ""
        task_count = len(context["pending_tasks"])

        insight = self._build_insight(
            stress_score=context["stress_score"],
            posture_score=context["posture_score"],
            task_count=task_count,
            dominant_recent=dominant_recent,
        )

        weekly_summary = self.weekly_summary(user)

        conversation_context = (
            f"User: {user}. "
            f"Recent mood trend: {dominant_recent or 'not enough data'}. "
            f"Stress score: {context['stress_score']:.2f}. "
            f"Posture score: {context['posture_score']:.2f}. "
            f"Pending tasks: {task_count}. "
            f"Weekly summary: {weekly_summary['headline']} "
            "Respond like a calm premium smart mirror: brief, observant, never clinical."
        )

        return {
            **context,
            "dominant_recent_emotion": dominant_recent,
            "insight": insight,
            "weekly_summary": weekly_summary,
            "conversation_context": conversation_context,
        }

    def weekly_summary(self, user: str) -> dict[str, Any]:
        if not user or user == "Unknown":
            return {
                "headline": "",
                "sessions": 0,
                "stress_score": 0.0,
                "posture_score": 1.0,
                "dominant_emotion": "",
            }

        sessions = self.memory.get_recent_sessions(user, days=7)
        if not sessions:
            return {
                "headline": "Not enough weekly history yet.",
                "sessions": 0,
                "stress_score": 0.0,
                "posture_score": 1.0,
                "dominant_emotion": "",
            }

        emotion_counts = Counter(row["dominant_emotion"] for row in sessions)
        dominant = emotion_counts.most_common(1)[0][0]
        stress_score = self.memory.get_stress_score(user)
        posture_score = self.memory.get_posture_score(user)
        session_count = len(sessions)

        if stress_score >= 0.65:
            headline = "This week has looked heavier than usual."
        elif posture_score < 0.55:
            headline = "Posture needs a little more attention this week."
        elif dominant == "happy":
            headline = "This week has had a lighter rhythm."
        else:
            headline = "Your week looks steady."

        return {
            "headline": headline,
            "sessions": session_count,
            "stress_score": round(stress_score, 2),
            "posture_score": round(posture_score, 2),
            "dominant_emotion": dominant,
        }

    def _build_insight(
        self,
        stress_score: float,
        posture_score: float,
        task_count: int,
        dominant_recent: str,
    ) -> str:
        if stress_score >= 0.65:
            return "Recent check-ins have looked heavier than usual."
        if posture_score < 0.55:
            return "Posture has been drifting lately."
        if task_count >= 3:
            return "A few loose ends are waiting."
        if dominant_recent == "happy":
            return "Your recent rhythm looks lighter."
        return ""
