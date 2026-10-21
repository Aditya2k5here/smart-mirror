"""Dependency bundle shared by the brain, proactive engine and runtime."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from assistant.llm import LLMRouter
from assistant.routines import BreathingSession, RoutineRunner
from assistant.timers import TimerService
from core.config import CONFIG
from memory.goal_service import GoalService
from memory.habit_service import HabitService
from memory.journal_service import JournalService
from memory.memory_service import MemoryService
from memory.nutrition_service import NutritionService
from memory.profile_service import ProfileService


@dataclass
class Services:
    memory: MemoryService
    profile: ProfileService
    goals: GoalService
    nutrition: NutritionService
    habits: HabitService
    journal: JournalService
    llm: LLMRouter = field(default_factory=LLMRouter)
    timers: TimerService = field(default_factory=TimerService)
    routines: RoutineRunner = field(default_factory=RoutineRunner)
    breathing: BreathingSession = field(default_factory=BreathingSession)

    @classmethod
    def for_db(cls, db_path: Path, llm: LLMRouter | None = None) -> "Services":
        return cls(
            memory=MemoryService(db_path), profile=ProfileService(db_path), goals=GoalService(db_path),
            nutrition=NutritionService(db_path), habits=HabitService(db_path), journal=JournalService(db_path),
            llm=llm or LLMRouter(),
        )

    @classmethod
    def default(cls) -> "Services":
        return cls.for_db(CONFIG.db_path)
