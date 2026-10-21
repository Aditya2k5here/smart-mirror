"""Small value types passed between the runtime, the brain and the proactive engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

GUESTS = ("Unknown", "Guest", "")


@dataclass
class Env:
    """What the mirror can perceive right now."""

    user: str = "Unknown"
    now: datetime = field(default_factory=datetime.now)
    emotion: str = "neutral"
    posture: str = "good"
    present: bool = True
    weather: dict[str, Any] | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    news: list[str] = field(default_factory=list)
    track: dict[str, Any] | None = None
    workout_active: bool = False
    privacy: bool = False
    capabilities: dict[str, bool] = field(default_factory=dict)

    @property
    def is_guest(self) -> bool:
        return self.user in GUESTS

    @property
    def day_part(self) -> str:
        hour = self.now.hour
        if 5 <= hour < 12:
            return "morning"
        if 12 <= hour < 17:
            return "afternoon"
        if 17 <= hour < 21:
            return "evening"
        return "night"


@dataclass
class Reply:
    speech: str = ""
    effects: list[dict[str, Any]] = field(default_factory=list)
    ui: dict[str, Any] = field(default_factory=dict)
    source: str = "local"  # local | llm | fallback
    handled: bool = True
    passthrough: bool = False  # side effects done; let the LLM word the answer if it can
    celebrate: str = ""  # milestone label for UI confetti
    keep_open: bool = False  # a question was asked; stay listening without the wake word
    override: bool = False  # this speech must replace the LLM's wording (e.g. a safety refusal)

    def merge(self, other: "Reply") -> "Reply":
        self.effects += other.effects
        self.ui.update(other.ui)
        self.celebrate = self.celebrate or other.celebrate
        return self


@dataclass
class Pending:
    kind: str
    data: dict[str, Any] = field(default_factory=dict)
    expires_at: float = 0.0
