from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import time


NEGATIVE_EMOTIONS = {"sad", "angry", "fear", "disgust"}

# State that describes the world, not the person: kept when someone walks away.
_PERSISTENT = (
    "weather", "events", "news", "track", "health", "diagnostics", "settings", "show_settings", "show_diagnostics",
    "privacy", "night_mode", "demo", "assistant_name", "capabilities", "timers",
)


@dataclass
class MirrorState:
    present: bool = False
    name: str = "Unknown"
    emotion: str = "neutral"
    posture: str = "good"
    events: list[dict[str, Any]] = field(default_factory=list)
    news: list[str] = field(default_factory=list)
    tasks: list[dict[str, Any]] = field(default_factory=list)
    weather: dict[str, Any] | None = None
    track: dict[str, Any] | None = None
    workout: dict[str, Any] = field(
        default_factory=lambda: {"active": False, "reps": 0, "timer": "00:00"}
    )
    assistant_listening: bool = False
    status: str = "idle"
    voice_state: str = "idle"
    insight: str = ""
    weekly_summary: dict[str, Any] = field(default_factory=dict)
    health: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    show_settings: bool = False
    show_diagnostics: bool = False
    stress_score: float = 0.0
    posture_score: float = 1.0
    session_started_at: datetime | None = None
    session_emotions: list[str] = field(default_factory=list)
    session_slouch_count: int = 0
    sequence: int = 0
    # --- personal assistant layer ---
    personal: dict[str, Any] = field(default_factory=dict)
    caption: dict[str, Any] = field(default_factory=lambda: {"user": "", "mirror": "", "at": 0})
    nudge: dict[str, Any] | None = None
    timers: list[dict[str, Any]] = field(default_factory=list)
    routine: dict[str, Any] = field(default_factory=lambda: {"active": False})
    breathing: dict[str, Any] = field(default_factory=lambda: {"active": False})
    focus: dict[str, Any] = field(default_factory=lambda: {"panel": "none"})
    celebrate: dict[str, Any] | None = None
    privacy: bool = False
    night_mode: bool = False
    alarm: bool = False
    enrolling: bool = False
    demo: bool = False
    assistant_name: str = "Mirror"
    capabilities: dict[str, Any] = field(default_factory=dict)
    conversation_open: bool = False


class StateManager:
    def __init__(self) -> None:
        self.state = MirrorState()

    def mark_dirty(self) -> None:
        self.state.sequence += 1

    def start_presence(self) -> None:
        if self.state.present:
            return
        self.state.present = True
        self.state.session_started_at = datetime.now()
        self.state.session_emotions = []
        self.state.session_slouch_count = 0
        self.state.status = "present"
        self.mark_dirty()

    def end_presence(self) -> dict[str, Any]:
        ended_at = datetime.now()
        session = {
            "user": self.state.name,
            "start_time": self.state.session_started_at,
            "end_time": ended_at,
            "emotions": list(self.state.session_emotions),
            "slouch_count": self.state.session_slouch_count,
        }
        kept = {key: getattr(self.state, key) for key in _PERSISTENT}
        sequence = self.state.sequence
        self.state = MirrorState(**kept)
        self.state.sequence = sequence
        self.mark_dirty()
        return session

    def set_identity(self, name: str) -> bool:
        name = name or "Unknown"
        if name == self.state.name:
            return False
        self.state.name = name
        self.mark_dirty()
        return True

    def set_emotion(self, emotion: str) -> bool:
        emotion = emotion or "neutral"
        if self.state.present:
            self.state.session_emotions.append(emotion)
        if emotion == self.state.emotion:
            return False
        self.state.emotion = emotion
        self.mark_dirty()
        return True

    def set_posture(self, posture: str) -> bool:
        posture = posture or "good"
        if posture == self.state.posture:
            return False
        self.state.posture = posture
        if posture == "slouching":
            self.state.session_slouch_count += 1
        self.mark_dirty()
        return True

    def update(self, **changes: Any) -> None:
        changed = False
        for key, value in changes.items():
            if hasattr(self.state, key) and getattr(self.state, key) != value:
                setattr(self.state, key, value)
                changed = True
        if changed:
            self.mark_dirty()

    def day_part(self) -> str:
        hour = datetime.now().hour
        if 5 <= hour < 12:
            return "morning"
        if 12 <= hour < 17:
            return "afternoon"
        if 17 <= hour < 21:
            return "evening"
        return "night"

    def tone(self) -> str:
        if self.state.workout.get("active"):
            return "training"
        if self.state.stress_score >= 0.55 or self.state.emotion in NEGATIVE_EMOTIONS:
            return "calm"
        if self.day_part() == "night" or self.state.night_mode:
            return "rest"
        return "focused"

    def scene(self) -> dict[str, str]:
        tone = self.tone()
        day_part = self.day_part()
        if self.state.workout.get("active"):
            return {"name": "training", "density": "focused", "message": "Training mode is ready."}
        if self.state.breathing.get("active"):
            return {"name": "calm", "density": "minimal", "message": "Slow down. Breathe."}
        if self.state.night_mode:
            return {"name": "night", "density": "quiet", "message": "Sleep well."}
        if self.state.stress_score >= 0.55:
            return {"name": "calm", "density": "minimal", "message": "Keep the pace gentle today."}
        if day_part == "morning":
            return {"name": "morning", "density": "briefing", "message": "Start clean. Move deliberately."}
        if day_part == "night":
            return {"name": "night", "density": "quiet", "message": "Soft landing for the day."}
        return {"name": tone, "density": "balanced", "message": "You are on track."}

    def ambient_message(self) -> str:
        name = "" if self.state.name in ("Unknown", "Guest") else self.state.name.split()[0]
        emotion = self.state.emotion
        day_part = self.day_part()

        if self.state.posture == "slouching":
            return "Shoulders back. Easy."
        if emotion == "sad":
            return "Long day?"
        if emotion == "angry":
            return "One slow breath."
        if emotion == "fear":
            return "Steady. You are okay."
        if self.state.stress_score >= 0.55:
            return "Keep the pace gentle today."
        if day_part == "morning":
            return f"Good morning{name and ', ' + name}."
        if day_part == "night":
            return "Soft landing for the day."
        return "You are on track."

    def snapshot(self) -> dict[str, Any]:
        s = self.state
        return {
            "present": s.present,
            "emotion": s.emotion,
            "posture": s.posture,
            "name": s.name,
            "events": s.events,
            "news": s.news,
            "track": s.track,
            "tasks": s.tasks,
            "weather": s.weather,
            "workout": s.workout,
            "assistant_listening": s.assistant_listening,
            "status": s.status,
            "voice_state": s.voice_state,
            "insight": s.insight,
            "weekly_summary": s.weekly_summary,
            "health": s.health,
            "diagnostics": s.diagnostics,
            "settings": s.settings,
            "show_settings": s.show_settings,
            "show_diagnostics": s.show_diagnostics,
            "personal": s.personal,
            "caption": s.caption,
            "nudge": s.nudge,
            "timers": s.timers,
            "routine": s.routine,
            "breathing": s.breathing,
            "focus": s.focus,
            "celebrate": s.celebrate,
            "privacy": s.privacy,
            "night_mode": s.night_mode,
            "alarm": s.alarm,
            "enrolling": s.enrolling,
            "demo": s.demo,
            "assistant_name": s.assistant_name,
            "capabilities": s.capabilities,
            "conversation_open": s.conversation_open,
            "context": {
                "day_part": self.day_part(),
                "tone": self.tone(),
                "ambient_message": self.ambient_message(),
                "stress_score": round(s.stress_score, 2),
                "posture_score": round(s.posture_score, 2),
                "scene": self.scene(),
            },
            "sequence": s.sequence,
            "server_time_ms": int(time.time() * 1000),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
