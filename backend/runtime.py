"""MirrorRuntime: wires perception, the assistant brain, proactive behaviour and the UI together.

The camera loop and the demo loop both drive the same `on_*` methods, so everything the
assistant does is identical with or without hardware.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Any
import asyncio
import logging
import os
import re
import time

import cv2

from assistant.brain import Brain
from assistant.models import Env, Reply
from assistant.proactive import Nudge, ProactiveEngine, Signals
from assistant.reflection import Reflector
from assistant.reports import Reporter
from assistant.services import Services
from core.config import CONFIG
from core.diagnostics import diagnostics_service
from core.event_bus import EventBus
from core.health_service import health_service
from core.state_manager import NEGATIVE_EMOTIONS, StateManager
from core.websocket_manager import WebSocketManager
from integrations.calendar_service import calendar_service
from integrations.home_assistant import home_assistant
from integrations.news_service import news_service
from integrations.spotify_service import spotify_service
from integrations.weather_service import weather_service
from memory.analytics_service import AnalyticsService
from memory.db import known_user
from memory.memory_service import memory_service
from vision.camera_service import CameraService
from vision.emotion_service import EmotionService
from vision.face_service import FaceService
from vision.posture_service import PostureService
from voice.tts_service import tts_service
from workout.exercises import exercise_title
from workout.workout_service import workout_service

CAPTION_HOLD_SECONDS = 14
NUDGE_HOLD_SECONDS = 18
CELEBRATE_HOLD_SECONDS = 6


def _wake_regex() -> re.Pattern[str]:
    names = {"mirror", "mirrors", "mira", "miro", CONFIG.assistant_name.lower()}
    prefixes = {w.split()[0] for w in CONFIG.wake_words} | {"hey", "hi", "ok", "okay", "hello"}
    return re.compile(rf"\b(?:{'|'.join(sorted(prefixes))})[, ]+(?:{'|'.join(sorted(names))})\b[,.!?]*\s*", re.I)


def format_timer(total_seconds: int) -> str:
    return f"{total_seconds // 60:02d}:{total_seconds % 60:02d}"


class IdentityTracker:
    """Smooths per-frame recognition so one bad frame doesn't flip who the mirror thinks you are."""

    def __init__(self) -> None:
        self.recent: deque[str] = deque(maxlen=4)
        self.locked_guest_until = 0.0

    def push(self, name: str) -> str | None:
        """Add a reading; return the identity to adopt, or None to keep the current one."""
        if time.monotonic() < self.locked_guest_until:
            return "Guest"
        self.recent.append(name or "Guest")
        if len(self.recent) >= 2 and self.recent[-1] == self.recent[-2]:
            return self.recent[-1]
        return None

    def reset(self) -> None:
        self.recent.clear()


class MirrorRuntime:
    def __init__(self, demo: bool = False) -> None:
        self.demo = demo
        self.loop: asyncio.AbstractEventLoop | None = None
        self.event_bus = EventBus()
        self.state = StateManager()
        self.state.state.demo = demo
        self.state.state.assistant_name = CONFIG.assistant_name
        self.services = Services.default()
        self.reporter = Reporter(self.services)
        self.brain = Brain(self.services, self.reporter)
        self.proactive = ProactiveEngine(self.services, self.reporter, self.brain)
        self.reflector = Reflector(self.services, self.reporter)
        self.analytics = AnalyticsService(memory_service)
        self.ws = WebSocketManager(snapshot_provider=self.state.snapshot, on_message=self.on_ui_message)

        self.camera: CameraService | None = None
        self.face: FaceService | None = None
        self.emotion = EmotionService()
        self.posture = PostureService()
        self.identity = IdentityTracker()

        self.latest_frame = None
        self.latest_faces: list[tuple[int, int, int, int]] = []
        self.camera_ok = False
        self.privacy = CONFIG.privacy_on_start
        self.command_lock = asyncio.Lock()
        self.conversation_open_until = 0.0
        self.last_seen_at = 0.0
        self.present_since = 0.0
        self.away_since: float | None = None
        self.negative_streak = 0
        self.slouch_episodes: deque[float] = deque(maxlen=20)
        self.slouch_since: float | None = None
        self.enroll_retry: tuple[str, float] | None = None
        self.last_reflect_at = 0.0
        self.last_personal_refresh = 0.0
        self.last_proactive_tick = 0.0
        self.last_system_refresh = 0.0
        self.workout_started_at: float | None = None
        self.spoke_reps = 0
        self.wake = _wake_regex()
        self._guest_since: float | None = None
        self._last_snapshot_sent = (-1, 0.0)
        self.background: dict[str, asyncio.Task] = {}
        self.last_started: dict[str, float] = {}

    # ================================================================== helpers
    def env(self) -> Env:
        s = self.state.state
        return Env(
            user=s.name, now=datetime.now(), emotion=s.emotion, posture=s.posture, present=s.present,
            weather=s.weather, events=s.events, news=s.news, track=s.track, workout_active=workout_service.active,
            privacy=self.privacy, capabilities=s.capabilities,
        )

    def capabilities(self) -> dict[str, Any]:
        return {
            "llm": self.services.llm.available(), "llm_providers": self.services.llm.configured(),
            "vision": self.services.llm.available(), "camera": self.camera_ok or self.demo,
            "face_backend": self.face.backend if self.face else "none", "tts": tts_service.engine_name,
            "spotify": bool(os.environ.get("SPOTIFY_CLIENT_ID")), "lights": home_assistant.configured,
        }

    async def push(self, force: bool = False) -> None:
        await self.ws.broadcast(self.state.snapshot())

    def say(self, text: str, priority: bool = False, caption: bool = True) -> None:
        if not text:
            return
        if caption:
            self.state.update(caption={**self.state.state.caption, "mirror": text, "at": int(time.time() * 1000)})
        self.proactive.note_spoken()
        tts_service.speak(text, priority=priority)

    def _on_tts_start(self, text: str) -> None:
        if self.loop:
            self.loop.call_soon_threadsafe(lambda: self.state.update(voice_state="speaking", status=self._base_status()))

    def _on_tts_end(self) -> None:
        def done() -> None:
            self.state.update(voice_state="idle", status=self._base_status())
            self.proactive.note_spoken()
            if self.conversation_open_until:
                self.conversation_open_until = max(self.conversation_open_until, time.monotonic() + CONFIG.follow_up_window_seconds)

        if self.loop:
            self.loop.call_soon_threadsafe(done)

    def _base_status(self) -> str:
        s = self.state.state
        if workout_service.active:
            return "training"
        return "present" if s.present else "idle"

    def schedule(self, key: str, interval: float, fn, *args) -> None:
        now = time.monotonic()
        existing = self.background.get(key)
        if existing and not existing.done():
            return
        if now - self.last_started.get(key, -9999.0) < interval:
            return
        self.last_started[key] = now
        self.background[key] = asyncio.create_task(asyncio.to_thread(fn, *args))

    @staticmethod
    def result(task: asyncio.Task, fallback=None):
        try:
            return task.result()
        except Exception:
            logging.exception("Background task failed.")
            return fallback

    # =============================================================== lifecycle
    async def start(self) -> None:
        self.loop = asyncio.get_running_loop()
        CONFIG.data_dir.mkdir(parents=True, exist_ok=True)
        CONFIG.known_faces_dir.mkdir(parents=True, exist_ok=True)
        tts_service.on_start, tts_service.on_end = self._on_tts_start, self._on_tts_end
        tts_service.start()
        self.event_bus.subscribe("*", self._log_event)
        self.state.update(privacy=self.privacy)
        if not self.demo:
            self.face = FaceService(CONFIG.known_faces_dir, CONFIG.face_match_tolerance)
        health_service.set_runtime("tts_runtime", "Voice output", "ready", tts_service.engine_name if tts_service.engine_name != "none" else "captions only")
        self.refresh_system_state(force=True)

    async def _log_event(self, event) -> None:
        user = event.payload.get("user", self.state.state.name)
        await asyncio.to_thread(memory_service.log_event, user, event.type, event.payload)

    def refresh_system_state(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_system_refresh < 1.0:
            return
        self.last_system_refresh = now
        caps = self.capabilities()
        if self.face:
            people = self.face.known_people
            health_service.set_runtime("faces", "Faces", "ready" if people else "warning",
                                       f"{self.face.backend}: {len(people)} known" if people else f"{self.face.backend}: say 'my name is...'")
        providers = caps["llm_providers"]
        health_service.set_runtime("ai", "AI brain", "ready" if providers else "warning",
                                   "+".join(providers) if providers else "offline mode (add a Groq/Gemini key)")
        health_service.set_runtime("lights", "Smart home", "ready" if caps["lights"] else "offline",
                                   "Home Assistant" if caps["lights"] else "not connected")
        self.state.update(
            health=health_service.snapshot(), diagnostics=diagnostics_service.snapshot(),
            settings=health_service.settings_snapshot(), capabilities=caps,
            show_diagnostics=CONFIG.show_diagnostics or self.state.state.show_diagnostics,
        )

    # ============================================================= personal state
    def _personal_sync(self, user: str) -> dict[str, Any]:
        context = self.analytics.build_user_context(user)
        return {
            "tasks": memory_service.get_tasks(user), "stress_score": context["stress_score"],
            "posture_score": context["posture_score"], "insight": context["insight"],
            "weekly_summary": context["weekly_summary"], "personal": self.reporter.ui_snapshot(user),
        }

    async def refresh_personal(self) -> None:
        user = self.state.state.name
        self.last_personal_refresh = time.monotonic()
        if not known_user(user):
            self.state.update(tasks=[], personal={}, insight="", weekly_summary={})
            return
        data = await asyncio.to_thread(self._personal_sync, user)
        self.state.update(**data)

    def _sync_live_widgets(self) -> None:
        """Cheap per-tick sync of in-memory widgets (timers, routine, breathing)."""
        self.state.update(
            timers=self.services.timers.snapshot(), routine=self.services.routines.snapshot(),
            breathing=self.services.breathing.snapshot(), alarm=bool(self.services.timers.ringing()),
        )

    # ============================================================ presence events
    async def on_arrival(self) -> None:
        now = time.monotonic()
        away = (now - self.away_since) if self.away_since else None
        self.state.start_presence()
        self.present_since = now
        self.away_since = None
        self.identity.reset()
        self._guest_since = now
        await self.event_bus.publish("USER_ARRIVED", user=self.state.state.name)

    async def on_departure(self) -> None:
        s = self.state.state
        session = self.state.end_presence()
        self.away_since = time.monotonic()
        stop = workout_service.stop() if workout_service.active else None
        if stop:
            await self._finish_workout(stop, speak=False)
        self.services.routines.stop()
        self.services.breathing.stop()
        user = session["user"]
        await asyncio.to_thread(
            memory_service.log_session, user, session["start_time"], session["end_time"], session["emotions"], session["slouch_count"])
        await self.event_bus.publish("USER_LEFT", user=user)
        self.brain.reset_session()
        self.conversation_open_until = 0
        self.negative_streak = 0
        self.slouch_episodes.clear()
        self.state.update(nudge=None, focus={"panel": "none"}, caption={"user": "", "mirror": "", "at": 0},
                          night_mode=False, personal={}, tasks=[], conversation_open=False)
        if known_user(user):
            self._reflect_in_background(user)

    def _reflect_in_background(self, user: str) -> None:
        if time.monotonic() - self.last_reflect_at < 600:
            return
        self.last_reflect_at = time.monotonic()
        asyncio.create_task(asyncio.to_thread(self.reflector.reflect, user))

    async def on_identity(self, name: str) -> None:
        previous = self.state.state.name
        if not self.state.set_identity(name):
            return
        s = self.state.state
        logging.info("Identity: %s -> %s", previous, name)
        await self.event_bus.publish("USER_RECOGNIZED", user=name, previous=previous)
        if known_user(name):
            self._guest_since = None
            self.brain.reset_session()
            await self.refresh_personal()
            away = (time.monotonic() - self.away_since) if self.away_since else None
            nudges = await asyncio.to_thread(self.proactive.arrival, self.env(), away)
            for nudge in nudges:
                await self.deliver(nudge)
        elif name in ("Guest", "Unknown"):
            self._guest_since = self._guest_since or time.monotonic()
            await self.refresh_personal()

    # ================================================================== delivery
    async def deliver(self, nudge: Nudge) -> None:
        now_ms = int(time.time() * 1000)
        if nudge.speak:
            self.state.update(nudge={"text": nudge.text, "kind": nudge.kind, "at": now_ms})
        if nudge.ui:
            self._apply_ui(nudge.ui)
        if nudge.celebrate:
            self.state.update(celebrate={"label": nudge.celebrate, "at": now_ms})
        if nudge.pending:
            self.brain.set_pending(nudge.pending[0], dict(nudge.pending[1]), ttl=90)
            self.conversation_open_until = time.monotonic() + 25
        if nudge.speak:
            self.say(nudge.text, priority=nudge.priority >= 3)
        await self.event_bus.publish("NUDGE", user=self.state.state.name, kind=nudge.kind, key=nudge.key)
        await self.push()

    def _apply_ui(self, ui: dict[str, Any]) -> None:
        if "focus" in ui:
            panel = ui["focus"]
            seconds = ui.get("seconds", 15)
            focus: dict[str, Any] = {"panel": panel}
            if panel not in ("none", "routine") and seconds:
                focus["until_ms"] = int((time.time() + seconds) * 1000)
            if "week" in ui:
                focus["week"] = ui["week"]
            self.state.update(focus=focus)

    # ============================================================== conversation
    def _is_addressed(self, text: str, source: str) -> bool:
        if source == "ui":
            return True
        now = time.monotonic()
        return bool(
            self.wake.search(text) or "hey mirror" in text.lower()
            or now < self.conversation_open_until or self.brain.pending
            or self.services.routines.active or self.services.breathing.active or self.services.timers.ringing()
        )

    async def handle_text(self, raw: str, source: str = "voice") -> None:
        """Entry point for everything the user says or types."""
        text = " ".join((raw or "").split())
        if not text:
            return
        if not self._is_addressed(text, source):
            return
        command = self.wake.sub("", text, count=1).strip(" ,.") if self.wake.search(text) else text
        command = re.sub(r"^hey mirror[, ]*", "", command, flags=re.I).strip()
        async with self.command_lock:
            await self._handle_command(command, text, source)

    async def _handle_command(self, command: str, original: str, source: str) -> None:
        s = self.state.state
        if re.fullmatch(r"(?i)(?:stop|quiet|be quiet|shut up|enough|silence)[.!]?", command or ""):
            tts_service.stop()
        self.state.update(
            assistant_listening=True, status="listening", voice_state="thinking" if command else "listening",
            caption={"user": command or original, "mirror": "", "at": int(time.time() * 1000)},
            conversation_open=True,
        )
        await self.push()
        if not command:
            self.conversation_open_until = time.monotonic() + CONFIG.follow_up_window_seconds
            self.say("I'm here.", priority=True)
            self.state.update(assistant_listening=False, voice_state="speaking")
            return
        try:
            env = self.env()
            reply: Reply = await asyncio.to_thread(self.brain.handle, command, env)
            await self.event_bus.publish("CONVERSATION", user=s.name, prompt=command, source=reply.source)
            follow_ups = await self.apply_effects(reply.effects)
            if reply.ui:
                self._apply_ui(reply.ui)
            if reply.celebrate:
                self.state.update(celebrate={"label": reply.celebrate, "at": int(time.time() * 1000)})
            spoken = " ".join(p for p in [reply.speech, *follow_ups] if p)
            if spoken:
                self.say(spoken, priority=True)
            window = CONFIG.follow_up_window_seconds * (2 if reply.keep_open else 1)
            self.conversation_open_until = time.monotonic() + window
            await self.refresh_personal()
        except Exception:
            logging.exception("Command failed")
            self.say("Sorry, something went wrong.", priority=True)
        finally:
            self.state.update(assistant_listening=False, voice_state="speaking" if tts_service.is_speaking else "idle",
                              status=self._base_status())
            await self.push()

    # ================================================================== effects
    async def apply_effects(self, effects: list[dict[str, Any]]) -> list[str]:
        """Run hardware/integration side effects. Returns extra sentences to speak."""
        said: list[str] = []
        for effect in effects:
            kind = effect.get("type")
            try:
                text = await self._effect(kind, effect)
            except Exception:
                logging.exception("Effect %s failed", kind)
                text = "Sorry, that didn't work."
            if text:
                said.append(text)
        return said

    async def _effect(self, kind: str | None, e: dict[str, Any]) -> str:
        s = self.state.state
        if kind == "enroll_face":
            return await self._enroll(e["name"])
        if kind == "reset_identity":
            self.identity.locked_guest_until = time.monotonic() + 60
            await self.on_identity("Guest")
            return ""
        if kind == "delete_user":
            if self.face:
                await asyncio.to_thread(self.face.delete_person, e["name"])
            self.identity.locked_guest_until = time.monotonic() + 30
            await self.on_identity("Guest")
            return ""
        if kind == "privacy":
            await self.set_privacy(bool(e["on"]))
            return ""
        if kind == "stop_speech":
            tts_service.stop()
            return ""
        if kind == "stop_alarm":
            self.services.timers.dismiss_ringing()
            tts_service.stop()
            return ""
        if kind == "start_workout":
            workout_service.start(e.get("exercise", "curls"))
            self.workout_started_at, self.spoke_reps = time.monotonic(), 0
            self.state.update(workout=workout_service.snapshot("00:00"), status="training")
            await self.event_bus.publish("WORKOUT_STARTED", user=s.name)
            line = f"{exercise_title(workout_service.exercise)} mode on."
            if self.demo:
                return line
            if not workout_service.tracking_available:
                return line + " I can't count reps without MediaPipe, but I'll time the session."
            return line + " Show me your full body."
        if kind == "stop_workout":
            if not workout_service.active:
                return "No workout is running."
            return await self._finish_workout(workout_service.stop(), speak=True)
        if kind == "music":
            return await asyncio.to_thread(spotify_service.control, e.get("command", ""), e.get("query", ""))
        if kind == "lights":
            return await asyncio.to_thread(home_assistant.lights, e.get("command", ""), e.get("room", ""))
        if kind == "vision":
            return await self._vision(e)
        if kind == "night_mode":
            self.state.update(night_mode=True)
            return ""
        if kind == "reflect":
            if known_user(s.name):
                asyncio.create_task(asyncio.to_thread(self.reflector.reflect, s.name))
            return ""
        if kind == "panel":
            key = "show_settings" if e["panel"] == "settings" else "show_diagnostics"
            self.state.update(**{key: bool(e["visible"])})
            return ""
        return ""

    async def _enroll(self, name: str) -> str:
        first = name.split()[0]
        self.state.update(enrolling=True)
        self.identity.locked_guest_until = 0
        if self.demo or not self.face or not self.camera_ok:
            await self.on_identity(name)
            self.state.update(enrolling=False)
            return "" if self.demo else f"I can't see through a camera right now, {first}, but I'll remember you by name."
        # Try straight away; the camera loop keeps retrying for a few seconds while they position themselves.
        await self.on_identity(name)
        self.enroll_retry = (name, time.monotonic() + 15)
        return ""

    async def try_pending_enroll(self) -> None:
        if not self.enroll_retry or not self.face or self.latest_frame is None:
            return
        name, deadline = self.enroll_retry
        if time.monotonic() > deadline:
            self.enroll_retry = None
            self.state.update(enrolling=False)
            self.say(f"I couldn't get a clear look at you, {name.split()[0]}. Say 'my name is {name.split()[0]}' once you're facing me.")
            return
        ok, reason = await asyncio.to_thread(self.face.enroll, name, self.latest_frame.copy())
        if ok:
            self.enroll_retry = None
            self.state.update(enrolling=False)
            await asyncio.to_thread(self.services.memory.log_event, name, "FACE_ENROLLED", {})
            self.say(f"Got it. I'll know you next time, {name.split()[0]}.", priority=True)
            await self.refresh_personal()

    async def _finish_workout(self, summary: dict[str, Any], speak: bool) -> str:
        s = self.state.state
        user = s.name
        self.state.update(workout=workout_service.snapshot("00:00"), status=self._base_status())
        self.workout_started_at = None
        reps, secs, hold = int(summary["reps"]), int(summary["duration_seconds"]), int(summary.get("hold_seconds") or 0)
        exercise = str(summary["exercise"])
        saved = False
        best_before = 0
        if known_user(user):
            rows = await asyncio.to_thread(
                self.services.habits.query, "SELECT MAX(reps) AS m FROM workouts WHERE user = ? AND exercise = ?", (user, exercise))
            best_before = int(rows[0]["m"] or 0) if rows else 0
            saved = await asyncio.to_thread(self.services.habits.log_workout, user, exercise, reps, secs or hold)
        await self.event_bus.publish("WORKOUT_ENDED", user=user, reps=reps)
        if not speak:
            return ""
        title = exercise_title(exercise).lower()
        if exercise == "plank":
            line = f"Plank held for {hold} seconds."
        elif reps:
            line = f"{reps} {title} in {max(1, secs // 60)} minute{'s' if secs >= 120 else ''}."
        else:
            line = f"{max(1, secs // 60)} minute {title} session."
        if saved and reps and reps > best_before > 0:
            line += " That's a personal best."
        elif saved:
            line += " Saved."
        return line + " Nice work."

    async def _vision(self, e: dict[str, Any]) -> str:
        if self.demo or self.latest_frame is None:
            return "I can't see anything right now." if not self.demo else \
                "In the demo I don't have a camera, but on your mirror I'd give you a proper outfit check."
        frame = self.latest_frame.copy()
        ok, buf = cv2.imencode(".jpg", cv2.resize(frame, (0, 0), fx=0.6, fy=0.6), [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            return "I couldn't capture a picture."
        system, prompt = self.brain.vision_prompt(e.get("kind", "outfit"), self.env(), e.get("question", ""))
        result = await asyncio.to_thread(self.services.llm.vision, prompt, buf.tobytes(), system)
        return result.text if result else "I need a Gemini or OpenAI key to look at your outfit."

    async def set_privacy(self, on: bool) -> None:
        self.privacy = on
        self.state.update(privacy=on)
        if self.camera and self.camera_ok:
            if on:
                self.camera.release()
                self.camera_ok = False
            else:
                await asyncio.to_thread(self._open_camera)
        await self.event_bus.publish("PRIVACY", user=self.state.state.name, on=on)

    # =============================================================== UI messages
    async def on_ui_message(self, msg: dict[str, Any]) -> None:
        kind = msg.get("type")
        if kind == "say":
            await self.handle_text(str(msg.get("text", ""))[:500], source="ui")
        elif kind == "action":
            await self._ui_action(str(msg.get("name", "")), msg)

    async def _ui_action(self, name: str, msg: dict[str, Any]) -> None:
        s = self.state.state
        user = s.name
        if name == "complete_task" and known_user(user):
            await asyncio.to_thread(memory_service.complete_task, user, int(msg["id"]))
            await self.refresh_personal()
        elif name == "log_water" and known_user(user):
            target = self.reporter.targets(user)["water_ml"]
            await asyncio.to_thread(self.services.habits.log_water, user, float(msg.get("ml", 250)), target)
            await self.refresh_personal()
        elif name == "habit_done" and known_user(user):
            await asyncio.to_thread(self.services.habits.log, user, int(msg["id"]), 1)
            await self.refresh_personal()
        elif name == "dismiss_focus":
            self.state.update(focus={"panel": "none"})
        elif name == "dismiss_nudge":
            self.state.update(nudge=None)
        elif name == "stop_alarm":
            await self._effect("stop_alarm", {})
        elif name == "privacy":
            await self.set_privacy(bool(msg.get("on")))
        elif name == "cancel_timer":
            self.services.timers.cancel(str(msg.get("label", "")))
        elif name == "routine_next":
            await self.handle_text("next", source="ui")
        elif name == "routine_stop":
            await self.handle_text("stop the routine", source="ui")
        elif name == "breathing_stop":
            self.services.breathing.stop()
        elif name == "focus":
            self._apply_ui({"focus": str(msg.get("panel", "none")), "seconds": 25})
        await self.push()

    # ================================================================ tick loop
    async def assistant_loop(self) -> None:
        """~2 Hz: timers, routines, nudges, widget expiry, periodic refreshes."""
        while True:
            try:
                await self._tick()
            except Exception:
                logging.exception("assistant tick failed")
            await asyncio.sleep(0.5)

    async def _tick(self) -> None:
        s = self.state.state
        now = time.monotonic()
        wall_ms = int(time.time() * 1000)
        changed = False

        for timer in self.services.timers.pop_finished():
            changed = True
            label = timer.label if timer.kind == "timer" and timer.label else ""
            if timer.kind == "focus":
                line = "Focus session complete. Take a five minute break. Stand up and move."
            elif label and not re.fullmatch(r"[\d ]+(?:hours?|minutes?|seconds?|and)+.*", label):
                line = f"Your {label} timer is done."
            else:
                line = "Your timer is done."
            tts_service.alarm()
            self.say(line, priority=True)
            self.state.update(nudge={"text": line, "kind": "timer", "at": wall_ms})

        event = self.services.routines.tick()
        if event:
            changed = True
            self.say(event.text, priority=True)
        if self.services.breathing.tick():
            changed = True
            self.say("Nicely done. How do you feel?", priority=True)
            self.brain.set_pending("breathing_after", {}, ttl=20)

        if s.nudge and wall_ms - s.nudge["at"] > NUDGE_HOLD_SECONDS * 1000:
            self.state.update(nudge=None)
        if s.celebrate and wall_ms - s.celebrate["at"] > CELEBRATE_HOLD_SECONDS * 1000:
            self.state.update(celebrate=None)
        focus = s.focus
        if focus.get("until_ms") and wall_ms > focus["until_ms"]:
            self.state.update(focus={"panel": "none"})
        if s.conversation_open and now > self.conversation_open_until and not tts_service.is_speaking:
            self.state.update(conversation_open=False)

        before = (s.timers, s.routine, s.breathing, s.alarm)
        self._sync_live_widgets()
        if before != (s.timers, s.routine, s.breathing, s.alarm):
            changed = True

        if known_user(s.name) and now - self.last_personal_refresh > 30:
            await self.refresh_personal()
            changed = True

        if now - self.last_proactive_tick >= 4:
            self.last_proactive_tick = now
            await self._proactive()

        self.refresh_system_state()
        if changed:
            await self.push()

    async def _proactive(self) -> None:
        s = self.state.state
        now = time.monotonic()
        env = self.env()
        busy = tts_service.is_speaking or s.assistant_listening or bool(self.brain.pending and now < self.conversation_open_until)
        if s.present and env.is_guest and self._guest_since and now - self._guest_since > 7 and self.camera_ok and not self.enroll_retry:
            for nudge in self.proactive.guest_prompt(env, busy):
                await self.deliver(nudge)
            return
        episodes = sum(1 for t in self.slouch_episodes if now - t < 300)
        if self.slouch_since and now - self.slouch_since > 45:
            episodes = max(episodes, 3)
        signals = Signals(
            present_seconds=(now - self.present_since) if s.present else 0.0, slouch_events_recent=episodes,
            negative_streak=self.negative_streak, workout_active=workout_service.active)
        nudges = await asyncio.to_thread(self.proactive.tick, env, signals, busy)
        for nudge in nudges:
            await self.deliver(nudge)
            if nudge.kind == "posture":
                self.slouch_episodes.clear()
                self.slouch_since = None
            if nudge.kind == "mood":
                self.negative_streak = 0

    # ============================================================ integrations
    def _integration_tick(self) -> None:
        self.schedule("weather", CONFIG.weather_interval_seconds, weather_service.get_weather)
        self.schedule("calendar", CONFIG.calendar_interval_seconds, calendar_service.get_calendar_events)
        self.schedule("news", CONFIG.news_interval_seconds, news_service.get_headlines)
        self.schedule("spotify", CONFIG.spotify_interval_seconds, spotify_service.get_current_track)

    def _apply_finished(self) -> None:
        for key, task in list(self.background.items()):
            if not task.done():
                continue
            result = self.result(task)
            del self.background[key]
            if key == "weather":
                self.state.update(weather=result)
            elif key == "calendar":
                self.state.update(events=result or [])
            elif key == "news":
                self.state.update(news=result or [])
            elif key == "spotify":
                self.state.update(track=result)

    async def integrations_loop(self) -> None:
        while True:
            if not self.demo:
                self._integration_tick()
                self._apply_finished()
            await asyncio.sleep(1.0)

    # ================================================================== camera
    def _open_camera(self) -> None:
        self.camera = CameraService(CONFIG.camera_index, CONFIG.camera_width, CONFIG.camera_height)
        try:
            self.camera.open()
            self.camera_ok = True
        except RuntimeError as exc:
            logging.error("%s", exc)
            self.camera_ok = False

    async def camera_loop(self) -> None:
        """Real perception. Falls back to a static-presence kiosk if there is no webcam."""
        await asyncio.to_thread(self._open_camera)
        health_service.set_runtime("camera", "Camera", "ready" if self.camera_ok else "offline",
                                   f"webcam {CONFIG.camera_index}" if self.camera_ok else "no webcam found", required=False)
        if not self.camera_ok and not self.privacy:
            logging.warning("No webcam: running in kiosk mode (always present, identity from last visit). Use typed or voice commands.")
            await self.static_presence_loop(self._last_known_user())
            return
        frame_delay = 1 / max(1, CONFIG.target_fps)
        try:
            while True:
                started = time.monotonic()
                if self.privacy or not self.camera or not self.camera_ok:
                    await asyncio.sleep(0.3)
                    if not self.privacy and not self.camera_ok:
                        await asyncio.to_thread(self._open_camera)
                    self._apply_finished()
                    continue
                ok, frame = await asyncio.to_thread(self.camera.read)
                if not ok:
                    await asyncio.sleep(frame_delay)
                    continue
                await self._process_frame(frame, started)
                diagnostics_service.record_frame(started, sum(1 for t in self.background.values() if not t.done()), self.ws.client_count)
                await asyncio.sleep(max(0.0, frame_delay - (time.monotonic() - started)))
        finally:
            if self.camera:
                self.camera.release()

    async def _process_frame(self, frame, started: float) -> None:
        s = self.state.state
        self.latest_frame = frame
        faces = self.face.detect_faces(frame)
        self.latest_faces = faces
        now = time.monotonic()

        if faces:
            self.last_seen_at = now
            if not s.present:
                await self.on_arrival()
            self.schedule("recognition", CONFIG.face_recognition_interval_seconds, self.face.recognize, frame.copy())
            self.schedule("emotion", CONFIG.emotion_interval_seconds, self.emotion.analyze, frame.copy())
            self.schedule("posture", CONFIG.posture_interval_seconds, self.posture.analyze, frame.copy(), faces)
            if workout_service.active:
                await self._workout_frame(frame, now)
            await self.try_pending_enroll()
        elif s.present and now - self.last_seen_at > CONFIG.absence_timeout_seconds + (10 if workout_service.active else 0):
            await self.on_departure()

        for key, task in list(self.background.items()):
            if not task.done():
                continue
            result = self.result(task)
            del self.background[key]
            if key == "recognition":
                if s.present:
                    adopted = self.identity.push(result or "Guest")
                    if adopted and adopted != "Guest" or (adopted == "Guest" and s.name in ("Unknown", "Guest")):
                        await self.on_identity(adopted)
                    elif adopted == "Guest" and known_user(s.name) and not self.enroll_retry:
                        # A known face stopped matching: trust the tracker only after repeated misses.
                        if list(self.identity.recent).count("Guest") >= 4:
                            await self.on_identity("Guest")
            elif key == "emotion":
                if s.present:
                    emotion = result or "neutral"
                    if self.state.set_emotion(emotion):
                        await self.event_bus.publish("MOOD_UPDATED", user=s.name, emotion=emotion)
                    self.negative_streak = self.negative_streak + 1 if emotion in NEGATIVE_EMOTIONS else 0
            elif key == "posture":
                if s.present:
                    posture = result or "good"
                    if self.state.set_posture(posture) and posture == "slouching":
                        self.slouch_episodes.append(now)
                    self.slouch_since = (self.slouch_since or now) if posture == "slouching" else None
            elif key in ("weather", "calendar", "news", "spotify"):
                self.background[key] = task  # handled by _apply_finished
        self._integration_tick()
        self._apply_finished()
        self.refresh_system_state()
        if CONFIG.camera_preview:
            cv2.imshow("Smart Mirror Camera Debug", frame)
            cv2.waitKey(1)

    async def _workout_frame(self, frame, now: float) -> None:
        started = self.workout_started_at or now
        self.workout_started_at = started
        counted = await asyncio.to_thread(workout_service.process_frame, frame)
        self.state.update(workout=workout_service.snapshot(format_timer(int(now - started))))
        if counted and workout_service.reps != self.spoke_reps:
            self.spoke_reps = workout_service.reps
            if self.spoke_reps % 5 == 0 or self.spoke_reps in (1,):
                self.say(str(self.spoke_reps), priority=True, caption=False)

    # =================================================================== demo
    async def static_presence_loop(self, name: str) -> None:
        """Someone is always at the mirror: demo mode and the no-webcam kiosk."""
        await self.on_arrival()
        self.present_since = time.monotonic()
        await self.on_identity(name)
        emotions = ["neutral", "neutral", "happy", "neutral", "happy", "neutral"]
        i = 0
        while True:
            now = time.monotonic()
            if workout_service.active and self.workout_started_at:
                self.state.update(workout=workout_service.snapshot(format_timer(int(now - self.workout_started_at))))
                if self.demo and int(now - self.workout_started_at) % 3 == 0:
                    workout_service.counter.reps = int((now - self.workout_started_at) // 3)
            if self.demo and i % 20 == 0:
                self.state.set_emotion(emotions[(i // 20) % len(emotions)])
            i += 1
            diagnostics_service.record_frame(now, 0, self.ws.client_count)
            await asyncio.sleep(0.5)

    def _last_known_user(self) -> str:
        rows = self.services.profile.query("SELECT user FROM profiles ORDER BY last_seen_at DESC LIMIT 1")
        return rows[0]["user"] if rows else "Guest"
