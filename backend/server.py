from __future__ import annotations

from datetime import datetime
import asyncio
import logging
import time

import cv2
import websockets

from core.config import CONFIG
from core.diagnostics import diagnostics_service
from core.event_bus import EventBus
from core.health_service import health_service
from core.state_manager import NEGATIVE_EMOTIONS, StateManager
from core.websocket_manager import WebSocketManager
from integrations.calendar_service import calendar_service
from integrations.news_service import news_service
from integrations.spotify_service import spotify_service
from integrations.weather_service import weather_service
from memory.analytics_service import AnalyticsService
from memory.memory_service import memory_service
from vision.camera_service import CameraService
from vision.emotion_service import EmotionService
from vision.face_service import FaceService
from vision.posture_service import PostureService
from voice.conversation_service import conversation_service
from voice.speech_service import SpeechService
from voice.tts_service import tts_service
from workout.workout_service import normalize_exercise, workout_service


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

event_bus = EventBus()
state_manager = StateManager()
websocket_manager = WebSocketManager(snapshot_provider=state_manager.snapshot)
analytics_service = AnalyticsService(memory_service)

camera_service = CameraService(
    camera_index=CONFIG.camera_index,
    width=CONFIG.camera_width,
    height=CONFIG.camera_height,
)
face_service = FaceService(CONFIG.known_faces_dir, CONFIG.face_match_tolerance)
emotion_service = EmotionService()
posture_service = PostureService()


WORKOUT_START_PHRASES = (
    "start workout",
    "begin workout",
    "workout mode",
    "turn on training mode",
    "training mode",
    "turn on training",
    "start training",
    "begin training",
    "curl mode",
    "squat mode",
    "pushup mode",
    "push up mode",
    "posture mode",
)

WORKOUT_STOP_PHRASES = (
    "stop workout",
    "end workout",
    "finish workout",
    "exit workout",
    "turn off training mode",
    "stop training",
    "end training",
    "finish training",
    "exit training",
)

SETTINGS_PHRASES = ("settings mode", "show settings", "hide settings")
DIAGNOSTICS_PHRASES = ("diagnostics mode", "show diagnostics", "hide diagnostics")
BRIEFING_PHRASES = ("brief me", "daily brief", "what is my day", "how is my day")


def seconds_since(started_at: float | None) -> int:
    if not started_at:
        return 0
    return int(time.monotonic() - started_at)


def format_timer(total_seconds: int) -> str:
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:02d}"


def command_matches(command: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in command for phrase in phrases)


async def refresh_personal_state(user: str) -> None:
    context = analytics_service.build_user_context(user)
    state_manager.update(
        tasks=memory_service.get_tasks(user),
        stress_score=context["stress_score"],
        posture_score=context["posture_score"],
        insight=context["insight"],
        weekly_summary=context["weekly_summary"],
    )


def refresh_system_state() -> None:
    state_manager.update(
        health=health_service.snapshot(),
        diagnostics=diagnostics_service.snapshot(),
        settings=health_service.settings_snapshot(),
        show_diagnostics=CONFIG.show_diagnostics or state_manager.state.show_diagnostics,
    )


def build_entry_summary(user: str) -> str:
    weather = state_manager.state.weather or {}
    events = state_manager.state.events or []
    tasks = state_manager.state.tasks or []
    track = state_manager.state.track or {}

    parts: list[str] = []
    if weather:
        temp = weather.get("temperature")
        desc = weather.get("description", "clear")
        high = weather.get("high")
        low = weather.get("low")
        temp_text = f"{temp} deg C" if temp is not None else "unavailable"
        if high is not None and low is not None:
            parts.append(
                f"Weather is {desc}, {temp_text}, with a high of {high} and a low of {low}."
            )
        else:
            parts.append(f"Weather is {desc}, {temp_text}.")

    if events:
        next_event = events[0]
        parts.append(
            f"Your next event is {next_event['title']} at {next_event['time']}.")
        if len(events) > 1:
            parts.append(
                f"Then {events[1]['title']} at {events[1]['time']}.")
    else:
        parts.append("No calendar events are scheduled for today.")

    if tasks:
        parts.append(f"You have {len(tasks)} reminders.")
    else:
        parts.append("No reminders are pending.")

    if track.get("title"):
        title = track.get("title")
        artist = track.get("artist", "unknown artist")
        parts.append(f"Currently playing {title} by {artist}.")

    return " ".join(parts).strip()


async def handle_voice_command(text: str) -> None:
    if "hey mirror" not in text:
        return

    command = text.replace("hey mirror", "", 1).strip()
    user = state_manager.state.name
    state_manager.update(assistant_listening=True, status="listening", voice_state="listening")
    await websocket_manager.broadcast(state_manager.snapshot())

    try:
        if not command:
            state_manager.update(status="speaking", voice_state="speaking")
            tts_service.speak("I'm here.", priority=True)
            return

        if "remind me to" in command:
            task = command.split("remind me to", 1)[-1].strip()
            if user == "Unknown":
                state_manager.update(status="speaking", voice_state="speaking")
                tts_service.speak("I need to recognize you before I save that.")
                return
            memory_service.add_task(user, task)
            await refresh_personal_state(user)
            await event_bus.publish("TASK_CREATED", user=user, task=task)
            state_manager.update(status="speaking", voice_state="speaking")
            tts_service.speak(f"Saved. {task}.", priority=True)
            return

        if "complete task" in command or "mark task done" in command:
            completed = memory_service.complete_latest_task(user)
            await refresh_personal_state(user)
            if completed:
                await event_bus.publish("TASK_COMPLETED", user=user, task=completed)
                state_manager.update(status="speaking", voice_state="speaking")
                tts_service.speak("Done.", priority=True)
            else:
                state_manager.update(status="speaking", voice_state="speaking")
                tts_service.speak("No pending task found.", priority=True)
            return

        if command_matches(command, SETTINGS_PHRASES):
            visible = "hide" not in command
            state_manager.update(show_settings=visible, status="speaking", voice_state="speaking")
            tts_service.speak("Settings shown." if visible else "Settings hidden.", priority=True)
            return

        if command_matches(command, DIAGNOSTICS_PHRASES):
            visible = "hide" not in command
            state_manager.update(show_diagnostics=visible, status="speaking", voice_state="speaking")
            tts_service.speak("Diagnostics shown." if visible else "Diagnostics hidden.", priority=True)
            return

        if command_matches(command, BRIEFING_PHRASES):
            state_manager.update(status="speaking", voice_state="speaking")
            tts_service.speak(build_entry_summary(user), priority=True)
            return

        if command_matches(command, WORKOUT_START_PHRASES):
            exercise = normalize_exercise(command)
            workout_service.start(exercise)
            state_manager.update(
                workout=workout_service.snapshot("00:00"),
                status="training",
                voice_state="speaking",
            )
            await event_bus.publish("WORKOUT_STARTED", user=user)
            tts_service.speak(f"{exercise.title()} mode on.", priority=True)
            return

        if command_matches(command, WORKOUT_STOP_PHRASES):
            workout_service.stop()
            state_manager.update(
                workout=workout_service.snapshot("00:00"),
                status="present" if state_manager.state.present else "idle",
                voice_state="speaking",
            )
            await event_bus.publish(
                "WORKOUT_ENDED", user=user, reps=workout_service.reps
            )
            tts_service.speak("Workout saved. Nice work.", priority=True)
            return

        context = analytics_service.build_user_context(user)
        await event_bus.publish("CONVERSATION_STARTED", user=user, prompt=command)
        state_manager.update(status="thinking", voice_state="thinking")
        await websocket_manager.broadcast(state_manager.snapshot())
        reply = await asyncio.to_thread(
            conversation_service.respond,
            command,
            user,
            context["conversation_context"],
        )
        state_manager.update(status="speaking", voice_state="speaking")
        tts_service.speak(reply, priority=True)
    finally:
        status = "idle"
        if state_manager.state.present:
            status = "training" if workout_service.active else "present"
        state_manager.update(
            assistant_listening=False,
            status=status,
            voice_state="idle",
        )
        await websocket_manager.broadcast(state_manager.snapshot())


def schedule_background_task(
    tasks: dict[str, asyncio.Task],
    last_started: dict[str, float],
    key: str,
    interval: float,
    function,
    *args,
) -> None:
    now = time.monotonic()
    existing = tasks.get(key)
    if existing and not existing.done():
        return
    if now - last_started.get(key, -9999.0) < interval:
        return
    last_started[key] = now
    tasks[key] = asyncio.create_task(asyncio.to_thread(function, *args))


def task_result(task: asyncio.Task, fallback=None):
    try:
        return task.result()
    except Exception:
        logging.exception("Background task failed.")
        return fallback


async def apply_finished_background_tasks(tasks: dict[str, asyncio.Task]) -> None:
    for key, task in list(tasks.items()):
        if not task.done():
            continue

        result = task_result(task)
        del tasks[key]

        if key == "recognition":
            if not state_manager.state.present:
                continue
            previous = state_manager.state.name
            changed = state_manager.set_identity(result or "Unknown")
            if changed:
                await event_bus.publish(
                    "USER_RECOGNIZED", user=state_manager.state.name, previous=previous
                )
                await refresh_personal_state(state_manager.state.name)

        elif key == "emotion":
            if not state_manager.state.present:
                continue
            changed = state_manager.set_emotion(result or "neutral")
            if changed:
                await event_bus.publish(
                    "MOOD_UPDATED",
                    user=state_manager.state.name,
                    emotion=state_manager.state.emotion,
                )

        elif key == "posture":
            if not state_manager.state.present:
                continue
            changed = state_manager.set_posture(result or "good")
            if changed and state_manager.state.posture == "slouching":
                await event_bus.publish("POSTURE_BAD", user=state_manager.state.name)

        elif key == "weather":
            state_manager.update(weather=result)

        elif key == "calendar":
            state_manager.update(events=result or [])

        elif key == "news":
            state_manager.update(news=result or [])

        elif key == "spotify":
            state_manager.update(track=result)

        elif key == "tasks":
            state_manager.update(tasks=result or [])


async def detection_loop() -> None:
    camera_service.open()
    health_service.set_runtime("camera", "Camera", "ready", f"webcam {CONFIG.camera_index}", required=True)
    frame_delay = 1 / max(1, CONFIG.target_fps)
    broadcast_delay = 1 / max(0.5, CONFIG.broadcast_hz)
    last_seen_at = 0.0
    last_broadcast_at = 0.0
    last_welcome_user = ""
    last_posture_voice_at = 0.0
    last_mood_voice_at = 0.0
    workout_started_at: float | None = None

    background_tasks: dict[str, asyncio.Task] = {}
    last_started: dict[str, float] = {}

    try:
        while True:
            loop_started_at = time.monotonic()
            ok, frame = camera_service.read()
            if not ok:
                logging.warning("Camera frame was empty.")
                await asyncio.sleep(frame_delay)
                continue

            faces = face_service.detect_faces(frame)
            now = time.monotonic()

            if faces:
                last_seen_at = now
                if not state_manager.state.present:
                    state_manager.start_presence()
                    last_welcome_user = ""
                    await event_bus.publish("USER_ARRIVED", user=state_manager.state.name)

                schedule_background_task(
                    background_tasks,
                    last_started,
                    "recognition",
                    CONFIG.face_recognition_interval_seconds,
                    face_service.recognize,
                    frame.copy(),
                )
                schedule_background_task(
                    background_tasks,
                    last_started,
                    "emotion",
                    CONFIG.emotion_interval_seconds,
                    emotion_service.analyze,
                    frame.copy(),
                )
                schedule_background_task(
                    background_tasks,
                    last_started,
                    "posture",
                    CONFIG.posture_interval_seconds,
                    posture_service.analyze,
                    frame.copy(),
                    faces,
                )

                if state_manager.state.name not in ("Unknown", "Guest"):
                    schedule_background_task(
                        background_tasks,
                        last_started,
                        "tasks",
                        CONFIG.tasks_interval_seconds,
                        memory_service.get_tasks,
                        state_manager.state.name,
                    )

                schedule_background_task(
                    background_tasks,
                    last_started,
                    "weather",
                    CONFIG.weather_interval_seconds,
                    weather_service.get_weather,
                )
                schedule_background_task(
                    background_tasks,
                    last_started,
                    "calendar",
                    CONFIG.calendar_interval_seconds,
                    calendar_service.get_calendar_events,
                )
                schedule_background_task(
                    background_tasks,
                    last_started,
                    "news",
                    CONFIG.news_interval_seconds,
                    news_service.get_headlines,
                )
                schedule_background_task(
                    background_tasks,
                    last_started,
                    "spotify",
                    CONFIG.spotify_interval_seconds,
                    spotify_service.get_current_track,
                )

                if workout_service.active:
                    if workout_started_at is None:
                        workout_started_at = now
                    reps = workout_service.process_frame(frame)
                    state_manager.update(
                        workout=workout_service.snapshot(
                            format_timer(seconds_since(workout_started_at))
                        )
                    )
                else:
                    workout_started_at = None

            elif state_manager.state.present and (
                now - last_seen_at > CONFIG.absence_timeout_seconds
            ):
                session = state_manager.end_presence()
                workout_service.stop()
                workout_started_at = None
                memory_service.log_session(
                    session["user"],
                    session["start_time"],
                    session["end_time"],
                    session["emotions"],
                    session["slouch_count"],
                )
                await event_bus.publish("USER_LEFT", user=session["user"])

            await apply_finished_background_tasks(background_tasks)
            diagnostics_service.record_frame(
                loop_started_at,
                active_tasks=sum(1 for task in background_tasks.values() if not task.done()),
                clients=websocket_manager.client_count,
            )
            refresh_system_state()

            current_user = state_manager.state.name
            if (
                state_manager.state.present
                and current_user not in ("Unknown",)
                and current_user != last_welcome_user
            ):
                last_welcome_user = current_user
                context = analytics_service.build_user_context(current_user)
                state_manager.update(
                    stress_score=context["stress_score"],
                    posture_score=context["posture_score"],
                    insight=context["insight"],
                )
                task_count = len(memory_service.get_tasks(current_user))
                welcome = f"Good to see you, {current_user}."
                if context["stress_score"] >= 0.55:
                    welcome = f"Good to see you, {current_user}. Let's keep it gentle."
                elif task_count:
                    welcome = f"Good to see you, {current_user}. {task_count} reminder pending."
                tts_service.speak(welcome)

            if state_manager.state.posture == "slouching" and now - last_posture_voice_at > 90:
                last_posture_voice_at = now
                tts_service.speak("Shoulders back. Easy.")

            if (
                state_manager.state.emotion in NEGATIVE_EMOTIONS
                and now - last_mood_voice_at > 180
            ):
                last_mood_voice_at = now
                tts_service.speak(state_manager.ambient_message())

            if now - last_broadcast_at >= broadcast_delay:
                await websocket_manager.broadcast(state_manager.snapshot())
                last_broadcast_at = now

            if CONFIG.camera_preview:
                cv2.imshow("Smart Mirror Camera Debug", frame)
                if cv2.waitKey(1) == 27:
                    break

            elapsed = time.monotonic() - loop_started_at
            await asyncio.sleep(max(0.0, frame_delay - elapsed))

    finally:
        health_service.set_runtime("camera", "Camera", "offline", "camera released", required=True)
        await websocket_manager.broadcast({"present": False, "status": "offline"})
        for task in background_tasks.values():
            task.cancel()
        camera_service.release()
        cv2.destroyAllWindows()


async def log_all_events(event) -> None:
    user = event.payload.get("user", state_manager.state.name)
    memory_service.log_event(user, event.type, event.payload)


async def main() -> None:
    CONFIG.data_dir.mkdir(parents=True, exist_ok=True)
    CONFIG.known_faces_dir.mkdir(parents=True, exist_ok=True)

    event_bus.subscribe("*", log_all_events)
    tts_service.start()
    health_service.set_runtime("tts_runtime", "Voice runtime", "ready", "speech queue started")

    loop = asyncio.get_running_loop()
    speech_service = SpeechService(
        lambda text: asyncio.run_coroutine_threadsafe(handle_voice_command(text), loop)
    )
    if speech_service.start():
        health_service.set_runtime("microphone", "Microphone", "ready", "listener started")
    else:
        health_service.set_runtime("microphone", "Microphone", "offline", "speech package missing")
    refresh_system_state()

    logging.info(
        "Starting Smart Mirror websocket on ws://%s:%s",
        CONFIG.websocket_host,
        CONFIG.websocket_port,
    )
    logging.info("Known faces directory: %s", CONFIG.known_faces_dir)
    logging.info("SQLite memory: %s", CONFIG.db_path)

    async with websockets.serve(
        websocket_manager.handler,
        CONFIG.websocket_host,
        CONFIG.websocket_port,
        ping_interval=20,
        ping_timeout=20,
    ):
        await detection_loop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Smart Mirror stopped.")
