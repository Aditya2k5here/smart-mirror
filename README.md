# AI Smart Mirror

Laptop-first AI Smart Mirror for local Windows development. The backend owns realtime vision, voice, memory, integrations, and websocket state. The frontend is a local browser mirror surface focused on calm, premium, ambient presentation.

## Architecture

```text
backend/
  server.py
  core/
    config.py
    event_bus.py
    state_manager.py
    websocket_manager.py
  vision/
    camera_service.py
    face_service.py
    emotion_service.py
    posture_service.py
  voice/
    conversation_service.py
    speech_service.py
    tts_service.py
  memory/
    memory_service.py
    analytics_service.py
  integrations/
    weather_service.py
    spotify_service.py
    calendar_service.py
    news_service.py
  workout/
    workout_service.py
  data/
    mirror.db
    known_faces/
frontend/
  index.html
  style.css
  script.js
```

## Why This Shape

The webcam loop must stay smooth. Expensive work is sampled instead of run every frame:

- Face detection: lightweight and frequent.
- Face recognition: about every 2 seconds.
- Emotion detection: about every 6 seconds.
- Posture detection: under 1 second, but isolated from the main loop.
- Websocket broadcast: capped at 4 updates per second.
- TTS: queued so speech does not overlap.
- Integrations: fetched in background threads on longer intervals.

This favors perceived smoothness and reliability over maximum AI processing.

## Setup

From `E:\PROJECTS\smart-mirror`:

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `PyAudio` fails on Windows, install a compatible wheel for your Python version, then rerun `pip install -r requirements.txt`.

Create your environment file:

```powershell
Copy-Item backend\.env.example backend\.env
notepad backend\.env
```

Fill only the integrations you want to use. The mirror still runs without OpenAI, Spotify, News, or Weather keys.

For fastest assistant replies, set:

```text
GROQ_API_KEY=your_groq_key
GROQ_MODEL=llama-3.1-8b-instant
GEMINI_API_KEY=your_gemini_key
GEMINI_MODEL=gemini-2.5-flash-lite
AI_ROUTER_MODE=auto
```

`AI_ROUTER_MODE=auto` uses Groq first for low-latency mirror responses, then Gemini as fallback. This avoids paying for two model calls for every sentence while still giving you provider redundancy.

## Register A Face

```powershell
.\.venv311\Scripts\Activate.ps1
python -m backend.register_face
```

Registered faces are saved in:

```text
backend\data\known_faces
```

Restart the backend after registering a new face.

## Run The Mirror

Terminal 1:

```powershell
.\.venv311\Scripts\Activate.ps1
python -m backend.server
```

Terminal 2:

```powershell
.\.venv311\Scripts\Activate.ps1
python -m http.server 8000 -d frontend
```

Open:

```text
http://localhost:8000
```

## Useful Debug Flags

Show the raw OpenCV camera window:

```powershell
$env:MIRROR_CAMERA_PREVIEW="true"
python -m backend.server
```

Use a different webcam:

```powershell
$env:MIRROR_CAMERA_INDEX="1"
python -m backend.server
```

Reduce CPU load:

```powershell
$env:MIRROR_TARGET_FPS="18"
$env:MIRROR_EMOTION_INTERVAL="10"
python -m backend.server
```

## Voice Commands

Say:

- `Hey mirror start workout`
- `Hey mirror workout mode`
- `Hey mirror training mode`
- `Hey mirror stop workout`
- `Hey mirror finish workout`
- `Hey mirror remind me to drink water`
- `Hey mirror complete task`
- `Hey mirror how is my day looking`

The assistant is intentionally brief. It should feel observant, not chatty.

## Troubleshooting

Camera fails:

- Close other apps using the webcam.
- Try `MIRROR_CAMERA_INDEX=1`.
- Enable `MIRROR_CAMERA_PREVIEW=true` to verify frames.

Mic fails:

- Confirm Windows microphone permissions.
- Confirm `SpeechRecognition` and `PyAudio` are installed.
- Use the default Windows input device.

Face is not recognized:

- Re-register with even lighting.
- Keep one clear face per image in `backend\data\known_faces`.
- Lower `MIRROR_FACE_MATCH_TOLERANCE` for stricter matching or raise it slightly for easier matching.

Frontend says reconnecting:

If `python -m backend.server` is running.
- Confirm it logs `ws://localhost:8765`.
- Refresh `http://localhost:8000`.
