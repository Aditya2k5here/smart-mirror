import sys
import types
from pathlib import Path

backend = Path(__file__).resolve().parent / "backend"
sys.path.insert(0, str(backend))
sys.modules.setdefault("cv2", types.ModuleType("cv2"))
cv2 = sys.modules["cv2"]
cv2.data = types.SimpleNamespace(haarcascades="")
cv2.CascadeClassifier = lambda *args, **kwargs: types.SimpleNamespace(detectMultiScale=lambda *args, **kwargs: [])
optional = [
    "websockets",
    "face_recognition",
    "mediapipe",
    "deepface",
    "pyttsx3",
    "speech_recognition",
    "openai",
    "spotipy",
    "google_auth_oauthlib",
    "google_auth_httplib2",
    "googleapiclient",
    "requests",
]
for name in optional:
    sys.modules.setdefault(name, types.ModuleType(name))

sys.modules.setdefault("google", types.ModuleType("google"))
sys.modules.setdefault("google.auth", types.ModuleType("google.auth"))
sys.modules.setdefault("google.auth.transport", types.ModuleType("google.auth.transport"))
sys.modules.setdefault("google.auth.transport.requests", types.ModuleType("google.auth.transport.requests"))
sys.modules.setdefault("google.oauth2", types.ModuleType("google.oauth2"))
sys.modules.setdefault("google.oauth2.credentials", types.ModuleType("google.oauth2.credentials"))
sys.modules.setdefault("googleapiclient", types.ModuleType("googleapiclient"))
sys.modules.setdefault("googleapiclient.discovery", types.ModuleType("googleapiclient.discovery"))

sys.modules["google.auth.transport.requests"].Request = type("Request", (), {})
sys.modules["google.oauth2.credentials"].Credentials = type("Credentials", (), {})

modules = [
    "core.config",
    "core.event_bus",
    "core.state_manager",
    "core.websocket_manager",
    "vision.camera_service",
    "vision.face_service",
    "vision.emotion_service",
    "vision.posture_service",
    "voice.speech_service",
    "voice.tts_service",
    "voice.conversation_service",
    "memory.memory_service",
    "memory.analytics_service",
    "integrations.weather_service",
    "integrations.news_service",
    "integrations.spotify_service",
    "integrations.calendar_service",
    "server",
]

for m in modules:
    __import__(m)

print("mocked imports ok")
