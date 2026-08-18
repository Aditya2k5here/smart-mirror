from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import os

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional during first setup.
    load_dotenv = None


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
SETTINGS_PATH = BACKEND_DIR / "settings.json"


def _load_plain_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


if load_dotenv:
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(BACKEND_DIR / ".env", override=True)
else:
    _load_plain_env(PROJECT_ROOT / ".env")
    _load_plain_env(BACKEND_DIR / ".env")


def _load_settings() -> dict[str, Any]:
    if not SETTINGS_PATH.exists():
        return {}
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


SETTINGS = _load_settings()


def _setting(name: str, default: Any) -> Any:
    return SETTINGS.get(name, default)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_or_setting(name: str, setting_name: str, default: Any) -> Any:
    raw = os.environ.get(name)
    if raw is not None:
        return raw
    return _setting(setting_name, default)


def _env_bool_or_setting(name: str, setting_name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is not None:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return bool(_setting(setting_name, default))


def _env_int_or_setting(name: str, setting_name: str, default: int) -> int:
    raw = _env_or_setting(name, setting_name, default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _env_float_or_setting(name: str, setting_name: str, default: float) -> float:
    raw = _env_or_setting(name, setting_name, default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _path_from_env(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    path = Path(raw).expanduser() if raw else default
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


@dataclass(frozen=True)
class MirrorConfig:
    websocket_host: str
    websocket_port: int
    camera_index: int
    camera_width: int
    camera_height: int
    camera_preview: bool
    target_fps: int
    broadcast_hz: float
    absence_timeout_seconds: float
    face_recognition_interval_seconds: float
    emotion_interval_seconds: float
    posture_interval_seconds: float
    spotify_interval_seconds: float
    weather_interval_seconds: float
    calendar_interval_seconds: float
    news_interval_seconds: float
    tasks_interval_seconds: float
    tts_min_gap_seconds: float
    data_dir: Path
    known_faces_dir: Path
    db_path: Path
    weather_city: str
    weather_api_key: str
    news_api_key: str
    openai_model: str
    groq_api_key: str
    groq_model: str
    gemini_api_key: str
    gemini_model: str
    ai_router_mode: str
    face_match_tolerance: float
    show_health_when_idle: bool
    show_diagnostics: bool


def load_config() -> MirrorConfig:
    data_dir = _path_from_env("MIRROR_DATA_DIR", BACKEND_DIR / "data")
    known_faces_dir = _path_from_env(
        "MIRROR_KNOWN_FACES_DIR", data_dir / "known_faces"
    )
    db_path = _path_from_env("MIRROR_DB_PATH", data_dir / "mirror.db")

    return MirrorConfig(
        websocket_host=os.environ.get("MIRROR_WS_HOST", "localhost"),
        websocket_port=_env_int("MIRROR_WS_PORT", 8765),
        camera_index=_env_int_or_setting("MIRROR_CAMERA_INDEX", "camera_index", 0),
        camera_width=_env_int_or_setting("MIRROR_CAMERA_WIDTH", "camera_width", 1280),
        camera_height=_env_int_or_setting("MIRROR_CAMERA_HEIGHT", "camera_height", 720),
        camera_preview=_env_bool_or_setting("MIRROR_CAMERA_PREVIEW", "camera_preview", False),
        target_fps=_env_int_or_setting("MIRROR_TARGET_FPS", "target_fps", 24),
        broadcast_hz=_env_float_or_setting("MIRROR_BROADCAST_HZ", "broadcast_hz", 4.0),
        absence_timeout_seconds=_env_float("MIRROR_ABSENCE_TIMEOUT", 3.0),
        face_recognition_interval_seconds=_env_float_or_setting(
            "MIRROR_FACE_RECOGNITION_INTERVAL", "face_recognition_interval_seconds", 2.0
        ),
        emotion_interval_seconds=_env_float_or_setting("MIRROR_EMOTION_INTERVAL", "emotion_interval_seconds", 6.0),
        posture_interval_seconds=_env_float_or_setting("MIRROR_POSTURE_INTERVAL", "posture_interval_seconds", 0.9),
        spotify_interval_seconds=_env_float_or_setting("MIRROR_SPOTIFY_INTERVAL", "spotify_interval_seconds", 5.0),
        weather_interval_seconds=_env_float_or_setting("MIRROR_WEATHER_INTERVAL", "weather_interval_seconds", 600.0),
        calendar_interval_seconds=_env_float_or_setting("MIRROR_CALENDAR_INTERVAL", "calendar_interval_seconds", 300.0),
        news_interval_seconds=_env_float_or_setting("MIRROR_NEWS_INTERVAL", "news_interval_seconds", 900.0),
        tasks_interval_seconds=_env_float_or_setting("MIRROR_TASKS_INTERVAL", "tasks_interval_seconds", 2.0),
        tts_min_gap_seconds=_env_float("MIRROR_TTS_MIN_GAP", 1.0),
        data_dir=data_dir,
        known_faces_dir=known_faces_dir,
        db_path=db_path,
        weather_city=str(_env_or_setting("WEATHER_CITY", "weather_city", "Bangalore")),
        weather_api_key=os.environ.get("WEATHER_API_KEY", ""),
        news_api_key=os.environ.get("NEWS_API_KEY", ""),
        openai_model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        groq_api_key=os.environ.get("GROQ_API_KEY", ""),
        groq_model=os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),
        gemini_api_key=os.environ.get("GEMINI_API_KEY", ""),
        gemini_model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite"),
        ai_router_mode=str(_env_or_setting("AI_ROUTER_MODE", "ai_router_mode", "auto")).strip().lower(),
        face_match_tolerance=_env_float("MIRROR_FACE_MATCH_TOLERANCE", 0.48),
        show_health_when_idle=_env_bool_or_setting("MIRROR_SHOW_HEALTH_WHEN_IDLE", "show_health_when_idle", True),
        show_diagnostics=_env_bool_or_setting("MIRROR_SHOW_DIAGNOSTICS", "show_diagnostics", False),
    )


CONFIG = load_config()
