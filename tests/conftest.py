"""Test bootstrap: isolate all data in a temp dir before backend modules import."""
import os
import sys
import tempfile
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="mirror-tests-")
os.environ["MIRROR_DATA_DIR"] = _TMP
os.environ["MIRROR_DB_PATH"] = str(Path(_TMP) / "test.db")
for key in ("GROQ_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "WEATHER_API_KEY", "NEWS_API_KEY"):
    os.environ[key] = ""
os.environ["MIRROR_TEST"] = "1"

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
for p in (str(ROOT), str(BACKEND)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest


@pytest.fixture()
def db_path(tmp_path):
    return tmp_path / "unit.db"
