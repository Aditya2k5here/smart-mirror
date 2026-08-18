from __future__ import annotations

import sys
from pathlib import Path

# Allow backend package execution while preserving the existing script-run import style.
# When running `python -m backend.server`, the package root must be importable for
# `from core.*`, `from vision.*`, and other local backend imports to resolve.
PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
