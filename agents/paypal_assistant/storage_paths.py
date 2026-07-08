"""Shared filesystem path helpers for persistent local storage (Chroma dirs, etc.)."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def chroma_dir() -> str:
    configured = os.getenv("CHROMA_DB_DIR", ".chroma")
    path = Path(configured)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return str(path)
