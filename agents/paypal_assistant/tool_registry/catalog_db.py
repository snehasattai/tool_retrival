"""SQLite-backed Tool Catalog: the actual metadata store for the Tool Registry.

Two distinct sync directions meet in this one table:

  1. Code -> catalog (`bootstrap_from_code`): specs.py/synthetic_specs.py
     (Python) remain authoritative for *which tools exist* and their
     category/description/is_real -- a function can't be invented by editing
     SQL, it has to exist in code. `bootstrap_from_code` is idempotent and
     re-runs on every sync, upserting name/category/description/is_real from
     the current code into this table (without touching `content_hash`,
     which belongs to sync direction 2).
  2. Catalog -> Chroma (`content_hash` tracking, read via `get_all_hashes`):
     once a row's description is in the catalog, tool_registry/index.py's
     sync_registry() diffs it against `content_hash` (the hash of whatever
     was last actually embedded into Chroma) to decide what needs
     (re-)embedding -- unchanged the same as before, just now comparing
     against catalog data instead of live Python objects.

The upshot: this table is the actual, fully-resolved metadata store (not a
lazy re-derivation of docstrings at runtime) -- inspectable directly and
independently of Chroma:

    sqlite3 .chroma/tool_catalog.db "SELECT name, category, description FROM tools WHERE category='payments'"

The one thing that deliberately never lives here is the callable itself --
see services/function_registry.py for why, and for the explicit consistency
check between "this row claims is_real=1" and "a matching Python function
actually exists."
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from ..storage_paths import REPO_ROOT

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tools (
    name TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    description TEXT NOT NULL,
    is_real INTEGER NOT NULL,
    content_hash TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tools_category ON tools(category);
"""


def _db_path() -> Path:
    configured = os.getenv("CHROMA_DB_DIR", ".chroma")
    path = Path(configured)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path / "tool_catalog.db"


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(_db_path())
    conn.executescript(_SCHEMA)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def bootstrap_from_code(rows: list[tuple[str, str, str, bool]]) -> None:
    """Sync direction 1: code -> catalog. rows: (name, category, description, is_real).

    Idempotent and safe to call every time sync_registry() runs -- upserts
    the descriptive columns from whatever specs.py/synthetic_specs.py
    currently say, but deliberately does NOT touch `content_hash` (that's
    sync direction 2's job, see mark_embedded below), so a description
    change here is exactly what makes the next Chroma-sync step notice and
    re-embed it.
    """
    if not rows:
        return
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.executemany(
            """
            INSERT INTO tools (name, category, description, is_real, content_hash, updated_at)
            VALUES (?, ?, ?, ?, '', ?)
            ON CONFLICT(name) DO UPDATE SET
                category=excluded.category,
                description=excluded.description,
                is_real=excluded.is_real,
                updated_at=excluded.updated_at
            """,
            [(name, category, description, int(is_real), now) for name, category, description, is_real in rows],
        )


def mark_embedded(rows: list[tuple[str, str]]) -> None:
    """Sync direction 2: record that `name`'s current description has now
    been embedded into Chroma with this content_hash. rows: (name, content_hash)."""
    if not rows:
        return
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.executemany(
            "UPDATE tools SET content_hash = ?, updated_at = ? WHERE name = ?",
            [(content_hash, now, name) for name, content_hash in rows],
        )


def get_all_hashes() -> dict[str, str]:
    """The hash of whatever was last actually embedded into Chroma, per tool."""
    with _connect() as conn:
        rows = conn.execute("SELECT name, content_hash FROM tools").fetchall()
    return dict(rows)


def get_all_specs() -> list[dict]:
    """The full, current, resolved metadata for every tool -- this is what
    tool_registry/index.py reads to build retrieval-time ToolSpec objects."""
    with _connect() as conn:
        cursor = conn.execute("SELECT name, category, description, is_real FROM tools ORDER BY category, name")
        columns = [d[0] for d in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
    for row in rows:
        row["is_real"] = bool(row["is_real"])
    return rows


def delete_many(names: list[str]) -> None:
    if not names:
        return
    with _connect() as conn:
        conn.executemany("DELETE FROM tools WHERE name = ?", [(n,) for n in names])


def all_rows() -> list[dict]:
    """Full catalog dump including sync bookkeeping -- e.g. for an admin/debug view."""
    with _connect() as conn:
        cursor = conn.execute(
            "SELECT name, category, description, is_real, content_hash, updated_at FROM tools ORDER BY category, name"
        )
        columns = [d[0] for d in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
