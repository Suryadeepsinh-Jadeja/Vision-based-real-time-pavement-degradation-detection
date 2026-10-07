"""SQLite access.

One connection per thread (SQLite objects are not shareable across threads),
WAL journalling so the dashboard can read while the agent writes, and foreign
keys enforced.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .. import paths

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"
_local = threading.local()


def connect(db_path: Path, *, read_only: bool = False) -> sqlite3.Connection:
    """Open a connection with the project's standard pragmas."""
    if not read_only:
        paths.ensure_dir(db_path.parent)
    conn = sqlite3.connect(str(db_path), timeout=15.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA busy_timeout = 15000")
    return conn


def thread_connection(db_path: Path, *, read_only: bool = False) -> sqlite3.Connection:
    """Return this thread's cached connection, creating it on first use."""
    key = f"{db_path}:{read_only}"
    cached: sqlite3.Connection | None = getattr(_local, key, None)
    if cached is None:
        cached = connect(db_path, read_only=read_only)
        setattr(_local, key, cached)
    return cached


def init_db(db_path: Path) -> sqlite3.Connection:
    """Apply the schema. Idempotent."""
    conn = connect(db_path)
    conn.executescript(SCHEMA_PATH.read_text())
    return conn


@contextmanager
def transaction(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Run a block inside an IMMEDIATE transaction, rolling back on error.

    IMMEDIATE takes the write lock up front, so a concurrent writer fails fast
    with a clear "database is locked" instead of deadlocking at commit time.
    """
    conn = thread_connection(db_path)
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except Exception:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


def close_thread_connection() -> None:
    """Close and forget this thread's cached connection."""
    for attr in [a for a in vars(_local) if ":" in a]:
        conn = getattr(_local, attr)
        try:
            conn.close()
        finally:
            delattr(_local, attr)
