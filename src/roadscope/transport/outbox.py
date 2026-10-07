"""SQLite-backed outbox.

The legacy system had no outbound path at all. The naive fix -- publish to MQTT
and hope -- loses data the moment a vehicle loses signal, which in a tunnel or
an urban canyon is most of the survey.

The outbox makes transmission reliable rather than merely attempted:

1. Enqueue the message locally. **Commit before publishing.**
2. A publisher loop drains pending rows and marks them sent.
3. Signal loss accumulates rows; they drain on reconnect.

Combined with ``UNIQUE(trip_id, seq)`` on the receiving side, at-least-once
delivery becomes idempotent ingestion.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import paths
from ..storage.db import connect, transaction

OUTBOX_SCHEMA = """
CREATE TABLE IF NOT EXISTS outbox (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    topic      TEXT NOT NULL,
    payload    TEXT NOT NULL,
    created_at REAL NOT NULL,
    sent_at    REAL,
    attempts   INTEGER NOT NULL DEFAULT 0,
    last_error TEXT
);
CREATE INDEX IF NOT EXISTS idx_outbox_pending ON outbox (sent_at) WHERE sent_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_outbox_created ON outbox (created_at);
"""


@dataclass(slots=True)
class OutboxRow:
    """One queued message."""

    id: int
    topic: str
    payload: dict[str, Any]
    created_at: float
    attempts: int


class Outbox:
    """Durable publish queue backed by SQLite."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        paths.ensure_dir(self.db_path.parent)
        conn = connect(self.db_path)
        conn.executescript(OUTBOX_SCHEMA)
        conn.close()

    def enqueue(self, topic: str, payload: dict[str, Any]) -> int:
        """Queue a message. Returns its row id. Committed before returning."""
        with transaction(self.db_path) as conn:
            cur = conn.execute(
                "INSERT INTO outbox (topic, payload, created_at) VALUES (?,?,?)",
                (topic, json.dumps(payload, separators=(",", ":")), time.time()),
            )
            return int(cur.lastrowid or 0)

    def pending(self, limit: int = 100) -> list[OutboxRow]:
        """Oldest-first undelivered messages."""
        conn = connect(self.db_path, read_only=True)
        try:
            rows = conn.execute(
                "SELECT id, topic, payload, created_at, attempts FROM outbox"
                " WHERE sent_at IS NULL ORDER BY created_at, id LIMIT ?",
                (limit,),
            ).fetchall()
        finally:
            conn.close()
        return [
            OutboxRow(
                id=r["id"],
                topic=r["topic"],
                payload=json.loads(r["payload"]),
                created_at=r["created_at"],
                attempts=r["attempts"],
            )
            for r in rows
        ]

    def mark_sent(self, row_id: int) -> None:
        with transaction(self.db_path) as conn:
            conn.execute(
                "UPDATE outbox SET sent_at = ?, last_error = NULL WHERE id = ?",
                (time.time(), row_id),
            )

    def increment_attempt(self, row_id: int, error: str | None = None) -> None:
        """Record a failed delivery. The row stays pending."""
        with transaction(self.db_path) as conn:
            conn.execute(
                "UPDATE outbox SET attempts = attempts + 1, last_error = ? WHERE id = ?",
                ((error or "")[:500], row_id),
            )

    def depth(self) -> int:
        """Number of undelivered messages. The link-loss indicator."""
        conn = connect(self.db_path, read_only=True)
        try:
            row = conn.execute("SELECT COUNT(*) AS n FROM outbox WHERE sent_at IS NULL").fetchone()
            return int(row["n"])
        finally:
            conn.close()

    def oldest_pending_age_s(self) -> float:
        """Age of the oldest undelivered message. Rises while offline."""
        conn = connect(self.db_path, read_only=True)
        try:
            row = conn.execute(
                "SELECT created_at FROM outbox WHERE sent_at IS NULL ORDER BY created_at LIMIT 1"
            ).fetchone()
            return 0.0 if row is None else max(0.0, time.time() - row["created_at"])
        finally:
            conn.close()

    def purge_sent(self, older_than_s: float = 3600.0) -> int:
        """Delete delivered rows. Keeps the table from growing without bound."""
        cutoff = time.time() - older_than_s
        with transaction(self.db_path) as conn:
            cur = conn.execute(
                "DELETE FROM outbox WHERE sent_at IS NOT NULL AND sent_at < ?", (cutoff,)
            )
            return int(cur.rowcount or 0)
