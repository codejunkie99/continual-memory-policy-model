"""Local staging store for sanitized history records.

Only scrubbed, non-sensitive text is stored here, keyed by a *salted* hash so a
plain dictionary attack on low-entropy content is not immediately feasible.
No absolute source paths and no original ids are ever persisted.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any, Iterable


_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    record_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    salted_hash TEXT NOT NULL UNIQUE,
    text        TEXT NOT NULL,
    category    TEXT NOT NULL,
    label       TEXT,
    source_kind TEXT NOT NULL,
    ts          REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_records_source ON records(source_kind);
CREATE INDEX IF NOT EXISTS idx_records_ts ON records(ts);
CREATE INDEX IF NOT EXISTS idx_records_category ON records(category);
"""


def salted_hash(text: str, salt: str) -> str:
    """Salted content digest for deduplication (not anonymization)."""
    return hashlib.sha256((salt + "\x00" + text).encode("utf-8")).hexdigest()


class HistoryStagingStore:
    """A single SQLite file (or ``:memory:``) of deduplicated, sanitized records."""

    def __init__(self, path: str | Path = ":memory:", *, salt: str = "mpm-history-v1"):
        self.path = str(path)
        self.salt = salt
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with self.conn:
            self.conn.executescript(_SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def insert(
        self,
        text: str,
        *,
        category: str,
        label: str | None,
        source_kind: str,
        ts: float,
    ) -> bool:
        """Insert one sanitized record; returns False if already present."""
        digest = salted_hash(text, self.salt)
        with self.conn:
            cur = self.conn.execute(
                "INSERT OR IGNORE INTO records (salted_hash, text, category, label, source_kind, ts) "
                "VALUES (?,?,?,?,?,?)",
                (digest, text, category, label, source_kind, float(ts)),
            )
        return cur.rowcount > 0

    def counts(self) -> dict[str, int]:
        """Aggregate counts: total plus per-source and per-category tallies."""
        total = self.conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
        by_source = {
            r["source_kind"]: r["n"]
            for r in self.conn.execute(
                "SELECT source_kind, COUNT(*) AS n FROM records GROUP BY source_kind"
            ).fetchall()
        }
        by_category = {
            r["category"]: r["n"]
            for r in self.conn.execute(
                "SELECT category, COUNT(*) AS n FROM records GROUP BY category"
            ).fetchall()
        }
        return {"total": total, "by_source": by_source, "by_category": by_category}

    def records(
        self,
        *,
        source_kinds: Iterable[str] | None = None,
        since: float | None = None,
        until: float | None = None,
    ) -> list[dict[str, Any]]:
        """Return sanitized records (optionally filtered), ordered by ts."""
        clauses: list[str] = []
        args: list[Any] = []
        if source_kinds is not None:
            kinds = list(source_kinds)
            clauses.append("source_kind IN (%s)" % ",".join("?" for _ in kinds))
            args.extend(kinds)
        if since is not None:
            clauses.append("ts >= ?")
            args.append(float(since))
        if until is not None:
            clauses.append("ts <= ?")
            args.append(float(until))
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self.conn.execute(
            f"SELECT * FROM records{where} ORDER BY ts ASC, salted_hash ASC", args
        ).fetchall()
        return [dict(r) for r in rows]
