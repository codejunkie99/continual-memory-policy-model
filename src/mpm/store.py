"""Event-sourced SQLite store for the external memory system.

Design contract
---------------
* ``events`` is the immutable append-only ledger; every mutation writes exactly
  one event *before* updating materialized state, inside the same transaction.
* ``memories`` holds current state (content lives here and is never exported).
* ``memory_revisions`` preserves full update history; DELETE is a soft tombstone.
* ``links`` and ``compact_*`` retain provenance (who produced a link/compact).
* All writes are transactional (``with self.conn:``).
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable

from .features import content_hash
from .safety import is_harmful_content
from .types import (
    CheckpointStatus,
    MemoryStatus,
    Op,
    PayloadError,
    Action,
    validate_action,
    validate_payload,
)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL    NOT NULL,
    txn_id      TEXT    NOT NULL,
    event_type  TEXT    NOT NULL,
    actor       TEXT    NOT NULL,
    payload     TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    scenario    TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    started_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS memories (
    memory_id        TEXT PRIMARY KEY,
    status           TEXT NOT NULL DEFAULT 'active',
    scope            TEXT,
    key              TEXT,
    content          TEXT NOT NULL,
    content_hash     TEXT NOT NULL,
    created_at       REAL NOT NULL,
    updated_at       REAL NOT NULL,
    current_revision INTEGER NOT NULL,
    created_event    INTEGER NOT NULL,
    UNIQUE(scope, key)
);

CREATE TABLE IF NOT EXISTS memory_revisions (
    revision_id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id   TEXT NOT NULL,
    rev         INTEGER NOT NULL,
    op          TEXT NOT NULL,
    content     TEXT NOT NULL,
    ts          REAL NOT NULL,
    event_seq   INTEGER NOT NULL,
    UNIQUE(memory_id, rev)
);

CREATE TABLE IF NOT EXISTS links (
    link_id       TEXT PRIMARY KEY,
    source_id     TEXT NOT NULL,
    target_id     TEXT NOT NULL,
    kind          TEXT NOT NULL,
    weight        REAL NOT NULL,
    status        TEXT NOT NULL DEFAULT 'active',
    created_at    REAL NOT NULL,
    created_event INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS compact_groups (
    compact_id       TEXT PRIMARY KEY,
    result_memory_id TEXT NOT NULL,
    strategy         TEXT NOT NULL,
    created_at       REAL NOT NULL,
    created_event    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS compact_members (
    compact_id       TEXT NOT NULL,
    source_memory_id TEXT NOT NULL,
    PRIMARY KEY (compact_id, source_memory_id)
);

CREATE TABLE IF NOT EXISTS interactions (
    interaction_id TEXT PRIMARY KEY,
    session_id     TEXT NOT NULL,
    ts             REAL NOT NULL,
    kind           TEXT NOT NULL,
    features       TEXT NOT NULL,
    raw            TEXT
);

CREATE TABLE IF NOT EXISTS policy_decisions (
    decision_id    TEXT PRIMARY KEY,
    policy_version TEXT NOT NULL,
    op             TEXT NOT NULL,
    target         TEXT,
    payload        TEXT NOT NULL,
    confidence     REAL NOT NULL,
    rationale      TEXT,
    features       TEXT NOT NULL,
    ts             REAL NOT NULL,
    event_seq      INTEGER NOT NULL,
    session_id     TEXT
);

CREATE TABLE IF NOT EXISTS retrievals (
    retrieval_id   TEXT PRIMARY KEY,
    session_id     TEXT NOT NULL,
    memory_id      TEXT NOT NULL,
    query_features TEXT NOT NULL,
    rank           INTEGER NOT NULL,
    score          REAL NOT NULL,
    ts             REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS outcomes (
    outcome_id   TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL,
    retrieval_id TEXT,
    kind         TEXT NOT NULL,
    value        REAL NOT NULL,
    confidence   REAL NOT NULL,
    ts           REAL NOT NULL,
    raw          TEXT
);

CREATE TABLE IF NOT EXISTS outcome_retrievals (
    outcome_id    TEXT NOT NULL,
    retrieval_id  TEXT NOT NULL,
    contribution  REAL NOT NULL,
    PRIMARY KEY (outcome_id, retrieval_id)
);

CREATE TABLE IF NOT EXISTS attributions (
    attribution_id       TEXT PRIMARY KEY,
    memory_id            TEXT NOT NULL,
    operation_event_seq  INTEGER NOT NULL,
    retrieval_id         TEXT,
    outcome_id           TEXT NOT NULL,
    weight               REAL NOT NULL,
    confidence           REAL NOT NULL,
    reward               REAL NOT NULL,
    ts                   REAL NOT NULL,
    audit                TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS checkpoints (
    version       TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    status        TEXT NOT NULL,
    created_at    REAL NOT NULL,
    metrics       TEXT,
    artifact_path TEXT,
    parent        TEXT,
    note          TEXT,
    gate_approved INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS training_examples (
    example_id INTEGER PRIMARY KEY AUTOINCREMENT,
    split      TEXT NOT NULL,
    kind       TEXT NOT NULL,
    stratum    TEXT NOT NULL,
    scenario   TEXT NOT NULL,
    user_id    TEXT NOT NULL,
    t          TEXT NOT NULL,
    features   TEXT NOT NULL,
    label      TEXT NOT NULL,
    chosen     TEXT,
    rejected   TEXT,
    reward     REAL,
    raw        TEXT
);

CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);
CREATE INDEX IF NOT EXISTS idx_revisions_memory ON memory_revisions(memory_id, rev);
CREATE INDEX IF NOT EXISTS idx_retrievals_memory ON retrievals(memory_id);
CREATE INDEX IF NOT EXISTS idx_outcomes_retrieval ON outcomes(retrieval_id);
CREATE INDEX IF NOT EXISTS idx_attributions_outcome ON attributions(outcome_id);
CREATE INDEX IF NOT EXISTS idx_attributions_memory ON attributions(memory_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_attributions_pair ON attributions(outcome_id, retrieval_id);
CREATE INDEX IF NOT EXISTS idx_decisions_session ON policy_decisions(session_id);
"""


def _now() -> float:
    return time.time()


def _uid() -> str:
    return uuid.uuid4().hex


class MemoryStore:
    """Wraps a single SQLite database file (or ``:memory:``)."""

    def __init__(self, path: str | Path = ":memory:", *, clock: Callable[[], float] = _now):
        self.path = str(path)
        self._clock = clock
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self._init_schema()

    def _init_schema(self) -> None:
        with self.conn:
            self.conn.executescript(_SCHEMA)
            checkpoint_columns = {
                row["name"] for row in self.conn.execute("PRAGMA table_info(checkpoints)").fetchall()
            }
            if "gate_approved" not in checkpoint_columns:
                self.conn.execute(
                    "ALTER TABLE checkpoints ADD COLUMN gate_approved INTEGER NOT NULL DEFAULT 0"
                )

    def close(self) -> None:
        self.conn.close()

    # -- low level -------------------------------------------------------

    def _append_event(self, event_type: str, payload: dict[str, Any], actor: str) -> tuple[int, float]:
        ts = self._clock()
        txn_id = _uid()
        cur = self.conn.execute(
            "INSERT INTO events (ts, txn_id, event_type, actor, payload) VALUES (?,?,?,?,?)",
            (ts, txn_id, event_type, actor, json.dumps(payload, sort_keys=True, default=str)),
        )
        return cur.lastrowid, ts

    def ensure_session(self, session_id: str, scenario: str, user_id: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO sessions (session_id, scenario, user_id, started_at) VALUES (?,?,?,?)",
                (session_id, scenario, user_id, self._clock()),
            )

    # -- memory operations ----------------------------------------------

    def _get_memory(self, memory_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM memories WHERE memory_id = ?", (memory_id,)).fetchone()
        return dict(row) if row else None

    def _require_active(self, memory_id: str, op: str) -> dict[str, Any]:
        mem = self._get_memory(memory_id)
        if mem is None:
            raise PayloadError(f"{op} target {memory_id!r} does not exist")
        return mem

    def write_memory(
        self,
        content: str,
        *,
        scope: str = "default",
        key: str | None = None,
        actor: str = "system",
        features: dict[str, Any] | None = None,
        session_id: str | None = None,
        confidence: float = 1.0,
        rationale: str = "",
        policy_version: str = "baseline",
    ) -> str:
        payload = {"op": Op.WRITE.value, "content": content, "scope": scope, "key": key}
        errors = validate_payload(Op.WRITE.value, payload)
        if errors:
            raise PayloadError("; ".join(errors))
        memory_id = _uid()
        with self.conn:
            seq, ts = self._append_event(
                "memory.write",
                {"memory_id": memory_id, "scope": scope, "key": key, "content_hash": content_hash(content)},
                actor,
            )
            self.conn.execute(
                """INSERT INTO memories
                   (memory_id, status, scope, key, content, content_hash, created_at, updated_at, current_revision, created_event)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (memory_id, MemoryStatus.ACTIVE.value, scope, key, content, content_hash(content), ts, ts, 1, seq),
            )
            self.conn.execute(
                "INSERT INTO memory_revisions (memory_id, rev, op, content, ts, event_seq) VALUES (?,?,?,?,?,?)",
                (memory_id, 1, Op.WRITE.value, content, ts, seq),
            )
            self._record_decision(
                Op.WRITE.value, memory_id, payload, features, actor, seq, ts, session_id, confidence, rationale, policy_version
            )
        return memory_id

    def update_memory(
        self,
        memory_id: str,
        content: str,
        *,
        actor: str = "system",
        session_id: str | None = None,
        confidence: float = 1.0,
        rationale: str = "",
        policy_version: str = "baseline",
    ) -> int:
        payload = {"op": Op.UPDATE.value, "memory_id": memory_id, "content": content}
        errors = validate_payload(Op.UPDATE.value, payload)
        if errors:
            raise PayloadError("; ".join(errors))
        with self.conn:
            mem = self._require_active(memory_id, "UPDATE")
            if mem["status"] != MemoryStatus.ACTIVE.value:
                raise PayloadError(f"UPDATE target {memory_id!r} is {mem['status']}, not active")
            new_rev = mem["current_revision"] + 1
            seq, ts = self._append_event("memory.update", {"memory_id": memory_id, "rev": new_rev}, actor)
            self.conn.execute(
                "UPDATE memories SET content=?, content_hash=?, updated_at=?, current_revision=? WHERE memory_id=?",
                (content, content_hash(content), ts, new_rev, memory_id),
            )
            self.conn.execute(
                "INSERT INTO memory_revisions (memory_id, rev, op, content, ts, event_seq) VALUES (?,?,?,?,?,?)",
                (memory_id, new_rev, Op.UPDATE.value, content, ts, seq),
            )
            self._record_decision(
                Op.UPDATE.value, memory_id, payload, None, actor, seq, ts, session_id, confidence, rationale, policy_version
            )
        return new_rev

    def delete_memory(
        self,
        memory_id: str,
        *,
        actor: str = "system",
        session_id: str | None = None,
        confidence: float = 1.0,
        rationale: str = "",
        policy_version: str = "baseline",
    ) -> None:
        payload = {"op": Op.DELETE.value, "memory_id": memory_id}
        errors = validate_payload(Op.DELETE.value, payload)
        if errors:
            raise PayloadError("; ".join(errors))
        with self.conn:
            mem = self._require_active(memory_id, "DELETE")
            if mem["status"] == MemoryStatus.TOMBSTONED.value:
                raise PayloadError(f"DELETE target {memory_id!r} is already tombstoned")
            seq, ts = self._append_event("memory.delete", {"memory_id": memory_id}, actor)
            self.conn.execute(
                "UPDATE memories SET status=?, updated_at=? WHERE memory_id=?",
                (MemoryStatus.TOMBSTONED.value, ts, memory_id),
            )
            self._record_decision(
                Op.DELETE.value, memory_id, payload, None, actor, seq, ts, session_id, confidence, rationale, policy_version
            )

    def link_memories(
        self,
        source_id: str,
        target_id: str,
        kind: str,
        weight: float,
        *,
        actor: str = "system",
        session_id: str | None = None,
        confidence: float = 1.0,
        rationale: str = "",
        policy_version: str = "baseline",
    ) -> str:
        payload = {"op": Op.LINK.value, "source_id": source_id, "target_id": target_id, "kind": kind, "weight": weight}
        errors = validate_payload(Op.LINK.value, payload)
        if errors:
            raise PayloadError("; ".join(errors))
        with self.conn:
            for mid, label in ((source_id, "source"), (target_id, "target")):
                self._require_active(mid, f"LINK {label}")
            link_id = _uid()
            seq, ts = self._append_event(
                "memory.link", {"link_id": link_id, "source_id": source_id, "target_id": target_id, "kind": kind}, actor
            )
            self.conn.execute(
                """INSERT INTO links (link_id, source_id, target_id, kind, weight, status, created_at, created_event)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (link_id, source_id, target_id, kind, float(weight), "active", ts, seq),
            )
            self._record_decision(
                Op.LINK.value, link_id, payload, None, actor, seq, ts, session_id, confidence, rationale, policy_version
            )
        return link_id

    def compact(
        self,
        memory_ids: list[str],
        strategy: str,
        *,
        actor: str = "system",
        content: str | None = None,
        session_id: str | None = None,
        confidence: float = 1.0,
        rationale: str = "",
        policy_version: str = "baseline",
    ) -> str:
        payload = {"op": Op.COMPACT.value, "memory_ids": list(memory_ids), "strategy": strategy}
        errors = validate_payload(Op.COMPACT.value, payload)
        if errors:
            raise PayloadError("; ".join(errors))
        with self.conn:
            sources: list[dict[str, Any]] = []
            for mid in memory_ids:
                mem = self._require_active(mid, "COMPACT")
                if mem["status"] != MemoryStatus.ACTIVE.value:
                    raise PayloadError(f"COMPACT source {mid!r} is {mem['status']}, not active")
                sources.append(mem)
            if content is None:
                content = f"[{strategy}] " + " || ".join(m["content"] for m in sources)[:2000]
            result_id = _uid()
            compact_id = _uid()
            seq, ts = self._append_event(
                "memory.compact",
                {"compact_id": compact_id, "result_memory_id": result_id, "memory_ids": memory_ids, "strategy": strategy},
                actor,
            )
            self.conn.execute(
                """INSERT INTO memories
                   (memory_id, status, scope, key, content, content_hash, created_at, updated_at, current_revision, created_event)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (result_id, MemoryStatus.ACTIVE.value, "compact", None, content, content_hash(content), ts, ts, 1, seq),
            )
            self.conn.execute(
                "INSERT INTO memory_revisions (memory_id, rev, op, content, ts, event_seq) VALUES (?,?,?,?,?,?)",
                (result_id, 1, Op.COMPACT.value, content, ts, seq),
            )
            self.conn.execute(
                "INSERT INTO compact_groups (compact_id, result_memory_id, strategy, created_at, created_event) VALUES (?,?,?,?,?)",
                (compact_id, result_id, strategy, ts, seq),
            )
            self.conn.executemany(
                "INSERT INTO compact_members (compact_id, source_memory_id) VALUES (?,?)",
                [(compact_id, mid) for mid in memory_ids],
            )
            self.conn.executemany(
                "UPDATE memories SET status=?, updated_at=? WHERE memory_id=?",
                [(MemoryStatus.COMPACTED.value, ts, mid) for mid in memory_ids],
            )
            self._record_decision(
                Op.COMPACT.value, result_id, payload, None, actor, seq, ts, session_id, confidence, rationale, policy_version
            )
        return result_id

    def record_noop(
        self,
        *,
        reason: str = "",
        observed: dict[str, Any] | None = None,
        actor: str = "system",
        features: dict[str, Any] | None = None,
        session_id: str | None = None,
        confidence: float = 1.0,
        rationale: str = "",
        policy_version: str = "baseline",
    ) -> None:
        payload = {"op": Op.NOOP.value, "reason": reason, "observed": observed or {}}
        errors = validate_payload(Op.NOOP.value, payload)
        if errors:
            raise PayloadError("; ".join(errors))
        with self.conn:
            seq, ts = self._append_event("policy.noop", {"reason": reason}, actor)
            self._record_decision(
                Op.NOOP.value, None, payload, features, actor, seq, ts, session_id, confidence, rationale, policy_version
            )

    # -- generic action dispatch ---------------------------------------

    def apply_action(
        self,
        action: Action,
        *,
        actor: str = "system",
        session_id: str | None = None,
        features: dict[str, Any] | None = None,
        observation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        errors = validate_action(action)
        if errors:
            raise PayloadError("; ".join(errors))
        op = action.op
        conf = float(action.confidence)
        rat = action.rationale
        pv = action.policy_version

        def resolved_content() -> str:
            if "content" in action.payload:
                return str(action.payload["content"])
            if action.payload.get("content_ref") == "observation.content" and observation is not None:
                content = observation.get("content")
                if isinstance(content, str) and content.strip():
                    harmful, reason = is_harmful_content(content)
                    if harmful:
                        raise PayloadError(f"refusing harmful memory content: {reason}")
                    return content
            raise PayloadError("content_ref requires a non-empty observation.content")

        # Defense in depth for callers that place content directly in a payload
        # instead of using the trusted observation.content reference.
        if op in {Op.WRITE.value, Op.UPDATE.value} and "content" in action.payload:
            harmful, reason = is_harmful_content(str(action.payload["content"]))
            if harmful:
                raise PayloadError(f"refusing harmful memory content: {reason}")

        if op == Op.WRITE.value:
            memory_id = self.write_memory(
                resolved_content(),
                scope=action.payload.get("scope", (observation or {}).get("scope", "default")),
                key=action.payload.get("key", (observation or {}).get("key")),
                actor=actor,
                features=features,
                session_id=session_id,
                confidence=conf,
                rationale=rat,
                policy_version=pv,
            )
            return {"op": op, "memory_id": memory_id}
        if op == Op.UPDATE.value:
            rev = self.update_memory(
                action.payload["memory_id"],
                resolved_content(),
                actor=actor,
                session_id=session_id,
                confidence=conf,
                rationale=rat,
                policy_version=pv,
            )
            return {"op": op, "memory_id": action.payload["memory_id"], "rev": rev}
        if op == Op.DELETE.value:
            self.delete_memory(
                action.payload["memory_id"],
                actor=actor,
                session_id=session_id,
                confidence=conf,
                rationale=rat,
                policy_version=pv,
            )
            return {"op": op, "memory_id": action.payload["memory_id"]}
        if op == Op.LINK.value:
            link_id = self.link_memories(
                action.payload["source_id"],
                action.payload["target_id"],
                action.payload["kind"],
                action.payload["weight"],
                actor=actor,
                session_id=session_id,
                confidence=conf,
                rationale=rat,
                policy_version=pv,
            )
            return {"op": op, "link_id": link_id}
        if op == Op.COMPACT.value:
            result_id = self.compact(
                action.payload["memory_ids"],
                action.payload["strategy"],
                actor=actor,
                session_id=session_id,
                confidence=conf,
                rationale=rat,
                policy_version=pv,
            )
            return {"op": op, "memory_id": result_id}
        if op == Op.NOOP.value:
            self.record_noop(
                reason=action.payload.get("reason", ""),
                observed=action.payload.get("observed"),
                actor=actor,
                features=features,
                session_id=session_id,
                confidence=conf,
                rationale=rat,
                policy_version=pv,
            )
            return {"op": op}
        raise PayloadError(f"unhandled op {op!r}")

    # -- decision recording --------------------------------------------

    def _record_decision(
        self,
        op: str,
        target: str | None,
        payload: dict[str, Any],
        features: dict[str, Any] | None,
        actor: str,
        seq: int,
        ts: float,
        session_id: str | None,
        confidence: float,
        rationale: str,
        policy_version: str,
    ) -> None:
        self.conn.execute(
            "INSERT INTO events (ts, txn_id, event_type, actor, payload) VALUES (?,?,?,?,?)",
            (ts, _uid(), "policy.decision", actor, json.dumps({"op": op, "target": target}, sort_keys=True)),
        )
        self.conn.execute(
            """INSERT INTO policy_decisions
               (decision_id, policy_version, op, target, payload, confidence, rationale, features, ts, event_seq, session_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                _uid(),
                policy_version,
                op,
                target,
                json.dumps(payload, sort_keys=True, default=str),
                float(confidence),
                rationale,
                json.dumps(features or {}, sort_keys=True),
                ts,
                seq,
                session_id,
            ),
        )

    def record_policy_decision(
        self,
        action: Action,
        *,
        actor: str = "system",
        session_id: str | None = None,
        features: dict[str, Any] | None = None,
    ) -> str:
        """Record a decision from any policy without mutating memory."""
        errors = validate_action(action)
        if errors:
            raise PayloadError("; ".join(errors))
        decision_id = _uid()
        with self.conn:
            seq, ts = self._append_event(
                "policy.decision",
                {"op": action.op, "target": action.target, "policy_version": action.policy_version},
                actor,
            )
            self.conn.execute(
                """INSERT INTO policy_decisions
                   (decision_id, policy_version, op, target, payload, confidence, rationale, features, ts, event_seq, session_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    decision_id,
                    action.policy_version,
                    action.op,
                    action.target,
                    json.dumps(action.payload, sort_keys=True, default=str),
                    float(action.confidence),
                    action.rationale,
                    json.dumps(features or {}, sort_keys=True),
                    ts,
                    seq,
                    session_id,
                ),
            )
        return decision_id

    # -- retrieval / outcome / attribution -----------------------------

    def record_retrieval(
        self,
        session_id: str,
        memory_id: str,
        query_features: dict[str, Any],
        rank: int,
        score: float,
    ) -> str:
        retrieval_id = _uid()
        with self.conn:
            seq, ts = self._append_event("retrieval", {"retrieval_id": retrieval_id, "memory_id": memory_id}, "system")
            self.conn.execute(
                """INSERT INTO retrievals (retrieval_id, session_id, memory_id, query_features, rank, score, ts)
                   VALUES (?,?,?,?,?,?,?)""",
                (retrieval_id, session_id, memory_id, json.dumps(query_features or {}, sort_keys=True), int(rank), float(score), ts),
            )
        return retrieval_id

    def record_outcome(
        self,
        session_id: str,
        *,
        retrieval_id: str | None,
        kind: str,
        value: float,
        confidence: float,
        raw: str | None = None,
        retrieval_weights: dict[str, float] | None = None,
    ) -> str:
        if kind not in {"positive", "negative", "neutral"}:
            raise PayloadError("outcome kind must be positive, negative, or neutral")
        value = float(value)
        confidence = float(confidence)
        if not math.isfinite(value) or not -1.0 <= value <= 1.0:
            raise PayloadError("outcome value must be finite and in [-1, 1]")
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise PayloadError("outcome confidence must be in [0, 1]")
        if kind == "positive" and value < 0:
            raise PayloadError("positive outcomes require a non-negative value")
        if kind == "negative" and value > 0:
            raise PayloadError("negative outcomes require a non-positive value")
        weights = dict(retrieval_weights or {})
        if retrieval_id is not None:
            weights.setdefault(retrieval_id, 1.0)
        if (
            any(not math.isfinite(float(weight)) or float(weight) < 0 for weight in weights.values())
            or (weights and sum(float(weight) for weight in weights.values()) <= 0)
        ):
            raise PayloadError("retrieval contribution weights must be non-negative with a positive sum")
        for rid in weights:
            row = self.conn.execute(
                "SELECT session_id FROM retrievals WHERE retrieval_id = ?", (rid,)
            ).fetchone()
            if row is None:
                raise PayloadError(f"retrieval {rid!r} does not exist")
            if row["session_id"] != session_id:
                raise PayloadError(f"retrieval {rid!r} belongs to a different session")
        outcome_id = _uid()
        with self.conn:
            seq, ts = self._append_event(
                "outcome", {"outcome_id": outcome_id, "retrieval_id": retrieval_id, "kind": kind, "value": value}, "system"
            )
            self.conn.execute(
                """INSERT INTO outcomes (outcome_id, session_id, retrieval_id, kind, value, confidence, ts, raw)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (outcome_id, session_id, retrieval_id, kind, float(value), float(confidence), ts, raw),
            )
            self.conn.executemany(
                "INSERT INTO outcome_retrievals (outcome_id, retrieval_id, contribution) VALUES (?,?,?)",
                [(outcome_id, rid, float(weight)) for rid, weight in weights.items()],
            )
        return outcome_id

    def add_attribution(
        self,
        *,
        memory_id: str,
        operation_event_seq: int,
        retrieval_id: str | None,
        outcome_id: str,
        weight: float,
        confidence: float,
        reward: float,
        audit: dict[str, Any],
    ) -> str:
        attribution_id = _uid()
        with self.conn:
            seq, ts = self._append_event(
                "attribution",
                {"attribution_id": attribution_id, "memory_id": memory_id, "outcome_id": outcome_id, "reward": reward},
                "system",
            )
            self.conn.execute(
                """INSERT INTO attributions
                   (attribution_id, memory_id, operation_event_seq, retrieval_id, outcome_id, weight, confidence, reward, ts, audit)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    attribution_id,
                    memory_id,
                    int(operation_event_seq),
                    retrieval_id,
                    outcome_id,
                    float(weight),
                    float(confidence),
                    float(reward),
                    ts,
                    json.dumps(audit, sort_keys=True, default=str),
                ),
            )
        return attribution_id

    def has_attribution(self, outcome_id: str, retrieval_id: str | None = None) -> bool:
        if retrieval_id is None:
            row = self.conn.execute("SELECT 1 FROM attributions WHERE outcome_id = ? LIMIT 1", (outcome_id,)).fetchone()
        else:
            row = self.conn.execute(
                "SELECT 1 FROM attributions WHERE outcome_id = ? AND retrieval_id = ? LIMIT 1",
                (outcome_id, retrieval_id),
            ).fetchone()
        return row is not None

    def pending_outcomes(self) -> list[dict[str, Any]]:
        # Per-retrieval idempotence is handled by the attribution layer; an
        # outcome may legitimately already have one attribution while another
        # linked retrieval is still pending.
        rows = self.conn.execute("SELECT * FROM outcomes ORDER BY ts ASC").fetchall()
        return [dict(r) for r in rows]

    def attribution_events_for_memory(self, memory_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM attributions WHERE memory_id = ? ORDER BY ts ASC", (memory_id,)).fetchall()
        return [dict(r) for r in rows]

    def get_retrieval(self, retrieval_id: str) -> dict[str, Any] | None:
        r = self.conn.execute("SELECT * FROM retrievals WHERE retrieval_id = ?", (retrieval_id,)).fetchone()
        return dict(r) if r else None

    def get_outcome(self, outcome_id: str) -> dict[str, Any] | None:
        r = self.conn.execute("SELECT * FROM outcomes WHERE outcome_id = ?", (outcome_id,)).fetchone()
        return dict(r) if r else None

    # -- checkpoints ----------------------------------------------------

    def add_checkpoint(
        self,
        version: str,
        name: str,
        *,
        metrics: dict[str, Any] | None = None,
        artifact_path: str | None = None,
        parent: str | None = None,
        note: str | None = None,
        status: str = CheckpointStatus.CANDIDATE.value,
        gate_approved: bool = False,
    ) -> None:
        valid_statuses = {item.value for item in CheckpointStatus}
        if status not in valid_statuses:
            raise PayloadError(f"invalid checkpoint status {status!r}")
        with self.conn:
            seq, ts = self._append_event("checkpoint.add", {"version": version, "status": status}, "system")
            self.conn.execute(
                """INSERT INTO checkpoints (version, name, status, created_at, metrics, artifact_path, parent, note, gate_approved)
                   VALUES (?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(version) DO UPDATE SET
                     name=excluded.name,
                     status=excluded.status,
                     metrics=excluded.metrics,
                     artifact_path=excluded.artifact_path,
                     parent=excluded.parent,
                     note=excluded.note,
                     gate_approved=excluded.gate_approved""",
                (
                    version,
                    name,
                    status,
                    ts,
                    json.dumps(metrics, sort_keys=True, default=str) if metrics is not None else None,
                    artifact_path,
                    parent,
                    note,
                    int(gate_approved),
                ),
            )

    def _mark_active(self, version: str, event_type: str, note: str | None) -> None:
        row = self.conn.execute("SELECT * FROM checkpoints WHERE version = ?", (version,)).fetchone()
        if row is None:
            raise PayloadError(f"checkpoint {version!r} does not exist")
        self._append_event(event_type, {"version": version}, "system")
        self.conn.execute(
            "UPDATE checkpoints SET status=? WHERE status=?", (CheckpointStatus.RETIRED.value, CheckpointStatus.ACTIVE.value)
        )
        self.conn.execute(
            "UPDATE checkpoints SET status=?, note=? WHERE version=?", (CheckpointStatus.ACTIVE.value, note, version)
        )

    def activate_checkpoint(self, version: str, *, note: str | None = None) -> None:
        with self.conn:
            row = self.conn.execute("SELECT * FROM checkpoints WHERE version = ?", (version,)).fetchone()
            if row is None:
                raise PayloadError(f"checkpoint {version!r} does not exist")
            if row["status"] != CheckpointStatus.ACTIVE.value and not bool(row["gate_approved"]):
                raise PayloadError(f"checkpoint {version!r} has not passed the promotion gate")
            self._mark_active(version, "checkpoint.activate", note)

    def rollback_checkpoint(self, version: str, *, note: str | None = None) -> None:
        with self.conn:
            row = self.conn.execute("SELECT * FROM checkpoints WHERE version = ?", (version,)).fetchone()
            if row is None:
                raise PayloadError(f"checkpoint {version!r} does not exist")
            if row["status"] in {CheckpointStatus.CANDIDATE.value, CheckpointStatus.REJECTED.value}:
                raise PayloadError(f"checkpoint {version!r} was never a trusted active checkpoint")
            self._mark_active(version, "checkpoint.rollback", note)

    def reject_checkpoint(self, version: str, *, note: str | None = None) -> None:
        """Mark a candidate rejected without ever making it active."""
        with self.conn:
            row = self.conn.execute("SELECT 1 FROM checkpoints WHERE version = ?", (version,)).fetchone()
            if row is None:
                raise PayloadError(f"checkpoint {version!r} does not exist")
            self._append_event("checkpoint.reject", {"version": version}, "system")
            self.conn.execute(
                "UPDATE checkpoints SET status=?, note=?, gate_approved=0 WHERE version=?",
                (CheckpointStatus.REJECTED.value, note, version),
            )

    def get_active_checkpoint(self) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM checkpoints WHERE status = ?", (CheckpointStatus.ACTIVE.value,)).fetchone()
        return dict(row) if row else None

    def list_checkpoints(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM checkpoints ORDER BY created_at ASC").fetchall()
        return [dict(r) for r in rows]

    # -- queries --------------------------------------------------------

    def get_memory(self, memory_id: str) -> dict[str, Any] | None:
        return self._get_memory(memory_id)

    def revisions(self, memory_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM memory_revisions WHERE memory_id = ? ORDER BY rev ASC", (memory_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def active_memories(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM memories WHERE status = ? ORDER BY created_at ASC", (MemoryStatus.ACTIVE.value,)
        ).fetchall()
        return [dict(r) for r in rows]

    def all_memories(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM memories ORDER BY created_at ASC").fetchall()
        return [dict(r) for r in rows]

    def links_for(self, memory_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM links WHERE source_id = ? OR target_id = ? ORDER BY created_at ASC",
            (memory_id, memory_id),
        ).fetchall()
        return [dict(r) for r in rows]

    def audit_events(
        self,
        *,
        since_seq: int = 0,
        limit: int = 100,
        event_types: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        q = "SELECT * FROM events WHERE seq > ?"
        args: list[Any] = [since_seq]
        if event_types:
            types = list(event_types)
            q += " AND event_type IN (%s)" % ",".join("?" for _ in types)
            args.extend(types)
        q += " ORDER BY seq ASC LIMIT ?"
        args.append(int(limit))
        rows = self.conn.execute(q, args).fetchall()
        return [dict(r) for r in rows]

    def storage_size_bytes(self) -> int:
        if self.path == ":memory:":
            page_count = self.conn.execute("PRAGMA page_count").fetchone()[0]
            page_size = self.conn.execute("PRAGMA page_size").fetchone()[0]
            return page_count * page_size
        path = Path(self.path)
        # SQLite WAL mode keeps recent pages outside the main file while the
        # connection is open, so report the complete on-disk footprint.
        return sum(
            candidate.stat().st_size
            for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm"))
            if candidate.exists()
        )

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for table in (
            "events",
            "memories",
            "memory_revisions",
            "links",
            "retrievals",
            "outcomes",
            "attributions",
            "checkpoints",
            "policy_decisions",
            "training_examples",
        ):
            out[table] = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return out

    # -- training examples ---------------------------------------------

    def add_training_example(
        self,
        *,
        split: str,
        kind: str,
        stratum: str,
        scenario: str,
        user_id: str,
        t: str,
        features: dict[str, Any],
        label: str,
        chosen: dict[str, Any] | None = None,
        rejected: dict[str, Any] | None = None,
        reward: float | None = None,
        raw: dict[str, Any] | None = None,
    ) -> None:
        with self.conn:
            self.conn.execute(
                """INSERT INTO training_examples
                   (split, kind, stratum, scenario, user_id, t, features, label, chosen, rejected, reward, raw)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    split,
                    kind,
                    stratum,
                    scenario,
                    user_id,
                    t,
                    json.dumps(features, sort_keys=True),
                    label,
                    json.dumps(chosen, sort_keys=True, default=str) if chosen is not None else None,
                    json.dumps(rejected, sort_keys=True, default=str) if rejected is not None else None,
                    reward,
                    json.dumps(raw, sort_keys=True, default=str) if raw is not None else None,
                ),
            )
