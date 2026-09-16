"""Read-only view layer for the Cognition Console.

This module turns the event-sourced :class:`~mpm.store.MemoryStore` into plain,
JSON-serializable dictionaries that the localhost console can render. It is
deliberately a *read* layer only: no method here mutates the store, and no
endpoint can promote, activate, reject, or roll back a checkpoint. Consequential
actions remain explicit, human-gated command-line operations.

Every external input is bounded before it touches a query: search text is
length-capped, pagination values are clamped to sane ranges, status/scope filters
are validated, and audit event-type filters are whitelisted in shape (count and
length) while still flowing through parameterized SQL. Raw database and adapter
paths are never emitted; checkpoint artifact paths are reduced to a boolean.
"""

from __future__ import annotations

import json
import math
from typing import Any, Iterable

from ..safety import sanitize_memory_for_prompt


# -- input bounds -------------------------------------------------------

MAX_SEARCH_CHARS = 200
MAX_SCOPE_CHARS = 64
MAX_ID_CHARS = 128
MAX_TYPE_CHARS = 64
MAX_TYPES = 32
MAX_VERSION_CHARS = 64
MAX_RATIONALE_CHARS = 1_000

DEFAULT_LIST_LIMIT = 100
MAX_LIST_LIMIT = 500
MAX_OFFSET = 100_000

DEFAULT_AUDIT_LIMIT = 100
MAX_AUDIT_LIMIT = 1_000

# -- output bounds ------------------------------------------------------

MAX_PREVIEW_CHARS = 220
MAX_DETAIL_CONTENT_CHARS = 4_000

_ACTIVE_STATUSES = frozenset({"active"})
_RETIRED_STATUSES = frozenset({"tombstoned", "compacted"})
_STATUS_ALIASES: dict[str, frozenset[str] | None] = {
    "active": _ACTIVE_STATUSES,
    "retired": _RETIRED_STATUSES,
    "all": None,
}
_SIGNALS = frozenset({"all", "helpful", "concerning", "untested"})
_OP_ORDER = ("WRITE", "UPDATE", "LINK", "COMPACT", "DELETE", "NOOP")
_OPS = frozenset(_OP_ORDER)


def _clamp_str(value: object, limit: int) -> str:
    text = str(value or "")
    return text[:limit]


def _clamp_int(value: object, lo: int, hi: int, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, parsed))


def _valid_id(value: object) -> bool:
    if not isinstance(value, str):
        return False
    if not value or len(value) > MAX_ID_CHARS:
        return False
    if "/" in value or "\\" in value or "\x00" in value:
        return False
    if any(ord(ch) < 32 for ch in value):
        return False
    return True


def _parse_json(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, (dict, list, int, float, bool)):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


def _injection_flags(content: object) -> list[str]:
    try:
        _, flags = sanitize_memory_for_prompt(str(content or ""), max_chars=10_000_000)
        return flags
    except Exception:  # noqa: BLE001 - the read layer must never raise on bad data
        return []


def _content_view(content: object, max_chars: int) -> dict[str, Any]:
    text = str(content or "")
    length = len(text)
    truncated = length > max_chars
    return {
        "content": text[:max_chars] if truncated else text,
        "length": length,
        "truncated": truncated,
        "injection_flags": _injection_flags(text),
    }


def _clamp_types(raw_types: Iterable[Any] | None) -> list[str]:
    if not raw_types:
        return []
    out: list[str] = []
    for chunk in raw_types:
        for piece in str(chunk).split(","):
            piece = piece.strip()
            if piece and len(piece) <= MAX_TYPE_CHARS:
                out.append(piece)
    seen: list[str] = []
    for piece in out:
        if piece not in seen:
            seen.append(piece)
    return seen[:MAX_TYPES]


class ConsoleView:
    """Read-only projection of a :class:`MemoryStore` for the web console."""

    def __init__(self, store: Any) -> None:
        self.store = store

    def status(self) -> dict[str, Any]:
        counts = self.store.counts()
        active = self.store.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE status = 'active'"
        ).fetchone()[0]
        retired = self.store.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE status != 'active'"
        ).fetchone()[0]
        active_credit = self.store.conn.execute(
            """SELECT m.memory_id,
                      COUNT(a.attribution_id) AS n_attributions,
                      COALESCE(SUM(a.reward), 0) AS total_reward
               FROM memories m
               LEFT JOIN attributions a ON a.memory_id = m.memory_id
               WHERE m.status = 'active'
               GROUP BY m.memory_id"""
        ).fetchall()
        helpful = sum(1 for row in active_credit if float(row["total_reward"]) > 0)
        concerning = sum(1 for row in active_credit if float(row["total_reward"]) < 0)
        untested = sum(1 for row in active_credit if int(row["n_attributions"]) == 0)
        mixed = sum(
            1
            for row in active_credit
            if int(row["n_attributions"]) > 0 and float(row["total_reward"]) == 0
        )
        active_checkpoint = self.store.get_active_checkpoint()
        decision_ops = {op: 0 for op in _OP_ORDER}
        for row in self.store.conn.execute(
            "SELECT op, COUNT(*) AS n FROM policy_decisions GROUP BY op"
        ).fetchall():
            decision_ops[str(row["op"])] = int(row["n"])
        last_event_ts = self.store.conn.execute("SELECT MAX(ts) FROM events").fetchone()[0]
        return {
            "ok": True,
            "counts": counts,
            "active_memories": int(active),
            "retired_memories": int(retired),
            "active_helpful_memories": helpful,
            "active_concerning_memories": concerning,
            "active_untested_memories": untested,
            "active_mixed_memories": mixed,
            "decision_ops": decision_ops,
            "decisions_total": sum(decision_ops.values()),
            "last_event_ts": float(last_event_ts) if last_event_ts is not None else None,
            "storage_size_bytes": int(self.store.storage_size_bytes()),
            "active_checkpoint": self._checkpoint_view(active_checkpoint) if active_checkpoint else None,
            "read_only": True,
            "live_training": False,
        }

    def list_memories(
        self,
        q: object = None,
        status: object = "all",
        scope: object = None,
        signal: object = "all",
        limit: object = None,
        offset: object = None,
    ) -> dict[str, Any]:
        q = _clamp_str(q, MAX_SEARCH_CHARS)
        status_key = str(status or "all")
        if status_key not in _STATUS_ALIASES:
            status_key = "all"
        scope = _clamp_str(scope, MAX_SCOPE_CHARS) or None
        signal_key = str(signal or "all")
        if signal_key not in _SIGNALS:
            signal_key = "all"
        limit = _clamp_int(limit, 1, MAX_LIST_LIMIT, DEFAULT_LIST_LIMIT)
        offset = _clamp_int(offset, 0, MAX_OFFSET, 0)

        where: list[str] = []
        args: list[Any] = []
        statuses = _STATUS_ALIASES[status_key]
        if statuses is not None:
            where.append("status IN (%s)" % ",".join("?" for _ in statuses))
            args.extend(sorted(statuses))
        if scope:
            where.append("scope = ?")
            args.append(scope)
        if q:
            needle = q.lower()
            where.append(
                "(instr(lower(content), ?) > 0 "
                "OR instr(lower(COALESCE(scope, '')), ?) > 0 "
                "OR instr(lower(COALESCE(key, '')), ?) > 0)"
            )
            args.extend([needle, needle, needle])
        if signal_key == "helpful":
            where.append(
                "(SELECT COALESCE(SUM(a.reward), 0) FROM attributions a "
                "WHERE a.memory_id = memories.memory_id) > 0"
            )
        elif signal_key == "concerning":
            where.append(
                "(SELECT COALESCE(SUM(a.reward), 0) FROM attributions a "
                "WHERE a.memory_id = memories.memory_id) < 0"
            )
        elif signal_key == "untested":
            where.append(
                "NOT EXISTS (SELECT 1 FROM attributions a "
                "WHERE a.memory_id = memories.memory_id)"
            )
        where_sql = (" WHERE " + " AND ".join(where)) if where else ""

        total = self.store.conn.execute(
            f"SELECT COUNT(*) FROM memories{where_sql}", args
        ).fetchone()[0]
        rows = self.store.conn.execute(
            f"SELECT * FROM memories{where_sql} ORDER BY created_at ASC LIMIT ? OFFSET ?",
            args + [limit, offset],
        ).fetchall()
        items = [self._memory_summary(dict(r)) for r in rows]
        return {
            "items": items,
            "total": int(total),
            "limit": limit,
            "offset": offset,
            "filters": {"q": q, "status": status_key, "scope": scope, "signal": signal_key},
        }

    def _memory_stats(self, memory_id: str) -> tuple[int, int, int, float]:
        n_retrievals = self.store.conn.execute(
            "SELECT COUNT(*) FROM retrievals WHERE memory_id = ?", (memory_id,)
        ).fetchone()[0]
        n_links = self.store.conn.execute(
            "SELECT COUNT(*) FROM links WHERE source_id = ? OR target_id = ?",
            (memory_id, memory_id),
        ).fetchone()[0]
        n_attributions, total_reward = self.store.conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(reward), 0) FROM attributions WHERE memory_id = ?",
            (memory_id,),
        ).fetchone()
        return int(n_retrievals), int(n_links), int(n_attributions), float(total_reward)

    def _memory_summary(self, mem: dict[str, Any]) -> dict[str, Any]:
        n_retrievals, n_links, n_attributions, total_reward = self._memory_stats(mem["memory_id"])
        return {
            "memory_id": mem["memory_id"],
            "status": mem["status"],
            "scope": mem.get("scope"),
            "key": mem.get("key"),
            "created_at": mem["created_at"],
            "updated_at": mem["updated_at"],
            "current_revision": mem["current_revision"],
            "preview": _content_view(mem["content"], MAX_PREVIEW_CHARS),
            "n_retrievals": n_retrievals,
            "n_links": n_links,
            "n_attributions": n_attributions,
            "total_reward": round(total_reward, 6),
        }

    def memory_detail(self, memory_id: object) -> dict[str, Any] | None:
        if not _valid_id(memory_id):
            return None
        mem = self.store.get_memory(str(memory_id))
        if mem is None:
            return None
        _, _, _, total_reward = self._memory_stats(mem["memory_id"])

        revisions = []
        for r in self.store.revisions(mem["memory_id"]):
            revisions.append(
                {
                    "rev": r["rev"],
                    "op": r["op"],
                    "ts": r["ts"],
                    "event_seq": r["event_seq"],
                    "content": _content_view(r["content"], MAX_DETAIL_CONTENT_CHARS),
                }
            )

        links = []
        for link in self.store.links_for(mem["memory_id"]):
            links.append(
                {
                    "link_id": link["link_id"],
                    "source_id": link["source_id"],
                    "target_id": link["target_id"],
                    "kind": link["kind"],
                    "weight": link["weight"],
                    "status": link["status"],
                    "created_at": link["created_at"],
                    "direction": "outbound" if link["source_id"] == mem["memory_id"] else "inbound",
                }
            )

        retrieval_rows = self.store.conn.execute(
            "SELECT * FROM retrievals WHERE memory_id = ? ORDER BY ts ASC",
            (mem["memory_id"],),
        ).fetchall()
        retrievals = [
            {
                "retrieval_id": r["retrieval_id"],
                "session_id": r["session_id"],
                "rank": r["rank"],
                "score": r["score"],
                "ts": r["ts"],
            }
            for r in retrieval_rows
        ]

        outcome_rows = self.store.conn.execute(
            """SELECT DISTINCT o.*, oretr.contribution
               FROM outcomes o
               JOIN outcome_retrievals oretr ON oretr.outcome_id = o.outcome_id
               JOIN retrievals r ON r.retrieval_id = oretr.retrieval_id
               WHERE r.memory_id = ?
               ORDER BY o.ts ASC""",
            (mem["memory_id"],),
        ).fetchall()
        outcomes = [
            {
                "outcome_id": o["outcome_id"],
                "retrieval_id": o["retrieval_id"],
                "contribution": o["contribution"],
                "kind": o["kind"],
                "value": o["value"],
                "confidence": o["confidence"],
                "ts": o["ts"],
            }
            for o in outcome_rows
        ]

        attributions = []
        for a in self.store.attribution_events_for_memory(mem["memory_id"]):
            attributions.append(
                {
                    "attribution_id": a["attribution_id"],
                    "outcome_id": a["outcome_id"],
                    "retrieval_id": a["retrieval_id"],
                    "weight": a["weight"],
                    "confidence": a["confidence"],
                    "reward": a["reward"],
                    "ts": a["ts"],
                    "audit": _parse_json(a["audit"]),
                }
            )

        return {
            "memory_id": mem["memory_id"],
            "status": mem["status"],
            "scope": mem.get("scope"),
            "key": mem.get("key"),
            "created_at": mem["created_at"],
            "updated_at": mem["updated_at"],
            "current_revision": mem["current_revision"],
            "total_reward": round(total_reward, 6),
            "created_event": mem["created_event"],
            "content": _content_view(mem["content"], MAX_DETAIL_CONTENT_CHARS),
            "revisions": revisions,
            "links": links,
            "retrievals": retrievals,
            "outcomes": outcomes,
            "attributions": attributions,
        }

    def audit(
        self,
        since: object = 0,
        limit: object = None,
        types: Iterable[Any] | None = None,
        newest: bool = False,
    ) -> dict[str, Any]:
        """Page through the event log oldest-first, or take the newest events.

        ``newest`` ignores ``since`` and returns the most recent ``limit``
        events in reverse order; the console's home page uses it for the
        recent-activity ledger.
        """
        since = _clamp_int(since, 0, 2**63 - 1, 0)
        limit = _clamp_int(limit, 1, MAX_AUDIT_LIMIT, DEFAULT_AUDIT_LIMIT)
        types = _clamp_types(types)
        if newest:
            events = self._newest_events(limit, types)
        else:
            events = self.store.audit_events(since_seq=since, limit=limit, event_types=types or None)
        return {
            "items": [self._event_view(e) for e in events],
            "since": since,
            "limit": limit,
            "types": types,
            "newest": bool(newest),
        }

    def _newest_events(self, limit: int, types: list[str]) -> list[dict[str, Any]]:
        q = "SELECT * FROM events"
        args: list[Any] = []
        if types:
            q += " WHERE event_type IN (%s)" % ",".join("?" for _ in types)
            args.extend(types)
        q += " ORDER BY seq DESC LIMIT ?"
        args.append(limit)
        return [dict(r) for r in self.store.conn.execute(q, args).fetchall()]

    def decisions(
        self,
        op: object = None,
        version: object = None,
        target: object = None,
        limit: object = None,
        offset: object = None,
    ) -> dict[str, Any]:
        """List recorded policy decisions, newest first."""
        op_key = str(op or "").upper()
        if op_key not in _OPS:
            op_key = ""
        version = _clamp_str(version, MAX_VERSION_CHARS) or None
        target = str(target) if _valid_id(target) else None
        limit = _clamp_int(limit, 1, MAX_LIST_LIMIT, DEFAULT_LIST_LIMIT)
        offset = _clamp_int(offset, 0, MAX_OFFSET, 0)

        where: list[str] = []
        args: list[Any] = []
        if op_key:
            where.append("op = ?")
            args.append(op_key)
        if version:
            where.append("policy_version = ?")
            args.append(version)
        if target:
            where.append("target = ?")
            args.append(target)
        where_sql = (" WHERE " + " AND ".join(where)) if where else ""

        total = self.store.conn.execute(
            f"SELECT COUNT(*) FROM policy_decisions{where_sql}", args
        ).fetchone()[0]
        rows = self.store.conn.execute(
            f"SELECT * FROM policy_decisions{where_sql} ORDER BY event_seq DESC LIMIT ? OFFSET ?",
            args + [limit, offset],
        ).fetchall()
        return {
            "items": [self._decision_view(dict(r)) for r in rows],
            "total": int(total),
            "limit": limit,
            "offset": offset,
            "filters": {"op": op_key, "version": version, "target": target},
        }

    def _decision_view(self, d: dict[str, Any]) -> dict[str, Any]:
        payload = _parse_json(d.get("payload"))
        preview = None
        if isinstance(payload, dict) and isinstance(payload.get("content"), str):
            preview = _content_view(payload["content"], MAX_PREVIEW_CHARS)
            payload = dict(payload)
            payload["content"] = _content_view(payload["content"], MAX_DETAIL_CONTENT_CHARS)["content"]
        return {
            "decision_id": d["decision_id"],
            "policy_version": d.get("policy_version"),
            "op": d.get("op"),
            "target": d.get("target"),
            "confidence": d.get("confidence"),
            "rationale": _clamp_str(d.get("rationale"), MAX_RATIONALE_CHARS),
            "ts": d.get("ts"),
            "event_seq": d.get("event_seq"),
            "session_id": d.get("session_id"),
            "payload": payload,
            "features": _parse_json(d.get("features")),
            "preview": preview,
        }

    def _event_view(self, event: dict[str, Any]) -> dict[str, Any]:
        return {
            "seq": event["seq"],
            "ts": event["ts"],
            "txn_id": event["txn_id"],
            "event_type": event["event_type"],
            "actor": event["actor"],
            "payload": _parse_json(event["payload"]),
        }

    def checkpoints(self) -> dict[str, Any]:
        items = [self._checkpoint_view(cp) for cp in self.store.list_checkpoints()]
        active = self.store.get_active_checkpoint()
        return {
            "items": items,
            "active": self._checkpoint_view(active) if active else None,
        }

    def _checkpoint_view(self, cp: dict[str, Any]) -> dict[str, Any]:
        return {
            "version": cp["version"],
            "name": cp["name"],
            "status": cp["status"],
            "created_at": cp["created_at"],
            "parent": cp.get("parent"),
            "note": cp.get("note"),
            "gate_approved": bool(cp.get("gate_approved")),
            "has_artifact": bool(cp.get("artifact_path")),
            "metrics": _parse_json(cp["metrics"]),
        }


def sanitize_payload(value: Any) -> Any:
    """Recursively coerce a value to valid-JSON primitives (no NaN/Infinity)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): sanitize_payload(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_payload(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)
