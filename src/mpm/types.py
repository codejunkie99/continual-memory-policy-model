"""Core types, operation enums, and strict payload validation.

Everything here uses only the standard library so the runtime and tests stay
dependency-free.  The ``Action`` dataclass is the single structured schema that
both the heuristic baseline and the future Liquid model must emit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Op(str, Enum):
    WRITE = "WRITE"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    LINK = "LINK"
    COMPACT = "COMPACT"
    NOOP = "NOOP"


ALL_OPS: frozenset[str] = frozenset(o.value for o in Op)


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    TOMBSTONED = "tombstoned"
    COMPACTED = "compacted"


class OutcomeKind(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class CheckpointStatus(str, Enum):
    ACTIVE = "active"
    CANDIDATE = "candidate"
    REJECTED = "rejected"
    RETIRED = "retired"
    ROLLED_BACK = "rolled_back"


@dataclass
class Action:
    """Structured decision emitted by a policy (baseline or learned model)."""

    op: str
    target: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    rationale: str = ""
    policy_version: str = "baseline"

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "target": self.target,
            "payload": self.payload,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "policy_version": self.policy_version,
        }


class PayloadError(ValueError):
    """Raised when an operation payload fails strict validation."""


def _is_nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_id(value: Any) -> bool:
    return _is_nonempty_str(value)


def validate_payload(op: str, payload: dict[str, Any]) -> list[str]:
    """Return a list of validation errors for ``op`` + ``payload`` (empty = valid)."""
    errors: list[str] = []
    if op not in ALL_OPS:
        errors.append(f"unknown op {op!r}; expected one of {sorted(ALL_OPS)}")
        return errors
    if not isinstance(payload, dict):
        return ["payload must be a JSON object"]

    if op == Op.WRITE.value:
        has_content = _is_nonempty_str(payload.get("content"))
        has_ref = payload.get("content_ref") == "observation.content"
        if not (has_content or has_ref):
            errors.append("WRITE requires non-empty 'content' or content_ref='observation.content'")
        scope = payload.get("scope")
        if scope is not None and not _is_nonempty_str(scope):
            errors.append("WRITE 'scope' must be a non-empty string when provided")
        key = payload.get("key")
        if key is not None and not _is_nonempty_str(key):
            errors.append("WRITE 'key' must be a non-empty string when provided")

    elif op == Op.UPDATE.value:
        if not _is_id(payload.get("memory_id")):
            errors.append("UPDATE requires a non-empty string 'memory_id'")
        has_content = "content" in payload and isinstance(payload["content"], str)
        has_ref = payload.get("content_ref") == "observation.content"
        if not (has_content or has_ref):
            errors.append("UPDATE requires string 'content' or content_ref='observation.content'")

    elif op == Op.DELETE.value:
        if not _is_id(payload.get("memory_id")):
            errors.append("DELETE requires a non-empty string 'memory_id'")

    elif op == Op.LINK.value:
        for field in ("source_id", "target_id"):
            if not _is_id(payload.get(field)):
                errors.append(f"LINK requires a non-empty string '{field}'")
        if payload.get("source_id") == payload.get("target_id") and payload.get("source_id"):
            errors.append("LINK source_id and target_id must differ")
        if not _is_nonempty_str(payload.get("kind")):
            errors.append("LINK requires a non-empty string 'kind'")
        weight = payload.get("weight")
        if not isinstance(weight, (int, float)) or isinstance(weight, bool) or not (0.0 <= float(weight) <= 1.0):
            errors.append("LINK 'weight' must be a number in [0, 1]")

    elif op == Op.COMPACT.value:
        ids = payload.get("memory_ids")
        if not isinstance(ids, list) or len(ids) < 2 or not all(_is_id(i) for i in ids):
            errors.append("COMPACT requires 'memory_ids' as a list of >=2 non-empty strings")
        if len(ids) != len(set(ids)):
            errors.append("COMPACT 'memory_ids' must not contain duplicates")
        if not _is_nonempty_str(payload.get("strategy")):
            errors.append("COMPACT requires a non-empty string 'strategy'")

    elif op == Op.NOOP.value:
        observed = payload.get("observed")
        if observed is not None and not isinstance(observed, dict):
            errors.append("NOOP 'observed' must be a JSON object when provided")

    return errors


def validate_action(action: Action) -> list[str]:
    """Validate a full structured action."""
    errors = validate_payload(action.op, action.payload)
    if not isinstance(action.confidence, (int, float)) or isinstance(action.confidence, bool):
        errors.append("action confidence must be numeric")
    elif not (0.0 <= float(action.confidence) <= 1.0):
        errors.append("action confidence must be in [0, 1]")
    if action.target is not None and not isinstance(action.target, str):
        errors.append("action target must be a string or null")
    return errors


def action_from_dict(d: dict[str, Any]) -> Action:
    """Coerce a JSON-like dict into an :class:`Action` (used for model outputs)."""
    return Action(
        op=str(d.get("op", "")),
        target=d.get("target"),
        payload=d.get("payload") if isinstance(d.get("payload"), dict) else {},
        confidence=float(d.get("confidence", 1.0)),
        rationale=str(d.get("rationale", "")),
        policy_version=str(d.get("policy_version", "learned")),
    )
