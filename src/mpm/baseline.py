"""Deterministic heuristic baseline policy.

The baseline emits exactly the same structured :class:`~mpm.types.Action`
schema the future Liquid model will emit.  It is fully deterministic given the
same observation + store state, so it can serve as a reproducible reference for
evaluation, credit assignment, and promotion gates.
"""

from __future__ import annotations

from typing import Any, Protocol

from .features import extract_features, tokenize
from .safety import is_harmful_content
from .store import MemoryStore
from .types import Action, Op


class Policy(Protocol):
    name: str

    def decide(self, observation: dict[str, Any], store: MemoryStore) -> Action: ...


def jaccard(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


_DUP_THRESHOLD = 0.9
_UPDATE_LOW = 0.45


class BaselinePolicy:
    """Heuristic policy: dedup via token overlap, refuse PII/secrets, honor intents."""

    name = "baseline"

    def decide(self, observation: dict[str, Any], store: MemoryStore) -> Action:
        content = str(observation.get("content") or "")
        scope = str(observation.get("scope") or "default")
        intent = observation.get("intent")
        context_ids = observation.get("context_ids") or []
        features = extract_features(content, scope=scope, extra=observation.get("extra_features"))

        harmful, reason = is_harmful_content(content)
        if harmful:
            return Action(
                op=Op.NOOP.value,
                payload={"reason": f"refuse-harmful:{reason}", "observed": {"harmful": True}},
                confidence=0.99,
                rationale="refuse to persist PII/secret",
            )

        if intent == "link" and len(context_ids) >= 2:
            return Action(
                op=Op.LINK.value,
                target=context_ids[0],
                payload={"source_id": context_ids[0], "target_id": context_ids[1], "kind": "related", "weight": 0.8},
                confidence=0.85,
                rationale="explicit link intent",
            )
        if intent == "delete" and context_ids:
            return Action(
                op=Op.DELETE.value,
                target=context_ids[0],
                payload={"memory_id": context_ids[0]},
                confidence=0.9,
                rationale="explicit delete intent",
            )
        if intent == "compact" and len(context_ids) >= 2:
            return Action(
                op=Op.COMPACT.value,
                payload={"memory_ids": list(context_ids), "strategy": "concat"},
                confidence=0.8,
                rationale="explicit compact intent",
            )

        if not content.strip():
            return Action(op=Op.NOOP.value, payload={"reason": "empty-content"}, confidence=1.0, rationale="nothing to store")

        incoming_tokens = tokenize(content)
        best_sim = 0.0
        best_id: str | None = None
        for mem in store.active_memories():
            if mem["scope"] != scope:
                continue
            sim = jaccard(incoming_tokens, tokenize(mem["content"]))
            if sim > best_sim:
                best_sim = sim
                best_id = mem["memory_id"]

        if best_id is not None and best_sim >= _DUP_THRESHOLD:
            return Action(
                op=Op.NOOP.value,
                payload={"reason": "duplicate", "similar_to": best_id, "similarity": round(best_sim, 4)},
                confidence=round(best_sim, 4),
                rationale="near-duplicate memory already exists",
            )
        if best_id is not None and best_sim >= _UPDATE_LOW:
            return Action(
                op=Op.UPDATE.value,
                target=best_id,
                payload={"memory_id": best_id, "content_ref": "observation.content"},
                confidence=round(0.5 + best_sim / 2, 4),
                rationale="merge into existing related memory",
            )
        return Action(
            op=Op.WRITE.value,
            payload={"content_ref": "observation.content"},
            confidence=0.7,
            rationale="new fact worth remembering",
        )
