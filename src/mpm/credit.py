"""Delayed credit assignment.

Connects a past memory operation to later retrievals and outcomes using an
explicit, auditable reward calculation.  No retrieved memory is ever given full
credit by default: the reward is discounted by the retrieval's contribution
score and confidence, and by a *gentle* time-decay term so that long delays
(the motivating step-40 -> step-8000 case) still receive meaningful credit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .store import MemoryStore


@dataclass
class CreditConfig:
    # Rational decay: 1 / (1 + decay_rate * delta).  A small rate keeps most of
    # the reward even after thousands of steps, but still expresses recency.
    decay_rate: float = 1e-5
    # Any contribution share below this is floored to zero.
    min_weight: float = 0.0


def time_decay(delta: float, rate: float) -> float:
    if delta < 0:
        delta = 0.0
    return 1.0 / (1.0 + rate * delta)


def _row(store: MemoryStore, sql: str, args: tuple[Any, ...]) -> dict[str, Any] | None:
    r = store.conn.execute(sql, args).fetchone()
    return dict(r) if r else None


def compute_attributions(
    store: MemoryStore,
    outcome_id: str,
    cfg: CreditConfig | None = None,
) -> list[dict[str, Any]]:
    """Compute every retrieval attribution for one outcome without writing.

    Explicit outcome contribution weights are normalized across retrieved
    memories, then discounted by retrieval relevance, rank, confidence, and
    time. This prevents every retrieved memory from receiving full credit.
    """
    cfg = cfg or CreditConfig()
    outcome = _row(store, "SELECT * FROM outcomes WHERE outcome_id = ?", (outcome_id,))
    if outcome is None:
        return []
    links = store.conn.execute(
        "SELECT retrieval_id, contribution FROM outcome_retrievals WHERE outcome_id = ? ORDER BY retrieval_id",
        (outcome_id,),
    ).fetchall()
    if not links and outcome.get("retrieval_id"):
        links = [{"retrieval_id": outcome["retrieval_id"], "contribution": 1.0}]
    total_contribution = sum(float(link["contribution"]) for link in links)
    if total_contribution <= 0:
        return []

    results: list[dict[str, Any]] = []
    for link in links:
        retrieval = _row(store, "SELECT * FROM retrievals WHERE retrieval_id = ?", (link["retrieval_id"],))
        if retrieval is None:
            continue
        memory = _row(store, "SELECT * FROM memories WHERE memory_id = ?", (retrieval["memory_id"],))
        if memory is None:
            continue
        origin = _row(store, "SELECT * FROM events WHERE seq = ?", (memory["created_event"],))
        operation_ts = origin["ts"] if origin else memory["created_at"]
        delta = max(0.0, retrieval["ts"] - operation_ts)
        decay = time_decay(delta, cfg.decay_rate)
        share = float(link["contribution"]) / total_contribution
        score = max(0.0, min(1.0, float(retrieval["score"])))
        rank = max(0, int(retrieval["rank"]))
        rank_discount = 1.0 / (1.0 + rank)
        weight = max(cfg.min_weight, min(1.0, share * score * rank_discount))
        confidence = float(outcome["confidence"])
        reward = float(outcome["value"]) * weight * confidence * decay
        audit = {
            "formula": "reward = outcome.value * normalized_contribution * retrieval_score * rank_discount * confidence * decay",
            "outcome_value": float(outcome["value"]),
            "outcome_kind": outcome["kind"],
            "raw_contribution": float(link["contribution"]),
            "normalized_contribution": round(share, 6),
            "retrieval_score": score,
            "retrieval_rank": rank,
            "rank_discount": round(rank_discount, 6),
            "weight": round(weight, 6),
            "confidence": confidence,
            "delta": round(delta, 6),
            "decay_rate": cfg.decay_rate,
            "decay": round(decay, 6),
            "reward": round(reward, 6),
            "operation_event_seq": memory["created_event"],
            "operation_event_type": origin["event_type"] if origin else None,
        }
        results.append({
            "memory_id": retrieval["memory_id"],
            "operation_event_seq": memory["created_event"],
            "retrieval_id": retrieval["retrieval_id"],
            "outcome_id": outcome_id,
            "weight": weight,
            "confidence": confidence,
            "reward": reward,
            "audit": audit,
        })
    return results


def compute_attribution(
    store: MemoryStore, outcome_id: str, cfg: CreditConfig | None = None
) -> dict[str, Any] | None:
    """Backward-compatible helper returning the first attribution, if any."""
    results = compute_attributions(store, outcome_id, cfg)
    return results[0] if results else None


def attribute_outcome(store: MemoryStore, outcome_id: str, cfg: CreditConfig | None = None) -> list[dict[str, Any]]:
    """Compute and persist all missing attributions for one outcome."""
    written: list[dict[str, Any]] = []
    for att in compute_attributions(store, outcome_id, cfg):
        if store.has_attribution(outcome_id, att["retrieval_id"]):
            continue
        store.add_attribution(
            memory_id=att["memory_id"],
            operation_event_seq=att["operation_event_seq"],
            retrieval_id=att["retrieval_id"],
            outcome_id=att["outcome_id"],
            weight=att["weight"],
            confidence=att["confidence"],
            reward=att["reward"],
            audit=att["audit"],
        )
        written.append(att)
    return written


def reconcile_all(store: MemoryStore, cfg: CreditConfig | None = None) -> list[dict[str, Any]]:
    """Attribute every currently-unattributed outcome; returns new attributions."""
    done: list[dict[str, Any]] = []
    for outcome in store.pending_outcomes():
        done.extend(attribute_outcome(store, outcome["outcome_id"], cfg))
    return done


def memory_credit_summary(store: MemoryStore) -> list[dict[str, Any]]:
    """Total credited reward per memory, plus the number of attributions."""
    rows = store.conn.execute(
        """SELECT memory_id, COUNT(*) AS n, SUM(reward) AS total_reward
           FROM attributions GROUP BY memory_id ORDER BY total_reward ASC"""
    ).fetchall()
    return [
        {"memory_id": r["memory_id"], "n": r["n"], "total_reward": r["total_reward"]}
        for r in rows
    ]


def audit_json(attribution: dict[str, Any]) -> str:
    return json.dumps(attribution.get("audit", {}), sort_keys=True)
