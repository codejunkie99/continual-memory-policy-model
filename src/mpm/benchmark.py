"""Synthetic environment / benchmark.

Exercises every operation and the motivating long-delay credit-assignment case:
a WRITE at step 40 is retrieved and positively rewarded at step 8000, while a
second retrieved memory receives negative credit.  Produces reproducible
baseline metrics with an explicit seed.
"""

from __future__ import annotations

import time
from typing import Any

from .baseline import BaselinePolicy, Policy
from .credit import CreditConfig, reconcile_all, memory_credit_summary
from .store import MemoryStore
from .types import Action, Op


class FakeClock:
    """A monotonic integer clock so 'step N' has a stable, inspectable meaning."""

    def __init__(self, start: float = 0.0):
        self.t = float(start)

    def __call__(self) -> float:
        return self.t

    def set(self, t: float) -> None:
        self.t = float(t)

    def advance(self, n: float = 1.0) -> None:
        self.t += n


def _exercise_operations(store: MemoryStore) -> dict[str, Any]:
    """Directly drive every operation so coverage is guaranteed and deterministic."""
    store.ensure_session("s0", "all-operations", "synthetic-user-0")
    a = store.write_memory("the meeting is on Tuesday at 3pm", scope="work", session_id="s0")
    b = store.write_memory("the budget for Q3 is $120k", scope="work", session_id="s0")
    store.update_memory(a, "the meeting moved to Wednesday at 3pm", session_id="s0")
    store.link_memories(a, b, "related", 0.7, session_id="s0")
    c = store.write_memory("Q3 revenue target is $1.2M", scope="work", session_id="s0")
    store.write_memory("Q3 hiring plan is 4 engineers", scope="work", session_id="s0")
    # COMPACT the two Q3 planning memories into one.
    q3_ids = [m["memory_id"] for m in store.active_memories() if "Q3" in m["content"]][:2]
    compacted = store.compact(q3_ids, "concat", session_id="s0")
    # Soft DELETE one memory.
    to_delete = store.write_memory("obsolete note about the old office", scope="work", session_id="s0")
    store.delete_memory(to_delete, session_id="s0")
    # NOOP: refuse a harmful write (the baseline does this; here we exercise the op directly).
    store.record_noop(reason="refuse-harmful:email", session_id="s0")
    return {"compacted_result": compacted}


def _seed_training_cohorts(store: MemoryStore, clock: FakeClock, count: int = 60) -> None:
    """Add diverse, synthetic policy labels across temporal/user cohorts."""
    templates = [
        lambda i: Action(Op.WRITE.value, payload={"content": f"synthetic fact {i}", "scope": "synthetic"}),
        lambda i: Action(Op.UPDATE.value, target=f"m{i}", payload={"memory_id": f"m{i}", "content": f"synthetic update {i}"}),
        lambda i: Action(Op.DELETE.value, target=f"m{i}", payload={"memory_id": f"m{i}"}),
        lambda i: Action(Op.LINK.value, payload={"source_id": f"m{i}", "target_id": f"m{i+1}", "kind": "related", "weight": 0.5}),
        lambda i: Action(Op.COMPACT.value, payload={"memory_ids": [f"m{i}", f"m{i+1}"], "strategy": "summary"}),
        lambda i: Action(Op.NOOP.value, payload={"reason": "duplicate"}),
    ]
    for i in range(count):
        clock.set(9_000.0 + i * 86_400.0)
        session_id = f"train-{i}"
        store.ensure_session(session_id, f"scenario-{i}", f"synthetic-user-{i % 13}")
        store.record_policy_decision(
            action := templates[i % len(templates)](i),
            session_id=session_id,
            features={
                "synthetic": True,
                "case": i % len(templates),
                "candidate_ids": [
                    memory_id
                    for memory_id in (
                        action.payload.get("memory_id"),
                        action.payload.get("source_id"),
                        action.payload.get("target_id"),
                        *(action.payload.get("memory_ids") or []),
                    )
                    if isinstance(memory_id, str)
                ],
            },
        )


def _long_delay_credit(store: MemoryStore, clock: FakeClock, seed: int = 42) -> dict[str, Any]:
    cfg = CreditConfig(decay_rate=1e-5)

    # Step 40: write the memory that will be rewarded 7960 steps later.
    clock.set(40.0)
    store.ensure_session("s-long", "synth", "u0")
    a = store.write_memory("the launch date is April 12th", scope="product", session_id="s-long")
    # Step 41: a second memory that turns out to be misleading (will earn negative credit).
    clock.set(41.0)
    b = store.write_memory("the launch date is March 1st (stale)", scope="product", session_id="s-long")

    # Step 8000: both memories are retrieved; A helps, B misleads.
    clock.set(8000.0)
    ra = store.record_retrieval("s-long", a, {"q": "launch"}, rank=0, score=0.9)
    rb = store.record_retrieval("s-long", b, {"q": "launch"}, rank=1, score=0.5)
    oa = store.record_outcome("s-long", retrieval_id=ra, kind="positive", value=1.0, confidence=1.0)
    ob = store.record_outcome("s-long", retrieval_id=rb, kind="negative", value=-1.0, confidence=0.9)

    reconcile_all(store, cfg)

    summary = {m["memory_id"]: m["total_reward"] for m in memory_credit_summary(store)}
    return {
        "memory_a": a,
        "memory_b": b,
        "reward_a": summary.get(a, 0.0),
        "reward_b": summary.get(b, 0.0),
        "positive_outcome": oa,
        "negative_outcome": ob,
        "attribution_count": store.counts()["attributions"],
    }


def run_benchmark(
    *,
    seed: int = 42,
    store: MemoryStore | None = None,
    policy: Policy | None = None,
    clock: FakeClock | None = None,
) -> dict[str, Any]:
    """Run the full benchmark and return a metrics dict (reproducible via seed)."""
    policy = policy or BaselinePolicy()
    clock = clock or FakeClock()
    own_store = store is None
    store = store or MemoryStore(":memory:", clock=clock)

    t0 = time.perf_counter()
    exercised = _exercise_operations(store)
    long_delay = _long_delay_credit(store, clock, seed)
    _seed_training_cohorts(store, clock)
    latency_s = time.perf_counter() - t0

    # Baseline self-check on a few canonical observations (deterministic).
    from .features import extract_features

    checks = [
        ("new-fact", {"content": "the vendor invoice is due on Friday", "scope": "work"}, Op.WRITE.value),
        ("duplicate", {"content": "the meeting is on Tuesday at 3pm", "scope": "work"}, Op.NOOP.value),
        ("harmful", {"content": "contact me at bob@example.com", "scope": "work"}, Op.NOOP.value),
    ]
    decisions = []
    for name, obs, _ in checks:
        action = policy.decide(obs, store)
        decisions.append({"name": name, "op": action.op, "features": extract_features(str(obs.get("content", "")))})

    metrics = {
        "seed": seed,
        "policy": policy.name,
        "counts": store.counts(),
        "active_memories": len(store.active_memories()),
        "total_memories": len(store.all_memories()),
        "links": len(store.conn.execute("SELECT 1 FROM links").fetchall()),
        "storage_bytes": store.storage_size_bytes(),
        "latency_s": round(latency_s, 6),
        "long_delay": long_delay,
        "canonical_decisions": decisions,
        "exercised": exercised,
    }
    if own_store:
        store.close()
    return metrics
