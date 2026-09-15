"""Privacy-safe dataset export and deterministic split builder.

Two hard rules are enforced here:

1. The default export carries *only* redacted features, operation labels, reward
   metadata, and redacted/synthetic context.  Raw user memory and interaction
   text are emitted only when ``allow_raw=True`` is explicitly passed.
2. Splits are assigned at the (scenario, user) group level so that an operation
   and its *later* downstream outcome can never land in different splits.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .store import MemoryStore
from .types import Op


DEFAULT_RATIOS: list[tuple[str, float]] = [("train", 0.7), ("val", 0.15), ("test", 0.15)]


_NOOP_REASONS = {
    "refuse-harmful": "refuse-harmful",
    "empty-content": "empty-content",
    "duplicate": "duplicate",
}

_SAFE_FEATURE_KEYS = {
    "n_tokens", "n_chars", "len_bucket", "token_len_bucket", "has_digit",
    "digit_density", "has_email", "has_url", "has_credit_card", "has_scope",
    "has_key", "synthetic", "case", "modality", "candidate_count",
    "requested_op", "context_count", "has_content", "exact_duplicate",
    "source_trust",
}


def _redact_reason(reason: str) -> str:
    for prefix, code in _NOOP_REASONS.items():
        if reason.startswith(prefix):
            return code
    return "noop"


def redact_payload(op: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Strip all raw content from an operation payload, keeping only structural
    (non-user-content) fields."""
    if op == Op.WRITE.value:
        return {
            "content_ref": "observation.content",
            "has_scope": bool(payload.get("scope")),
            "has_key": bool(payload.get("key")),
        }
    if op == Op.UPDATE.value:
        return {"memory_id": payload.get("memory_id"), "content_ref": "observation.content"}
    if op == Op.DELETE.value:
        return {"memory_id": payload.get("memory_id")}
    if op == Op.LINK.value:
        return {
            "source_id": payload.get("source_id"),
            "target_id": payload.get("target_id"),
            "kind": payload.get("kind"),
            "weight": payload.get("weight"),
        }
    if op == Op.COMPACT.value:
        return {"memory_ids": list(payload.get("memory_ids", [])), "strategy": payload.get("strategy")}
    if op == Op.NOOP.value:
        return {"reason": _redact_reason(str(payload.get("reason", "")))}
    return {}


def _payload_ids(payload: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("memory_id", "source_id", "target_id"):
        if isinstance(payload.get(key), str):
            values.append(payload[key])
    if isinstance(payload.get("memory_ids"), list):
        values.extend(x for x in payload["memory_ids"] if isinstance(x, str))
    return list(dict.fromkeys(values))


def _alias_action_ids(action: dict[str, Any], candidate_ids: list[str]) -> dict[str, Any]:
    aliases = {memory_id: f"candidate_{i}" for i, memory_id in enumerate(candidate_ids)}
    payload = dict(action.get("payload") or {})
    for key in ("memory_id", "source_id", "target_id"):
        if payload.get(key) in aliases:
            payload[key] = aliases[payload[key]]
    if isinstance(payload.get("memory_ids"), list):
        payload["memory_ids"] = [aliases.get(x, x) for x in payload["memory_ids"]]
    return {
        **action,
        "target": aliases.get(action.get("target"), action.get("target")),
        "payload": payload,
    }


def sanitize_features(features: dict[str, Any], candidate_ids: list[str]) -> dict[str, Any]:
    """Allowlist non-text policy features; never trust caller-provided extras."""
    safe: dict[str, Any] = {}
    for key, value in features.items():
        if key not in _SAFE_FEATURE_KEYS:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
    safe["candidate_count"] = len(candidate_ids)
    safe["candidate_ids"] = [f"candidate_{i}" for i in range(len(candidate_ids))]
    return safe


def _group_hash(seed: str, scenario: str, user_id: str, time_bucket: int = 0) -> int:
    raw = f"{seed}\x00{scenario}\x00{user_id}\x00{time_bucket}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def _opaque_group(seed: str, scenario: str, user_id: str, time_bucket: int = 0) -> str:
    """Stable pseudonym for grouping; raw scenario/user IDs stay in SQLite."""
    raw = f"{seed}\x00{scenario}\x00{user_id}\x00{time_bucket}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def assign_split(
    scenario: str,
    user_id: str,
    ratios: list[tuple[str, float]] | None = None,
    seed: str = "mpm-v1",
    time_bucket: int = 0,
) -> str:
    ratios = ratios or DEFAULT_RATIOS
    total = sum(r for _, r in ratios)
    bucket = (_group_hash(seed, scenario, user_id, time_bucket) / 2**64) * total
    acc = 0.0
    for name, r in ratios:
        acc += r
        if bucket < acc:
            return name
    return ratios[-1][0]


def _reward_map(store: MemoryStore) -> dict[str, float]:
    rows = store.conn.execute(
        "SELECT memory_id, SUM(reward) AS total FROM attributions GROUP BY memory_id"
    ).fetchall()
    return {r["memory_id"]: r["total"] for r in rows}


def _stratum(reward: float | None) -> str:
    if reward is None:
        return "hard"
    if reward < -1e-9:
        return "negative"
    if abs(reward) < 1e-9:
        return "hard"
    return "easy"


def _decisions(store: MemoryStore) -> list[dict[str, Any]]:
    rows = store.conn.execute(
        """SELECT d.*, s.scenario, s.user_id, s.started_at
           FROM policy_decisions d
           LEFT JOIN sessions s ON d.session_id = s.session_id
           ORDER BY d.ts ASC"""
    ).fetchall()
    return [dict(r) for r in rows]


def build_examples(
    store: MemoryStore,
    *,
    allow_raw: bool = False,
    ratios: list[tuple[str, float]] | None = None,
    seed: str = "mpm-v1",
) -> list[dict[str, Any]]:
    """Return a list of SFT examples (with strata) plus preference pairs."""
    rewards = _reward_map(store)
    examples: list[dict[str, Any]] = []
    for d in _decisions(store):
        op = d["op"]
        payload = json.loads(d["payload"]) if d["payload"] else {}
        scenario = d["scenario"] or "unknown"
        user = d["user_id"] or "unknown"
        # A coarse temporal cohort is part of the split key. All decisions and
        # delayed outcomes from the same session remain together, while later
        # cohorts can be held out without exporting wall-clock timestamps.
        time_bucket = int(float(d["started_at"] or 0) // 86_400)
        split = assign_split(scenario, user, ratios, seed, time_bucket)
        group_id = _opaque_group(seed, scenario, user, time_bucket)
        reward = rewards.get(d["target"]) if d["target"] else None
        stratum = _stratum(reward)
        stored_features = json.loads(d["features"]) if d["features"] else {}
        red = redact_payload(op, payload)
        raw_candidate_ids = []
        if isinstance(stored_features.get("candidate_ids"), list):
            raw_candidate_ids.extend(x for x in stored_features["candidate_ids"] if isinstance(x, str))
        raw_candidate_ids.extend(_payload_ids(red))
        policy_target = None
        if op in {Op.UPDATE.value, Op.DELETE.value}:
            policy_target = red.get("memory_id")
        candidate_ids = list(dict.fromkeys(raw_candidate_ids))
        features = sanitize_features(stored_features, candidate_ids)
        action = _alias_action_ids(
            {"op": op, "target": policy_target, "payload": red, "confidence": d["confidence"]},
            candidate_ids,
        )
        ex: dict[str, Any] = {
            "split": split,
            "kind": "sft",
            "stratum": stratum,
            "group_id": group_id,
            "t": d["event_seq"],
            "modality": "text",
            "features": features,
            "label": op,
            "reward": reward,
            "action": action,
        }
        if allow_raw:
            ex["raw"] = {"payload": payload}
        examples.append(ex)

        # Preference pairs (DPO) from clear positive/negative signals.
        if stratum == "easy" and op != Op.NOOP.value:
            examples.append(
                {
                    "split": split,
                    "kind": "preference",
                    "stratum": "easy",
                    "group_id": group_id,
                    "t": d["event_seq"],
                    "modality": "text",
                    "features": features,
                    "chosen": action,
                    "rejected": {"op": Op.NOOP.value, "target": None, "payload": {"reason": "noop"}},
                    "reward": reward,
                }
            )
        elif stratum == "negative":
            examples.append(
                {
                    "split": split,
                    "kind": "preference",
                    "stratum": "negative",
                    "group_id": group_id,
                    "t": d["event_seq"],
                    "modality": "text",
                    "features": features,
                    "chosen": {"op": Op.NOOP.value, "target": None, "payload": {"reason": "noop"}},
                    "rejected": action,
                    "reward": reward,
                }
            )
    return examples


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows, 1):
            f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    return len(rows)


def export_all(
    store: MemoryStore,
    outdir: str | Path,
    *,
    allow_raw: bool = False,
    ratios: list[tuple[str, float]] | None = None,
    seed: str = "mpm-v1",
) -> dict[str, Any]:
    """Write SFT and DPO JSONL files, persist examples to the store, and report counts."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    examples = build_examples(store, allow_raw=allow_raw, ratios=ratios, seed=seed)

    sft = [e for e in examples if e["kind"] == "sft"]
    pref = [e for e in examples if e["kind"] == "preference"]

    report: dict[str, Any] = {"allow_raw": allow_raw, "seed": seed, "files": {}, "counts": {}}
    for split in [r[0] for r in (ratios or DEFAULT_RATIOS)]:
        sft_split = [e for e in sft if e["split"] == split]
        pref_split = [e for e in pref if e["split"] == split]
        sft_path = outdir / f"sft-{split}.jsonl"
        pref_path = outdir / f"dpo-{split}.jsonl"
        report["files"][f"sft-{split}.jsonl"] = _write_jsonl(sft_path, sft_split)
        report["files"][f"dpo-{split}.jsonl"] = _write_jsonl(pref_path, pref_split)
        report["counts"][f"sft-{split}"] = len(sft_split)
        report["counts"][f"dpo-{split}"] = len(pref_split)

    # Persist examples into the store for audit / consolidation replay.
    for e in sft:
        store.add_training_example(
            split=e["split"],
            kind="sft",
            stratum=e["stratum"],
            scenario="redacted",
            user_id=e["group_id"],
            t=str(e["t"]),
            features=e["features"],
            label=e["label"],
            chosen=None,
            rejected=None,
            reward=e["reward"],
            raw=e.get("raw"),
        )
    for e in pref:
        store.add_training_example(
            split=e["split"],
            kind="preference",
            stratum=e["stratum"],
            scenario="redacted",
            user_id=e["group_id"],
            t=str(e["t"]),
            features=e["features"],
            label=e.get("label", ""),
            chosen=e["chosen"],
            rejected=e["rejected"],
            reward=e["reward"],
            raw=None,
        )

    report["totals"] = {"sft": len(sft), "preference": len(pref)}
    report["outdir"] = str(outdir)
    return report
