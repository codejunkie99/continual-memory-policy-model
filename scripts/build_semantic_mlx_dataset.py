#!/usr/bin/env python3
"""Build privacy-safe semantic policy examples for the MLX runtime contract.

The dataset contains no memory text, real identifiers, user ids, or timestamps.
It teaches the operation policy from a closed vocabulary of structural signals
that the production ``MLXPolicy`` computes locally.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from mpm.train.leap import to_lfm_messages
from mpm.types import Op


def _shape(rng: random.Random, *, has_content: bool) -> dict:
    if not has_content:
        return {
            "n_tokens": 0,
            "n_chars": 0,
            "len_bucket": 0,
            "token_len_bucket": 0,
            "has_digit": False,
            "digit_density": 0.0,
            "has_email": False,
            "has_url": False,
            "has_credit_card": False,
            "has_scope": True,
        }
    n_tokens = rng.randint(3, 22)
    n_chars = rng.randint(max(9, n_tokens * 3), min(128, n_tokens * 8))
    return {
        "n_tokens": n_tokens,
        "n_chars": n_chars,
        "len_bucket": 2 if n_chars <= 32 else 3,
        "token_len_bucket": 1 if n_tokens <= 8 else 2,
        "has_digit": bool(rng.getrandbits(1)),
        "digit_density": round(rng.uniform(0.0, 0.15), 4),
        "has_email": False,
        "has_url": False,
        "has_credit_card": False,
        "has_scope": True,
    }


def _features(rng: random.Random, op: str, index: int) -> dict:
    if op == Op.WRITE.value:
        requested, context_count, has_content, duplicate = (
            ("WRITE" if index % 4 == 0 else "AUTO"),
            0,
            True,
            False,
        )
    elif op == Op.UPDATE.value:
        requested, context_count, has_content, duplicate = "UPDATE", 1, True, False
    elif op == Op.DELETE.value:
        requested, context_count, has_content, duplicate = "DELETE", 1, False, False
    elif op == Op.LINK.value:
        requested, context_count, has_content, duplicate = "LINK", 2, False, False
    elif op == Op.COMPACT.value:
        requested, context_count, has_content, duplicate = "COMPACT", 2 + (index % 2), False, False
    else:
        # Most model-visible NOOPs are exact duplicates. The runtime itself
        # short-circuits empty and sensitive observations before inference.
        requested, context_count, has_content, duplicate = "AUTO", 0, True, True

    candidate_count = max(context_count, rng.randint(0 if op == Op.WRITE.value else 1, 8))
    features = _shape(rng, has_content=has_content)
    features.update(
        {
            "requested_op": requested,
            "context_count": context_count,
            "has_content": has_content,
            "exact_duplicate": duplicate,
            "source_trust": rng.choice(["high", "high", "medium"]),
            "candidate_count": candidate_count,
            "candidate_ids": [f"candidate_{i}" for i in range(candidate_count)],
        }
    )
    return features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples-per-op", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows: list[dict] = []
    for op in (item.value for item in Op):
        for index in range(args.examples_per_op):
            example = {
                "modality": "text",
                "features": _features(rng, op, index),
                "label": op,
                "action": {"op": op, "target": None, "payload": {}},
            }
            messages = to_lfm_messages(example)["messages"][:3]
            messages[-1]["tool_calls"][0]["function"]["arguments"] = json.dumps(
                {"op": op, "payload": {}}, sort_keys=True
            )
            rows.append({"messages": messages})

    rng.shuffle(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    counts = Counter(
        json.loads(row["messages"][-1]["tool_calls"][0]["function"]["arguments"])["op"]
        for row in rows
    )
    print(json.dumps({"path": str(args.output), "count": len(rows), "labels": dict(sorted(counts.items()))}, indent=2))


if __name__ == "__main__":
    main()
