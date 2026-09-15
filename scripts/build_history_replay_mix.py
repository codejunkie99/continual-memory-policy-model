#!/usr/bin/env python3
"""Mix bounded history-derived MLX rows with balanced policy replay.

The history rows are already feature-only; this script never accepts or emits
raw memory text. Per-label caps keep weak WRITE/UPDATE supervision from
overwriting the six-operation policy learned by the balanced replay set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from mpm.types import ALL_OPS


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _label(row: dict) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        raise ValueError("every row must be an MLX messages record")
    arguments = messages[2]["tool_calls"][0]["function"]["arguments"]
    label = json.loads(arguments)["op"]
    if label not in ALL_OPS:
        raise ValueError(f"unknown operation label: {label!r}")
    return label


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-per-label", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()

    selected: list[dict] = []
    history_counts: Counter[str] = Counter()
    for row in _read(args.history):
        label = _label(row)
        if history_counts[label] >= args.history_per_label:
            continue
        history_counts[label] += 1
        selected.append(row)

    replay = _read(args.replay)
    rows = replay + selected
    random.Random(args.seed).shuffle(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    args.output.write_text(payload, encoding="utf-8")

    total_counts = Counter(_label(row) for row in rows)
    print(
        json.dumps(
            {
                "count": len(rows),
                "history_count": len(selected),
                "history_labels": dict(sorted(history_counts.items())),
                "labels": dict(sorted(total_counts.items())),
                "output_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
