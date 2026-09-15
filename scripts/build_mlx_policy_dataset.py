#!/usr/bin/env python3
"""Build a class-balanced, completion-focused MLX policy dataset."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from mpm.train.leap import to_lfm_messages


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hard-label", choices=["WRITE", "UPDATE", "DELETE", "LINK", "COMPACT", "NOOP"])
    parser.add_argument("--hard-extra-copies", type=int, default=0)
    args = parser.parse_args()

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in read_jsonl(args.input):
        grouped[str(row.get("label") or row["action"]["op"]).upper()].append(row)

    target = max(len(rows) for rows in grouped.values())
    balanced: list[dict] = []
    for op, rows in sorted(grouped.items()):
        for index in range(target):
            row = rows[index % len(rows)]
            messages = to_lfm_messages(row)["messages"][:3]
            # Keep a native LFM tool call, but make the supervised completion
            # focus on the policy decision rather than copying incidental
            # payload fields. `payload` remains because the tool schema requires it.
            messages[-1]["tool_calls"][0]["function"]["arguments"] = json.dumps(
                {"op": op, "payload": {}}, sort_keys=True
            )
            balanced.append({"messages": messages})

    if args.hard_label and args.hard_extra_copies > 0:
        hard_rows = [row for row in balanced if json.loads(
            row["messages"][-1]["tool_calls"][0]["function"]["arguments"]
        )["op"] == args.hard_label]
        for _ in range(args.hard_extra_copies):
            balanced.extend(hard_rows)

    random.Random(args.seed).shuffle(balanced)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in balanced)
    )
    print(json.dumps({
        "input_counts": dict(sorted((op, len(rows)) for op, rows in grouped.items())),
        "output_count": len(balanced),
        "output_counts": dict(sorted(Counter(
            json.loads(row["messages"][-1]["tool_calls"][0]["function"]["arguments"])["op"]
            for row in balanced
        ).items())),
        "path": str(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
