#!/usr/bin/env python3
"""Compare the base LFM2.5-VL checkpoint with an MLX LoRA adapter.

The evaluator intentionally scores only the memory-operation label. Payload and
target quality require a larger task-grounded suite and are retained in the raw
generation records for later inspection.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

import mlx.core as mx
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

from mpm.train.leap import to_lfm_messages


OPS = ("WRITE", "UPDATE", "DELETE", "LINK", "COMPACT", "NOOP")
OP_PATTERN = re.compile(r"\b(" + "|".join(OPS) + r")\b", re.IGNORECASE)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_op(text: str) -> str | None:
    match = OP_PATTERN.search(text)
    return match.group(1).upper() if match else None


def evaluate(model_path: str, adapter_path: str | None, rows: list[dict]) -> dict:
    model, processor = load(model_path, adapter_path=adapter_path)
    config = model.config.__dict__
    predictions = []
    started = time.monotonic()

    for index, row in enumerate(rows):
        messages = to_lfm_messages(row)["messages"][:2]
        prompt = apply_chat_template(
            processor,
            config,
            messages,
            add_generation_prompt=True,
            num_images=0,
            num_audios=0,
        )
        result = generate(
            model,
            processor,
            prompt,
            image=None,
            max_tokens=96,
            temperature=0,
            verbose=False,
        )
        expected = str(row.get("label") or row["action"]["op"]).upper()
        predicted = extract_op(result.text)
        predictions.append(
            {
                "index": index,
                "expected": expected,
                "predicted": predicted,
                "correct": predicted == expected,
                "text": result.text,
                "prompt_tokens": result.prompt_tokens,
                "generation_tokens": result.generation_tokens,
                "generation_tps": result.generation_tps,
                "peak_memory_gb": result.peak_memory,
            }
        )

    correct = sum(item["correct"] for item in predictions)
    valid = sum(
        item["predicted"] is not None
        and "<|tool_call_start|>" in item["text"]
        and "memory_action(" in item["text"]
        and "<|tool_call_end|>" in item["text"]
        for item in predictions
    )
    labels = Counter(item["expected"] for item in predictions)
    f1s = []
    for op in labels:
        tp = sum(item["predicted"] == op and item["expected"] == op for item in predictions)
        fp = sum(item["predicted"] == op and item["expected"] != op for item in predictions)
        fn = sum(item["predicted"] != op and item["expected"] == op for item in predictions)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    elapsed = time.monotonic() - started
    report = {
        "adapter_path": adapter_path,
        "accuracy": correct / len(predictions) if predictions else 0.0,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "structured_output_validity": valid / len(predictions) if predictions else 0.0,
        "correct": correct,
        "count": len(predictions),
        "expected_label_counts": dict(sorted(labels.items())),
        "elapsed_seconds": elapsed,
        "predictions": predictions,
    }

    del model, processor
    gc.collect()
    mx.clear_cache()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.test)
    report = {
        "model": args.model,
        "adapter": str(args.adapter),
        "test_file": str(args.test.resolve()),
        "test_sha256": sha256(args.test),
        "base": evaluate(args.model, None, rows),
        "fine_tuned": evaluate(args.model, args.adapter, rows),
    }
    report["accuracy_delta"] = (
        report["fine_tuned"]["accuracy"] - report["base"]["accuracy"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "base_accuracy": report["base"]["accuracy"],
        "fine_tuned_accuracy": report["fine_tuned"]["accuracy"],
        "accuracy_delta": report["accuracy_delta"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
