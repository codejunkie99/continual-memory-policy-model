"""Deterministic replay mixing for longer-term policy consolidation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .train.config import TrainConfig
from .train.leap import render_leap_yaml, to_lfm_messages


def _read_jsonl(path: Path, *, expected_split: str = "train") -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
        if row.get("split") != expected_split:
            raise ValueError(f"{path}:{line_no}: expected split {expected_split!r}")
        rows.append(row)
    return rows


def _fingerprint(row: dict[str, Any]) -> str:
    encoded = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rank(seed: str, row: dict[str, Any]) -> tuple[int, str]:
    # Difficult and negative historical examples are replayed first; hashing
    # makes selection deterministic without depending on filesystem order.
    priority = {"negative": 0, "hard": 1, "easy": 2}.get(str(row.get("stratum")), 1)
    digest = hashlib.sha256((seed + _fingerprint(row)).encode("utf-8")).hexdigest()
    return priority, digest


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in materialized),
        encoding="utf-8",
    )
    return len(materialized)


def build_consolidation(
    current_dir: str | Path,
    replay_dirs: Iterable[str | Path],
    outdir: str | Path,
    *,
    max_replay_per_kind: int = 1000,
    seed: str = "mpm-consolidation-v1",
) -> dict[str, Any]:
    """Mix all new train rows with bounded, difficult-first historical replay."""
    if max_replay_per_kind < 0:
        raise ValueError("max_replay_per_kind must be non-negative")
    current_dir = Path(current_dir).resolve()
    replay_dirs = [Path(path).resolve() for path in replay_dirs]
    outdir = Path(outdir).resolve()
    report: dict[str, Any] = {
        "seed": seed,
        "current_dir": str(current_dir),
        "replay_dirs": [str(path) for path in replay_dirs],
        "max_replay_per_kind": max_replay_per_kind,
        "kinds": {},
    }

    for kind, filename, output_name in (
        ("sft", "sft-train.jsonl", "consolidated-sft.jsonl"),
        ("preference", "dpo-train.jsonl", "consolidated-dpo.jsonl"),
    ):
        current = _read_jsonl(current_dir / filename)
        history = [row for directory in replay_dirs for row in _read_jsonl(directory / filename)]
        seen: set[str] = set()
        current_unique: list[dict[str, Any]] = []
        for row in current:
            fp = _fingerprint(row)
            if fp not in seen:
                seen.add(fp)
                current_unique.append(row)
        replay_unique: list[dict[str, Any]] = []
        for row in sorted(history, key=lambda item: _rank(seed, item)):
            fp = _fingerprint(row)
            if fp not in seen:
                seen.add(fp)
                replay_unique.append(row)
            if len(replay_unique) >= max_replay_per_kind:
                break
        combined = current_unique + replay_unique
        count = _write_jsonl(outdir / output_name, combined)
        report["kinds"][kind] = {
            "current_input": len(current),
            "current_unique": len(current_unique),
            "historical_input": len(history),
            "historical_selected": len(replay_unique),
            "output": count,
            "strata": dict(sorted(Counter(str(row.get("stratum", "unknown")) for row in combined).items())),
            "sha256": hashlib.sha256((outdir / output_name).read_bytes()).hexdigest(),
        }

        if kind == "sft":
            leap_train = outdir / "leap-vlm-sft-train.jsonl"
            _write_jsonl(leap_train, (to_lfm_messages(row) for row in combined))

    validation = _read_jsonl(current_dir / "sft-val.jsonl", expected_split="val")
    _write_jsonl(outdir / "leap-vlm-sft-val.jsonl", (to_lfm_messages(row) for row in validation))
    consolidation_cfg = TrainConfig(adapter_version="v1.0.0", parent_version="v0.1.0")
    leap_config_path = outdir / "leap-consolidation.yaml"
    leap_config_path.write_text(render_leap_yaml(consolidation_cfg, str(outdir)), encoding="utf-8")
    report["leap_sft"] = {
        "status": "ready" if validation else "not_ready",
        "config": str(leap_config_path),
        "train_rows": report["kinds"]["sft"]["output"],
        "validation_rows": len(validation),
        "adapter_version": consolidation_cfg.adapter_version,
        "parent_version": consolidation_cfg.parent_version,
    }
    if not validation:
        report["leap_sft"]["reason"] = "a non-empty held-out validation split is required"
    report["leap_preference"] = {
        "status": "not_ready",
        "reason": "LEAP vlm_dpo requires a loadable image or images field for every preference row; current policy trajectories are text-only",
    }

    report["policy"] = "all unique current rows plus bounded negative-first, hard-first historical replay"
    report["auto_promote"] = False
    manifest = outdir / "consolidation-manifest.json"
    manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["manifest"] = str(manifest)
    return report
