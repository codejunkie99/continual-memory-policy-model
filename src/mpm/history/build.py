"""Build a weak-supervision policy dataset from sanitized history records.

Records are converted to a *closed-vocabulary* feature vector (the same numeric/
boolean summaries used by the MLX runtime) plus one of the six operation labels.
Raw memory text, original identifiers, paths, secrets, and wall-clock
timestamps never enter the emitted JSONL or any MLX prompt.  Splits are
deterministic and time-ordered so later data is held out rather than leaked
into training.  The exporter refuses to write anything if a privacy check fails.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..features import extract_features, feature_digest
from ..train.leap import to_lfm_messages
from ..types import ALL_OPS
from .redact import (
    _ADDRESS_RE,
    _CC_RE,
    _EMAIL_RE,
    _LONG_HEX_RE,
    _PHONE_RE,
    _POSIX_PATH_RE,
    _UUID_RE,
    _WINDOWS_PATH_RE,
    SECRET_KEYWORDS,
    _normalize,
)
from .staging import HistoryStagingStore


class PrivacyError(RuntimeError):
    """Raised when a dataset would leak raw content, ids, paths, or secrets."""


CATEGORY_LABELS: dict[str, str] = {
    "preference": "WRITE",
    "decision": "WRITE",
    "convention": "WRITE",
    "lesson": "WRITE",
    "fact": "WRITE",
    "correction": "UPDATE",
    "other": "NOOP",
}

DEFAULT_RATIOS: list[tuple[str, float]] = [("train", 0.7), ("val", 0.15), ("test", 0.15)]


def resolve_label(category: str, label_override: str | None) -> str:
    """Resolve a record to one of the six policy labels."""
    if label_override and label_override in ALL_OPS:
        return label_override
    return CATEGORY_LABELS.get(category, "WRITE")


def assign_time_ordered_splits(
    records: list[dict[str, Any]],
    ratios: list[tuple[str, float]] | None = None,
    *,
    time_bucket: float = 86_400.0,
) -> list[tuple[dict[str, Any], str]]:
    """Deterministic, leak-safe temporal split.

    Records are grouped into time cohorts (fixed-width buckets), the cohorts are
    ordered chronologically, and whole cohorts are assigned to train/val/test in
    time order.  A later record can therefore never land in an earlier split.
    """
    ratios = ratios or DEFAULT_RATIOS
    total_ratio = sum(r for _, r in ratios)

    cohorts: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        bucket = int(float(record["ts"]) // time_bucket) if time_bucket > 0 else 0
        cohorts[bucket].append(record)

    ordered_cohorts = sorted(cohorts.items())
    n_cohorts = len(ordered_cohorts)
    cohort_split: dict[int, str] = {}
    for i, (bucket, _) in enumerate(ordered_cohorts):
        pos = (i + 1) / n_cohorts if n_cohorts else 1.0
        acc = 0.0
        chosen = ratios[-1][0]
        for name, r in ratios:
            acc += r / total_ratio
            if pos <= acc + 1e-9:
                chosen = name
                break
        cohort_split[bucket] = chosen

    result: list[tuple[dict[str, Any], str]] = []
    for bucket, recs in ordered_cohorts:
        for record in sorted(recs, key=lambda r: (float(r["ts"]), r["salted_hash"])):
            result.append((record, cohort_split[bucket]))
    return result


def _apply_source_caps(
    assigned: list[tuple[dict[str, Any], str]],
    max_per_source: int | None,
) -> list[tuple[dict[str, Any], str]]:
    if max_per_source is None:
        return assigned
    by_split_source: dict[tuple[str, str], int] = Counter()
    kept: list[tuple[dict[str, Any], str]] = []
    for record, split in assigned:
        key = (split, record["source_kind"])
        if by_split_source[key] >= max_per_source:
            continue
        by_split_source[key] += 1
        kept.append((record, split))
    return kept


def _feature_vector(record: dict[str, Any]) -> dict[str, Any]:
    scope = record.get("category") or "default"
    features = extract_features(record["text"], scope=scope)
    label = str(record["label"])
    context_count = {
        "UPDATE": 1,
        "DELETE": 1,
        "LINK": 2,
        "COMPACT": 2,
    }.get(label, 0)
    features.update(
        {
            "requested_op": label,
            "context_count": context_count,
            "has_content": label in {"WRITE", "UPDATE"},
            "exact_duplicate": label == "NOOP",
            "source_trust": "high" if record["source_kind"] in {"brain", "mpm"} else "medium",
            "candidate_count": context_count,
            "candidate_ids": [f"candidate_{i}" for i in range(context_count)],
        }
    )
    return features


def _contains_secret(text: str) -> bool:
    """Detect secret material without false-positive on closed feature keys.

    Single-word keywords use word boundaries (so ``n_tokens`` does not trigger
    ``token``); multi-word phrases use a space-insensitive normalized substring.
    """
    lowered = text.lower()
    for kw in SECRET_KEYWORDS:
        if " " in kw or "_" in kw:
            if _normalize(kw) in _normalize(lowered):
                return True
        elif re.search(r"\b" + re.escape(kw) + r"\b", lowered):
            return True
    return False


def _privacy_violations(serialized: str, raw_texts: list[str]) -> list[str]:
    violations: list[str] = []
    for raw in raw_texts:
        if raw and raw in serialized:
            violations.append("raw_text")
            break
    if _POSIX_PATH_RE.search(serialized) or _WINDOWS_PATH_RE.search(serialized):
        violations.append("absolute_path")
    if (
        _EMAIL_RE.search(serialized)
        or _PHONE_RE.search(serialized)
        or _ADDRESS_RE.search(serialized)
        or _CC_RE.search(serialized)
    ):
        violations.append("pii")
    if _contains_secret(serialized):
        violations.append("secret")
    if _UUID_RE.search(serialized) or _LONG_HEX_RE.search(serialized):
        violations.append("original_id")
    return sorted(set(violations))


def _read_replay_rows(replay_dirs: list[str] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for d in replay_dirs or []:
        path = Path(d) / "history-train.jsonl"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if isinstance(row.get("features"), dict) and row.get("label") in ALL_OPS:
                    rows.append(row)
    return rows


def build_dataset(
    staging: HistoryStagingStore,
    outdir: str | Path,
    *,
    ratios: list[tuple[str, float]] | None = None,
    time_bucket: float = 86_400.0,
    max_per_source: int | None = None,
    seed: str = "mpm-history-v1",
    replay_dirs: list[str] | None = None,
    max_replay: int = 0,
) -> dict[str, Any]:
    """Build the dataset, write JSONL + manifest, and return a manifest dict."""
    ratios = ratios or DEFAULT_RATIOS
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    records = staging.records()
    raw_texts = [r["text"] for r in records]
    resolved: list[dict[str, Any]] = []
    for record in records:
        resolved.append(
            {
                "text": record["text"],
                "category": record["category"],
                "label": resolve_label(record["category"], record.get("label")),
                "source_kind": record["source_kind"],
                "ts": float(record["ts"]),
                "salted_hash": record["salted_hash"],
            }
        )

    assigned = _apply_source_caps(
        assign_time_ordered_splits(resolved, ratios, time_bucket=time_bucket),
        max_per_source,
    )

    # Build row objects (features + label only; no raw text / ids / ts).
    rows_by_split: dict[str, list[dict[str, Any]]] = {name: [] for name, _ in ratios}
    seen_digests: set[str] = set()
    for record, split in assigned:
        features = _feature_vector(record)
        digest = f"{record['label']}:{feature_digest(features)}"
        if digest in seen_digests:
            continue
        seen_digests.add(digest)
        rows_by_split[split].append(
            {
                "split": split,
                "kind": "sft",
                "source_kind": record["source_kind"],
                "features": features,
                "label": record["label"],
            }
        )

    # Replay prior feature-only rows into train (bounded, deduplicated).
    replay_rows = _read_replay_rows(replay_dirs)
    if max_replay > 0 and replay_rows:
        replay_rows = replay_rows[:max_replay]
        for row in replay_rows:
            digest = f"{row['label']}:{feature_digest(row['features'])}"
            if digest in seen_digests:
                continue
            seen_digests.add(digest)
            rows_by_split["train"].append(
                {
                    "split": "train",
                    "kind": "sft",
                    "source_kind": "replay",
                    "features": row["features"],
                    "label": row["label"],
                }
            )

    # Write feature-only audit files and MLX-ready message files. Both contain
    # only the closed feature vocabulary and synthetic candidate identifiers.
    file_hashes: dict[str, str] = {}
    counts: dict[str, int] = {}
    for split, _ in ratios:
        path = outdir / f"history-{split}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in rows_by_split[split]:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
        file_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        mlx_path = outdir / f"mlx-{split}.jsonl"
        with mlx_path.open("w", encoding="utf-8") as fh:
            for row in rows_by_split[split]:
                converted = to_lfm_messages(row)
                arguments = {"op": row["label"], "payload": {}}
                converted["messages"][2]["tool_calls"][0]["function"]["arguments"] = json.dumps(
                    arguments, sort_keys=True
                )
                fh.write(json.dumps(converted, sort_keys=True) + "\n")
        file_hashes[mlx_path.name] = hashlib.sha256(mlx_path.read_bytes()).hexdigest()
        counts[split] = len(rows_by_split[split])

    # Privacy gate: refuse the export if anything sensitive leaked.
    corpus = "\n".join(
        "\n".join(
            [
                (outdir / f"history-{split}.jsonl").read_text(encoding="utf-8"),
                (outdir / f"mlx-{split}.jsonl").read_text(encoding="utf-8"),
            ]
        )
        for split, _ in ratios
    )
    violations = _privacy_violations(corpus, raw_texts)
    if violations:
        for split, _ in ratios:
            (outdir / f"history-{split}.jsonl").unlink(missing_ok=True)
            (outdir / f"mlx-{split}.jsonl").unlink(missing_ok=True)
        (outdir / "manifest.json").unlink(missing_ok=True)
        raise PrivacyError(f"dataset would leak: {', '.join(violations)}")

    # Time boundaries per split (from the assigned, not the raw records).
    boundaries: dict[str, dict[str, float | None]] = {}
    for split, _ in ratios:
        ts_values = [float(r["ts"]) for r, s in assigned if s == split]
        boundaries[split] = {
            "min_ts": min(ts_values) if ts_values else None,
            "max_ts": max(ts_values) if ts_values else None,
        }

    counts_by_source: dict[str, dict[str, int]] = {}
    counts_by_label: dict[str, int] = Counter()
    for split, _ in ratios:
        source_counter: Counter = Counter()
        for row in rows_by_split[split]:
            source_counter[row["source_kind"]] += 1
            counts_by_label[row["label"]] += 1
        counts_by_source[split] = dict(sorted(source_counter.items()))

    manifest: dict[str, Any] = {
        "version": "history-dataset-v1",
        "weak_supervision": True,
        "note": "history-derived labels are weak supervision unless downstream outcomes exist",
        "seed": seed,
        "ratios": list(ratios),
        "time_bucket": time_bucket,
        "max_per_source": max_per_source,
        "max_replay": max_replay,
        "counts": counts,
        "counts_by_source": counts_by_source,
        "counts_by_label": dict(sorted(counts_by_label.items())),
        "time_boundaries": boundaries,
        "file_sha256": file_hashes,
        "privacy_checks": {
            "no_raw_text": True,
            "no_absolute_path": True,
            "no_pii": True,
            "no_secret": True,
            "no_original_id": True,
        },
        "replay_sources": len(replay_dirs or []),
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
