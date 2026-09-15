"""Privacy-safe, streaming history ingestion across local sources.

Sources:
* Codex session JSONL (``~/.codex/sessions/**/*.jsonl``)
* Claude Code project JSONL (``~/.claude/projects/**/*.jsonl``)
* Brain notes (markdown / JSONL, optionally git-backed)
* Existing MPM SQLite stores

Parsers are deliberately defensive: malformed rows are counted and skipped,
schema variants are handled by probing common field names, and every text block
is passed through the default-deny :func:`mpm.history.redact.redact` gate before
anything reaches the staging store.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from .redact import redact
from .staging import HistoryStagingStore


def _coerce_ts(value: Any) -> float:
    """Best-effort conversion of a timestamp to a float epoch."""
    def normalize_epoch(number: float) -> float:
        magnitude = abs(number)
        if magnitude >= 1e17:  # nanoseconds
            return number / 1_000_000_000.0
        if magnitude >= 1e14:  # microseconds
            return number / 1_000_000.0
        if magnitude >= 1e11:  # milliseconds
            return number / 1_000.0
        return number

    if value is None:
        return 0.0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return normalize_epoch(float(value))
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0.0
        try:
            return normalize_epoch(float(text))
        except ValueError:
            pass
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return 0.0
    return 0.0


def _text_from_content(content: Any) -> list[str]:
    """Extract plain-text strings from a content field that may be a str or a list."""
    out: list[str] = []
    if isinstance(content, str):
        out.append(content)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, str):
                out.append(block)
            elif isinstance(block, dict):
                block_type = block.get("type")
                if block_type not in {"text", "input_text", "output_text"}:
                    continue
                text = block.get("text") or block.get("content")
                if isinstance(text, str):
                    out.append(text)
    elif isinstance(content, dict):
        block_type = content.get("type")
        if block_type not in {"text", "input_text", "output_text"}:
            return out
        text = content.get("text") or content.get("content")
        if isinstance(text, str):
            out.append(text)
    return out


def _jsonl_files(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    if root.is_file():
        return [root] if root.suffix == ".jsonl" else []
    return sorted(p for p in root.rglob("*.jsonl") if p.is_file())


# ---------------------------------------------------------------------------
# Source iterators
# ---------------------------------------------------------------------------


def iter_codex(
    root: str | Path, malformed: list[int] | None = None
) -> Iterator[tuple[str, str | None, float]]:
    """Yield ``(text, role, ts)`` from Codex session JSONL records."""
    for path in _jsonl_files(root):
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    if malformed is not None:
                        malformed[0] += 1
                    continue
                if not isinstance(record, dict):
                    if malformed is not None:
                        malformed[0] += 1
                    continue
                rtype = record.get("type")
                payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
                ts = _coerce_ts(
                    payload.get("timestamp")
                    or record.get("timestamp")
                    or record.get("created_at")
                    or payload.get("created_at")
                    or record.get("ts")
                ) or float(path.stat().st_mtime)
                if rtype == "response_item":
                    if payload.get("type") in {"function_call", "function_call_output", "reasoning", "tool_call"}:
                        continue  # tool output / reasoning is default-deny
                    role = str(payload.get("role") or "user").lower()
                    for text in _text_from_content(payload.get("content")):
                        yield text, role, ts
                elif rtype == "message" and isinstance(record.get("message"), dict):
                    msg = record["message"]
                    role = str(msg.get("role") or "user").lower()
                    for text in _text_from_content(msg.get("content")):
                        yield text, role, ts


def iter_claude(
    root: str | Path, malformed: list[int] | None = None
) -> Iterator[tuple[str, str | None, float]]:
    """Yield ``(text, role, ts)`` from Claude Code project JSONL history."""
    for path in _jsonl_files(root):
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    if malformed is not None:
                        malformed[0] += 1
                    continue
                if not isinstance(record, dict):
                    if malformed is not None:
                        malformed[0] += 1
                    continue
                rtype = record.get("type")
                if rtype not in {"user", "assistant", "system"}:
                    continue
                role = str(rtype)
                ts = _coerce_ts(
                    record.get("timestamp") or record.get("created_at") or record.get("date") or record.get("ts")
                ) or float(path.stat().st_mtime)
                content = record.get("content")
                if content is None and isinstance(record.get("message"), dict):
                    content = record["message"].get("content")
                for text in _text_from_content(content):
                    yield text, role, ts


def iter_brain(
    root: str | Path, malformed: list[int] | None = None
) -> Iterator[tuple[str, str | None, float]]:
    """Yield ``(text, role, ts)`` from Brain markdown / JSONL notes."""
    root = Path(root)
    if not root.exists():
        return
    files: list[Path] = []
    if root.is_file():
        files = [root]
    else:
        files = sorted(
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in {".md", ".jsonl", ".json", ".sqlite"}
        )
    for path in files:
        try:
            if path.suffix.lower() == ".md":
                yield from _brain_markdown(path)
            elif path.suffix.lower() == ".sqlite" and path.stat().st_size:
                yield from _brain_sqlite(path, malformed)
            else:
                yield from _brain_jsonl(path, malformed)
        except OSError:
            continue


def _brain_markdown(path: Path) -> Iterator[tuple[str, str | None, float]]:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        raw = fh.read()
    # Drop a leading YAML frontmatter block.
    if raw.startswith("---\n"):
        parts = raw.split("\n---\n", 1)
        if len(parts) == 2:
            raw = parts[1]
    ts = _coerce_ts(path.stat().st_mtime)
    for paragraph in raw.split("\n\n"):
        paragraph = paragraph.strip()
        if paragraph:
            yield paragraph, None, ts


def _brain_jsonl(path: Path, malformed: list[int] | None = None) -> Iterator[tuple[str, str | None, float]]:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if malformed is not None:
                    malformed[0] += 1
                continue
            if not isinstance(record, dict):
                if malformed is not None:
                    malformed[0] += 1
                continue
            ts = _coerce_ts(record.get("ts") or record.get("timestamp") or record.get("created_at"))
            for key in ("note", "text", "content", "summary"):
                value = record.get(key)
                if isinstance(value, str) and value.strip():
                    yield value, None, ts
                    break


def _brain_sqlite(path: Path, malformed: list[int] | None = None) -> Iterator[dict[str, Any]]:
    """Read public note summaries from a Brain index without mutating it."""
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "events" not in tables:
            conn.close()
            return
        rows = conn.execute(
            "SELECT payload_json, time_observed FROM events "
            "WHERE is_redacted = 0 ORDER BY time_observed ASC, event_id ASC"
        )
        for row in rows:
            try:
                payload = json.loads(row["payload_json"])
            except (TypeError, json.JSONDecodeError):
                if malformed is not None:
                    malformed[0] += 1
                continue
            summary = payload.get("summary") if isinstance(payload, dict) else None
            if isinstance(summary, str) and summary.strip():
                yield {
                    "text": summary,
                    "role": "user",
                    "category": "lesson",
                    "ts": _coerce_ts(row["time_observed"]),
                }
        conn.close()
    except sqlite3.Error:
        if malformed is not None:
            malformed[0] += 1


def iter_mpm(db_path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield structured ``{text, category, label, ts}`` items from an MPM store."""
    conn = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM memories ORDER BY created_at ASC").fetchall()
        for row in rows:
            mem = dict(row)
            status = mem.get("status")
            scope = str(mem.get("scope") or "")
            category = "preference" if scope.startswith("preference") else "fact"
            label = None
            ts = _coerce_ts(mem.get("created_at"))
            if status == "tombstoned":
                category = "decision"
                label = "DELETE"
                ts = _coerce_ts(mem.get("updated_at")) or ts
            yield {"text": mem["content"], "category": category, "label": label, "ts": ts}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass
class IngestConfig:
    salt: str = "mpm-history-v1"
    max_per_source: int | None = None
    since: float | None = None
    until: float | None = None


@dataclass
class IngestReport:
    dry_run: bool = False
    seen: int = 0
    malformed: int = 0
    accepted: int = 0
    duplicates: int = 0
    rejected_by_reason: Counter = field(default_factory=Counter)
    accepted_by_source: Counter = field(default_factory=Counter)
    seen_by_source: Counter = field(default_factory=Counter)
    files_by_source: Counter = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "seen": self.seen,
            "malformed": self.malformed,
            "accepted": self.accepted,
            "duplicates": self.duplicates,
            "rejected_by_reason": dict(sorted(self.rejected_by_reason.items())),
            "accepted_by_source": dict(sorted(self.accepted_by_source.items())),
            "seen_by_source": dict(sorted(self.seen_by_source.items())),
            "files_by_source": dict(sorted(self.files_by_source.items())),
        }


def _count_files(root: str | Path | None, suffix_filter: str) -> int:
    if not root:
        return 0
    root = Path(root)
    if not root.exists():
        return 0
    if root.is_file():
        return 1 if root.suffix == suffix_filter else 0
    return sum(1 for p in root.rglob("*") if p.is_file() and p.suffix == suffix_filter)


def ingest_sources(
    staging: HistoryStagingStore,
    *,
    codex_dir: str | Path | None = None,
    claude_dir: str | Path | None = None,
    brain_dir: str | Path | None = None,
    mpm_db: str | Path | list[str | Path] | None = None,
    dry_run: bool = False,
    cfg: IngestConfig | None = None,
) -> IngestReport:
    """Run all configured sources into the staging store and return a report."""
    cfg = cfg or IngestConfig()
    report = IngestReport(dry_run=dry_run)
    malformed = [0]

    sources: list[tuple[str, Iterator[Any]]] = []
    if codex_dir:
        report.files_by_source["codex"] = _count_files(codex_dir, ".jsonl")
        sources.append(("codex", iter_codex(codex_dir, malformed)))
    if claude_dir:
        report.files_by_source["claude"] = _count_files(claude_dir, ".jsonl")
        sources.append(("claude", iter_claude(claude_dir, malformed)))
    if brain_dir:
        report.files_by_source["brain"] = _count_files(brain_dir, ".jsonl")
        report.files_by_source["brain"] += _count_files(brain_dir, ".md")
        report.files_by_source["brain"] += _count_files(brain_dir, ".json")
        report.files_by_source["brain"] += _count_files(brain_dir, ".sqlite")
        sources.append(("brain", iter_brain(brain_dir, malformed)))
    if mpm_db:
        db_paths = mpm_db if isinstance(mpm_db, list) else [mpm_db]
        for db_path in db_paths:
            if Path(db_path).exists():
                report.files_by_source["mpm"] += 1
                sources.append(("mpm", iter_mpm(db_path)))

    for source_kind, iterator in sources:
        for item in iterator:
            report.seen += 1
            report.seen_by_source[source_kind] += 1

            if isinstance(item, dict):
                text = item.get("text")
                role = item.get("role")
                category_override = item.get("category")
                label_override = item.get("label")
                ts = _coerce_ts(item.get("ts"))
            else:
                text, role, ts = item
                category_override = None
                label_override = None

            if cfg.since is not None and ts < cfg.since:
                report.rejected_by_reason["before_since"] += 1
                continue
            if cfg.until is not None and ts > cfg.until:
                report.rejected_by_reason["after_until"] += 1
                continue

            result = redact(
                text,
                role=role,
                require_durable=source_kind not in {"brain", "mpm"},
            )
            if not result.accept:
                report.rejected_by_reason[result.reason or "unknown"] += 1
                continue

            if cfg.max_per_source is not None and report.accepted_by_source[source_kind] >= cfg.max_per_source:
                report.rejected_by_reason["per_source_cap"] += 1
                continue

            category = category_override or result.category or (
                "lesson" if source_kind == "brain" else "fact"
            )
            if dry_run:
                report.accepted += 1
                report.accepted_by_source[source_kind] += 1
                continue

            inserted = staging.insert(
                result.sanitized,
                category=category,
                label=label_override,
                source_kind=source_kind,
                ts=ts,
            )
            if inserted:
                report.accepted += 1
                report.accepted_by_source[source_kind] += 1
            else:
                report.duplicates += 1

    report.malformed = malformed[0]
    return report
