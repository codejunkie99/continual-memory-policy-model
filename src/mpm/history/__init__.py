"""Privacy-safe history ingestion and policy-dataset building.

This subpackage turns local, consented interaction histories (Codex JSONL,
Claude Code JSONL, Brain notes, and existing MPM SQLite stores) into a
closed-vocabulary, weak-supervision policy dataset without training any raw
facts, identifiers, secrets, or paths into model weights.
"""

from __future__ import annotations

from .redact import RedactResult, redact
from .staging import HistoryStagingStore, salted_hash
from .ingest import IngestConfig, ingest_sources
from .build import PrivacyError, build_dataset

__all__ = [
    "RedactResult",
    "redact",
    "HistoryStagingStore",
    "salted_hash",
    "IngestConfig",
    "ingest_sources",
    "PrivacyError",
    "build_dataset",
]
