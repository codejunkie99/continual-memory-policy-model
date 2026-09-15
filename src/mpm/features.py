"""Privacy-preserving feature extraction.

Features are deliberately *not* raw text.  They are numeric/boolean summaries
plus a non-invertible content digest, so the default training export can carry
enough signal for a policy model without leaking user memory content.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any


_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_CC_RE = re.compile(r"\b(?:\d[ -]?){13,16}\b")
_DIGIT_RE = re.compile(r"\d")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def content_hash(text: str) -> str:
    """Content-integrity digest for the private store, not anonymization.

    A plain digest of low-entropy text can be guessed with a dictionary attack,
    so this value must never be included in the default training export.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _length_bucket(n: int) -> int:
    # Coarse buckets: 0, 1-8, 9-32, 33-128, 129-512, 513+
    for i, hi in enumerate((0, 8, 32, 128, 512)):
        if n <= hi:
            return i
    return 5


def extract_features(
    content: str,
    *,
    scope: str | None = None,
    key: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a JSON-serializable feature vector that contains no raw user text."""
    tokens = tokenize(content)
    n_chars = len(content)
    n_digits = len(_DIGIT_RE.findall(content))
    has_email = bool(_EMAIL_RE.search(content))
    has_url = bool(_URL_RE.search(content))
    has_cc = bool(_CC_RE.search(content))

    features: dict[str, Any] = {
        "n_tokens": len(tokens),
        "n_chars": n_chars,
        "len_bucket": _length_bucket(n_chars),
        "token_len_bucket": _length_bucket(len(tokens)),
        "has_digit": n_digits > 0,
        "digit_density": round(n_digits / max(n_chars, 1), 4),
        "has_email": has_email,
        "has_url": has_url,
        "has_credit_card": has_cc,
    }
    if scope is not None:
        features["has_scope"] = bool(scope)
    if key is not None:
        features["has_key"] = bool(key)
    if extra:
        # extra must already be redacted; callers are responsible for that.
        features.update(extra)
    return features


def feature_digest(features: dict[str, Any]) -> str:
    """Stable, order-independent digest of a feature vector for split keys."""
    canon = repr(sorted((str(k), repr(v)) for k, v in features.items()))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()
