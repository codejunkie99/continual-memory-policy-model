"""Default-deny redaction and category classification for history ingestion.

The ingestion path is intentionally conservative.  It extracts only candidate
*durable* statements (preferences, decisions, corrections, conventions, and
completed-task lessons) and drops everything else.  Sensitive material is
rejected outright; benign structural material (absolute paths, raw ids) is
scrubbed to placeholders so it never enters the staging store.

Every signal here is a heuristic.  The resulting records are weak supervision:
they become trustworthy training labels only once a downstream outcome exists.
"""

from __future__ import annotations

import base64
import binascii
import math
import re
from collections import Counter
from dataclasses import dataclass

from ..features import _CC_RE, _EMAIL_RE
from ..safety import is_harmful_content


# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

_PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]?\d{3}[ .-]?\d{4}"
)
_ADDRESS_RE = re.compile(
    r"\b\d{1,5}\s+[A-Za-z][A-Za-z .'-]{2,40}\b"
    r"\s+(?:street|st\.?|avenue|ave\.?|road|rd\.?|boulevard|blvd\.?|"
    r"lane|ln\.?|drive|dr\.?|court|ct\.?|place|pl\.?|way|circle|cir\.?)\b",
    re.IGNORECASE,
)
_POBOX_RE = re.compile(r"\b(?:p\.?o\.?\s*box|post office box)\s+\d+\b", re.IGNORECASE)
# Absolute POSIX and Windows paths.
_POSIX_PATH_RE = re.compile(r"(?<!\w)/(?:Users|home|Volumes|opt|private|var|tmp|mnt)/[^\s]*")
_WINDOWS_PATH_RE = re.compile(r"\b[A-Za-z]:\\[^\s]*")

# Original identifiers: UUIDs and long raw hex digests.
_UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE)
_LONG_HEX_RE = re.compile(r"\b[0-9a-f]{32,}\b", re.IGNORECASE)

# A single URL whose query string carries an obviously secret parameter.
_URL_QUERY_SECRET_RE = re.compile(
    r"https?://[^\s?#]+\?[^\s]*\b(?:token|key|secret|password|passwd|"
    r"auth|sig|signature|credential|access|apikey|api_key|client_secret)\b\s*=",
    re.IGNORECASE,
)

# A single base64-looking token (no spaces, base64 alphabet, optional padding).
_BASE64_TOKEN_RE = re.compile(r"\A[A-Za-z0-9+/]{32,}={0,2}\Z")

# Stack-trace fingerprints.
_STACK_TRACE_PATTERNS = (
    re.compile(r"Traceback\s*\(most recent call last\)", re.IGNORECASE),
    re.compile(r"File\s+\"[^\"]+\"\s*,\s*line\s+\d+", re.IGNORECASE),
    re.compile(r"\bgoroutine\s+\d+", re.IGNORECASE),
    re.compile(r"^\s*at\s+\S+\s+\(.+:\d+:\d+\)", re.MULTILINE),
    re.compile(r"\bCaused by:\b", re.IGNORECASE),
)

# Strong code / shell-command fingerprints.
_CODE_MARKERS = (
    re.compile(r"\bdef\s+\w+\s*\("),
    re.compile(r"\bfunction\s+\w+\s*\("),
    re.compile(r"\bclass\s+\w+\b"),
    re.compile(r"\bimport\s+\w+"),
    re.compile(r"\bfrom\s+\w+\s+import\b"),
    re.compile(r"^\s*#!/"),
    re.compile(r"\bpublic\s+(?:static\s+)?(?:void|class|int|String)\b"),
    re.compile(r"=>\s*\{"),
    re.compile(r"\b(?:npm|yarn|pnpm)\s+(?:install|run|add|test)\b"),
    re.compile(r"\bpip(?:3)?\s+install\b"),
    re.compile(r"\bgit\s+(?:clone|commit|push|pull|checkout)\b"),
    re.compile(r"\b(?:sudo|brew|curl|wget|docker|kubectl)\s"),
)


SECRET_KEYWORDS = frozenset(
    {
        "password",
        "passwd",
        "api_key",
        "apikey",
        "api key",
        "secret",
        "token",
        "ssn",
        "credit card number",
        "private key",
        "access key",
        "bearer",
        "authorization",
        "client_secret",
    }
)

PAYMENT_KEYWORDS = frozenset(
    {
        "card number",
        "cvv",
        "cvc",
        "expiry",
        "routing number",
        "account number",
        "iban",
        "billing address",
        "swift code",
    }
)


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------


def shannon_entropy(text: str) -> float:
    """Bits-per-character Shannon entropy of a string."""
    if not text:
        return 0.0
    n = len(text)
    counts = Counter(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def looks_like_high_entropy(text: str, *, threshold: float = 4.8, min_len: int = 32) -> bool:
    """Flag a single long, whitespace-free token with unusually high entropy."""
    cleaned = text.strip()
    if len(cleaned) < min_len or " " in cleaned:
        return False
    return shannon_entropy(cleaned) >= threshold


def looks_like_base64(text: str) -> bool:
    cleaned = text.strip()
    if not _BASE64_TOKEN_RE.match(cleaned):
        return False
    try:
        decoded = base64.b64decode(cleaned, validate=True)
    except (binascii.Error, ValueError):
        return False
    if len(decoded) < 16:
        return False
    printable = sum(byte in {9, 10, 13} or 32 <= byte < 127 for byte in decoded)
    return printable / len(decoded) >= 0.8


def looks_like_stack_trace(text: str) -> bool:
    if "\n" not in text and len(text) < 200:
        return False
    return any(p.search(text) for p in _STACK_TRACE_PATTERNS)


def looks_like_code(text: str) -> bool:
    return any(p.search(text) for p in _CODE_MARKERS)


def has_url_with_secret(text: str) -> bool:
    return bool(_URL_QUERY_SECRET_RE.search(text))


def detect_pii(text: str) -> list[str]:
    """Return a list of PII categories present in ``text``."""
    reasons: list[str] = []
    if _EMAIL_RE.search(text):
        reasons.append("email")
    if _CC_RE.search(text):
        reasons.append("card-number")
    if _PHONE_RE.search(text):
        reasons.append("phone")
    if _ADDRESS_RE.search(text) or _POBOX_RE.search(text):
        reasons.append("address")
    return reasons


def _contains_keyword(text: str, keywords: frozenset[str]) -> bool:
    normalized = _normalize(text)
    return any(_normalize(k) in normalized for k in keywords)


def contains_binary(text: str) -> bool:
    return any(ord(ch) < 9 or (13 < ord(ch) < 32) or ord(ch) == 0x7F for ch in text)


# ---------------------------------------------------------------------------
# Category classification
# ---------------------------------------------------------------------------


_CATEGORY_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        (
            "we always", "always do", "standard", "convention", "rule of thumb",
            "we do", "we usually", "the team always", "normally", "as a rule",
        ),
        "convention",
    ),
    (
        (
            "i prefer", "prefer ", "i'd prefer", "i like", "i want", "i don't like",
            "i dislike", "always use", "never use", "use ... instead", "my workflow",
            "i'd rather", "please use", "default to",
        ),
        "preference",
    ),
    (
        (
            "decided", "decision", "we will", "i chose", "going with", "go with",
            "let's go with", "resolved", "we'll go", "final answer", "we are going",
        ),
        "decision",
    ),
    (
        (
            "correction", "actually", "that's wrong", "no, ", "not ... but",
            "scratch that", "instead of", "i meant", "fix that", "that's incorrect",
        ),
        "correction",
    ),
    (
        (
            "lesson", "learned", "root cause", "turned out", "the problem was",
            "took me", "avoid", "what went wrong", "the fix was", "it was caused by",
        ),
        "lesson",
    ),
)


def classify(text: str) -> str:
    """Assign a coarse, deterministic category for weak-supervision labelling."""
    lowered = text.lower()
    for markers, category in _CATEGORY_RULES:
        if any(m in lowered for m in markers):
            return category
    return "other"


# ---------------------------------------------------------------------------
# Scrubber
# ---------------------------------------------------------------------------


def scrub_sensitive(text: str) -> str:
    """Replace absolute paths and original ids with opaque placeholders."""
    out = _POSIX_PATH_RE.sub("<path>", text)
    out = _WINDOWS_PATH_RE.sub("<path>", out)
    out = _UUID_RE.sub("<id>", out)
    out = _LONG_HEX_RE.sub("<id>", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@dataclass
class RedactResult:
    accept: bool
    reason: str | None
    sanitized: str | None
    category: str | None


_ROLE_REJECT = {
    "assistant": "assistant_output",
    "system": "system_instruction",
    "developer": "system_instruction",
    "tool": "tool_output",
    "tool_output": "tool_output",
    "tool_result": "tool_output",
}


def redact(
    text: str | None,
    *,
    role: str | None = None,
    require_durable: bool = True,
) -> RedactResult:
    """Classify and sanitize one candidate text block.

    Returns an :class:`RedactResult`; ``accept=False`` means the block was
    dropped under the default-deny policy, with ``reason`` describing why.
    """
    if text is None or not str(text).strip():
        return RedactResult(False, "empty", None, None)

    raw = str(text)

    if role and role in _ROLE_REJECT:
        return RedactResult(False, _ROLE_REJECT[role], None, None)

    if contains_binary(raw):
        return RedactResult(False, "binary", None, None)

    harmful, reason = is_harmful_content(raw)
    if harmful:
        return RedactResult(False, f"pii-or-secret:{reason}", None, None)

    pii = detect_pii(raw)
    if pii:
        return RedactResult(False, "pii:" + "+".join(sorted(set(pii))), None, None)

    if _contains_keyword(raw, SECRET_KEYWORDS):
        return RedactResult(False, "credential", None, None)

    if _contains_keyword(raw, PAYMENT_KEYWORDS):
        return RedactResult(False, "payment", None, None)

    if looks_like_base64(raw):
        return RedactResult(False, "base64", None, None)

    if looks_like_high_entropy(raw):
        return RedactResult(False, "high_entropy", None, None)

    if looks_like_stack_trace(raw):
        return RedactResult(False, "stack_trace", None, None)

    if looks_like_code(raw):
        return RedactResult(False, "code", None, None)

    if has_url_with_secret(raw):
        return RedactResult(False, "url_secret", None, None)

    sanitized = scrub_sensitive(raw)
    if not sanitized or sanitized in {"<path>", "<id>"} or re.fullmatch(r"(<path>|<id>\s*)+", sanitized):
        return RedactResult(False, "path_or_id_only", None, None)

    category = classify(sanitized)
    if require_durable and category == "other":
        return RedactResult(False, "not_durable", None, None)
    return RedactResult(True, None, sanitized, category)
