"""MLX-backed policy driving the stateful memory runtime.

This module wires the locally trained MLX LoRA adapter (``outputs/mlx-lora-v4``)
into the same :class:`~mpm.types.Action` contract the heuristic baseline emits,
so a learned policy can replace the baseline without changing the runtime.

Privacy contract
----------------
* Raw observation content is *never* placed in the model prompt.  The prompt is
  built only from :func:`~mpm.features.extract_features` (numeric/boolean
  summaries) plus positional ``candidate_i`` aliases for the current memory
  state, matching the privacy-safe training export.
* PII / secrets are blocked *before* the model is ever called: harmful content
  short-circuits to a ``NOOP`` refusal and is never stored.
* The model output contributes only the operation *op*.  The executable
  ``payload`` is reconstructed deterministically by this trusted policy from the
  observation's structured ``context_ids`` and ``content_ref``, so a model can
  never inject raw content, arbitrary memory ids, or otherwise-unvalidated data.
* Every output is validated with :func:`~mpm.types.validate_action` before it is
  returned to the executor.

Imports are lazy and optional: importing this module, or constructing a policy
with :class:`FakeMLXBackend`, requires no MLX packages.  Only constructing and
using :class:`MlxBackend` imports ``mlx`` / ``mlx_vlm``.
"""

from __future__ import annotations

import re
from typing import Any, Protocol

from .export import sanitize_features
from .features import content_hash, extract_features
from .safety import is_harmful_content
from .store import MemoryStore
from .types import Action, Op


_NATIVE_TOOL_CALL = re.compile(
    r"^\s*<\|tool_call_start\|>\s*\[\s*memory_action\(\s*"
    r"op\s*=\s*(['\"])(WRITE|UPDATE|DELETE|LINK|COMPACT|NOOP)\1\s*,\s*"
    r"payload\s*=\s*\{\s*\}\s*\)\s*\]\s*<\|tool_call_end\|>\s*$",
    re.IGNORECASE,
)


def parse_op(text: str) -> str | None:
    """Parse only the exact native LFM tool-call envelope.

    Prose that merely mentions an operation is rejected. This keeps arbitrary
    model text from being treated as an executable decision.
    """
    match = _NATIVE_TOOL_CALL.fullmatch(text or "")
    return match.group(2).upper() if match else None


def tool_call_text(op: str) -> str:
    """Render the LFM-native bracketed tool-call form for an op."""
    return f'<|tool_call_start|>[memory_action(op="{op}", payload={{}})]<|tool_call_end|>'


class PolicyBackend(Protocol):
    """The model-facing surface a policy depends on (kept tiny for testability)."""

    name: str

    def generate(self, observation_summary: dict[str, Any], *, max_tokens: int = 96) -> str: ...


class FakeMLXBackend:
    """Dependency-free, deterministic test double for the MLX backend.

    ``decide`` is an optional callable ``(observation_summary) -> str`` returning
    raw model text.  Every invocation is recorded in ``prompts`` so tests can
    assert privacy (no raw content ever reaches the model).
    """

    name = "fake-mlx"

    def __init__(self, decide: Any | None = None):
        self.decide_fn = decide
        self.prompts: list[dict[str, Any]] = []

    def generate(self, observation_summary: dict[str, Any], *, max_tokens: int = 96) -> str:
        self.prompts.append(observation_summary)
        if self.decide_fn is None:
            return tool_call_text(Op.NOOP.value)
        return self.decide_fn(observation_summary)


class MlxBackend:
    """Real MLX-VLM backend with lazy, optional imports and prompt templating."""

    name = "mlx"

    def __init__(
        self,
        model_path: str,
        adapter_path: str | None = None,
        *,
        max_tokens: int = 96,
        temperature: float = 0.0,
    ):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.prompts: list[dict[str, Any]] = []
        self._model: Any = None
        self._processor: Any = None
        self._config: dict[str, Any] = {}

    def _load(self) -> None:
        if self._model is not None:
            return
        from mlx_vlm import load  # type: ignore

        self._model, self._processor = load(self.model_path, adapter_path=self.adapter_path)
        self._config = self._model.config.__dict__

    def generate(self, observation_summary: dict[str, Any], *, max_tokens: int | None = None) -> str:
        self._load()
        # Record only the redacted summary: the system text and tool definition
        # are fixed constants, so the summary is the sole user-derived input.
        self.prompts.append(observation_summary)

        from mlx_vlm import generate  # type: ignore
        from mlx_vlm.prompt_utils import apply_chat_template  # type: ignore

        from .train.leap import to_lfm_messages

        messages = to_lfm_messages(
            {
                "modality": "text",
                "features": observation_summary.get("features", {}),
                "label": Op.NOOP.value,
                "action": {"op": Op.NOOP.value, "target": None, "payload": {}},
            }
        )["messages"][:2]
        prompt = apply_chat_template(
            self._processor,
            self._config,
            messages,
            add_generation_prompt=True,
            num_images=0,
            num_audios=0,
        )
        result = generate(
            self._model,
            self._processor,
            prompt,
            image=None,
            max_tokens=max_tokens or self.max_tokens,
            temperature=self.temperature,
            verbose=False,
        )
        return result.text


class MLXPolicy:
    """A policy that delegates the operation decision to an MLX backend.

    The trusted executor (this class) owns payload construction and privacy
    gating; the model only selects the operation.
    """

    name = "mlx"

    def __init__(self, backend: PolicyBackend):
        self.backend = backend
        self.prompts: list[dict[str, Any]] = []
        self.last_raw_text: str = ""
        self.last_structured_valid: bool = True

    def decide(self, observation: dict[str, Any], store: MemoryStore) -> Action:
        content = str(observation.get("content") or "")
        scope = str(observation.get("scope") or "default")
        self.last_raw_text = ""
        self.last_structured_valid = True

        # Privacy gate: refuse PII / secrets before any model call.
        harmful, reason = is_harmful_content(content)
        if harmful:
            return Action(
                op=Op.NOOP.value,
                payload={"reason": f"refuse-harmful:{reason}", "observed": {"harmful": True}},
                confidence=0.99,
                rationale="refuse to persist PII/secret",
                policy_version=self.name,
            )

        # A context-bearing observation (intent / context_ids) is actionable even
        # when its free-text content is empty (e.g. LINK, DELETE, COMPACT).
        if not content.strip() and not (observation.get("intent") or observation.get("context_ids")):
            return Action(
                op=Op.NOOP.value,
                payload={"reason": "empty-content"},
                confidence=1.0,
                rationale="nothing to store",
                policy_version=self.name,
            )

        # Build a redacted observation summary. Raw content and real memory ids
        # are never included; semantic control signals use a closed vocabulary.
        real_ids = self._candidate_ids(store, observation)
        active_hashes = {m["content_hash"] for m in store.active_memories()}
        requested = str(observation.get("intent") or "AUTO").upper()
        if requested not in {op.value for op in Op}:
            requested = "UNKNOWN"
        trust = str(observation.get("source_trust") or "unknown").lower()
        if trust not in {"high", "medium", "low", "unknown"}:
            trust = "unknown"
        safe_extra = {
            "requested_op": requested,
            "context_count": len([mid for mid in observation.get("context_ids") or [] if isinstance(mid, str)]),
            "has_content": bool(content.strip()),
            "exact_duplicate": bool(content.strip()) and content_hash(content) in active_hashes,
            "source_trust": trust,
        }
        features = sanitize_features(
            extract_features(content, scope=scope, key=observation.get("key"), extra=safe_extra),
            real_ids,
        )
        observation_summary = {"features": features}

        raw_text = self.backend.generate(observation_summary)
        self.prompts.append(observation_summary)
        self.last_raw_text = raw_text

        op = parse_op(raw_text)
        action, valid = self._build_action(op, observation, content)
        if valid:
            from .types import validate_action

            valid = not validate_action(action)
            if not valid:
                action = self._fallback("hydrated action failed validation")
        self.last_structured_valid = valid
        return action

    def _candidate_ids(self, store: MemoryStore, observation: dict[str, Any]) -> list[str]:
        """Return the deduplicated, order-stable candidate memory ids."""
        ids: list[str] = []
        seen: set[str] = set()
        for mid in observation.get("context_ids") or []:
            if isinstance(mid, str) and mid not in seen:
                seen.add(mid)
                ids.append(mid)
        for mem in store.active_memories():
            mid = mem["memory_id"]
            if mid not in seen:
                seen.add(mid)
                ids.append(mid)
        return ids

    def _build_action(self, op: str | None, observation: dict[str, Any], content: str) -> tuple[Action, bool]:
        """Reconstruct a validated action deterministically from the chosen op."""
        context_ids = [x for x in (observation.get("context_ids") or []) if isinstance(x, str)]
        content_ref = {"content_ref": "observation.content"}

        if op == Op.WRITE.value:
            if not content.strip():
                return self._fallback("model requested WRITE with empty content"), False
            return (
                Action(Op.WRITE.value, payload=dict(content_ref), confidence=0.8, rationale="mlx policy", policy_version=self.name),
                True,
            )
        if op == Op.UPDATE.value:
            if not content.strip() or not context_ids:
                return self._fallback("model requested UPDATE without content or context"), False
            return (
                Action(
                    Op.UPDATE.value,
                    target=context_ids[0],
                    payload={"memory_id": context_ids[0], **content_ref},
                    confidence=0.8,
                    rationale="mlx policy",
                    policy_version=self.name,
                ),
                True,
            )
        if op == Op.DELETE.value:
            if not context_ids:
                return self._fallback("model requested DELETE without context"), False
            return (
                Action(Op.DELETE.value, target=context_ids[0], payload={"memory_id": context_ids[0]}, confidence=0.8, rationale="mlx policy", policy_version=self.name),
                True,
            )
        if op == Op.LINK.value:
            if len(context_ids) < 2:
                return self._fallback("model requested LINK without two context ids"), False
            return (
                Action(
                    Op.LINK.value,
                    payload={"source_id": context_ids[0], "target_id": context_ids[1], "kind": "related", "weight": 0.8},
                    confidence=0.8,
                    rationale="mlx policy",
                    policy_version=self.name,
                ),
                True,
            )
        if op == Op.COMPACT.value:
            if len(context_ids) < 2:
                return self._fallback("model requested COMPACT without two context ids"), False
            return (
                Action(
                    Op.COMPACT.value,
                    payload={"memory_ids": list(context_ids), "strategy": "concat"},
                    confidence=0.8,
                    rationale="mlx policy",
                    policy_version=self.name,
                ),
                True,
            )
        if op == Op.NOOP.value:
            return (
                Action(Op.NOOP.value, payload={"reason": "policy-noop"}, confidence=1.0, rationale="mlx policy", policy_version=self.name),
                True,
            )
        return self._fallback("unparsable model output"), False

    def _fallback(self, reason: str) -> Action:
        return Action(
            op=Op.NOOP.value,
            payload={"reason": "invalid-output"},
            confidence=1.0,
            rationale=reason,
            policy_version=self.name,
        )
