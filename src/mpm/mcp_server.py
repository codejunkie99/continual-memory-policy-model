"""MCP adapter exposing the memory system to Codex and Claude Code.

The :class:`MemoryService` is a plain, dependency-free service object whose five
tools (:func:`search`, :func:`observe`, :func:`get`, :func:`feedback`, and
:func:`status`) are directly callable, which keeps them fully testable without
an MCP transport.  The FastMCP server is built lazily so importing this module
never requires the optional ``mcp`` package.

Privacy and safety contracts reused here:
* Harmful (PII / secret) content is refused before any model call or write.
* Model loading is lazy; the baseline backend is dependency-free.
* Tool output never exposes raw database or adapter paths.
"""

from __future__ import annotations

import argparse
import os
import uuid
from typing import Any

from .baseline import BaselinePolicy, jaccard
from .credit import attribute_outcome
from .features import extract_features, tokenize
from .mlx_policy import FakeMLXBackend, MlxBackend, MLXPolicy
from .store import MemoryStore
from .types import PayloadError, validate_action


def _lexical_score(query: str, text: str) -> float:
    """Dependency-free lexical relevance: token overlap plus substring bonus."""
    qt = tokenize(query)
    mt = tokenize(text)
    overlap = jaccard(qt, mt)
    bonus = 0.5 if query.strip().lower() in text.lower() else 0.0
    return round(min(1.0, overlap + 0.3 * bonus), 6)


class MemoryService:
    """Callable service layer implementing the five MCP tools."""

    def __init__(
        self,
        store: MemoryStore,
        policy: Any,
        *,
        adapter: str | None = None,
        model: str | None = None,
        session_id: str | None = None,
    ):
        self.store = store
        self.policy = policy
        self.adapter = adapter
        self.model = model
        self._session_id = session_id or f"s-mcp-{uuid.uuid4().hex}"
        self.store.ensure_session(self._session_id, "mcp", "user")

    # -- memory_search ---------------------------------------------------

    def search(self, query: str, *, scope: str | None = None, limit: int = 10) -> dict[str, Any]:
        """Deterministic, scoped, bounded retrieval; logs one retrieval per hit."""
        limit = max(1, min(int(limit), 100))
        query = str(query or "")
        if not query.strip():
            return {"retrieval_ids": [], "results": []}

        memories = self.store.active_memories()
        if scope:
            memories = [m for m in memories if m.get("scope") == scope]

        scored = []
        for memory in memories:
            score = _lexical_score(query, memory["content"])
            if score > 0:
                scored.append((score, memory))
        # Stable ordering: highest score first, then memory_id for determinism.
        scored.sort(key=lambda pair: (-pair[0], pair[1]["memory_id"]))

        retrieval_ids: list[str] = []
        results: list[dict[str, Any]] = []
        for rank, (score, mem) in enumerate(scored[:limit]):
            rid = self.store.record_retrieval(
                self._session_id,
                mem["memory_id"],
                extract_features(query, scope=scope or "default"),
                rank=rank,
                score=score,
            )
            retrieval_ids.append(rid)
            results.append(
                {"memory_id": mem["memory_id"], "score": score, "scope": mem.get("scope"), "content": mem["content"]}
            )
        return {"retrieval_ids": retrieval_ids, "results": results}

    # -- memory_observe --------------------------------------------------

    def observe(
        self,
        content: str,
        *,
        scope: str = "default",
        intent: str | None = None,
        context_ids: list[str] | None = None,
        source_trust: str | None = None,
    ) -> dict[str, Any]:
        """Choose and execute a safe memory action for one observation."""
        observation: dict[str, Any] = {
            "content": content,
            "scope": scope,
            "intent": intent,
            "context_ids": [c for c in (context_ids or []) if isinstance(c, str)],
            "source_trust": source_trust,
        }
        action = self.policy.decide(observation, self.store)
        errors = validate_action(action)
        if errors:
            return {"op": action.op, "ok": False, "error": "; ".join(errors)}
        try:
            result = self.store.apply_action(
                action,
                session_id=self._session_id,
                features=extract_features(str(content or ""), scope=str(scope)),
                observation=observation,
            )
        except (PayloadError, KeyError, ValueError) as exc:
            return {"op": action.op, "ok": False, "error": str(exc)}
        return {"op": action.op, "ok": True, **result}

    # -- memory_get ------------------------------------------------------

    def get(self, memory_id: str) -> dict[str, Any]:
        mem = self.store.get_memory(memory_id)
        if mem is None:
            return {"memory_id": memory_id, "found": False}
        return {
            "memory_id": memory_id,
            "found": True,
            "status": mem["status"],
            "scope": mem.get("scope"),
            "content": mem["content"],
        }

    # -- memory_feedback -------------------------------------------------

    def feedback(
        self,
        retrieval_id: str,
        *,
        kind: str = "positive",
        value: float = 1.0,
        confidence: float = 1.0,
    ) -> dict[str, Any]:
        try:
            outcome_id = self.store.record_outcome(
                self._session_id,
                retrieval_id=retrieval_id,
                kind=kind,
                value=float(value),
                confidence=float(confidence),
            )
        except PayloadError as exc:
            return {"ok": False, "error": str(exc)}
        written = attribute_outcome(self.store, outcome_id)
        return {
            "ok": True,
            "outcome_id": outcome_id,
            "attributions": [{"memory_id": a["memory_id"], "reward": a["reward"]} for a in written],
        }

    # -- memory_status ---------------------------------------------------

    def status(self) -> dict[str, Any]:
        return {
            "policy": self.policy.name,
            "adapter": "configured" if self.adapter else None,
            "model": self.model,
            "active_memories": len(self.store.active_memories()),
            "counts": self.store.counts(),
        }


def create_service(
    db: str,
    *,
    policy: str = "baseline",
    backend: str = "fake",
    adapter: str | None = None,
    model: str = "LiquidAI/LFM2.5-VL-3B-MLX-8bit",
) -> MemoryService:
    """Construct a service with a lazy, configurable policy backend."""
    store = MemoryStore(db)
    if policy == "mlx":
        if backend == "fake":
            pol = MLXPolicy(FakeMLXBackend())
        else:
            pol = MLXPolicy(MlxBackend(model, adapter_path=adapter))
    else:
        pol = BaselinePolicy()
    return MemoryService(store, pol, adapter=adapter, model=model)


def build_fastmcp(
    service: MemoryService,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> Any:
    """Build and return a FastMCP server bound to the service (lazy import)."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "the MCP server requires the optional 'mcp' extra; install with "
            "`pip install 'mpm[mcp]'`"
        ) from exc

    mcp = FastMCP("mpm-memory", host=host, port=port)

    @mcp.tool()
    def memory_search(query: str, scope: str | None = None, limit: int = 10) -> dict[str, Any]:
        return service.search(query, scope=scope, limit=limit)

    @mcp.tool()
    def memory_observe(
        content: str,
        scope: str = "default",
        intent: str | None = None,
        context_ids: list[str] | None = None,
        source_trust: str | None = None,
    ) -> dict[str, Any]:
        return service.observe(
            content, scope=scope, intent=intent, context_ids=context_ids, source_trust=source_trust
        )

    @mcp.tool()
    def memory_get(memory_id: str) -> dict[str, Any]:
        return service.get(memory_id)

    @mcp.tool()
    def memory_feedback(
        retrieval_id: str,
        kind: str = "positive",
        value: float = 1.0,
        confidence: float = 1.0,
    ) -> dict[str, Any]:
        return service.feedback(retrieval_id, kind=kind, value=value, confidence=confidence)

    @mcp.tool()
    def memory_status() -> dict[str, Any]:
        return service.status()

    return mcp


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mpm-mcp", description="Memory Policy Model MCP server")
    p.add_argument("--db", default=os.environ.get("MPM_DB", "outputs/mpm.db"))
    p.add_argument("--adapter", default=os.environ.get("MPM_ADAPTER"))
    p.add_argument("--model", default=os.environ.get("MPM_MODEL", "LiquidAI/LFM2.5-VL-3B-MLX-8bit"))
    p.add_argument("--policy", choices=["baseline", "mlx"], default=os.environ.get("MPM_POLICY", "baseline"))
    p.add_argument("--backend", choices=["fake", "mlx"], default=os.environ.get("MPM_BACKEND", "fake"))
    p.add_argument("--transport", choices=["stdio", "http"], default=os.environ.get("MPM_TRANSPORT", "stdio"))
    p.add_argument("--host", default=os.environ.get("MPM_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("MPM_PORT", "8000")))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    service = create_service(
        args.db,
        policy=args.policy,
        backend=args.backend,
        adapter=args.adapter,
        model=args.model,
    )
    mcp = build_fastmcp(service, host=args.host, port=args.port)
    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="streamable-http")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
