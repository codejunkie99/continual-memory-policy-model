"""Command-line interface for the Memory Policy Model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .baseline import BaselinePolicy
from .benchmark import FakeClock, run_benchmark
from .credit import reconcile_all
from .consolidate import build_consolidation
from .eval import PromotionGate, evaluate
from .export import export_all
from .live_eval import run_live_evaluation
from .mlx_policy import FakeMLXBackend, MlxBackend, MLXPolicy
from .store import MemoryStore
from .train.config import TrainConfig
from .train.dryrun import dry_run
from .types import Action, Op


def _write_report(name: str, data: dict[str, Any], outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / name
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


class AlwaysWritePolicy:
    """Naive candidate: write everything, including harmful content (unsafe)."""

    name = "always-write"

    def decide(self, observation: dict[str, Any], store: MemoryStore) -> Action:
        return Action(
            op=Op.WRITE.value,
            payload={"content_ref": "observation.content"},
            confidence=0.9,
            rationale="candidate: always persist",
            policy_version=self.name,
        )


class NoopPolicy:
    """Naive candidate: never store anything (safe but useless)."""

    name = "noop-all"

    def decide(self, observation: dict[str, Any], store: MemoryStore) -> Action:
        return Action(op=Op.NOOP.value, payload={"reason": "noop-all"}, confidence=1.0, policy_version=self.name)


def _cmd_init(args: argparse.Namespace) -> int:
    store = MemoryStore(args.db)
    print(f"initialized database at {args.db}")
    print(json.dumps(store.counts(), indent=2))
    store.close()
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    db = Path(args.db)
    clock = FakeClock()
    store = MemoryStore(db, clock=clock)
    metrics = run_benchmark(seed=args.seed, store=store, clock=clock)
    metrics["long_delay_reward_a"] = metrics["long_delay"]["reward_a"]
    metrics["long_delay_reward_b"] = metrics["long_delay"]["reward_b"]
    store.close()
    path = _write_report("demo.json", metrics, outdir)
    print(f"demo complete -> {path}")
    print(json.dumps(metrics, indent=2, sort_keys=True, default=str))
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    outdir = Path(args.outdir)
    store = MemoryStore(args.db)
    reconcile_all(store)
    report = export_all(store, outdir, allow_raw=args.unsafe_raw, seed=args.seed)
    store.close()
    path = _write_report("export.json", report, outdir)
    print(f"export complete -> {path}")
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    outdir = Path(args.outdir)
    store = MemoryStore(args.db)
    baseline = BaselinePolicy()
    current = evaluate(store, baseline, seed=args.seed)

    # Candidate policy evaluation (on a separate store to avoid cross-state).
    cand_store = MemoryStore(":memory:")
    candidate_policy = AlwaysWritePolicy()
    candidate = evaluate(cand_store, candidate_policy, seed=args.seed)
    cand_store.close()

    gate = PromotionGate()
    decision = gate.decide(candidate, current)

    # Persist checkpoints + gate outcome into the audit store.
    store.add_checkpoint("baseline-v1", "heuristic baseline", metrics=current, status="active", gate_approved=True, note="current policy")
    candidate_note = "passed gate; awaiting explicit promotion" if decision["approved"] else None
    store.add_checkpoint(
        "candidate-v1",
        "always-write candidate",
        metrics=candidate,
        note=candidate_note,
        gate_approved=bool(decision["approved"]),
    )
    if not decision["approved"]:
        store.reject_checkpoint("candidate-v1", note="rejected by promotion gate; baseline remains active")

    report = {
        "current": current,
        "candidate": candidate,
        "promotion_gate": decision,
        "checkpoints": store.list_checkpoints(),
    }
    store.close()
    path = _write_report("evaluation.json", report, outdir)
    print(f"evaluation complete -> {path}")
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    store = MemoryStore(args.db)
    events = store.audit_events(
        since_seq=args.since,
        limit=args.limit,
        event_types=args.types or None,
    )
    store.close()
    for e in events:
        print(json.dumps(e, sort_keys=True, default=str))
    return 0


def _cmd_dry_run(args: argparse.Namespace) -> int:
    cfg = TrainConfig(model_name=args.model)
    report = dry_run(args.data_dir, cfg, outdir=args.outdir, seed=args.seed)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["status"] == "ok" else 1


def _cmd_promote(args: argparse.Namespace) -> int:
    store = MemoryStore(args.db)
    store.activate_checkpoint(args.version, note="explicit promotion")
    active = store.get_active_checkpoint()
    store.close()
    print(f"promoted {args.version}; active checkpoint: {json.dumps(active, default=str)}")
    return 0


def _cmd_rollback(args: argparse.Namespace) -> int:
    store = MemoryStore(args.db)
    store.rollback_checkpoint(args.version, note="explicit rollback")
    active = store.get_active_checkpoint()
    store.close()
    print(f"rolled back to {args.version}; active checkpoint: {json.dumps(active, default=str)}")
    return 0


def _cmd_consolidate(args: argparse.Namespace) -> int:
    report = build_consolidation(
        args.current_dir,
        args.replay_dir,
        args.outdir,
        max_replay_per_kind=args.max_replay,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _build_live_policy(args: argparse.Namespace):
    """Construct the policy for a live trajectory run without eager MLX imports."""
    if args.policy == "baseline":
        return BaselinePolicy()
    if args.backend == "fake":
        return MLXPolicy(FakeMLXBackend())
    return MLXPolicy(MlxBackend(args.model, adapter_path=args.adapter))


def _cmd_live_evaluate(args: argparse.Namespace) -> int:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    clock = FakeClock()
    store = MemoryStore(args.db, clock=clock)
    policy = _build_live_policy(args)
    metrics = run_live_evaluation(store, policy, seed=args.seed, clock=clock)

    baseline_clock = FakeClock()
    baseline_store = MemoryStore(":memory:", clock=baseline_clock)
    current = run_live_evaluation(
        baseline_store, BaselinePolicy(), seed=args.seed, clock=baseline_clock
    )
    baseline_store.close()
    gate = PromotionGate()
    decision = gate.decide(metrics, current)

    store.add_checkpoint(
        args.version,
        f"{policy.name} live trajectory",
        metrics=metrics,
        artifact_path=args.adapter if args.policy == "mlx" else None,
        note=("passed full live gate" if decision["approved"] else "failed full live gate"),
        gate_approved=bool(decision["approved"]),
    )
    if args.activate:
        if not decision["approved"]:
            raise RuntimeError("cannot activate: live promotion gate failed")
        store.activate_checkpoint(args.version, note="explicit live activation")
    active = store.get_active_checkpoint()
    store.close()
    report = {
        **metrics,
        "current_baseline": current,
        "promotion_gate": decision,
        "checkpoint_version": args.version,
        "activated": bool(active and active["version"] == args.version),
    }
    path = _write_report("live-evaluation.json", report, outdir)
    print(f"live evaluation complete -> {path}")
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if decision["approved"] else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mpm", description="Memory Policy Model")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("init", help="initialize a fresh event-sourced database")
    sp.add_argument("--db", default="outputs/mpm.db")
    sp.set_defaults(func=_cmd_init)

    sp = sub.add_parser("demo", help="run the synthetic benchmark")
    sp.add_argument("--db", default="outputs/mpm.db")
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--outdir", default="outputs")
    sp.set_defaults(func=_cmd_demo)

    sp = sub.add_parser("export", help="export privacy-safe SFT/DPO datasets")
    sp.add_argument("--db", default="outputs/mpm.db")
    sp.add_argument("--outdir", default="outputs")
    sp.add_argument("--seed", default="mpm-v1")
    sp.add_argument("--unsafe-raw", action="store_true", help="include raw user content (explicit, unsafe)")
    sp.set_defaults(func=_cmd_export)

    sp = sub.add_parser("evaluate", help="run the evaluation harness + promotion gate")
    sp.add_argument("--db", default="outputs/mpm.db")
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--outdir", default="outputs")
    sp.set_defaults(func=_cmd_evaluate)

    sp = sub.add_parser("inspect", help="dump audit events")
    sp.add_argument("--db", default="outputs/mpm.db")
    sp.add_argument("--since", type=int, default=0)
    sp.add_argument("--limit", type=int, default=100)
    sp.add_argument("--types", nargs="*")
    sp.set_defaults(func=_cmd_inspect)

    sp = sub.add_parser("train-dry-run", help="validate datasets/config without ML imports")
    sp.add_argument("--data-dir", required=True)
    sp.add_argument("--outdir", default="outputs")
    sp.add_argument("--model", default="LiquidAI/LFM2.5-VL-3B")
    sp.add_argument("--seed", default="mpm-v1")
    sp.set_defaults(func=_cmd_dry_run)

    sp = sub.add_parser("promote", help="explicitly activate a checkpoint")
    sp.add_argument("--db", default="outputs/mpm.db")
    sp.add_argument("--version", required=True)
    sp.set_defaults(func=_cmd_promote)

    sp = sub.add_parser("rollback", help="explicitly roll back to a checkpoint")
    sp.add_argument("--db", default="outputs/mpm.db")
    sp.add_argument("--version", required=True)
    sp.set_defaults(func=_cmd_rollback)

    sp = sub.add_parser("consolidate", help="mix new trajectories with bounded historical replay")
    sp.add_argument("--current-dir", required=True)
    sp.add_argument("--replay-dir", action="append", default=[], help="historical export directory; repeatable")
    sp.add_argument("--outdir", default="outputs/consolidation")
    sp.add_argument("--max-replay", type=int, default=1000)
    sp.add_argument("--seed", default="mpm-consolidation-v1")
    sp.set_defaults(func=_cmd_consolidate)

    sp = sub.add_parser("live-evaluate", help="run an end-to-end live trajectory evaluation")
    sp.add_argument("--db", default="outputs/mpm.db")
    sp.add_argument("--policy", choices=["baseline", "mlx"], default="baseline")
    sp.add_argument("--backend", choices=["fake", "mlx"], default="fake", help="MLX backend selection (fake avoids MLX imports)")
    sp.add_argument("--model", default="LiquidAI/LFM2.5-VL-3B-MLX-8bit")
    sp.add_argument("--adapter", default="outputs/mlx-lora-v4")
    sp.add_argument("--version", default="mlx-live-candidate")
    sp.add_argument("--activate", action="store_true", help="explicitly activate only if the full live gate passes")
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--outdir", default="outputs")
    sp.set_defaults(func=_cmd_live_evaluate)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
