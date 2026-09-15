"""Dependency-free training dry-run.

Validates datasets, config, deterministic splits, LoRA budgets, and the LEAP
config/contract without importing any ML framework.  This is the safe first
gate before any real (CUDA LEAP or experimental PEFT) training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..types import ALL_OPS
from .config import (
    TrainConfig,
    MODEL_PROFILES,
    estimate_lora_params,
    validate_trainable_params,
)
from .leap import (
    build_dataset_contract,
    build_modal_leap_config,
    render_config_yaml,
    render_leap_yaml,
    to_lfm_messages,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{i} invalid JSON: {exc}") from exc
    return rows


def _validate_sft(row: dict[str, Any], path: Path, i: int) -> list[str]:
    errors: list[str] = []
    for key in ("split", "features", "label"):
        if key not in row:
            errors.append(f"{path}:{i} missing {key!r}")
    if row.get("label") not in ALL_OPS:
        errors.append(f"{path}:{i} invalid label {row.get('label')!r}")
    if row.get("kind") != "sft":
        errors.append(f"{path}:{i} unexpected kind {row.get('kind')!r}")
    return errors


def _validate_dpo(row: dict[str, Any], path: Path, i: int) -> list[str]:
    errors: list[str] = []
    for key in ("chosen", "rejected", "features"):
        if key not in row:
            errors.append(f"{path}:{i} missing {key!r}")
    if row.get("kind") != "preference":
        errors.append(f"{path}:{i} unexpected kind {row.get('kind')!r}")
    return errors


def dry_run(
    data_dir: str | Path,
    cfg: TrainConfig | None = None,
    *,
    outdir: str | Path = "outputs",
    seed: str = "mpm-v1",
) -> dict[str, Any]:
    """Validate everything and return a report dict."""
    cfg = cfg or TrainConfig()
    data_dir = Path(data_dir).resolve()
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {"config": cfg.to_dict()}
    errors: list[str] = []

    # 1. Config validation.
    errors.extend(cfg.validate())

    # 2. Dataset presence and schema.
    splits = ["train", "val", "test"]
    totals: dict[str, int] = {}
    dataset_errors: list[str] = []
    groups_by_split: dict[str, set[str]] = {split: set() for split in splits}
    example_count = 0
    sft_rows_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in splits}
    for split in splits:
        sft_path = data_dir / f"sft-{split}.jsonl"
        dpo_path = data_dir / f"dpo-{split}.jsonl"
        if not sft_path.exists():
            dataset_errors.append(f"missing {sft_path}")
            continue
        if not dpo_path.exists():
            dataset_errors.append(f"missing {dpo_path}")
        sft_rows = _read_jsonl(sft_path) if sft_path.exists() else []
        dpo_rows = _read_jsonl(dpo_path) if dpo_path.exists() else []
        totals[f"sft-{split}"] = len(sft_rows)
        totals[f"dpo-{split}"] = len(dpo_rows)
        for i, row in enumerate(sft_rows, 1):
            dataset_errors.extend(_validate_sft(row, sft_path, i))
            group_id = str(row.get("group_id", ""))
            if not group_id:
                dataset_errors.append(f"{sft_path}:{i} missing 'group_id'")
            groups_by_split[split].add(group_id)
            sft_rows_by_split[split].append(row)
            example_count += 1
        for i, row in enumerate(dpo_rows, 1):
            dataset_errors.extend(_validate_dpo(row, dpo_path, i))
            group_id = str(row.get("group_id", ""))
            if not group_id:
                dataset_errors.append(f"{dpo_path}:{i} missing 'group_id'")
            groups_by_split[split].add(group_id)
            example_count += 1

    # 3. Deterministic / leak-safe split re-check.
    split_mismatches = 0
    all_groups = set().union(*groups_by_split.values())
    for group_id in all_groups:
        if sum(group_id in groups for groups in groups_by_split.values()) > 1:
            split_mismatches += 1

    if split_mismatches:
        dataset_errors.append(f"{split_mismatches} scenario/user groups leaked across splits")

    # 4. LoRA budget estimate.
    trainable = estimate_lora_params(cfg)
    profile = MODEL_PROFILES.get(cfg.model_name)
    budget_errors = validate_trainable_params(trainable, total_params=profile["total_params"] if profile else None)
    dataset_errors.extend(budget_errors)

    # 5. LEAP config + contract render.
    for split in ("train", "val"):
        leap_dataset = data_dir / f"leap-vlm-sft-{split}.jsonl"
        with leap_dataset.open("w", encoding="utf-8") as f:
            for row in sft_rows_by_split[split]:
                f.write(json.dumps(to_lfm_messages(row), sort_keys=True) + "\n")
    leap_yaml = render_leap_yaml(cfg, str(data_dir))
    contract = build_dataset_contract()
    leap_out = outdir / "leap-finetune.yaml"
    leap_out.write_text(leap_yaml, encoding="utf-8")
    modal_out = outdir / "leap-finetune-modal.yaml"
    modal_out.write_text(render_config_yaml(build_modal_leap_config(cfg)), encoding="utf-8")
    modal_upload = outdir / "modal-upload-commands.txt"
    modal_upload.write_text(
        "# Run only after Modal authentication and cloud-spend authorization.\n"
        "uvx modal setup\n"
        "uvx modal volume create mpm-lfm25-vl\n"
        f"uvx modal volume put -f mpm-lfm25-vl {data_dir / 'leap-vlm-sft-train.jsonl'} /mpm-data/leap-vlm-sft-train.jsonl\n"
        f"uvx modal volume put -f mpm-lfm25-vl {data_dir / 'leap-vlm-sft-val.jsonl'} /mpm-data/leap-vlm-sft-val.jsonl\n",
        encoding="utf-8",
    )
    contract_out = outdir / "dataset-contract.json"
    contract_out.write_text(json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8")

    # 6. Demonstrate the LFM2.5 chat/tool mapping on one synthetic example.
    sample_messages = None
    if example_count > 0:
        first_path = data_dir / "sft-train.jsonl"
        if first_path.exists():
            rows = _read_jsonl(first_path)
            if rows:
                sample_messages = to_lfm_messages(rows[0])

    report.update(
        {
            "status": "ok" if not errors and not dataset_errors else "invalid",
            "config_errors": errors,
            "dataset_errors": dataset_errors,
            "totals": totals,
            "example_count": example_count,
            "scenario_user_groups": len(all_groups),
            "split_mismatches": split_mismatches,
            "trainable_params_estimate": trainable,
            "model_profile": profile,
            "leap_config": leap_out.name,
            "leap_modal_config": modal_out.name,
            "modal_upload_commands": modal_upload.name,
            "dataset_contract": contract_out.name,
            "sample_messages": sample_messages,
        }
    )
    report_path = outdir / "train-dry-run.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
