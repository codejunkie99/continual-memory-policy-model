"""Experimental Transformers/PEFT path (16 GB Apple M5), fail-safe by design.

This is *not* the recommended path.  The supported path is Liquid's LEAP
fine-tuner, which requires visible CUDA devices and does not train on Apple
silicon.  This module exists only as a guarded, clearly-labelled experiment for
a 16 GB Apple M5: every heavy import is lazy, every failure returns a structured
error instead of crashing, and training never auto-promotes an adapter.

Important caveat: the public checkpoint requires Transformers 5 or newer, and
official LEAP local training still requires CUDA. Official MLX checkpoints only
prove *inference* through ``mlx-vlm``, not training.
"""

from __future__ import annotations

import platform
import sys
from typing import Any

from .config import TrainConfig


class ExperimentalTrainingError(RuntimeError):
    """Raised (or caught and reported) when the experimental path cannot proceed."""


def platform_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
    }
    try:
        import torch  # type: ignore

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_device_count"] = torch.cuda.device_count()
        info["mps_available"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception as exc:  # pragma: no cover - import guard
        info["torch"] = None
        info["import_error"] = str(exc)
    return info


def _require_imports() -> tuple[Any, Any, Any]:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import peft  # noqa: F401
    except Exception as exc:
        raise ExperimentalTrainingError(
            "Experimental training deps are missing. Install `mpm[train]` "
            f"or run the dependency-free `--dry-run` instead. ({exc})"
        ) from exc
    import torch
    import transformers
    import peft
    return torch, transformers, peft


def apply_lora(model: Any, cfg: TrainConfig, peft: Any) -> Any:
    """Apply LoRA to LFM2.5's documented target modules and freeze everything else."""
    from peft import LoraConfig, get_peft_model  # type: ignore

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def prepare_model(cfg: TrainConfig, torch: Any, transformers: Any, peft: Any) -> dict[str, Any]:
    """Load the model, freeze the vision encoder by default, and apply LoRA."""
    if not torch.cuda.is_available():
        raise ExperimentalTrainingError(
            "No CUDA device visible. Official LEAP training requires CUDA; the "
            "Transformers/PEFT path here is only an experiment for an Apple M5 "
            "MPS backend, which is not a supported LEAP target."
        )
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(cfg.model_name, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            cfg.model_name,
            dtype=torch.bfloat16,
        )
    except Exception as exc:
        raise ExperimentalTrainingError(
            f"Failed to load {cfg.model_name}. LFM2.5 requires a matching "
            f"Transformers build and attention implementation. ({exc})"
        ) from exc

    if cfg.freeze_vision:
        # SigLIP2 vision tower is frozen for the first text-policy phase.
        for n, p in model.named_parameters():
            if "vision" in n or "siglip" in n:
                p.requires_grad = False

    model = apply_lora(model, cfg, peft)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"model": model, "processor": processor, "trainable_params": trainable}


def run_experimental(cfg: TrainConfig, train_file: str, *, max_steps: int = 10) -> dict[str, Any]:
    """Guarded entry point.  Returns a status dict and never raises."""
    try:
        torch, transformers, peft = _require_imports()
    except ExperimentalTrainingError as exc:
        return {"status": "blocked", "reason": str(exc), "platform": platform_info()}

    try:
        prepared = prepare_model(cfg, torch, transformers, peft)
    except ExperimentalTrainingError as exc:
        return {"status": "blocked", "reason": str(exc), "platform": platform_info()}

    # A deliberately minimal, auditable smoke loop.  Real training is out of
    # scope for this entry point and is left to LEAP / the main agent.
    return {
        "status": "prepared",
        "model": cfg.model_name,
        "trainable_params": prepared["trainable_params"],
        "vision_frozen": cfg.freeze_vision,
        "max_steps": max_steps,
        "auto_promote": False,
        "note": "adapter is never auto-promoted; use the evaluation gate + explicit promotion.",
    }
