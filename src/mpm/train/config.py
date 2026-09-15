"""Training configuration and adapter versioning.

Hard facts this module is allowed to state:
* The primary target model is ``LiquidAI/LFM2.5-VL-3B``.
* LFM2.5's documented LoRA target modules are the eight listed below.
* The SigLIP2 vision encoder is frozen for the first text-policy phase.
* Official LEAP local training requires visible CUDA devices (it does *not*
  train on Apple silicon).  Official MLX checkpoints only prove *inference*
  through ``mlx-vlm``; they do not prove training support.

Architecture numbers are marked ``declared`` (estimated) and must be verified
against the actual model at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


DEFAULT_MODEL = "LiquidAI/LFM2.5-VL-3B"
FALLBACK_MODEL = "LiquidAI/LFM2.5-VL-1.6B"

# LFM2.5 documented LoRA target modules.
LORA_TARGET_MODULES: list[str] = [
    "w1",
    "w2",
    "w3",
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "in_proj",
]

VISION_ENCODER_NAME = "siglip2"

# Public-config profiles used to sanity-check LoRA budgets. The generated
# adapter count remains an estimate until PEFT prints the runtime model count.
MODEL_PROFILES: dict[str, dict[str, Any]] = {
    DEFAULT_MODEL: {
        "total_params": 3_100_000_000,
        "hidden_size": 2048,
        "intermediate_size": 10752,
        "num_hidden_layers": 30,
        "vision_encoder": VISION_ENCODER_NAME,
        "note": "architecture fields verified from the public model config; LoRA count remains an estimate until runtime",
    },
    FALLBACK_MODEL: {
        "total_params": 1_600_000_000,
        "hidden_size": 2048,
        "intermediate_size": 12288,
        "num_hidden_layers": 16,
        "vision_encoder": VISION_ENCODER_NAME,
        "note": "architecture fields verified from the public model config; LoRA count remains an estimate until runtime",
    },
}


@dataclass
class TrainConfig:
    model_name: str = DEFAULT_MODEL
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: list(LORA_TARGET_MODULES))
    freeze_vision: bool = True
    vision_adapter_phase: bool = False
    phases: list[str] = field(default_factory=lambda: ["text-policy"])
    max_seq_len: int = 4096
    batch_size: int = 1
    learning_rate: float = 2e-4
    epochs: int = 1
    output_dir: str = "adapters"
    adapter_version: str = "v0.1.0"
    parent_version: str | None = None
    modality: str = "multimodal"  # "text" or "multimodal"

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.lora_r <= 0:
            errors.append("lora_r must be positive")
        if self.lora_alpha <= 0:
            errors.append("lora_alpha must be positive")
        if not (0.0 <= self.lora_dropout < 1.0):
            errors.append("lora_dropout must be in [0, 1)")
        if not self.lora_target_modules:
            errors.append("lora_target_modules must not be empty")
        unknown = [m for m in self.lora_target_modules if m not in LORA_TARGET_MODULES]
        if unknown:
            errors.append(f"unknown LoRA target modules: {unknown}")
        if not self.freeze_vision and not self.vision_adapter_phase:
            errors.append("unfreezing the vision encoder requires vision_adapter_phase=True")
        if self.model_name not in MODEL_PROFILES:
            errors.append(
                f"model {self.model_name!r} has no declared profile; use {DEFAULT_MODEL!r} or {FALLBACK_MODEL!r}"
            )
        return errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AdapterVersion:
    version: str
    model_name: str
    parent_version: str | None
    trainable_params: int
    vision_frozen: bool
    artifact_path: str
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_train_config(**overrides: Any) -> TrainConfig:
    cfg = TrainConfig()
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise ValueError(f"unknown TrainConfig field {k!r}")
        setattr(cfg, k, v)
    return cfg


def estimate_lora_params(cfg: TrainConfig) -> int:
    """Estimate LoRA trainable parameters from the declared hidden size.

    LoRA adds ``r * (in_features + out_features)`` per adapted matrix (plus a
    small bias term).  This is an *estimate* used for budget sanity checks;
    the real count comes from the model at load time.
    """
    profile = MODEL_PROFILES.get(cfg.model_name)
    h = profile["hidden_size"] if profile else 4096
    intermediate = profile.get("intermediate_size", 4 * h) if profile else 4 * h
    layers = profile.get("num_hidden_layers", 1) if profile else 1
    total = 0
    for module in cfg.lora_target_modules:
        if module in {"w1", "w2", "w3"}:
            total += cfg.lora_r * (h + intermediate)
        else:
            total += cfg.lora_r * (h + h)
    # This intentionally over-approximates by assuming every selected suffix
    # appears in every layer. Runtime PEFT counts are the promotion authority.
    return total * layers


def validate_trainable_params(count: int, *, max_ratio: float = 0.25, total_params: int | None = None) -> list[str]:
    errors: list[str] = []
    if count <= 0:
        errors.append("trainable parameter count must be positive")
    if total_params and count > total_params * max_ratio:
        errors.append(f"trainable parameters exceed {max_ratio:.0%} of declared model size")
    return errors
