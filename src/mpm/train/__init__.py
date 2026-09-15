"""Training entry points for the Memory Policy Model.

The primary target is ``LiquidAI/LFM2.5-VL-3B`` (a genuinely multimodal VLM)
fine-tuned with LoRA via Liquid's LEAP stack.  A dependency-free ``dry_run``
validates datasets and config without importing any ML framework.  An optional,
clearly experimental Transformers/PEFT path exists for a 16 GB Apple M5 and is
designed to fail safely.
"""

from .config import (
    DEFAULT_MODEL,
    LORA_TARGET_MODULES,
    TrainConfig,
    AdapterVersion,
    build_train_config,
)

__all__ = [
    "DEFAULT_MODEL",
    "LORA_TARGET_MODULES",
    "TrainConfig",
    "AdapterVersion",
    "build_train_config",
]
