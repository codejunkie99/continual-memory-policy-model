"""Liquid LEAP-Finetune configuration and dataset contract.

Generates a LEAP-compatible VLM SFT config and the LFM2.5-native chat/tool
format mapping for memory-policy examples.  This module is dependency-free: it
only produces configuration text and a JSON contract.
"""

from __future__ import annotations

import json
from typing import Any

from .config import TrainConfig, LORA_TARGET_MODULES, VISION_ENCODER_NAME


TOOL_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "memory_action",
        "description": "Choose the memory operation to perform next.",
        "parameters": {
            "type": "object",
            "properties": {
                "op": {"type": "string", "enum": ["WRITE", "UPDATE", "DELETE", "LINK", "COMPACT", "NOOP"]},
                "target": {"type": ["string", "null"]},
                "payload": {"type": "object"},
                "confidence": {"type": "number"},
            },
            "required": ["op", "payload"],
        },
    },
}


def build_leap_config(cfg: TrainConfig, data_dir: str) -> dict[str, Any]:
    """Return a config matching LEAP's public VLM-SFT schema."""
    return {
        "project_name": "mpm-lfm25-vl-policy",
        "model_name": cfg.model_name,
        "training_type": "vlm_sft",
        "dataset": {
            "train_path": f"{data_dir}/leap-vlm-sft-train.jsonl",
            "val_path": f"{data_dir}/leap-vlm-sft-val.jsonl",
            "type": "vlm_sft",
            "limit": None,
        },
        "training_config": {
            "extends": "DEFAULT_VLM_SFT",
            "num_train_epochs": cfg.epochs,
            "per_device_train_batch_size": cfg.batch_size,
            "gradient_accumulation_steps": 8,
            "learning_rate": cfg.learning_rate,
            "freeze_vision_encoder": cfg.freeze_vision,
            "output_dir": f"{cfg.output_dir}/{cfg.adapter_version}",
        },
        "peft_config": {
            "extends": "DEFAULT_VLM_LORA",
            "use_peft": True,
            "r": cfg.lora_r,
            "lora_alpha": cfg.lora_alpha,
            "lora_dropout": cfg.lora_dropout,
            "target_modules": cfg.lora_target_modules,
        },
    }


def _yaml_scalar(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, list):
        return "[" + ", ".join(json.dumps(x) for x in v) + "]"
    return json.dumps(v)


def render_config_yaml(c: dict[str, Any]) -> str:
    """Serialize a LEAP config to YAML without adding a runtime dependency."""
    lines: list[str] = []

    def emit(d: dict[str, Any], indent: int) -> None:
        pad = "  " * indent
        for k, v in d.items():
            if isinstance(v, dict):
                lines.append(f"{pad}{k}:")
                emit(v, indent + 1)
            else:
                lines.append(f"{pad}{k}: {_yaml_scalar(v)}")

    emit(c, 0)
    return "\n".join(lines) + "\n"


def render_leap_yaml(cfg: TrainConfig, data_dir: str) -> str:
    return render_config_yaml(build_leap_config(cfg, data_dir))


def build_modal_leap_config(cfg: TrainConfig, *, volume: str = "mpm-lfm25-vl") -> dict[str, Any]:
    """Build an inert Modal submission config whose data lives on its mounted volume."""
    config = build_leap_config(cfg, "/outputs/mpm-data")
    config["modal"] = {
        "app_name": "mpm-lfm25-vl",
        "gpu": "H100",
        "timeout": 86_400,
        "output_volume": volume,
        "output_dir": "/outputs",
        "detach": False,
    }
    return config


def build_dataset_contract() -> dict[str, Any]:
    """Document the feature -> LFM2.5 message/tool mapping."""
    return {
        "format": "lfm2.5-chat-tool",
        "tool": TOOL_DEFINITION,
        "messages": {
            "system": "You are a Memory Policy Model. Given redacted memory features and the current memory state, emit exactly one memory_action tool call.",
            "user": "features rendered as a compact JSON observation (may include an optional image/document reference for multimodal runs).",
            "assistant": "a single tool_call to memory_action whose arguments match the structured Action schema.",
        },
        "privacy": "default export carries only redacted features, op labels, and reward metadata; raw content requires an explicit unsafe flag.",
        "multimodal": "examples may carry a `modality` field of 'text' or 'multimodal'; multimodal observations reference an image/document URL without embedding raw user text.",
    }


def to_lfm_messages(example: dict[str, Any]) -> dict[str, Any]:
    """Convert one exported example into LFM2.5 chat/tool messages."""
    features = example.get("features", {})
    label = example.get("label", example.get("action", {}).get("op", "NOOP"))
    action = example.get("action") or {"op": label, "target": None, "payload": {}}
    observation = {"features": features}
    user_content: list[dict[str, Any]] = []
    image_ref = example.get("image_ref")
    if example.get("modality") == "multimodal" and image_ref:
        user_content.append({"type": "image", "image": image_ref})
    user_content.append({"type": "text", "text": json.dumps(observation, sort_keys=True)})
    tool_json = json.dumps([TOOL_DEFINITION], sort_keys=True)
    system_text = f"List of tools: {tool_json}\n" + build_dataset_contract()["messages"]["system"]
    return {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": system_text}]},
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_memory_action",
                        "type": "function",
                        "function": {
                            "name": "memory_action",
                            "arguments": json.dumps(
                                {
                                    "op": action.get("op"),
                                    "target": action.get("target"),
                                    "payload": action.get("payload", {}),
                                    "confidence": action.get("confidence"),
                                },
                                sort_keys=True,
                            ),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "content": json.dumps({"status": "recorded"}, sort_keys=True),
            },
        ]
    }
