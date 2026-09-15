# LFM2.5-VL-3B MLX fine-tuning report

## Outcome

A real local LoRA/QLoRA-style adapter was trained for
`LiquidAI/LFM2.5-VL-3B-MLX-8bit` on an Apple M5 with 16 GB unified memory.
The final `mlx-lora-v3` adapter is 47 MB and remains a **candidate**, not the
active policy.

MLX Studio itself is an inference/chat/serving application. The training run
used the separate `mlx-vlm` Python trainer, which implements both LoRA training
and the `lfm2_vl` architecture.

## Environment

- macOS 26.7, Apple M5, arm64, 16 GB unified memory
- Python 3.12.12
- `mlx` 0.32.2
- `mlx-vlm` 0.7.1
- `datasets` 5.0.1
- `transformers` 5.17.0
- Base checkpoint download: 3.74 GB reconstructed

## Training evidence

1. One-step smoke test completed a real forward/backward update at 5.693 GB
   peak memory and wrote a 23 MB rank-4 adapter.
2. The first rank-8 pass trained 51 steps but collapsed to always predicting
   `WRITE`. Its test accuracy was 0/9, so it was rejected as a useful policy.
3. A class-balanced, completion-only tool-call dataset was built: 96 examples,
   16 for each of the six operations. After 192 steps, the adapter reached 7/9
   on the diagnostic test set.
4. The two errors were both `LINK -> COMPACT`. A 112-example hard-replay pass
   resumed from v2 with 32 `LINK` examples and 16 examples for every other
   operation. It completed 112 more optimizer steps with 5.901 GB peak memory.

The final run trained 12,230,656 LoRA parameters, or 0.392% of the 3.123B
parameters exposed by MLX-VLM.

## Evaluation evidence

The 9-example test split was used diagnostically while developing v3 and is
therefore not treated as untouched final evidence. V3 scored 9/9 on it.

The separate 12-example validation split was evaluated only after v3 was
frozen:

| Metric | Base 8-bit checkpoint | MLX LoRA v3 |
|---|---:|---:|
| Operation accuracy | 1/12 (8.33%) | 12/12 (100%) |
| Macro-F1 | 0.0256 | 1.0000 |
| Native tool-call validity | 12/12 (100%) | 12/12 (100%) |

Every v3 output used the native LFM bracketed form:

```text
<|tool_call_start|>[memory_action(op='...', payload={})]<|tool_call_end|>
```
## Artifact receipts

- `mlx-lora-v3/adapters.safetensors`
  - SHA-256: `3a94b2b1b3bd90c041050b3dc24e105e56e2c2ac9ea516308b7843e7c8b6a853`
- `mlx-policy-hard-replay.jsonl`
  - SHA-256: `69058a659e6f16a55041eac672423d3306b763299975a682a215d1611c965463`
- `mlx-validation-v3.json`
  - SHA-256: `49e1e84372e51dde3e6e2aea9d01bb495f9a75278dcea2f68bb206a3ed4b3185`

## Promotion decision

The adapter clears the project's 0.80 operation-accuracy threshold on this
small synthetic validation set. It has **not** been promoted because the MLX
model has not yet been exercised inside the stateful memory runtime to measure
downstream utility and harmful-memory rate. The checkpoint registry records
`mlx-lora-v3` as `candidate`, `gate_approved=0`; `baseline-v1` remains active.

This is proof that local gradient training works, not evidence of a
research-quality memory policy. Promotion requires a larger untouched temporal
test set, real consented trajectories, adversarial privacy cases, and the full
downstream/safety gate.

## Reproduction

The project uses a local Python 3.12 environment at `.venv-mlx`. The local
MLX-VLM checkout contains a compatibility patch for loading a direct JSONL path
with the current `datasets` release.

```bash
PYTHONPATH=src:work/mlx-vlm-upstream .venv-mlx/bin/python \
  scripts/evaluate_mlx_adapter.py \
  --model LiquidAI/LFM2.5-VL-3B-MLX-8bit \
  --adapter outputs/mlx-lora-v3 \
  --test outputs/sft-val.jsonl \
  --output outputs/mlx-validation-v3.json
```
