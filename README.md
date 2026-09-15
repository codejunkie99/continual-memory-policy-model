# Continual Memory Policy Model (MPM)

An executable research scaffold for training `LiquidAI/LFM2.5-VL-3B` to manage
an external memory system. User facts remain in SQLite; the model learns the
policy for `WRITE`, `UPDATE`, `DELETE`, `LINK`, `COMPACT`, and `NOOP` from later
consequences.

```text
interaction / image / document
              |
              v
     LFM2.5-VL policy decision
              |
              v
 WRITE UPDATE DELETE LINK COMPACT NOOP ----> SQLite event log + current state
              ^                                      |
              |                                      v
       adapter vNext <---- SFT / DPO buffer <---- retrieval -> outcome
              |
              v
   promotion gate -> explicit activate / rollback
```

## What works now

- Transactional, event-sourced SQLite memory with revision history, soft
  deletion, links, compaction provenance, retrievals, outcomes, credit, and
  checkpoint history.
- Delayed positive and negative credit, including one outcome shared across
  multiple retrieved memories with normalized contribution weights.
- Privacy-safe export by default. Raw memory, raw identifiers, plain content
  hashes, and wall-clock timestamps are excluded; `--unsafe-raw` is explicit.
- Policy labels refer to `observation.content`; only the trusted executor copies
  raw content into SQLite. Candidate memory IDs are per-example aliases.
- Deterministic scenario/user/temporal-cohort splits, SFT and preference data,
  a synthetic step-40 to step-8000 benchmark, evaluation metrics, and a
  promotion gate that never auto-activates a candidate.
- A LEAP VLM-SFT YAML and dataset in Liquid's current public schema, with the
  SigLIP2 vision encoder frozen for the first text-policy phase.
- A real MLX-backed runtime that accepts only privacy-safe semantic features,
  strictly parses native LFM tool calls, hydrates payloads in trusted code, and
  blocks PII/secrets again at the executor boundary.
- A 24-step stateful live gate covering all six operations, exact duplicates,
  URLs, preferences, five delayed outcomes (including a negative outcome), and
  four harmful-memory probes.

## Run the lightweight system

Python 3.11 or newer is sufficient for the runtime and tests.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v

mpm demo --db outputs/mpm.db --outdir outputs
mpm export --db outputs/mpm.db --outdir outputs
mpm evaluate --db outputs/mpm.db --outdir outputs
mpm train-dry-run --data-dir outputs --outdir outputs
mpm consolidate --current-dir outputs --replay-dir /path/to/older/export \
  --outdir outputs/consolidation
```

The generated LEAP artifacts are `outputs/leap-finetune.yaml`,
`outputs/leap-vlm-sft-train.jsonl`, and
`outputs/leap-vlm-sft-val.jsonl`.

`outputs/leap-finetune-modal.yaml` is the equivalent one-H100 Modal job. Its
dataset paths point into a mounted Modal Volume; run the commands in
`outputs/modal-upload-commands.txt` only after authenticating and approving
cloud spend.

The consolidation command keeps all new training rows, deduplicates historical
rows, prioritizes negative and difficult replay examples, emits a lineage
manifest, and prepares a versioned `leap-consolidation.yaml`. It does not mix
validation/test rows into training and never promotes the resulting adapter.

## Run the actual LoRA job locally with MLX

A real local adapter has now been trained with `mlx-vlm` 0.7.1 against
`LiquidAI/LFM2.5-VL-3B-MLX-8bit`. The active local adapter is v4. It was trained
for one balanced 288-step epoch on a closed, privacy-safe feature vocabulary.
On the 24-step stateful live gate it achieved 24/24 operation accuracy, 1.0
macro-F1, 100% native tool-call validity, 0 harmful memories, 0 prompt leakage,
and 0.77 normalized downstream utility. The same gate passed twice before a
third passing run explicitly activated the checkpoint.

MLX Studio is useful for local inference and serving, but it does not expose
fine-tuning. Use the MLX-VLM trainer directly:

```bash
uv venv .venv-mlx --python 3.12
uv pip install --python .venv-mlx/bin/python 'mlx-vlm[train]==0.7.1'

git clone --depth 1 https://github.com/Blaizzy/mlx-vlm.git work/mlx-vlm-upstream
git -C work/mlx-vlm-upstream apply --unidiff-zero ../../patches/mlx-vlm-direct-jsonl.patch

PYTHONPATH=src .venv-mlx/bin/python scripts/build_semantic_mlx_dataset.py \
  --output outputs/mlx-policy-v4-train.jsonl

PYTHONPATH=src:work/mlx-vlm-upstream .venv-mlx/bin/python -m mlx_vlm.lora \
  --model-path LiquidAI/LFM2.5-VL-3B-MLX-8bit \
  --dataset outputs/mlx-policy-v4-train.jsonl --split train --iters 288 \
  --batch-size 1 --learning-rate 1e-5 --train-on-completions \
  --lora-rank 8 --lora-alpha 16 --output-path outputs/mlx-lora-v4

PYTHONPATH=src:work/mlx-vlm-upstream .venv-mlx/bin/python -m mpm live-evaluate \
  --db outputs/mpm-live.db --policy mlx --backend mlx \
  --adapter outputs/mlx-lora-v4 --version mlx-lora-v4 --activate \
  --outdir outputs/live-v4-active
```

The current MLX-VLM/Datasets combination needs the included small direct-JSONL
loader compatibility patch.
Training and evaluation receipts are in `outputs/MLX_FINETUNE_REPORT.md`.

## Alternative LEAP CUDA job

Liquid's current LEAP local backend requires visible CUDA devices. On a CUDA
machine, or after adding a documented Modal/SLURM/KubeRay backend block:

```bash
git clone https://github.com/Liquid4All/leap-finetune.git
cd leap-finetune
uv sync
uv run leap-finetune /absolute/path/to/outputs/leap-finetune.yaml
```

For Modal, upload the generated datasets first and then run the Modal YAML from
the LEAP checkout. The backend uses the `mpm-lfm25-vl` volume for both data and
checkpoints.

The config targets `LiquidAI/LFM2.5-VL-3B`, uses `DEFAULT_VLM_SFT` plus LoRA,
and writes a versioned adapter directory. Training does not promote it. Run the
held-out evaluation first, register the resulting metrics, then explicitly use
`mpm promote --version ...` only if the gate passes.

## Important limitations

- The 3.74 GB MLX checkpoint was downloaded and local gradient training was
  completed. No paid/cloud GPU job was launched.
- The v4 adapter is active in the local test runtime after passing the full
  gate. This is not a production deployment or a claim of broad generalization.
- The included 24-step trajectory is realistic but authored test data. A
  research-quality result still needs hundreds to thousands of consented,
  outcome-labelled production trajectories and a separately governed temporal
  holdout.
- Hash-based group IDs are pseudonyms, not anonymity. The raw SQLite database
  remains sensitive and should be encrypted and access-controlled in production.
- Multimodal records are supported by the LEAP message contract, but the first
  dataset is text-policy data. A later vision phase needs image/document
  examples and an explicit decision about which vision/projector parameters to
  train.
- LEAP's current `vlm_dpo` path requires a loadable image for every preference
  row. The present text-only preference buffer is therefore retained for
  research/replay but intentionally not mislabelled as a runnable VLM-DPO job.

## Staged research plan

1. Collect real-time memory operations and downstream outcomes without updating
   weights online.
2. Train a LoRA adapter periodically with SFT, then preference learning on clear
   positive/negative pairs; keep replay strata for difficult old cases.
3. Evaluate utility, operation F1, harmful-memory rate, latency, and storage on
   held-out temporal cohorts. Promote only through the explicit gate.
4. Add multimodal screen/document trajectories with the vision tower still
   frozen; unfreeze or add vision adapters only after an ablation demonstrates
   that visual learning is necessary.
5. Periodically distill new trajectories plus replay and adversarial examples
   into a consolidated checkpoint, retaining rollback artifacts and audit logs.
