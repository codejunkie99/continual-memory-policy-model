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
- A local MCP server for Codex and Claude Code with scoped search, safe
  observation, point reads, delayed feedback, and status tools.
- Default-deny Codex/Claude/Brain/MPM history ingestion and time-ordered,
  feature-only MLX datasets. Historical statements never enter model prompts.

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

## Private history policy update on an external SSD

Keep source code in Git and private artifacts outside the checkout. The example
below uses an environment variable so no machine-specific path is committed:

```bash
export MPM_SSD_ROOT=/Volumes/YourSSD/continual-memory-policy-model
mkdir -p "$MPM_SSD_ROOT"/{private,datasets,adapters,runtime,cache/huggingface,reports}
chmod 700 "$MPM_SSD_ROOT" "$MPM_SSD_ROOT/private"

mpm ingest-history \
  --staging-db "$MPM_SSD_ROOT/private/history-staging.db" \
  --codex-dir "$HOME/.codex/sessions" \
  --claude-dir "$HOME/.claude/projects" \
  --brain-dir "$HOME/.brain" \
  --mpm-db /path/to/current-mpm.db \
  --report ingest.json --outdir "$MPM_SSD_ROOT/reports"

mpm build-history-dataset \
  --staging-db "$MPM_SSD_ROOT/private/history-staging.db" \
  --outdir "$MPM_SSD_ROOT/datasets/history-v1" \
  --max-per-source 500

PYTHONPATH=src python scripts/build_history_replay_mix.py \
  --history "$MPM_SSD_ROOT/datasets/history-v1/mlx-train.jsonl" \
  --replay outputs/mlx-policy-v4-train.jsonl \
  --output "$MPM_SSD_ROOT/datasets/history-v1/mlx-train-replay.jsonl"
```

`ingest-history --dry-run` emits aggregate counts only. A fresh 256-bit salt is
created beside the staging database with mode `0600`; it is never included in
the dataset manifest. Unknown chat, assistant/tool/reasoning output, secrets,
PII, stack traces, code blobs, and secret-bearing URLs are rejected. Brain and
existing MPM stores are read without mutation. `--mpm-db` is repeatable.

The generated `history-*.jsonl` audit rows and `mlx-*.jsonl` training rows carry
only closed-vocabulary structural features, operation labels, and synthetic
candidate IDs. Splits use whole chronological cohorts. These labels remain
weak supervision until downstream outcomes provide stronger credit.

Do not fine-tune on the entire imbalanced history export directly. Mix a bounded
number of history examples per label with balanced six-operation replay, train
a candidate adapter, evaluate the temporal test partition, and then run
`mpm live-evaluate`. Activation is explicit and refused when the live promotion
gate fails.

## Use the memory policy from Codex and Claude Code

Install the optional MCP runtime into the same Python environment as MLX. A
normal wheel install is preferable to an editable install for long-lived MCP
registrations:

```bash
uv pip install --python .venv-mlx/bin/python '.[mcp]'

codex mcp add continual-memory-policy \
  --env HF_HOME="$MPM_SSD_ROOT/cache/huggingface" -- \
  "$PWD/.venv-mlx/bin/mpm-mcp" \
  --db "$MPM_SSD_ROOT/runtime/memory.db" \
  --policy mlx --backend mlx \
  --model /path/to/LFM2.5-VL-3B-MLX-8bit-snapshot \
  --adapter "$MPM_SSD_ROOT/adapters/active"

claude mcp add --scope user continual-memory-policy \
  -e HF_HOME="$MPM_SSD_ROOT/cache/huggingface" -- \
  "$PWD/.venv-mlx/bin/mpm-mcp" \
  --db "$MPM_SSD_ROOT/runtime/memory.db" \
  --policy mlx --backend mlx \
  --model /path/to/LFM2.5-VL-3B-MLX-8bit-snapshot \
  --adapter "$MPM_SSD_ROOT/adapters/active"
```

The server exposes `memory_search`, `memory_observe`, `memory_get`,
`memory_feedback`, and `memory_status` over stdio or streamable HTTP. Model
loading is lazy, database/adapter paths are not returned by tools, searches are
scoped and bounded, and each retrieval gets an ID that can receive delayed
positive or negative feedback.

The first local history update trained 483 steps from the SSD-resident model:
195 bounded history-derived rows plus 288 balanced replay rows. It retained
25/25 accuracy on a deterministic temporal sample. On the full stateful gate it
scored 23/24 operation accuracy, 0.979 macro-F1, 100% structured validity,
0 harmful memories, 0 prompt leakage, and 0.9625 downstream utility. That is a
local engineering gate, not evidence of general research-quality performance.

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
