# LFM2.5-VL-3B Memory Policy Model: live validation

## Result

`mlx-lora-v4` passed the complete local live gate and was explicitly activated
in `outputs/mpm-live-v3.db` after prior passing evaluation runs.

| Metric | Result | Gate |
|---|---:|---:|
| Operation accuracy | 24/24 (100%) | >= 80% |
| Macro-F1 | 1.000 | >= 0.75 |
| Native structured-output validity | 100% | >= 99% |
| Action execution validity | 100% | >= 99% |
| Normalized downstream utility | 0.770 | >= 0.50 |
| Net delayed reward | 3.270708 | informational |
| Harmful-memory rate | 0% | <= 5% |
| Harmful-write attempt rate | 0% | <= 5% |
| Harmful content found in memory | 0 | 0 |
| Raw prompt privacy leakage | 0% | 0% |
| Delayed outcome attributions | 5 | > 0 |
| Live runtime latency | 22.35 s / 24 steps | informational |

The stateful run executed `WRITE`, `UPDATE`, `DELETE`, `LINK`, `COMPACT`, and
`NOOP` against SQLite. It later retrieved five memories, attached four positive
and one negative delayed outcomes, and reconciled all five back to their
originating memory operations.

## Privacy and safety boundary

The model never receives raw memory text, scope values, keys, or real memory
identifiers. It sees coarse text-shape features, a closed operation-intent enum,
structural counts, exact-duplicate status, source-trust class, and per-request
candidate aliases. The model selects only the operation. Trusted code creates
the executable payload from local state and validates it.

Email addresses, API-key language, credit-card-like numbers, and private-key
language were refused before inference. The executor independently rejects
harmful direct or referenced write/update content as a second boundary.

## Training receipt

- Base model: `LiquidAI/LFM2.5-VL-3B-MLX-8bit`
- Runtime: Apple MLX 0.32.2 and MLX-VLM 0.7.1
- Adapter: rank 8 LoRA, 12,230,656 trainable parameters (0.392%)
- Training: 288 balanced completion-only steps; 96,294 tokens
- Peak unified memory: 7.213 GB
- Final training loss: 0.00012194
- Adapter SHA-256: `90b70f93c20a6a441db79fc0dede8a507202d78f012787f8e5e21b5967520084`
- Dataset SHA-256: `93d887699d556a453d27ed75c36ed30e7053e2739b51b813cdd2e3d6da49933b`
- Activation evidence SHA-256: `2ec7adfa90e2d71b83d41660d7c69df9d9c229639d57d4f1c25ec352fd6ff55d`

The prior v3 adapter was also tested through the same runtime contract before
retraining. It failed live promotion at 41.7% operation accuracy and 50% native
output validity, which confirmed that its earlier synthetic case-ID result did
not transfer to realistic runtime features.

## Scope

This proves the end-to-end mechanism locally with authored, realistic test
trajectories. It does not prove production generalization. The next research
stage is a consented, versioned trajectory corpus with a temporal holdout,
poisoning review, and periodic replay-based consolidation.
