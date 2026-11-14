# 0003: Training loop and checkpoint recovery

**Status:** Accepted · **Date:** 2026-11-14

## Context

Affordable GPUs (free notebook tiers, spot instances) get interrupted. Losing progress, or silently resuming into a different run, wastes the compute budget.

## Decision

- **AdamW** (β = 0.9/0.95, weight decay 0.1 on matrices only), **linear warmup + cosine decay**, gradient clipping at 1.0, gradient accumulation, bf16 autocast on GPUs that support it.
- **Full-state checkpoints:** weights, optimizer, step, tokens seen, best validation loss, and the Python, NumPy, Torch, CUDA and batch-sampler RNG states.
- **Atomic writes:** write to a temp file, then `os.replace`. A crash mid-save leaves the previous checkpoint intact.
- **Bit-exact resume**, enforced by a test that compares a 12-step run with a 6 + 6 interrupted run.
- **Graceful stop:** SIGINT/SIGTERM and `--session-steps` save after the current step and exit.
- **Guards:** training refuses to resume across a model-shape or tokenizer change. `--resume never` protects existing runs. A non-finite loss stops training without overwriting the last good checkpoint.
- **Evaluation:** a fixed set of validation batches during training, so evals are comparable over time. `thor.evaluate` gives a deterministic full pass reporting loss, perplexity and bits per byte.

## Consequences

Checkpoints are about 3× the model size, since they include optimizer state (≈1.2 GB for THOR-100M). `model_best.pt` is weights-only.
