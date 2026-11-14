# 0001: Model architecture

**Status:** Accepted · **Date:** 2026-11-14

## Context

THOR needs its own decoder-only language model at 30M–150M parameters. It must be trainable on a single GPU, understandable end to end, and a sound base for later instruction tuning.

## Decision

A pre-norm decoder-only Transformer built from widely published, well-understood components:

| Choice | Why |
|---|---|
| **RMSNorm, pre-norm** | Stable training at depth; cheaper than LayerNorm |
| **Rotary position embeddings (RoPE)** | Relative positions with no learned position table; a path to longer contexts later |
| **SwiGLU feed-forward**, hidden ≈ 8/3·d | Better quality per parameter than a GELU MLP at equal parameter count |
| **No bias terms** | Fewer parameters, no observed quality cost |
| **Tied input/output embeddings** | At 16k vocab, untied embeddings would add 6–15M parameters, a large share of a 30M model |
| **PyTorch `scaled_dot_product_attention`, causal** | Uses FlashAttention-style kernels on GPU automatically |
| **Init:** N(0, 0.02), residual projections scaled by 1/√(2·layers) | Keeps the residual stream variance stable at initialization |
| **Multi-head attention (no GQA yet)** | KV-cache savings from grouped-query attention don't matter at this size; it can be added later |

Presets are specified in `models/thor/config.py`. Their parameter counts are computed analytically and checked against the real module in the tests. Each preset must be within 10% of its name.

## Consequences

- One code path serves every model size; presets differ only in shape.
- Changing any shape field makes checkpoints incompatible. Training refuses to resume across such a change.
