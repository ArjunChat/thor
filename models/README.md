# THOR Models

THOR's language-model stack, written from scratch in PyTorch:

| Module | Purpose |
|---|---|
| `thor/tokenizer.py` | Byte-level BPE tokenizer: training, encoding, decoding, versioned save format |
| `thor/model.py` | Decoder-only Transformer (RMSNorm, RoPE, SwiGLU, fused causal attention, tied embeddings) |
| `thor/config.py` | Model configuration and the THOR presets |
| `thor/prepare.py` | Corpus → tokenizer + `train.bin` / `val.bin` + `meta.json` |
| `thor/train.py` | Training loop with checkpointing, exact resume, and safe interruption |
| `thor/evaluate.py` | Deterministic held-out loss, perplexity and bits per byte |
| `thor/sample.py` | Text generation from a checkpoint |
| `thor/specs.py` | Prints the preset specification table and compute estimates |

Design rationale: [`docs/decisions/`](../docs/decisions/).

## Presets

Output of `python -m thor.specs`:

| Preset | Parameters | Non-embedding | Layers | d_model | Heads | FFN | Context | Vocab |
|---|---|---|---|---|---|---|---|---|
| thor-nano | 1.1M | 0.8M | 4 | 128 | 4 | 352 | 256 | 2,048 |
| THOR-30M | 31.1M | 24.8M | 14 | 384 | 6 | 1,024 | 1,024 | 16,384 |
| THOR-100M | 97.5M | 85.0M | 12 | 768 | 12 | 2,048 | 1,024 | 16,384 |
| THOR-150M | 151.2M | 136.5M | 14 | 896 | 14 | 2,432 | 1,024 | 16,384 |

`thor-nano` exists only for tests and CPU smoke runs; it is not a THOR model. The vocabulary size of a real run is taken from its tokenizer, so counts shift slightly if the vocabulary differs from 16,384.

### Compute estimates

These are **estimates from arithmetic, not measurements.** They assume 20 training tokens per parameter, ≈6·N FLOPs per token plus the attention term, and the stated hardware utilization.

| Model | Tokens | FLOPs | 1× A100 at 40% util | 1× T4 at 25% util |
|---|---|---|---|---|
| THOR-30M | 0.62B | 1.4e17 | ~0.3 h | ~2.3 h |
| THOR-100M | 1.95B | 1.3e18 | ~2.8 h | ~21 h |
| THOR-150M | 3.02B | 3.0e18 | ~6.6 h | ~51 h |

CPU training is only practical for `thor-nano` (≈10k tokens/s measured on a 4-core cloud CPU).

## Usage

### 1. Install

```bash
pip install -e ".[dev]"   # CPU or CUDA build of PyTorch, whichever your machine has
pytest -q
```

### 2. Prepare data

Input is plain UTF-8 text. Each file is one document, or use `--doc-sep` to split files into documents.

```bash
python -m thor.prepare --input ../data/raw/TinyStoriesV2-GPT4-train.txt \
  --out ../data/tinystories --vocab-size 16384 --doc-sep "<|endoftext|>"
```

This trains a tokenizer on the training split only and writes `tokenizer.json`, `train.bin`, `val.bin` and `meta.json`. The train/validation split is content-hashed, so re-running produces identical files. It refuses to overwrite a prepared directory unless you pass `--overwrite`.

**Before using any dataset, record its source and license in [`DATA.md`](DATA.md).**

### 3. Train

```bash
python -m thor.train --config configs/thor-30m-tinystories.json
```

- Every `TrainConfig` field is a CLI flag, and flags override `--config`.
- `ckpt_last.pt` holds the full training state (weights, optimizer, step, all RNG states, tokenizer fingerprint). It is written atomically every `ckpt_interval` steps.
- **Resume:** run the same command again. Resume is bit-exact (this is tested).
- **Interrupt safely:** Ctrl-C or SIGTERM finishes the current step, saves, and exits.
- **Time-limited GPU sessions** (Colab, Kaggle): `--session-steps 1000` trains 1,000 steps, checkpoints, and exits. The learning-rate schedule is unaffected.
- `model_best.pt` keeps the weights with the best validation loss. Metrics go to `metrics.jsonl`.
- Training refuses to resume if the model shape or tokenizer changed. `--resume never` refuses to touch an existing run at all.
- The configs in `configs/` are untuned starting points.

### 4. Evaluate and sample

```bash
python -m thor.evaluate --ckpt ../runs/thor-30m-tinystories/model_best.pt --data-dir ../data/tinystories
python -m thor.sample   --ckpt ../runs/thor-30m-tinystories/model_best.pt \
  --tokenizer ../data/tinystories/tokenizer.json --prompt "Once upon a time"
```

Bits per byte can be compared across tokenizers; loss and perplexity can only be compared between models that share a tokenizer.

## Known limitations

- The tokenizer is pure Python. It is fine for corpora up to a few GB with `--workers`, but slow at multi-billion-token scale. A compiled encoder is a planned follow-up.
- No KV cache yet, so generation recomputes the full context each token. That is acceptable for these model sizes.
- Single-GPU training only (no distributed data parallel yet).
- No instruction tuning yet (THOR-Instruct is planned).
