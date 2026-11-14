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
| `thor/chat_format.py` | Chat format: role tokens and the assistant-only loss mask |
| `thor/sft_data.py` | Conversations (JSONL) → instruction-tuning data |
| `thor/chat.py` | Terminal chat with a checkpoint |
| `thor/specs.py` | Prints the preset specification table and compute estimates |
| `notebooks/train_thor_30m_colab.ipynb` | End-to-end THOR-30M training on a free Colab GPU |

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

### Train THOR-30M on Google Colab (free T4 GPU)

Open [`notebooks/train_thor_30m_colab.ipynb`](notebooks/train_thor_30m_colab.ipynb) in Colab (*File → Open notebook → GitHub*, or [this link](https://colab.research.google.com/github/ArjunChat/thor/blob/main/models/notebooks/train_thor_30m_colab.ipynb) if the repo is public), choose a T4 GPU runtime, and *Run all*. The notebook:

1. clones this repo (for a private repo, add a `GITHUB_TOKEN` Colab secret),
2. downloads TinyStories, trains the THOR tokenizer and tokenizes the data once, saving the result to Google Drive,
3. trains with checkpoints on Drive. After a disconnect, *Run all* again and it resumes,
4. plots loss, evaluates the best checkpoint, and generates samples.

CI runs the whole notebook on CPU with a tiny model (`THOR_SMOKE_TEST=1`), so it stays in sync with the code. The real T4 run has not been tested yet.

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

### 5. Instruction tuning and chat

A base model only continues text. To make it chat, fine-tune it on conversations ([decision record](../docs/decisions/0004-chat-format-and-instruction-tuning.md)):

```bash
# conversations.jsonl: one {"messages": [{"role": "user", "content": "…"}, {"role": "assistant", "content": "…"}]} per line
python -m thor.sft_data --input conversations.jsonl --tokenizer ../data/tinystories/tokenizer.json --out ../data/chat-sft
python -m thor.train --init-from ../runs/thor-30m-tinystories/model_best.pt --data-dir ../data/chat-sft \
  --out-dir ../runs/thor-30m-chat --lr 1e-4 --min-lr 1e-5 --max-steps 2000 --warmup-steps 100
python -m thor.chat --ckpt ../runs/thor-30m-chat/model_best.pt --tokenizer ../data/chat-sft/tokenizer.json
```

- Use the base model's tokenizer. Tokenizers include the chat tokens by default.
- Only assistant replies are trained on. Resume, interruption and checkpointing work as in pretraining.
- `thor.chat` keeps the conversation history (`/reset` clears it) and warns if the checkpoint is a base model.
- The learning rates above are untuned starting points. **Choose and register a conversation dataset in [`DATA.md`](DATA.md) first.**

## Known limitations

- The tokenizer is pure Python. It is fine for corpora up to a few GB with `--workers`, but slow at multi-billion-token scale. A compiled encoder is a planned follow-up.
- No KV cache yet, so generation recomputes the full context each token. That is acceptable for these model sizes.
- Single-GPU training only (no distributed data parallel yet).
- Instruction tuning is single-turn-loss SFT only (no preference tuning yet). Chat quality depends entirely on the trained base model and the conversation data.
