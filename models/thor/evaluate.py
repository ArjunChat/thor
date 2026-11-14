"""Deterministic held-out evaluation: loss, perplexity and bits per byte.

Example:
    python -m thor.evaluate --ckpt runs/thor-30m/model_best.pt --data-dir data/tinystories

Uses non-overlapping windows over val.bin in order, so the same checkpoint
always gets the same numbers. Bits per byte is comparable across tokenizers;
loss and perplexity are only comparable between models that share one.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from .data import TokenDataset, load_meta
from .tokenizer import Tokenizer
from .train import load_model


@torch.no_grad()
def evaluate(ckpt_path: str, data_dir: str, split: str = "val", batch_size: int = 8,
             max_tokens: int | None = None, device: str = "cpu") -> dict:
    meta = load_meta(data_dir)
    tok = Tokenizer.load(Path(data_dir) / "tokenizer.json")
    model, ckpt = load_model(ckpt_path, device)
    if ckpt["tokenizer_sha256"] != meta["tokenizer_sha256"]:
        raise SystemExit("Checkpoint and dataset use different tokenizers")
    ds = TokenDataset(Path(data_dir) / f"{split}.bin", model.cfg.max_seq_len)

    total_nll, total_tokens, total_bytes = 0.0, 0, 0
    for x, y in ds.sequential_batches(batch_size, max_tokens):
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        nll = torch.nn.functional.cross_entropy(
            logits.float().view(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        total_nll += nll.item()
        total_tokens += y.numel()
        total_bytes += sum(len(tok.vocab.get(t, b"")) for t in y.reshape(-1).tolist())
    loss = total_nll / total_tokens
    return {
        "checkpoint": ckpt_path,
        "step": ckpt.get("step"),
        "split": split,
        "tokens": total_tokens,
        "loss": round(loss, 4),
        "perplexity": round(math.exp(loss), 3),
        "bits_per_byte": round(total_nll / math.log(2) / max(total_bytes, 1), 4),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    print(json.dumps(evaluate(args.ckpt, args.data_dir, args.split, args.batch_size,
                              args.max_tokens, args.device), indent=2))


if __name__ == "__main__":
    main()
