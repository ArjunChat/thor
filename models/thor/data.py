"""Tokenized datasets: flat binary token files plus a metadata file.

A prepared data directory contains:
    train.bin, val.bin   uint16 token ids, documents separated by <|endoftext|>
    tokenizer.json       the tokenizer used to produce them
    meta.json            vocab size, token counts, tokenizer fingerprint, sources
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

TOKEN_DTYPE = np.uint16
MAX_VOCAB = int(np.iinfo(TOKEN_DTYPE).max) + 1


def load_meta(data_dir: str | Path) -> dict:
    path = Path(data_dir) / "meta.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `python -m thor.prepare` first")
    return json.loads(path.read_text(encoding="utf-8"))


class TokenDataset:
    """Random fixed-length windows over a memory-mapped token file."""

    def __init__(self, path: str | Path, seq_len: int) -> None:
        self.path = Path(path)
        self.seq_len = seq_len
        self.data = np.memmap(self.path, dtype=TOKEN_DTYPE, mode="r")
        if len(self.data) <= seq_len + 1:
            raise ValueError(
                f"{self.path} has {len(self.data)} tokens; need more than seq_len + 1 = {seq_len + 1}"
            )

    def __len__(self) -> int:
        return len(self.data)

    def get_batch(
        self, batch_size: int, generator: torch.Generator, device: torch.device | str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.seq_len
        starts = torch.randint(len(self.data) - t - 1, (batch_size,), generator=generator).tolist()
        chunk = np.stack([self.data[s : s + t + 1].astype(np.int64) for s in starts])
        chunk = torch.from_numpy(chunk)
        x, y = chunk[:, :-1], chunk[:, 1:]
        if torch.device(device).type == "cuda":
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)

    def sequential_batches(self, batch_size: int, max_tokens: int | None = None):
        """Non-overlapping windows in order, for deterministic evaluation."""
        t = self.seq_len
        limit = len(self.data) - 1 if max_tokens is None else min(len(self.data) - 1, max_tokens)
        starts = list(range(0, limit - t + 1, t))
        for i in range(0, len(starts), batch_size):
            group = starts[i : i + batch_size]
            chunk = torch.from_numpy(np.stack([self.data[s : s + t + 1].astype(np.int64) for s in group]))
            yield chunk[:, :-1], chunk[:, 1:]
