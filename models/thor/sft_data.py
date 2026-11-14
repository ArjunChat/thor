"""Instruction-tuning (SFT) data: conversations with a loss mask.

Prepare from JSONL, one conversation per line:
    {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}

    python -m thor.sft_data --input chats.jsonl --tokenizer data/tinystories/tokenizer.json \\
        --out data/chat-sft --max-len 1024 --license Apache-2.0

--license is required and must be on the curation allowlist (see thor/curate.py).
Personal data in every message is replaced with fictitious surrogates (thor/pii.py)
unless --no-scrub-pii is given.

Use the *same tokenizer as the base model* you will fine-tune. Writes:
    {train,val}_tokens.bin   uint16 token ids, conversations back to back
    {train,val}_mask.bin     uint8, 1 where the token is part of an assistant reply
    {train,val}_offsets.npy  int64 start of each conversation (plus the end)
    tokenizer.json, meta.json ("kind": "sft")
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
import shutil
from pathlib import Path

import numpy as np
import torch

from . import pii
from .chat_format import render, require_chat
from .curate import DEFAULT_ALLOWED_LICENSES, OBLIGATIONS
from .data import TOKEN_DTYPE
from .prepare import is_val
from .tokenizer import Tokenizer


class SFTDataset:
    """Samples whole conversations; targets outside assistant replies are -1 (ignored by the loss)."""

    def __init__(self, data_dir: str | Path, split: str, seq_len: int) -> None:
        d = Path(data_dir)
        self.seq_len = seq_len
        self.tokens = np.memmap(d / f"{split}_tokens.bin", dtype=TOKEN_DTYPE, mode="r")
        self.mask = np.memmap(d / f"{split}_mask.bin", dtype=np.uint8, mode="r")
        self.offsets = np.load(d / f"{split}_offsets.npy")
        if len(self.offsets) < 2:
            raise ValueError(f"{d} has no {split} conversations")
        # Only conversations with at least one reply token inside the context window can teach anything.
        self.usable = [
            i for i in range(len(self.offsets) - 1)
            if self.mask[int(self.offsets[i]) + 1 : min(int(self.offsets[i + 1]), int(self.offsets[i]) + seq_len + 1)].any()
        ]
        if not self.usable:
            raise ValueError(f"No {split} conversation has a reply within seq_len={seq_len}")

    def __len__(self) -> int:
        return len(self.usable)

    def example(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        a, b = int(self.offsets[i]), int(self.offsets[i + 1])
        b = min(b, a + self.seq_len + 1)  # truncate to the context window
        return self.tokens[a:b].astype(np.int64), self.mask[a:b]

    def _collate(self, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        examples = [self.example(i) for i in indices]
        width = max(len(t) for t, _ in examples) - 1
        x = np.zeros((len(examples), width), dtype=np.int64)
        y = np.full((len(examples), width), -1, dtype=np.int64)
        for row, (t, m) in enumerate(examples):
            n = len(t) - 1
            x[row, :n] = t[:-1]
            y[row, :n] = np.where(m[1:] == 1, t[1:], -1)  # learn only assistant tokens
        return torch.from_numpy(x), torch.from_numpy(y)

    def get_batch(self, batch_size: int, generator: torch.Generator, device: torch.device | str = "cpu"):
        picks = torch.randint(len(self.usable), (batch_size,), generator=generator).tolist()
        x, y = self._collate([self.usable[i] for i in picks])
        return x.to(device), y.to(device)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", nargs="+", required=True, help="JSONL files of conversations")
    ap.add_argument("--tokenizer", required=True, help="The base model's tokenizer.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=1024, help="Conversations are truncated to this many tokens")
    ap.add_argument("--val-fraction", type=float, default=0.02)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--license", required=True, help="License of the input data (SPDX id, or THOR-owned)")
    ap.add_argument("--allow-license", action="append", default=[],
                    help="Add a license to the allowlist (only after checking its terms)")
    ap.add_argument("--no-scrub-pii", action="store_true", help="Keep personal data as is (not recommended)")
    args = ap.parse_args(argv)
    if args.license not in DEFAULT_ALLOWED_LICENSES | set(args.allow_license):
        raise SystemExit(f"License {args.license!r} is not on the allowlist; check the dataset's terms first")

    out = Path(args.out)
    if (out / "meta.json").exists() and not args.overwrite:
        raise SystemExit(f"{out} already contains prepared data; pass --overwrite to replace it")
    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.load(args.tokenizer)
    require_chat(tok)
    shutil.copyfile(args.tokenizer, out / "tokenizer.json")

    files = {s: (open(out / f"{s}_tokens.bin", "wb"), open(out / f"{s}_mask.bin", "wb")) for s in ("train", "val")}
    offsets = {"train": [0], "val": [0]}
    pii_counts: Counter[str] = Counter()
    stats = {"conversations": 0, "skipped_invalid": 0, "skipped_no_reply": 0, "truncated": 0,
             "train_tokens": 0, "val_tokens": 0, "train_reply_tokens": 0, "val_reply_tokens": 0}
    try:
        for path in args.input:
            with open(path, encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    try:
                        messages = json.loads(line)["messages"]
                        if not args.no_scrub_pii:
                            scrubbed = []
                            for m in messages:
                                content, found = pii.scrub(m["content"]) if isinstance(m.get("content"), str) else (m.get("content"), {})
                                pii_counts.update(found)
                                scrubbed.append({**m, "content": content})
                            messages = scrubbed
                        ids, mask = render(tok, messages, add_eot=True)
                    except (ValueError, KeyError, TypeError) as e:
                        stats["skipped_invalid"] += 1
                        print(f"  skipping {path}:{line_no}: {e}")
                        continue
                    if len(ids) > args.max_len + 1:
                        ids, mask = ids[: args.max_len + 1], mask[: args.max_len + 1]
                        stats["truncated"] += 1
                    if not any(mask[1:]):
                        stats["skipped_no_reply"] += 1  # nothing to learn (no reply, or truncated away)
                        continue
                    split = "val" if is_val(line, args.val_fraction) else "train"
                    np.asarray(ids, dtype=TOKEN_DTYPE).tofile(files[split][0])
                    np.asarray(mask, dtype=np.uint8).tofile(files[split][1])
                    offsets[split].append(offsets[split][-1] + len(ids))
                    stats["conversations"] += 1
                    stats[f"{split}_tokens"] += len(ids)
                    stats[f"{split}_reply_tokens"] += sum(mask)
    finally:
        for a, b in files.values():
            a.close()
            b.close()
    for split, offs in offsets.items():
        np.save(out / f"{split}_offsets.npy", np.asarray(offs, dtype=np.int64))
        if len(offs) < 2:
            raise SystemExit(f"No {split} conversations; add data or adjust --val-fraction")

    meta = {
        "kind": "sft",
        "vocab_size": tok.vocab_size,
        "eot_id": tok.eot_id,
        "dtype": np.dtype(TOKEN_DTYPE).name,
        "tokenizer_sha256": tok.fingerprint(),
        "max_len": args.max_len,
        "license": args.license,
        "license_obligations": OBLIGATIONS.get(args.license, "none recorded"),
        "pii_scrubbed": not args.no_scrub_pii,
        "pii_replaced": dict(pii_counts),
        "train_conversations": len(offsets["train"]) - 1,
        "val_conversations": len(offsets["val"]) - 1,
        **stats,
        "sources": [{"path": str(p), "bytes": Path(p).stat().st_size} for p in args.input],
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "sources"}, indent=2))


if __name__ == "__main__":
    main()
