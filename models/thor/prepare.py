"""Prepare a training corpus: train (or reuse) a tokenizer and write token files.

Example:
    python -m thor.prepare --input data/raw/*.txt --out data/tinystories \\
        --vocab-size 16384 --doc-sep "<|endoftext|>"

Input is plain UTF-8 text files. By default each file is one document;
with --doc-sep, files are split into documents on that separator (the
TinyStories .txt release uses "<|endoftext|>"). Documents are assigned to
train/val deterministically, so re-running produces identical files.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import multiprocessing as mp
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from .data import MAX_VOCAB, TOKEN_DTYPE
from .tokenizer import Tokenizer


def iter_documents(paths: list[Path], doc_sep: str | None, block_chars: int = 1 << 24) -> Iterator[str]:
    """Yield stripped, non-empty documents, streaming so multi-GB files never sit in memory whole."""
    for path in paths:
        with open(path, encoding="utf-8", errors="replace", newline="") as f:
            if not doc_sep:
                doc = f.read().strip()
                if doc:
                    yield doc
                continue
            buffer = ""
            while True:
                block = f.read(block_chars)
                if not block:
                    break
                buffer += block
                *docs, buffer = buffer.split(doc_sep)
                for doc in docs:
                    doc = doc.strip()
                    if doc:
                        yield doc
            buffer = buffer.strip()
            if buffer:
                yield buffer


def is_val(doc: str, val_fraction: float) -> bool:
    """Stable, content-based split: the same document always lands in the same set."""
    h = int.from_bytes(hashlib.blake2b(doc.encode("utf-8"), digest_size=8).digest(), "big")
    return (h / 2**64) < val_fraction


_worker_tok: Tokenizer | None = None
_worker_val_fraction = 0.0


def _init_worker(tokenizer_path: str, val_fraction: float) -> None:
    global _worker_tok, _worker_val_fraction
    _worker_tok = Tokenizer.load(tokenizer_path)
    _worker_val_fraction = val_fraction


def _encode_doc(doc: str) -> tuple[str, np.ndarray]:
    assert _worker_tok is not None
    ids = _worker_tok.encode_ordinary(doc)
    ids.append(_worker_tok.eot_id)
    split = "val" if is_val(doc, _worker_val_fraction) else "train"
    return split, np.asarray(ids, dtype=TOKEN_DTYPE)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", nargs="+", required=True, help="Text files or glob patterns")
    ap.add_argument("--out", required=True, help="Output data directory")
    ap.add_argument("--doc-sep", default=None, help="Split files into documents on this string")
    ap.add_argument("--tokenizer", default=None, help="Reuse an existing tokenizer.json instead of training")
    ap.add_argument("--vocab-size", type=int, default=16384)
    ap.add_argument("--tokenizer-train-chars", type=int, default=200_000_000,
                    help="Train the tokenizer on at most this many characters of training documents")
    ap.add_argument("--val-fraction", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 1) - 1))
    ap.add_argument("--overwrite", action="store_true", help="Replace an existing prepared directory")
    args = ap.parse_args(argv)

    paths = sorted({Path(p) for pattern in args.input for p in glob.glob(pattern)})
    if not paths:
        raise SystemExit(f"No input files match {args.input}")
    out = Path(args.out)
    if (out / "meta.json").exists() and not args.overwrite:
        raise SystemExit(f"{out} already contains prepared data; pass --overwrite to replace it")
    out.mkdir(parents=True, exist_ok=True)

    tok_path = out / "tokenizer.json"
    if args.tokenizer:
        tok = Tokenizer.load(args.tokenizer)
        print(f"Using tokenizer {args.tokenizer} (vocab {tok.vocab_size})")
    else:
        def sample() -> Iterator[str]:
            budget = args.tokenizer_train_chars
            for doc in iter_documents(paths, args.doc_sep):
                if is_val(doc, args.val_fraction):
                    continue  # never let validation text shape the tokenizer
                yield doc[:budget]
                budget -= len(doc)
                if budget <= 0:
                    return

        print(f"Training tokenizer (target vocab {args.vocab_size})...")
        tok = Tokenizer.train(sample(), args.vocab_size, verbose=True)
        print(f"Tokenizer vocab size: {tok.vocab_size}")
    if tok.vocab_size > MAX_VOCAB:
        raise SystemExit(f"vocab size {tok.vocab_size} does not fit in {TOKEN_DTYPE.__name__}")
    tok.save(tok_path)

    counts = {"train": 0, "val": 0}
    docs = {"train": 0, "val": 0}
    files = {split: open(out / f"{split}.bin", "wb") for split in counts}
    try:
        docs_iter = iter_documents(paths, args.doc_sep)
        init_args = (str(tok_path), args.val_fraction)
        # "spawn" avoids forking a process that already has PyTorch threads running.
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers, initializer=_init_worker, initargs=init_args) as pool:
            # imap preserves order, so output files are identical across runs.
            for split, ids in pool.imap(_encode_doc, docs_iter, chunksize=64):
                ids.tofile(files[split])
                counts[split] += len(ids)
                docs[split] += 1
    finally:
        for f in files.values():
            f.close()

    for split, n in docs.items():
        if n == 0:
            raise SystemExit(
                f"The {split} split is empty ({docs['train']} train / {docs['val']} val documents). "
                "The split is by document content, so add more distinct documents or adjust --val-fraction."
            )

    meta = {
        "vocab_size": tok.vocab_size,
        "eot_id": tok.eot_id,
        "dtype": np.dtype(TOKEN_DTYPE).name,
        "tokenizer_sha256": tok.fingerprint(),
        "train_tokens": counts["train"],
        "val_tokens": counts["val"],
        "train_documents": docs["train"],
        "val_documents": docs["val"],
        "val_fraction": args.val_fraction,
        "sources": [{"path": str(p), "bytes": p.stat().st_size} for p in paths],
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "sources"}, indent=2))


if __name__ == "__main__":
    main()
