# 0002: Tokenizer

**Status:** Accepted · **Date:** 2026-11-14

## Decision

THOR trains its own **byte-level BPE** tokenizer (`models/thor/tokenizer.py`) instead of reusing another model's vocabulary.

- **Byte-level base (256 ids):** any UTF-8 string round-trips exactly, so there is never an unknown token.
- **Pre-tokenization regex** splits text into contractions, letters, 1–3 digit groups, other symbols, and whitespace. Merges never cross these boundaries. Every character falls into exactly one class; a randomized test checks this.
- **Default vocabulary 16,384:** small enough that the embedding table doesn't dominate a 30M model, and ids fit in `uint16` token files.
- **Training** is incremental: a priority queue plus a pair → document index, so each merge touches only affected words. A test checks it against a brute-force reference implementation. Ties are broken deterministically.
- **Special tokens** (`<|endoftext|>`) take the ids after the merges and are stripped from training text, so they can't be learned as ordinary merges.
- **Format:** versioned JSON. A SHA-256 fingerprint is stored in `meta.json` and in every checkpoint, so data, tokenizer and weights can't be mismatched silently.

## Limitations

Encoding is pure Python, parallelized over documents. That is adequate up to a few GB of text; a compiled encoder producing identical output is the planned upgrade for multi-billion-token corpora.
