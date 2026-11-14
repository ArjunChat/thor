"""THOR byte-level BPE tokenizer.

A from-scratch implementation: text is split into chunks with a
pre-tokenization regex, each chunk is encoded as UTF-8 bytes, and learned
merges combine byte pairs into larger tokens. Any string round-trips
exactly because the base vocabulary is all 256 byte values.

Token ids:   0..255              raw bytes
             256..256+M-1        merges, in the order they were learned
             256+M..             special tokens (e.g. <|endoftext|>)

Design notes and limitations: docs/decisions/0002-tokenizer.md.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

# Contractions | letters | 1-3 digits | other symbols (incl. underscore) | whitespace.
# Every character falls into exactly one class, so chunks concatenate back to the input.
SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d{1,3}| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+"""
EOT = "<|endoftext|>"
FORMAT_VERSION = 1
_CACHE_LIMIT = 100_000


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out: list[int] = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class Tokenizer:
    def __init__(
        self,
        merges: list[tuple[int, int]],
        special_tokens: Iterable[str] = (EOT,),
        pattern: str = SPLIT_PATTERN,
    ) -> None:
        self.pattern = pattern
        self._split = re.compile(pattern)
        self.merges = [tuple(m) for m in merges]
        self.ranks: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for rank, (a, b) in enumerate(self.merges):
            new_id = 256 + rank
            if a >= new_id or b >= new_id:
                raise ValueError(f"Merge {rank} references a token that does not exist yet")
            self.ranks[(a, b)] = new_id
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
        self.special_tokens: dict[str, int] = {}
        for tok in special_tokens:
            self.special_tokens[tok] = len(self.vocab) + len(self.special_tokens)
        self.special_ids = {v: k for k, v in self.special_tokens.items()}
        self._special_split = (
            re.compile("(" + "|".join(re.escape(s) for s in self.special_tokens) + ")")
            if self.special_tokens
            else None
        )
        self._cache: dict[str, list[int]] = {}

    # ------------------------------------------------------------------ training
    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int,
        special_tokens: Iterable[str] = (EOT,),
        min_frequency: int = 2,
        verbose: bool = False,
    ) -> "Tokenizer":
        """Learn merges from an iterable of documents.

        Special-token strings inside the training text are removed so they
        can never be learned as ordinary merges.
        """
        special_tokens = tuple(special_tokens)
        n_merges = vocab_size - 256 - len(special_tokens)
        if n_merges < 0:
            raise ValueError("vocab_size is smaller than 256 bytes + special tokens")
        split = re.compile(SPLIT_PATTERN)

        chunk_counts: Counter[str] = Counter()
        for text in texts:
            for tok in special_tokens:
                text = text.replace(tok, " ")
            chunk_counts.update(split.findall(text))

        words = [list(chunk.encode("utf-8")) for chunk in chunk_counts]
        freqs = list(chunk_counts.values())
        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        where: dict[tuple[int, int], set[int]] = defaultdict(set)
        for i, w in enumerate(words):
            for pair in zip(w, w[1:]):
                pair_counts[pair] += freqs[i]
                where[pair].add(i)
        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        merges: list[tuple[int, int]] = []
        while len(merges) < n_merges and heap:
            neg, pair = heapq.heappop(heap)
            if pair_counts.get(pair, 0) != -neg:
                continue  # stale heap entry
            if -neg < min_frequency:
                break
            new_id = 256 + len(merges)
            merges.append(pair)
            changed: set[tuple[int, int]] = set()
            for i in where.pop(pair, ()):
                w, f = words[i], freqs[i]
                for p in zip(w, w[1:]):
                    pair_counts[p] -= f
                    changed.add(p)
                w = _merge(w, pair, new_id)
                words[i] = w
                for p in zip(w, w[1:]):
                    pair_counts[p] += f
                    where[p].add(i)
                    changed.add(p)
            for p in changed:
                c = pair_counts.get(p, 0)
                if c <= 0:
                    pair_counts.pop(p, None)
                elif p != pair:
                    heapq.heappush(heap, (-c, p))
            pair_counts.pop(pair, None)
            if verbose and len(merges) % 1000 == 0:
                print(f"  learned {len(merges)}/{n_merges} merges")
        return cls(merges, special_tokens)

    # ------------------------------------------------------------------ encode/decode
    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.special_tokens)

    @property
    def eot_id(self) -> int:
        return self.special_tokens[EOT]

    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            best_rank, best_pair = None, None
            for pair in zip(ids, ids[1:]):
                rank = self.ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_pair = rank, pair
            if best_pair is None:
                break
            ids = _merge(ids, best_pair, best_rank)
        if len(self._cache) < _CACHE_LIMIT:
            self._cache[chunk] = ids
        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode text, treating special-token strings as plain text."""
        out: list[int] = []
        for chunk in self._split.findall(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def encode(self, text: str, allow_special: bool = True) -> list[int]:
        if not allow_special or self._special_split is None:
            return self.encode_ordinary(text)
        out: list[int] = []
        for part in self._special_split.split(text):
            if part in self.special_tokens:
                out.append(self.special_tokens[part])
            elif part:
                out.extend(self.encode_ordinary(part))
        return out

    def decode(self, ids: Iterable[int]) -> str:
        parts: list[bytes] = []
        for i in ids:
            i = int(i)
            if i in self.vocab:
                parts.append(self.vocab[i])
            elif i in self.special_ids:
                parts.append(self.special_ids[i].encode("utf-8"))
            else:
                raise ValueError(f"Unknown token id {i}")
        return b"".join(parts).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {
            "format": "thor-bpe",
            "version": FORMAT_VERSION,
            "pattern": self.pattern,
            "merges": [list(m) for m in self.merges],
            "special_tokens": list(self.special_tokens),
        }

    def fingerprint(self) -> str:
        """SHA-256 of the canonical serialization; stored in data and checkpoints."""
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != "thor-bpe" or data.get("version") != FORMAT_VERSION:
            raise ValueError(f"{path} is not a THOR tokenizer (format v{FORMAT_VERSION})")
        return cls([tuple(m) for m in data["merges"]], data["special_tokens"], data["pattern"])
