import random
import re

import pytest

from thor.tokenizer import EOT, SPLIT_PATTERN, Tokenizer

CORPUS = [
    "The quick brown fox jumps over the lazy dog. " * 20,
    "Intelligence, engineered. THOR builds language models from first principles.",
    "def train(model, data):\n    for step in range(100):\n        loss = model(data)\n",
    "Numbers like 1234567 and 3.14159 and snake_case_names and émigré café naïve.",
    "Emoji 🚀🔬 and CJK 機械学習 and Arabic تعلم الآلة mixed together.",
]


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return Tokenizer.train(CORPUS, vocab_size=400, min_frequency=2)


def test_split_pattern_covers_every_character():
    rng = random.Random(0)
    split = re.compile(SPLIT_PATTERN)
    for _ in range(200):
        s = "".join(chr(rng.randint(1, 0x2FFFF)) for _ in range(rng.randint(0, 40)))
        s = s.encode("utf-8", "replace").decode("utf-8", "replace")
        assert "".join(split.findall(s)) == s


@pytest.mark.parametrize("text", CORPUS + ["", " ", "\n\n\t ", "unseen wörds ∑∫ 🤖 \x00 end"])
def test_roundtrip(tok, text):
    assert tok.decode(tok.encode(text)) == text


def test_merges_compress(tok):
    text = CORPUS[0]
    assert len(tok.encode(text)) < len(text.encode("utf-8")) / 2


def test_special_tokens(tok):
    ids = tok.encode(f"hello{EOT}world")
    assert tok.eot_id in ids
    assert tok.decode(ids) == f"hello{EOT}world"
    # encode_ordinary treats the special string as plain text
    assert tok.eot_id not in tok.encode_ordinary(f"hello{EOT}world")
    assert tok.vocab_size == 256 + len(tok.merges) + 1
    assert tok.eot_id == tok.vocab_size - 1


def test_training_is_deterministic():
    a = Tokenizer.train(CORPUS, vocab_size=350)
    b = Tokenizer.train(CORPUS, vocab_size=350)
    assert a.merges == b.merges
    assert a.fingerprint() == b.fingerprint()


def test_matches_naive_reference():
    """The incremental trainer must learn the same merges as a brute-force one."""
    text_chunks = re.findall(SPLIT_PATTERN, " ".join(CORPUS))
    words = [list(c.encode()) for c in text_chunks]
    merges = []
    for m in range(60):
        counts = {}
        for w in words:
            for p in zip(w, w[1:]):
                counts[p] = counts.get(p, 0) + 1
        if not counts:
            break
        best = min(counts, key=lambda p: (-counts[p], p))
        if counts[best] < 2:
            break
        merges.append(best)
        new_id = 256 + m
        out = []
        for w in words:
            r, i = [], 0
            while i < len(w):
                if i < len(w) - 1 and (w[i], w[i + 1]) == best:
                    r.append(new_id)
                    i += 2
                else:
                    r.append(w[i])
                    i += 1
            out.append(r)
        words = out
    fast = Tokenizer.train([" ".join(CORPUS)], vocab_size=256 + 60 + 1)
    assert fast.merges == merges


def test_save_load(tok, tmp_path):
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = Tokenizer.load(path)
    assert loaded.fingerprint() == tok.fingerprint()
    assert loaded.encode(CORPUS[2]) == tok.encode(CORPUS[2])


def test_special_strings_not_learned():
    t = Tokenizer.train([f"a{EOT}b" * 50], vocab_size=300)
    for i in range(256, 256 + len(t.merges)):
        assert b"<|" not in t.vocab[i] or b"endoftext" not in t.vocab[i]


def test_vocab_too_small():
    with pytest.raises(ValueError):
        Tokenizer.train(CORPUS, vocab_size=100)
