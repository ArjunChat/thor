"""Personal-data scrubbing, curation, dataset conversion and SFT data licensing."""

import json

import pytest

from thor import convert, curate, pii
from thor.sft_data import main as sft_prepare
from thor.tokenizer import Tokenizer


# ------------------------------------------------------------------ PII
@pytest.mark.parametrize("text,kind", [
    ("Write to jane.doe@gmail.com today", "email"),
    ("Call me at (415) 555-2671 tonight", "phone"),
    ("Call 415-867-5309", "phone"),
    ("Ring +44 20 7946 0958 please", "phone"),
    ("Server at 172.16.254.1 is down", "ip"),
    ("Card 4111 1111 1111 1111 exp 12/27", "card"),
    ("IBAN GB82 WEST 1234 5698 7654 32", "iban"),
    ("SSN 123-45-6789", "ssn"),
])
def test_pii_detected_and_replaced(text, kind):
    out, counts = pii.scrub(text)
    assert counts[kind] == 1, (out, counts)
    assert out != text


@pytest.mark.parametrize("text", [
    "The run finished on 2026-11-14 at step 1,024.",
    "Version 1.2.3 of the tokenizer, 16,384 tokens.",
    "THOR-30M has 31,075,200 parameters.",
    "Contact person1234@example.com or (202) 555-0142.",  # already fictitious
    "Docs use 192.0.2.10 as an example address.",
    "Order 1234 5678 9012 3456 failed",  # not Luhn-valid
])
def test_pii_leaves_ordinary_text_alone(text):
    out, counts = pii.scrub(text)
    assert out == text and not counts, counts


def test_surrogates_are_fictitious_and_consistent():
    text = "a@real.org wrote to b@real.org, then a@real.org replied"
    out, counts = pii.scrub(text)
    assert counts["email"] == 3
    assert "real.org" not in out and out.count("@example.com") == 3
    first = out.split()[0]
    assert out.split()[-2] == first  # the same person keeps the same surrogate
    assert pii.scrub(out)[1] == {}  # surrogates aren't detected again


def test_tag_mode_for_replies():
    out, _ = pii.scrub("mail me: x@y.com", mode="tag")
    assert out == "mail me: [email removed]"


# ------------------------------------------------------------------ curation
def write(p, text):
    p.write_text(text, encoding="utf-8")
    return str(p)


GOOD = "The little fox walked to the river and drank cool water under the bright morning sun."


def test_curate_end_to_end(tmp_path):
    a = write(tmp_path / "a.txt", "<|endoftext|>".join([
        GOOD,
        GOOD.upper(),  # duplicate after normalization
        "Email the author at someone@university.edu for the data. " + GOOD,
        "short",
        "%%%% #### $$$$ 1234 5678 !!!! ???? &&&& **** ^^^^ (((( )))) [[[[ ]]]]",
        "\n".join(["same line here"] * 10),
    ]))
    b = write(tmp_path / "b.txt", GOOD + "<|endoftext|>" + "A completely different story about a kind dragon who shared cake.")
    manifest = {"sources": [
        {"name": "a", "paths": [a], "doc_sep": "<|endoftext|>", "license": "CC0-1.0"},
        {"name": "b", "paths": [b], "doc_sep": "<|endoftext|>", "license": "CC-BY-SA-4.0"},
    ]}
    report = curate.curate(manifest, tmp_path / "out", curate.DEFAULT_ALLOWED_LICENSES)
    docs = (tmp_path / "out" / "curated.txt").read_text().split("<|endoftext|>")
    assert len(docs) == 3
    assert not any("university.edu" in d for d in docs)
    sa, sb = report["sources"]
    assert sa["stats"]["dropped_duplicate"] == 1 and sb["stats"]["dropped_duplicate"] == 1  # dedup across sources
    assert sa["stats"]["dropped_too_short"] == 1
    assert sa["stats"]["dropped_too_few_letters"] == 1
    assert sa["stats"]["dropped_repetitive_lines"] == 1
    assert sa["pii_replaced"] == {"email": 1}
    assert sb["obligations"] == "attribution, share-alike"
    raw = (tmp_path / "out" / "report.json").read_text()
    assert "someone@university.edu" not in raw  # the report never contains removed data


def test_curate_refuses_unlisted_or_missing_license(tmp_path):
    p = write(tmp_path / "x.txt", GOOD)
    for lic in ("CC-BY-NC-4.0", ""):
        with pytest.raises(SystemExit):
            curate.curate({"sources": [{"name": "x", "paths": [p], "license": lic}]}, tmp_path / "o",
                          curate.DEFAULT_ALLOWED_LICENSES)


def flag_dragons(texts):
    return [1.0 if "dragon" in t else 0.0 for t in texts]


def test_curate_toxicity_hook(tmp_path):
    p = write(tmp_path / "x.txt", GOOD + "<|endoftext|>" + "A story about a dragon that is long enough to keep around.")
    scorer = curate.load_scorer("tests.test_data_governance:flag_dragons")
    report = curate.curate({"sources": [{"name": "x", "paths": [p], "doc_sep": "<|endoftext|>", "license": "MIT"}]},
                           tmp_path / "o", curate.DEFAULT_ALLOWED_LICENSES, scorer=scorer)
    assert report["totals"]["dropped_toxicity"] == 1 and report["totals"]["kept"] == 1


# ------------------------------------------------------------------ converters
@pytest.mark.parametrize("fmt,record,expected_roles", [
    ("alpaca", {"instruction": "Add", "input": "2 and 3", "output": "5"}, ["user", "assistant"]),
    ("dolly", {"instruction": "Summarize", "context": "Long text", "response": "Short", "category": "summarization"},
     ["user", "assistant"]),
    ("sharegpt", {"conversations": [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hello"},
                                    {"from": "human", "value": "Bye"}]}, ["user", "assistant"]),
    ("messages", {"messages": [{"role": "system", "content": "Be kind"}, {"role": "user", "content": "Hi"},
                               {"role": "assistant", "content": "Hello"}]}, ["system", "user", "assistant"]),
    ("hh", {"chosen": "\n\nHuman: How are you?\n\nAssistant: Fine, thanks.\n\nHuman: Great\n\nAssistant: :)"},
     ["user", "assistant", "user", "assistant"]),
    ("oasst", {"prompt": {"role": "prompter", "text": "Q", "replies": [
        {"role": "assistant", "text": "worse", "rank": 1, "replies": []},
        {"role": "assistant", "text": "best", "rank": 0, "replies": [
            {"role": "prompter", "text": "Q2", "replies": [{"role": "assistant", "text": "A2", "rank": 0, "replies": []}]}]},
    ]}}, ["user", "assistant", "user", "assistant"]),
])
def test_converters(tmp_path, fmt, record, expected_roles):
    src = tmp_path / "in.jsonl"
    src.write_text(json.dumps(record) + "\n" + json.dumps({"unrelated": True}) + "\n")
    convert.main(["--format", fmt, "--input", str(src), "--out", str(tmp_path / "out.jsonl")])
    lines = (tmp_path / "out.jsonl").read_text().splitlines()
    assert len(lines) == 1  # the malformed record is skipped
    msgs = json.loads(lines[0])["messages"]
    assert [m["role"] for m in msgs] == expected_roles
    if fmt == "oasst":
        assert msgs[1]["content"] == "best"


# ------------------------------------------------------------------ SFT data governance
@pytest.fixture
def chat_tok(tmp_path):
    tok = Tokenizer.train(["hello there my email is a real person"] * 10, vocab_size=320)
    path = tmp_path / "tok.json"
    tok.save(path)
    return path


def test_sft_requires_allowed_license(tmp_path, chat_tok):
    src = tmp_path / "c.jsonl"
    src.write_text(json.dumps({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}))
    with pytest.raises(SystemExit, match="allowlist"):
        sft_prepare(["--input", str(src), "--tokenizer", str(chat_tok), "--out", str(tmp_path / "o"),
                     "--license", "CC-BY-NC-4.0"])


def test_sft_scrubs_personal_data(tmp_path, chat_tok):
    rows = [{"messages": [{"role": "user", "content": f"my email is person{i}@realmail.com"},
                          {"role": "assistant", "content": "Thanks, I won't store it."}]} for i in range(40)]
    src = tmp_path / "c.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in rows))
    sft_prepare(["--input", str(src), "--tokenizer", str(chat_tok), "--out", str(tmp_path / "o"),
                 "--license", "THOR-owned", "--val-fraction", "0.2"])
    meta = json.loads((tmp_path / "o" / "meta.json").read_text())
    assert meta["pii_replaced"] == {"email": 40} and meta["license"] == "THOR-owned"
    tok = Tokenizer.load(chat_tok)
    import numpy as np
    text = tok.decode(np.fromfile(tmp_path / "o" / "train_tokens.bin", dtype=np.uint16).tolist())
    assert "realmail.com" not in text and "@example.com" in text
