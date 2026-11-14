"""Convert public instruction datasets to THOR conversation JSONL.

    python -m thor.convert --format dolly --input databricks-dolly-15k.jsonl --out dolly.jsonl

Output lines look like {"messages": [...], "source": "..."} and feed thor.sft_data.

Formats (from datasets catalogued in LLMDataHub):
  messages   {"messages": [{"role", "content"}]}                    e.g. HuggingFaceH4/no_robots
  alpaca     {"instruction", "input", "output"}                     Alpaca-style datasets
  dolly      {"instruction", "context", "response", "category"}     databricks/databricks-dolly-15k
  sharegpt   {"conversations": [{"from": "human"|"gpt", "value"}]}  ShareGPT-style datasets
  oasst      OpenAssistant oasst1 message *trees* (…trees.jsonl)    one conversation per tree,
             following the highest-ranked reply at each turn
  hh         Anthropic/hh-rlhf: {"chosen": "\\n\\nHuman: …\\n\\nAssistant: …"}

Converting a dataset doesn't make it usable: check its license and provenance first
(models/DATA.md). Inputs may be .jsonl or a single .json array.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

Messages = list[dict[str, str]]


def _read(path: str) -> Iterator[dict]:
    text = Path(path).read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        yield from json.loads(text)
        return
    for line in text.splitlines():
        if line.strip():
            yield json.loads(line)


def from_messages(r: dict) -> Messages:
    return [{"role": m["role"], "content": m["content"]} for m in r["messages"]]


def from_alpaca(r: dict) -> Messages:
    prompt = r["instruction"].strip()
    if (r.get("input") or "").strip():
        prompt += "\n\n" + r["input"].strip()
    return [{"role": "user", "content": prompt}, {"role": "assistant", "content": r["output"].strip()}]


def from_dolly(r: dict) -> Messages:
    prompt = r["instruction"].strip()
    if (r.get("context") or "").strip():
        prompt += "\n\n" + r["context"].strip()
    return [{"role": "user", "content": prompt}, {"role": "assistant", "content": r["response"].strip()}]


_SHAREGPT_ROLES = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant",
                   "chatgpt": "assistant", "bard": "assistant", "bing": "assistant", "system": "system"}


def from_sharegpt(r: dict) -> Messages:
    return [{"role": _SHAREGPT_ROLES[t["from"].lower()], "content": t["value"]} for t in r["conversations"]]


def from_oasst(r: dict) -> Messages:
    node = r.get("prompt", r)
    out: Messages = []
    while node:
        out.append({"role": "user" if node["role"] == "prompter" else "assistant", "content": node["text"]})
        replies = [c for c in node.get("replies", []) if not c.get("deleted") and c.get("review_result", True)]
        replies.sort(key=lambda c: (c.get("rank") is None, c.get("rank") or 0))
        node = replies[0] if replies else None
    return out


_HH_TURN = re.compile(r"\n\n(Human|Assistant): ")


def from_hh(r: dict) -> Messages:
    parts = _HH_TURN.split("\n\n" + r["chosen"].lstrip("\n").removeprefix("\n\n"))
    out: Messages = []
    for speaker, text in zip(parts[1::2], parts[2::2]):
        out.append({"role": "user" if speaker == "Human" else "assistant", "content": text.strip()})
    return out


FORMATS = {"messages": from_messages, "alpaca": from_alpaca, "dolly": from_dolly,
           "sharegpt": from_sharegpt, "oasst": from_oasst, "hh": from_hh}


def clean(messages: Messages) -> Messages | None:
    """Merge consecutive same-role turns, drop empty ones; keep only conversations ending with a reply."""
    merged: Messages = []
    for m in messages:
        content = m["content"].strip()
        if not content:
            continue
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["content"] += "\n\n" + content
        else:
            merged.append({"role": m["role"], "content": content})
    while merged and merged[-1]["role"] != "assistant":
        merged.pop()
    if not any(m["role"] == "user" for m in merged):
        return None
    return merged or None


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--format", required=True, choices=sorted(FORMATS))
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", default=None, help="Name recorded on every conversation (default: format)")
    ap.add_argument("--max", type=int, default=None, help="Stop after this many conversations")
    args = ap.parse_args(argv)

    convert = FORMATS[args.format]
    stats: Counter[str] = Counter()
    with open(args.out, "w", encoding="utf-8") as f:
        for path in args.input:
            for record in _read(path):
                if args.max is not None and stats["written"] >= args.max:
                    break
                try:
                    messages = clean(convert(record))
                except (KeyError, TypeError, AttributeError):
                    messages = None
                if not messages:
                    stats["skipped"] += 1
                    continue
                f.write(json.dumps({"messages": messages, "source": args.source or args.format}, ensure_ascii=False) + "\n")
                stats["written"] += 1
    print(json.dumps(dict(stats)))


if __name__ == "__main__":
    main()
