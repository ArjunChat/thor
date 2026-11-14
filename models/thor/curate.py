"""Curate raw text into clean, documented pretraining data.

    python -m thor.curate --manifest sources.json --out ../data/curated

The manifest lists every source with its license and origin:

    {"sources": [
      {"name": "common-corpus-sample", "paths": ["raw/cc/*.txt"], "doc_sep": "<|endoftext|>",
       "license": "CC0-1.0", "url": "https://huggingface.co/datasets/PleIAs/common_corpus"}
    ]}

Pipeline (after the Common Corpus approach, arXiv:2506.01732, section 5):
  1. License gate: sources whose license is not on the allowlist are refused.
  2. Quality filters: drop very short, symbol-heavy or repetitive documents.
  3. Personal data: replaced with fictitious surrogates (see thor/pii.py).
  4. Deduplication: exact duplicates after normalization, across all sources.
  5. Toxicity (optional): a pluggable classifier, e.g. a wrapper around PleIAs/celadon.

Writes curated.txt (documents separated by <|endoftext|>, ready for thor.prepare)
and report.json (provenance, licenses and counts). The report never contains
any removed personal data.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import importlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path

from . import pii
from .prepare import iter_documents
from .tokenizer import EOT

# Licenses that allow training and commercial use. Share-alike and attribution
# licenses are allowed but their obligations are recorded in the report.
DEFAULT_ALLOWED_LICENSES = {
    "public-domain", "CC0-1.0", "CC-PDDC", "CC-BY-4.0", "CC-BY-3.0", "CC-BY-SA-3.0", "CC-BY-SA-4.0",
    "ODC-By-1.0", "ODbL-1.0", "PDDL-1.0", "Apache-2.0", "MIT", "BSD-3-Clause", "CDLA-Permissive-2.0",
    "THOR-owned",
}
OBLIGATIONS = {
    "CC-BY-4.0": "attribution", "CC-BY-3.0": "attribution", "ODC-By-1.0": "attribution",
    "CC-BY-SA-3.0": "attribution, share-alike", "CC-BY-SA-4.0": "attribution, share-alike",
    "ODbL-1.0": "attribution, share-alike (database)", "Apache-2.0": "attribution, notice",
    "MIT": "attribution, notice", "BSD-3-Clause": "attribution, notice",
}


def quality_issue(doc: str, min_chars: int = 50) -> str | None:
    """Return why a document should be dropped, or None to keep it. Heuristics in the spirit of Gopher/C4."""
    if len(doc) < min_chars:
        return "too_short"
    non_space = [c for c in doc if not c.isspace()]
    if non_space:
        letters = sum(c.isalpha() for c in non_space) / len(non_space)
        if letters < 0.55:
            return "too_few_letters"
    lines = [l.strip() for l in doc.splitlines() if l.strip()]
    if len(lines) >= 5:
        repeated = 1 - len(set(lines)) / len(lines)
        if repeated > 0.3:
            return "repetitive_lines"
    words = doc.split()
    if len(words) >= 20 and len(set(words)) / len(words) < 0.2:
        return "repetitive_words"
    return None


def normalize_for_dedup(doc: str) -> str:
    doc = unicodedata.normalize("NFKC", doc).lower()
    doc = re.sub(r"[^\w\s]", "", doc)
    return re.sub(r"\s+", " ", doc).strip()


def load_scorer(spec: str) -> Callable[[list[str]], list[float]]:
    """'package.module:function'. The function takes a list of texts and returns toxicity scores in [0, 1]."""
    module, _, name = spec.partition(":")
    if not name:
        raise ValueError("--toxicity-scorer must look like 'package.module:function'")
    return getattr(importlib.import_module(module), name)


def curate(manifest: dict, out: Path, allowed: set[str], min_chars: int = 50,
           scorer: Callable[[list[str]], list[float]] | None = None, toxicity_threshold: float = 0.5,
           batch: int = 64) -> dict:
    sources = manifest.get("sources") or []
    if not sources:
        raise SystemExit("Manifest has no sources")
    for s in sources:
        for key in ("name", "license"):
            if not s.get(key):
                raise SystemExit(f"Every source needs a {key!r} (problem in {s})")
        if s["license"] not in allowed:
            raise SystemExit(
                f"Source {s['name']!r} has license {s['license']!r}, which is not on the allowlist. "
                "Check the dataset card; pass --allow-license only after confirming the terms permit training."
            )

    out.mkdir(parents=True, exist_ok=True)
    seen: set[bytes] = set()
    report: dict = {"sources": [], "totals": Counter()}
    with open(out / "curated.txt", "w", encoding="utf-8") as f:
        first = True
        for s in sources:
            paths = sorted({Path(p) for pattern in s.get("paths", [s.get("path")]) if pattern for p in glob.glob(pattern)})
            if not paths:
                raise SystemExit(f"Source {s['name']!r}: no files match {s.get('paths') or s.get('path')}")
            stats: Counter[str] = Counter()
            pii_counts: Counter[str] = Counter()
            pending: list[str] = []

            def flush() -> Iterator[str]:
                scores = scorer(pending) if scorer else [0.0] * len(pending)
                for doc, score in zip(pending, scores):
                    if score >= toxicity_threshold:
                        stats["dropped_toxicity"] += 1
                    else:
                        yield doc
                pending.clear()

            def emit(docs: Iterator[str]) -> None:
                nonlocal first
                for doc in docs:
                    if not first:
                        f.write(EOT)
                    f.write(doc)
                    first = False
                    stats["kept"] += 1
                    stats["kept_chars"] += len(doc)

            for doc in iter_documents(paths, s.get("doc_sep")):
                stats["documents"] += 1
                doc = doc.replace(EOT, " ")  # a literal separator inside a document would split it
                if s.get("quality", True):
                    issue = quality_issue(doc, min_chars)
                    if issue:
                        stats[f"dropped_{issue}"] += 1
                        continue
                doc, found = pii.scrub(doc, mode="surrogate")
                pii_counts.update(found)
                stats["documents_with_pii"] += bool(found)
                key = hashlib.blake2b(normalize_for_dedup(doc).encode(), digest_size=16).digest()
                if key in seen:
                    stats["dropped_duplicate"] += 1
                    continue
                seen.add(key)
                pending.append(doc)
                if len(pending) >= batch:
                    emit(flush())
            if pending:
                emit(flush())

            report["sources"].append({
                "name": s["name"], "license": s["license"], "obligations": OBLIGATIONS.get(s["license"], "none recorded"),
                "url": s.get("url"), "files": [{"path": str(p), "bytes": p.stat().st_size} for p in paths],
                "stats": dict(stats), "pii_replaced": dict(pii_counts),
            })
            report["totals"].update(stats)
    report["totals"] = dict(report["totals"])
    report["config"] = {"min_chars": min_chars, "toxicity_filter": bool(scorer),
                        "toxicity_threshold": toxicity_threshold if scorer else None,
                        "pii_mode": "surrogate", "dedup": "exact, normalized"}
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-chars", type=int, default=50)
    ap.add_argument("--allow-license", action="append", default=[],
                    help="Add a license to the allowlist (only after checking its terms)")
    ap.add_argument("--toxicity-scorer", default=None, help="package.module:function returning scores in [0, 1]")
    ap.add_argument("--toxicity-threshold", type=float, default=0.5)
    args = ap.parse_args(argv)

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    scorer = load_scorer(args.toxicity_scorer) if args.toxicity_scorer else None
    report = curate(manifest, Path(args.out), DEFAULT_ALLOWED_LICENSES | set(args.allow_license),
                    args.min_chars, scorer, args.toxicity_threshold)
    print(json.dumps(report["totals"], indent=2))
    print(f"Wrote {Path(args.out) / 'curated.txt'} and report.json")


if __name__ == "__main__":
    main()
