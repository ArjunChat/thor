# Model card: THOR-<SIZE>

> Publish a card only after the model has actually been trained. Every number below must come from a recorded run (`metrics.jsonl`, `thor.evaluate` output); never estimate or round up.

| | |
|---|---|
| **Status** | Trained (internal) / Released |
| **Parameters** | from `python -m thor.specs` |
| **Architecture** | decoder-only Transformer; see docs/decisions/0001 |
| **Context length** | |
| **Tokenizer** | THOR BPE, vocab N, fingerprint `sha256:…` |
| **Training data** | dataset(s), versions, token counts; see models/DATA.md |
| **Training compute** | hardware, hours, tokens seen, steps |
| **Checkpoint** | step, file, SHA-256 |

## Evaluation

| Benchmark | Split | Metric | Result | Command |
|---|---|---|---|---|
| Held-out validation | val | loss / perplexity / bits per byte | | `python -m thor.evaluate …` |

## Intended use

Research and education.

## Limitations

State plainly what the model cannot do. Small models produce fluent but often incorrect or incoherent text, have no reliable factual knowledge, and are not instruction-following unless tuned.

## License

State the license for the weights explicitly at release. The training code is Apache-2.0 (see the repository LICENSE), but that does not automatically cover weights.
