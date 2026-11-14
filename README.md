# THOR

**Intelligence, engineered.**

THOR is an independent AI company building its own language models: tokenizer, architecture, training pipeline and weights, developed in-house rather than resold from another provider.

This repository is THOR's monorepo.

| Path | What it is | Status |
|---|---|---|
| [`models/`](models/) | THOR Models: tokenizer, Transformer, training, evaluation, sampling | **Working code, tested.** No THOR model has been trained yet. |
| [`docs/decisions/`](docs/decisions/) | Architecture decision records | Active |
| [`docs/model-cards/`](docs/model-cards/) | Model card template; a card is published per model only once it is trained | Template only |
| [`brand/`](brand/) | Brand guidelines, wordmark, symbol, design tokens | Version 1 |
| [`web/`](web/) | Company website (Next.js, fully static) | Built, not deployed |

## Model program status

Status values are deliberately strict: **Planned → In development → Training → Trained (internal) → Released.** Nothing moves forward until it has actually happened.

| Model | Specification | Status |
|---|---|---|
| THOR-30M | 31.1M parameters, 14 layers, d=384, 1,024-token context | In development (pipeline ready, not trained) |
| THOR-100M | 97.5M parameters, 12 layers, d=768, 1,024-token context | In development (pipeline ready, not trained) |
| THOR-150M | 151.2M parameters, 14 layers, d=896, 1,024-token context | Planned |
| THOR-Instruct | Instruction-tuned family | Planned |
| THOR-Embed | Embedding model | Planned |

Parameter counts are specifications (the size of the network), not claims about capability. Models of this size produce fluent but shallow text and are not comparable to frontier commercial LLMs.

## Principles

- Every THOR model is trained by THOR. Output from any external API is never presented as a THOR model.
- No benchmark numbers, partners, customers or funding are published unless they are real and verifiable.
- Training data sources and their licenses are recorded before use.
- No secrets in the repository. No paid infrastructure without explicit approval.

## Quick start

```bash
cd models
pip install -e ".[dev]"
pytest -q
python -m thor.specs
```

Full instructions: [`models/README.md`](models/README.md).
