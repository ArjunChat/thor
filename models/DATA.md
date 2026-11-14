# Training data register

Every dataset used to train or evaluate a THOR model is listed here **before** it is used, with its license and how it was obtained. Datasets are never committed to git.

| Dataset | Used for | Source | License | Status |
|---|---|---|---|---|
| Tiny Shakespeare (~1.1 MB) | CPU pipeline smoke test only; not used for any THOR model | `raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt` | Shakespeare's works are public domain | Used for smoke test |
| TinyStories | Candidate for THOR-30M | huggingface.co/datasets/roneneldan/TinyStories | Verify the license on the dataset card before use | Candidate, not yet used |
| FineWeb-Edu (sample) | Candidate for THOR-100M | huggingface.co/datasets/HuggingFaceFW/fineweb-edu | Verify the license on the dataset card before use | Candidate, not yet used |

When a dataset is adopted, add the exact file names, download date, size, and `meta.json` token counts.
