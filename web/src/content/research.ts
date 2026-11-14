import type { Status } from "./status";

export type ResearchKind = "Engineering note" | "Experiment" | "Technical report";

export interface ResearchSection {
  heading: string;
  paragraphs?: string[];
  table?: { head: string[]; rows: string[][] };
  list?: string[];
}

export interface ResearchEntry {
  slug: string;
  kind: ResearchKind;
  title: string;
  /** Publication date, or undefined for planned work. */
  date?: string;
  status: Status;
  summary: string;
  /** Path in the THOR repository with the full source document. */
  source?: string;
  sections: ResearchSection[];
}

export const research: ResearchEntry[] = [
  {
    slug: "pipeline-validation",
    kind: "Experiment",
    title: "Validating the training pipeline end to end",
    date: "2026-11-14",
    status: "released",
    summary:
      "A 1.1M-parameter test network trained on CPU on Tiny Shakespeare to check that tokenizer, model, training, checkpointing and evaluation work together. It is a pipeline check, not a THOR model.",
    sections: [
      {
        heading: "Purpose",
        paragraphs: [
          "Before spending GPU time on THOR-30M, every stage of the pipeline was run together on real text: train a tokenizer, tokenize a corpus, train a model with checkpoints, then evaluate and sample from it.",
          "The network used here (thor-nano) exists only for tests and smoke runs. It is far too small to be useful and is not a THOR model release.",
        ],
      },
      {
        heading: "Setup",
        table: {
          head: ["", ""],
          rows: [
            ["Data", "Tiny Shakespeare, 1.1 MB, public domain. 6,864 training / 358 validation passages"],
            ["Tokenizer", "THOR BPE, 2,048 tokens, trained on the training split only (≈2.9 bytes per token)"],
            ["Tokens", "363,164 train · 18,154 validation"],
            ["Model", "thor-nano: 1,066,112 parameters, 4 layers, d_model 128, context 256"],
            ["Training", "1,500 steps × 4,096 tokens (6.1M tokens), AdamW, cosine schedule, dropout 0.1"],
            ["Hardware", "4-core cloud CPU, ≈10k tokens/s"],
          ],
        },
      },
      {
        heading: "Results",
        paragraphs: ["Held-out results from a full deterministic pass over the validation set (thor.evaluate, step 1,500):"],
        table: {
          head: ["Metric", "Value"],
          rows: [
            ["Validation loss", "3.6275"],
            ["Perplexity", "37.62"],
            ["Bits per byte", "1.82"],
            ["Loss at initialization", "7.65"],
          ],
        },
      },
      {
        heading: "What it shows",
        list: [
          "Loss fell steadily and samples follow the corpus's line structure and vocabulary but are not coherent, which is expected at this size.",
          "Training loss pulled below validation loss over time (3.17 vs 3.62), the expected sign of a small corpus being seen many times.",
          "Separately, the test suite verifies that an interrupted and resumed run produces bit-identical weights to an uninterrupted one.",
        ],
      },
    ],
  },
  {
    slug: "architecture",
    kind: "Engineering note",
    title: "The architecture of the THOR model family",
    date: "2026-11-14",
    status: "released",
    summary: "Why THOR models are pre-norm decoder-only Transformers with RoPE, SwiGLU, RMSNorm and tied embeddings, and how the three sizes are specified.",
    source: "docs/decisions/0001-model-architecture.md",
    sections: [
      {
        heading: "Design",
        paragraphs: [
          "THOR models are decoder-only Transformers built from well-understood components, chosen for stable training and quality per parameter at the 30M–150M scale.",
        ],
        table: {
          head: ["Choice", "Reason"],
          rows: [
            ["RMSNorm, pre-norm", "Stable training at depth; cheaper than LayerNorm"],
            ["Rotary position embeddings", "Relative positions without a learned table; room to extend context later"],
            ["SwiGLU feed-forward", "Better quality per parameter than a GELU MLP of the same size"],
            ["No bias terms", "Fewer parameters with no observed quality cost"],
            ["Tied input/output embeddings", "Untied, a 16k vocabulary would add 6–15M parameters: a large share of a 30M model"],
            ["Fused causal attention", "PyTorch's scaled-dot-product kernel uses FlashAttention-style kernels on GPU"],
          ],
        },
      },
      {
        heading: "Verified specifications",
        paragraphs: [
          "Parameter counts are computed analytically and checked against the instantiated network in the test suite. Every size must be within 10% of its name. A count is a specification of size, not a claim about capability.",
        ],
      },
    ],
  },
  {
    slug: "tokenizer",
    kind: "Engineering note",
    title: "A byte-level BPE tokenizer, built in-house",
    date: "2026-11-14",
    status: "released",
    summary: "THOR trains its own tokenizer rather than borrowing another model's vocabulary. How it works, and how it is kept in lockstep with data and weights.",
    source: "docs/decisions/0002-tokenizer.md",
    sections: [
      {
        heading: "How it works",
        list: [
          "The base vocabulary is all 256 byte values, so any UTF-8 text round-trips exactly and there is never an unknown token.",
          "A pre-tokenization pattern splits text into words, numbers, symbols and whitespace; merges never cross those boundaries. A randomized test checks that every character is covered.",
          "Training is incremental: a priority queue and a pair-to-document index mean each merge only touches the words it affects. A brute-force reference implementation in the tests must learn identical merges.",
          "The default vocabulary is 16,384 tokens: small enough that embeddings don't dominate a 30M model, and token ids fit in 16 bits.",
        ],
      },
      {
        heading: "Consistency",
        paragraphs: [
          "Every tokenizer has a SHA-256 fingerprint, recorded in the prepared dataset and in every checkpoint. Training, evaluation and sampling refuse to combine mismatched pieces.",
        ],
      },
    ],
  },
  {
    slug: "training-system",
    kind: "Engineering note",
    title: "Training that survives interruption",
    date: "2026-11-14",
    status: "released",
    summary: "Affordable GPUs get interrupted. THOR's training loop checkpoints atomically and resumes bit-for-bit, so no compute is wasted or silently changed.",
    source: "docs/decisions/0003-training-and-checkpoints.md",
    sections: [
      {
        heading: "Mechanics",
        list: [
          "AdamW, linear warmup with cosine decay, gradient clipping and accumulation; bf16 on GPUs with native support, fp16 with loss scaling on older ones.",
          "Checkpoints hold the full training state: weights, optimizer, step, and every random-number generator, including the batch sampler.",
          "Writes are atomic (write, then rename), so a crash mid-save never corrupts the last good checkpoint.",
          "Ctrl-C, SIGTERM or a per-session step limit finish the current step, save and exit cleanly.",
        ],
      },
      {
        heading: "Guarantees",
        list: [
          "Resume is bit-exact: a test compares a 12-step run against a 6 + 6 interrupted run and requires identical weights.",
          "Training refuses to resume across a change of model shape or tokenizer, and stops without overwriting anything if the loss becomes non-finite.",
        ],
      },
    ],
  },
  {
    slug: "thor-30m-report",
    kind: "Technical report",
    title: "THOR-30M: training report",
    status: "planned",
    summary: "Data, compute, training curves, evaluation and samples for THOR-30M. Will be published from the recorded run once training is complete.",
    sections: [
      {
        heading: "Not yet published",
        paragraphs: ["THOR-30M has not been trained. This report will contain only measured results from the actual run."],
      },
    ],
  },
];

export function getResearch(slug: string): ResearchEntry | undefined {
  return research.find((r) => r.slug === slug);
}
