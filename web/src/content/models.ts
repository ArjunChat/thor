import type { Status } from "./status";

export interface ModelSpec {
  /** Must match `name` in models/thor/config.py (checked by tests). */
  preset: string;
  params: number;
  nonEmbeddingParams: number;
  layers: number;
  dModel: number;
  heads: number;
  ffnHidden: number;
  context: number;
  vocab: number;
}

export interface Model {
  slug: string;
  name: string;
  status: Status;
  role: string;
  summary: string;
  spec?: ModelSpec;
  /** Plain statements of what has and hasn't happened. No projections. */
  training: string[];
  evaluation: string;
}

const ARCH_FAMILY = "Decoder-only Transformer · RMSNorm · RoPE · SwiGLU · tied embeddings";

export const models: Model[] = [
  {
    slug: "thor-30m",
    name: "THOR-30M",
    status: "in-development",
    role: "Educational foundation model",
    summary:
      "The first THOR model: small enough to train on a single GPU in hours, and the end-to-end proof of THOR's tokenizer, architecture and training pipeline.",
    spec: {
      preset: "thor-30m", params: 31_075_200, nonEmbeddingParams: 24_783_744, layers: 14, dModel: 384,
      heads: 6, ffnHidden: 1024, context: 1024, vocab: 16384,
    },
    training: [
      "Training pipeline built and tested; training has not started.",
      "Planned first run: TinyStories, ~655M tokens, single GPU.",
    ],
    evaluation: "No results yet. Held-out loss, perplexity and bits per byte will be published from the recorded run.",
  },
  {
    slug: "thor-100m",
    name: "THOR-100M",
    status: "in-development",
    role: "Main development target",
    summary:
      "THOR's primary model at this stage: large enough for broader general-text training, still small enough to iterate on quickly.",
    spec: {
      preset: "thor-100m", params: 97_536_768, nonEmbeddingParams: 84_953_856, layers: 12, dModel: 768,
      heads: 12, ffnHidden: 2048, context: 1024, vocab: 16384,
    },
    training: [
      "Configuration defined; not trained.",
      "Follows THOR-30M. Planned on an educational web-text corpus of ~2B tokens.",
    ],
    evaluation: "No results yet.",
  },
  {
    slug: "thor-150m",
    name: "THOR-150M",
    status: "planned",
    role: "Stretch research model",
    summary: "A larger configuration for scaling experiments once THOR-100M is trained and evaluated.",
    spec: {
      preset: "thor-150m", params: 151_184_768, nonEmbeddingParams: 136_504_704, layers: 14, dModel: 896,
      heads: 14, ffnHidden: 2432, context: 1024, vocab: 16384,
    },
    training: ["Configuration defined; not trained."],
    evaluation: "No results yet.",
  },
  {
    slug: "thor-instruct",
    name: "THOR-Instruct",
    status: "in-development",
    role: "Instruction-tuned model family",
    summary: "Instruction-tuned versions of trained THOR base models, intended to power THOR Chat.",
    training: [
      "Chat format and instruction-tuning pipeline built and tested; no model tuned yet.",
      "Needs a trained base model (THOR-30M first) and a chosen conversation dataset.",
    ],
    evaluation: "No results yet.",
  },
  {
    slug: "thor-embed",
    name: "THOR-Embed",
    status: "planned",
    role: "Embedding model",
    summary: "Text embedding models for search and retrieval, built on the THOR stack.",
    training: ["Not started."],
    evaluation: "No results yet.",
  },
];

export const architectureFamily = ARCH_FAMILY;

export function getModel(slug: string): Model | undefined {
  return models.find((m) => m.slug === slug);
}

export function formatParams(n: number): string {
  return `${(n / 1e6).toFixed(1)}M`;
}
