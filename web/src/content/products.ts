import type { ComponentStatus, Status } from "./status";

export interface Product {
  slug: string;
  name: string;
  status: Status;
  summary: string;
  detail: string;
}

/** The THOR ecosystem, in the order it is being built. */
export const products: Product[] = [
  {
    slug: "models",
    name: "THOR Models",
    status: "in-development",
    summary: "THOR's own language models, trained from scratch.",
    detail: "Tokenizer, architecture and training pipeline are built and tested. The first model, THOR-30M, has not been trained yet.",
  },
  {
    slug: "research",
    name: "THOR Research",
    status: "in-development",
    summary: "Experiments, engineering notes and technical reports.",
    detail: "Engineering notes on the architecture, tokenizer and training system are published. Technical reports follow trained models.",
  },
  {
    slug: "chat",
    name: "THOR Chat",
    status: "planned",
    summary: "A public conversational AI application.",
    detail: "Will run on THOR's own instruction-tuned models once they are trained. Any external model used in the meantime would be clearly labelled as such.",
  },
  {
    slug: "studio",
    name: "THOR Studio",
    status: "planned",
    summary: "A platform for developing with THOR models.",
    detail: "Tools for evaluating, fine-tuning and integrating THOR models.",
  },
  {
    slug: "cloud",
    name: "THOR Cloud",
    status: "planned",
    summary: "Inference and AI infrastructure.",
    detail: "Hosted inference for THOR models, and the API behind THOR Chat and THOR Studio.",
  },
];

export interface PipelineStage {
  name: string;
  status: ComponentStatus;
  detail: string;
}

/** What exists in code today. Every "working" stage has tests in models/tests. */
export const pipeline: PipelineStage[] = [
  { name: "Tokenizer", status: "working", detail: "Byte-level BPE trained in-house. Every string round-trips exactly." },
  { name: "Architecture", status: "working", detail: "Decoder-only Transformer with RoPE, SwiGLU and RMSNorm, in three sizes." },
  { name: "Training", status: "working", detail: "Atomic checkpoints and bit-exact resume after any interruption." },
  { name: "Evaluation", status: "working", detail: "Deterministic held-out loss, perplexity and bits per byte." },
  { name: "Instruction tuning", status: "working", detail: "Chat format and fine-tuning that learns only from assistant replies." },
  { name: "Inference service", status: "planned", detail: "An independent serving layer for THOR Chat and the API." },
];
