# 0004: Chat format and instruction tuning

**Status:** Accepted · **Date:** 2026-11-14

## Context

A pretrained base model only continues text. To chat, it must learn a conversation format and how to write a reply, via supervised fine-tuning on conversations (THOR-Instruct). This is the foundation for THOR Chat.

## Decision

**Format** (`models/thor/chat_format.py`): every message is its role token, the content, then `<|end|>`:

```
<|system|>…<|end|><|user|>…<|end|><|assistant|>…<|end|><|endoftext|>
```

- Four special tokens (`<|system|>`, `<|user|>`, `<|assistant|>`, `<|end|>`) are part of every new THOR tokenizer by default, alongside `<|endoftext|>`.
- Message content is encoded as plain text, so a user typing `<|assistant|>` can't inject a role token.
- To generate, the prompt ends with `<|assistant|>`; generation stops at `<|end|>`.

**Loss mask:** the model learns only the assistant's reply tokens and the `<|end|>` that closes each reply. It never trains to imitate users or system prompts. Learning `<|end|>` teaches it when to stop.

**Training:** instruction tuning reuses the pretraining loop (`thor.train --init-from <base checkpoint>` on data from `thor.sft_data`), so it inherits atomic checkpoints, bit-exact resume and safe interruption. Checkpoints record `stage: "sft"`; `thor.chat` warns when given a base model.

**Data:** JSONL conversations (`{"messages": [...]}`), split train/validation by content hash. Conversations with no learnable reply are skipped; over-length ones are truncated and counted.

## Consequences

- Tokenizers trained before this change have no chat tokens and can't be instruction-tuned. THOR has not trained a real model yet, so nothing is lost.
- The chat dataset is a separate decision with licensing implications; see `models/DATA.md`.
- Chat context is the model's context window (1,024 tokens for THOR presets); `thor.chat` drops the oldest turns when a conversation outgrows it.
