# 0006: External models as clearly labelled test engines

**Status:** Accepted · **Date:** 2026-11-20

## Context

THOR Chat needs to be usable and testable before any THOR model is trained. Free LLM APIs (Google Gemini's free tier, and others with OpenAI-compatible endpoints) make that possible. Their terms constrain how they can be used. Gemini's Additional Terms (checked 2026-11-20) say:

- "You may not use the Services to develop models that compete with the Services."
- On unpaid services, Google uses submitted content and responses to improve its products, "human reviewers may read, annotate, and process your API input and output", and users should not submit sensitive, confidential or personal information.
- Users must be 18+, and only paid services may serve users in the EEA, Switzerland or the UK.

## Decision

1. **Engines.** THOR Chat (`thor.chat`, `thor.serve`) talks to swappable backends: `ThorBackend` (a THOR checkpoint) or `ExternalBackend` (any OpenAI-compatible API: Gemini, Groq, OpenRouter).
2. **Labelling.** Every external reply is marked "External · not THOR" with the model and provider name, in the terminal and in the browser. External models get a system prompt telling them not to present themselves as THOR. Replies written by the safety layer are labelled as such.
3. **No training on external output.** External engines are for testing the product only. Their outputs are never used as THOR training data (provider terms, and decision 0005).
4. **Privacy.** Before a conversation leaves the machine, personal data in every message is replaced with tags (`[email removed]`). The safety layer screens requests first: a blocked request is never sent to the provider.
5. **Scope.** Free tiers are for local, personal testing. `thor.serve` binds to 127.0.0.1 by default and has no authentication. Offering an external engine to the public would need a paid tier, region and age compliance, and a server with authentication and rate limiting: a separate decision.

## Consequences

- People can test THOR Chat today, and the UI makes clear they're not talking to a THOR model.
- When a THOR chat model is trained, it slots in as another engine, and the two can be compared side by side.
- API keys stay in environment variables on the user's machine and are never written to the repository.
