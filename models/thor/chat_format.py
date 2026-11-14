"""THOR chat format: how a conversation becomes tokens.

    <|system|>You are THOR.<|end|><|user|>Hi!<|end|><|assistant|>Hello.<|end|>...<|endoftext|>

Each message is its role token, the content, then <|end|>. To ask the model
for a reply, the prompt ends with <|assistant|>, and generation stops at <|end|>.

For instruction tuning, `render` also returns a loss mask: 1 only on assistant
content and the <|end|> that closes it. The model learns to write replies,
never to imitate the user or the system prompt.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

from .tokenizer import ASSISTANT, END, EOT, SYSTEM, USER, Tokenizer

CHAT_TOKENS = (SYSTEM, USER, ASSISTANT, END)
ROLE_TOKENS = {"system": SYSTEM, "user": USER, "assistant": ASSISTANT}


class Message(TypedDict):
    role: str
    content: str


def supports_chat(tok: Tokenizer) -> bool:
    return all(t in tok.special_tokens for t in CHAT_TOKENS)


def require_chat(tok: Tokenizer) -> None:
    if not supports_chat(tok):
        raise ValueError(
            "This tokenizer has no chat tokens. Tokenizers trained by `thor.prepare` since the chat "
            "format was added include them; re-run `thor.prepare` (and pretraining) with a new tokenizer."
        )


def validate(messages: Sequence[Message]) -> None:
    if not messages:
        raise ValueError("A conversation needs at least one message")
    for i, m in enumerate(messages):
        if m.get("role") not in ROLE_TOKENS:
            raise ValueError(f"Message {i}: role must be system, user or assistant, got {m.get('role')!r}")
        if not isinstance(m.get("content"), str):
            raise ValueError(f"Message {i}: content must be a string")
        if m["role"] == "system" and i != 0:
            raise ValueError("A system message may only come first")


def render(
    tok: Tokenizer, messages: Sequence[Message], add_generation_prompt: bool = False, add_eot: bool = False
) -> tuple[list[int], list[int]]:
    """Token ids and a same-length loss mask (1 = learn to predict this token)."""
    require_chat(tok)
    validate(messages)
    sp = tok.special_tokens
    ids: list[int] = []
    mask: list[int] = []
    for m in messages:
        learn = 1 if m["role"] == "assistant" else 0
        ids.append(sp[ROLE_TOKENS[m["role"]]])
        mask.append(0)  # the role token is given, not predicted
        content = tok.encode_ordinary(m["content"])  # role strings inside content stay plain text
        ids.extend(content)
        mask.extend([learn] * len(content))
        ids.append(sp[END])
        mask.append(learn)  # learning to stop is part of learning to reply
    if add_generation_prompt:
        ids.append(sp[ASSISTANT])
        mask.append(0)
    if add_eot:
        ids.append(sp[EOT])
        mask.append(0)
    return ids, mask
