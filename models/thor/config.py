"""Model configurations and the THOR model presets.

Parameter counts here are *specifications* — the size of the network —
not statements about capability. ``num_params`` is computed analytically
and checked against the real module in the test suite.
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    """Shape of a decoder-only THOR Transformer."""

    name: str = "custom"
    vocab_size: int = 16384
    d_model: int = 384
    n_layer: int = 12
    n_head: int = 6
    ffn_hidden: int = 1024
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    dropout: float = 0.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_head != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_head ({self.n_head})")
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim ({self.head_dim}) must be even for rotary embeddings")
        for field in ("vocab_size", "d_model", "n_layer", "n_head", "ffn_hidden", "max_seq_len"):
            if getattr(self, field) <= 0:
                raise ValueError(f"{field} must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    def num_params(self, include_embedding: bool = True) -> int:
        """Exact trainable-parameter count (tied embeddings counted once)."""
        d, h = self.d_model, self.ffn_hidden
        per_layer = (
            4 * d * d  # q, k, v, output projections (no biases)
            + 3 * d * h  # SwiGLU: gate, up, down
            + 2 * d  # two RMSNorm weights
        )
        body = self.n_layer * per_layer + d  # + final RMSNorm
        if not include_embedding:
            return body
        embedding = self.vocab_size * d
        head = 0 if self.tie_embeddings else self.vocab_size * d
        return body + embedding + head

    def flops_per_token(self, seq_len: int | None = None) -> int:
        """Approximate training FLOPs per token (forward + backward).

        Uses the standard 6N estimate plus the attention term
        6 * n_layer * seq_len * d_model.
        """
        t = seq_len or self.max_seq_len
        return 6 * self.num_params() + 6 * self.n_layer * t * self.d_model

    def with_vocab_size(self, vocab_size: int) -> "ModelConfig":
        return dataclasses.replace(self, vocab_size=vocab_size)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown model config keys: {sorted(unknown)}")
        return cls(**data)


# Presets. vocab_size is the default for a freshly trained THOR tokenizer;
# training replaces it with the actual tokenizer's vocabulary size.
PRESETS: dict[str, ModelConfig] = {
    # Tiny network for tests and CPU smoke runs. Not a THOR model release.
    "thor-nano": ModelConfig(
        name="thor-nano", vocab_size=2048, d_model=128, n_layer=4, n_head=4,
        ffn_hidden=352, max_seq_len=256,
    ),
    # Educational foundational model.
    "thor-30m": ModelConfig(
        name="thor-30m", vocab_size=16384, d_model=384, n_layer=14, n_head=6,
        ffn_hidden=1024, max_seq_len=1024,
    ),
    # Main development target.
    "thor-100m": ModelConfig(
        name="thor-100m", vocab_size=16384, d_model=768, n_layer=12, n_head=12,
        ffn_hidden=2048, max_seq_len=1024,
    ),
    # Stretch research model.
    "thor-150m": ModelConfig(
        name="thor-150m", vocab_size=16384, d_model=896, n_layer=14, n_head=14,
        ffn_hidden=2432, max_seq_len=1024,
    ),
}


def get_preset(name: str) -> ModelConfig:
    try:
        return PRESETS[name]
    except KeyError:
        raise ValueError(f"Unknown preset {name!r}. Available: {', '.join(PRESETS)}") from None
