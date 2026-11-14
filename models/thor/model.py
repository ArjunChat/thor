"""THOR decoder-only Transformer.

Architecture (see docs/decisions/0001-model-architecture.md):
pre-norm blocks with RMSNorm, rotary position embeddings (RoPE),
causal multi-head attention via PyTorch's fused SDPA kernel, SwiGLU
feed-forward, no biases, and input/output embeddings tied by default.
"""

from __future__ import annotations

import math
from collections.abc import Collection

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        normed = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed.type_as(x) * self.weight


def rope_tables(head_dim: int, max_seq_len: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Cos/sin tables of shape (max_seq_len, head_dim // 2)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate interleaved pairs of x (B, H, T, head_dim) by position."""
    x1 = x[..., 0::2].float()
    x2 = x[..., 1::2].float()
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.stack((out1, out2), dim=-1).flatten(-2).type_as(x)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.dropout = cfg.dropout
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=2)
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_drop(self.proj(y))


class FeedForward(nn.Module):
    """SwiGLU: down(silu(gate(x)) * up(x))."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = FeedForward(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.ffn(self.ffn_norm(x))


class ThorModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        cos, sin = rope_tables(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Scaled init for projections that write into the residual stream.
        residual_std = 0.02 / math.sqrt(2 * cfg.n_layer)
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.ffn.down.weight, mean=0.0, std=residual_std)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        # .parameters() de-duplicates shared (tied) tensors.
        return sum(p.numel() for p in self.parameters())

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, t = idx.shape
        if t > self.cfg.max_seq_len:
            raise ValueError(f"Sequence length {t} exceeds max_seq_len {self.cfg.max_seq_len}")
        cos, sin = self.rope_cos[:t], self.rope_sin[:t]
        x = self.drop(self.tok_emb(idx))
        for block in self.blocks:
            x = block(x, cos, sin)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
            )
        return logits, loss

    def configure_optimizer(
        self, lr: float, weight_decay: float, betas: tuple[float, float], device_type: str
    ) -> torch.optim.AdamW:
        """AdamW with weight decay on matrices only (not norms)."""
        params = [p for p in self.parameters() if p.requires_grad]
        groups = [
            {"params": [p for p in params if p.dim() >= 2], "weight_decay": weight_decay},
            {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=device_type == "cuda")

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_id: int | Collection[int] | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Autoregressive sampling. temperature == 0 means greedy decoding.

        Stops early once every sequence has produced a stop token (eos_id: one id or several).
        """
        stop_ids = [eos_id] if isinstance(eos_id, int) else list(eos_id or ())
        stop = torch.tensor(stop_ids, device=idx.device, dtype=torch.long) if stop_ids else None
        for _ in range(max_new_tokens):
            context = idx[:, -self.cfg.max_seq_len :]
            logits, _ = self(context)
            logits = logits[:, -1, :].float()
            if temperature <= 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    kth = torch.topk(logits, k).values[:, [-1]]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1, generator=generator)
            idx = torch.cat((idx, next_id), dim=1)
            if stop is not None and bool(torch.isin(next_id, stop).all()):
                break
        return idx
