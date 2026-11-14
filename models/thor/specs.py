"""Print the THOR preset specification table:  python -m thor.specs"""

from __future__ import annotations

from .config import PRESETS


def _format_count(n: int) -> str:
    return f"{n / 1e6:.1f}M"


def main() -> None:
    """Print the preset specification table (used for docs and the website)."""
    header = f"{'preset':<11} {'params':>8} {'non-emb':>8} {'layers':>6} {'d_model':>7} {'heads':>5} {'ffn':>5} {'ctx':>5} {'vocab':>6}"
    print(header)
    print("-" * len(header))
    for cfg in PRESETS.values():
        print(
            f"{cfg.name:<11} {_format_count(cfg.num_params()):>8} "
            f"{_format_count(cfg.num_params(include_embedding=False)):>8} {cfg.n_layer:>6} "
            f"{cfg.d_model:>7} {cfg.n_head:>5} {cfg.ffn_hidden:>5} {cfg.max_seq_len:>5} {cfg.vocab_size:>6}"
        )

    print("\nCompute to train each preset on 20 tokens per parameter (estimates, not measurements):")
    for cfg in PRESETS.values():
        if cfg.name == "thor-nano":
            continue
        tokens = 20 * cfg.num_params()
        flops = cfg.flops_per_token() * tokens
        print(f"  {cfg.name:<10} {tokens / 1e9:5.2f}B tokens  {flops:.2e} FLOPs  "
              f"~{flops / (312e12 * 0.4) / 3600:4.1f} h on 1x A100 (40% util)  "
              f"~{flops / (65e12 * 0.25) / 3600:5.1f} h on 1x T4 (25% util)")


if __name__ == "__main__":
    main()
