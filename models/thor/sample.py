"""Generate text from a trained THOR checkpoint.

Example:
    python -m thor.sample --ckpt runs/thor-30m/model_best.pt \\
        --tokenizer data/tinystories/tokenizer.json --prompt "Once upon a time"
"""

from __future__ import annotations

import argparse

import torch

from .tokenizer import Tokenizer
from .train import load_model


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    tok = Tokenizer.load(args.tokenizer)
    model, ckpt = load_model(args.ckpt, args.device)
    if ckpt["tokenizer_sha256"] != tok.fingerprint():
        raise SystemExit("This tokenizer is not the one the checkpoint was trained with")

    prompt_ids = tok.encode(args.prompt) or [tok.eot_id]
    gen = torch.Generator(device=args.device).manual_seed(args.seed)
    for i in range(args.num_samples):
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)
        out = model.generate(idx, args.max_new_tokens, args.temperature, args.top_k,
                             eos_id=tok.eot_id, generator=gen)
        text = tok.decode(t for t in out[0].tolist() if t != tok.eot_id)
        print(f"--- sample {i + 1} (step {ckpt['step']}) ---\n{text}\n")


if __name__ == "__main__":
    main()
