"""Chat with a THOR checkpoint in the terminal.

    python -m thor.chat --ckpt runs/thor-30m-chat/model_best.pt --tokenizer data/chat-sft/tokenizer.json

Commands: /reset clears the conversation, /exit quits. Older turns are dropped
when the conversation no longer fits the model's context window.

Only an instruction-tuned checkpoint (trained with thor.sft_data + thor.train
--init-from) is trained to reply. A base model will just continue the text.
"""

from __future__ import annotations

import argparse
import sys

import torch

from .chat_format import END, Message, render, require_chat
from .model import ThorModel
from .tokenizer import Tokenizer
from .train import load_model


class ChatSession:
    def __init__(self, model: ThorModel, tok: Tokenizer, system: str | None = None,
                 max_new_tokens: int = 256, temperature: float = 0.7, top_k: int | None = 50,
                 seed: int = 0, device: str = "cpu") -> None:
        require_chat(tok)
        self.model, self.tok = model, tok
        self.system = system
        self.max_new_tokens = max_new_tokens
        self.temperature, self.top_k = temperature, top_k
        self.device = device
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.history: list[Message] = []

    def reset(self) -> None:
        self.history = []

    def _prompt_ids(self) -> list[int]:
        """Render the conversation, dropping the oldest turns until it fits with room to reply."""
        budget = self.model.cfg.max_seq_len - min(self.max_new_tokens, self.model.cfg.max_seq_len // 2)
        turns = list(self.history)
        while True:
            messages: list[Message] = ([{"role": "system", "content": self.system}] if self.system else []) + turns
            ids, _ = render(self.tok, messages, add_generation_prompt=True)
            if len(ids) <= budget or len(turns) <= 1:
                return ids[-budget:]
            turns = turns[2:] if len(turns) > 2 else turns[1:]  # drop the oldest user/assistant pair

    def reply(self, user_text: str) -> str:
        self.history.append({"role": "user", "content": user_text})
        prompt = self._prompt_ids()
        idx = torch.tensor([prompt], dtype=torch.long, device=self.device)
        stop = [self.tok.special_tokens[END], self.tok.eot_id]
        out = self.model.generate(idx, self.max_new_tokens, self.temperature, self.top_k,
                                  eos_id=stop, generator=self.generator)[0, len(prompt):].tolist()
        out = [t for t in out if t not in stop and t not in self.tok.special_ids]
        text = self.tok.decode(out).strip()
        self.history.append({"role": "assistant", "content": text})
        return text


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--system", default=None, help="Optional system message")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    tok = Tokenizer.load(args.tokenizer)
    model, ckpt = load_model(args.ckpt, args.device)
    if ckpt["tokenizer_sha256"] != tok.fingerprint():
        raise SystemExit("This tokenizer is not the one the checkpoint was trained with")
    try:
        session = ChatSession(model, tok, args.system, args.max_new_tokens, args.temperature,
                              args.top_k, args.seed, args.device)
    except ValueError as e:
        raise SystemExit(str(e)) from None

    print(f"THOR chat · {model.cfg.name} · {model.num_params() / 1e6:.1f}M parameters · step {ckpt.get('step')}")
    if ckpt.get("stage") != "sft":
        print("Note: this is a base model, not instruction-tuned. It will continue text rather than reply.")
    print("Type /reset to start over, /exit to quit.\n")
    while True:
        try:
            user = input("you › ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/exit", "/quit"):
            break
        if user == "/reset":
            session.reset()
            print("(conversation cleared)\n")
            continue
        print(f"thor › {session.reply(user)}\n", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
