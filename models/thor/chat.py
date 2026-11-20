"""Chat in the terminal with a THOR checkpoint, or with an external model for testing.

    python -m thor.chat --ckpt runs/thor-30m-chat/model_best.pt --tokenizer data/chat-sft/tokenizer.json
    GEMINI_API_KEY=... python -m thor.chat --provider gemini      # external model, labelled as such

Commands: /reset clears the conversation, /exit quits. Older turns are dropped
when the conversation no longer fits the model's context window.

The safety layer (thor/safety.py) is on by default: it screens requests and
replies and redacts personal data from replies. --no-safety turns it off for
research use only.

Only an instruction-tuned checkpoint (trained with thor.sft_data + thor.train
--init-from) is trained to reply. A base model will just continue the text.
"""

from __future__ import annotations

import argparse
import sys

from . import pii
from .chat_format import Message
from .model import ThorModel
from .providers import PROVIDERS, BackendError, ExternalBackend, ThorBackend
from .safety import SafetyGuard
from .tokenizer import Tokenizer
from .train import load_model


class ChatSession:
    """A conversation with any backend, behind the safety layer.

    Pass a THOR model and tokenizer (the usual case), or any backend from
    thor.providers. With an external backend, personal data in the conversation
    is replaced with tags before it is sent off the machine.
    """

    def __init__(self, model: ThorModel | None = None, tok: Tokenizer | None = None, system: str | None = None,
                 max_new_tokens: int = 256, temperature: float = 0.7, top_k: int | None = 50,
                 seed: int = 0, device: str = "cpu", guard: SafetyGuard | None = None,
                 backend: ThorBackend | ExternalBackend | None = None, scrub_outbound: bool = True) -> None:
        if backend is None:
            if model is None or tok is None:
                raise ValueError("Pass a model and tokenizer, or a backend")
            backend = ThorBackend(model, tok, max_new_tokens, temperature, top_k, seed, device)
        self.backend = backend
        self.system = system
        self.history: list[Message] = []
        self.guard = guard
        self.scrub_outbound = scrub_outbound
        self.last_safety: dict = {}  # what the safety layer did on the last turn

    def reset(self) -> None:
        self.history = []

    def _outbound(self) -> list[Message]:
        messages: list[Message] = ([{"role": "system", "content": self.system}] if self.system else []) + self.history
        if self.backend.is_thor or not self.scrub_outbound:
            return messages
        withheld: dict[str, int] = {}
        safe: list[Message] = []
        for m in messages:
            content, found = pii.scrub(m["content"], mode="tag")
            for k, v in found.items():
                withheld[k] = withheld.get(k, 0) + v
            safe.append({"role": m["role"], "content": content})
        if withheld:
            self.last_safety["pii_withheld"] = withheld
        return safe

    def reply(self, user_text: str) -> str:
        self.history.append({"role": "user", "content": user_text})
        self.last_safety = {}
        if self.guard is not None:
            verdict = self.guard.check_request(user_text)
            if not verdict.allowed:
                self.last_safety = {"request_blocked": verdict.category}
                self.history.append({"role": "assistant", "content": verdict.response or ""})
                return verdict.response or ""
        try:
            text = self.backend.complete(self._outbound())
        except BackendError:
            self.history.pop()  # keep history consistent: the turn didn't happen
            raise
        if self.guard is not None:
            text, changes = self.guard.filter_reply(text)
            self.last_safety.update(changes)
        self.history.append({"role": "assistant", "content": text})
        return text


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", help="A THOR checkpoint")
    src.add_argument("--provider", choices=sorted(PROVIDERS), help="An external model API (not a THOR model)")
    ap.add_argument("--tokenizer", help="Tokenizer for --ckpt")
    ap.add_argument("--model", default=None, help="Model name for --provider")
    ap.add_argument("--system", default=None, help="Optional system message")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-safety", action="store_true", help="Disable the safety layer (research only)")
    args = ap.parse_args(argv)

    guard = None if args.no_safety else SafetyGuard()
    if args.ckpt:
        if not args.tokenizer:
            raise SystemExit("--ckpt needs --tokenizer")
        tok = Tokenizer.load(args.tokenizer)
        model, ckpt = load_model(args.ckpt, args.device)
        if ckpt["tokenizer_sha256"] != tok.fingerprint():
            raise SystemExit("This tokenizer is not the one the checkpoint was trained with")
        try:
            backend = ThorBackend(model, tok, args.max_new_tokens, args.temperature, args.top_k, args.seed,
                                  args.device, stage=ckpt.get("stage"))
        except ValueError as e:
            raise SystemExit(str(e)) from None
        prompt = "thor ›"
        print(f"THOR chat · {model.cfg.name} · {model.num_params() / 1e6:.1f}M parameters · step {ckpt.get('step')}")
        if ckpt.get("stage") != "sft":
            print("Note: this is a base model, not instruction-tuned. It will continue text rather than reply.")
    else:
        backend = ExternalBackend(args.provider, args.model, temperature=args.temperature,
                                  max_tokens=args.max_new_tokens)
        prompt = f"{backend.model} (external) ›"
        print(f"THOR chat · {backend.label}")
        print("Replies come from a third-party model, NOT a THOR model. Personal data in your messages is")
        print(f"replaced before sending, but {backend.provider.name}'s terms apply to everything you send.")
    session = ChatSession(guard=guard, backend=backend)
    if args.no_safety:
        print("Warning: the safety layer is OFF. Replies are not screened or redacted.")
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
        try:
            print(f"{prompt} {session.reply(user)}\n", flush=True)
        except BackendError as e:
            print(f"(error) {e}\n", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
