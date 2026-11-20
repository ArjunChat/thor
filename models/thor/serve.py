"""THOR Chat in the browser, for local testing.

    GEMINI_API_KEY=... python -m thor.serve --provider gemini
    python -m thor.serve --thor-ckpt runs/thor-30m-chat/model_best.pt --thor-tokenizer data/chat-sft/tokenizer.json
    # then open http://127.0.0.1:8800

Serves one page and a small JSON API, using only the Python standard library.
Every reply is labelled with the engine that produced it: a THOR model, or an
external model that is not THOR. The safety layer is on for every engine.

It listens on 127.0.0.1 only by default. It is a testing tool, not a production
server: there is no authentication or rate limiting.
"""

from __future__ import annotations

import argparse
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .chat import ChatSession
from .providers import PROVIDERS, BackendError, ExternalBackend, ThorBackend
from .safety import SafetyGuard

PAGE = Path(__file__).with_name("static") / "chat.html"
MAX_BODY = 256 * 1024
MAX_MESSAGES = 100


class App:
    def __init__(self, backends: dict[str, ThorBackend | ExternalBackend], safety: bool = True) -> None:
        self.backends = backends
        self.safety = safety
        self.locks = {k: threading.Lock() for k in backends}  # a local THOR model handles one request at a time

    def describe(self) -> list[dict]:
        out = [{"id": b.id, "label": b.label, "is_thor": b.is_thor, "available": True} for b in self.backends.values()]
        if not any(b.is_thor for b in self.backends.values()):
            out.insert(0, {"id": "thor:none", "label": "THOR model · no trained chat model yet", "is_thor": True,
                           "available": False})
        return out

    def chat(self, payload: dict) -> tuple[int, dict]:
        backend = self.backends.get(payload.get("backend", ""))
        if backend is None:
            return HTTPStatus.BAD_REQUEST, {"error": "Unknown or unavailable engine"}
        messages = payload.get("messages")
        if (not isinstance(messages, list) or not messages or len(messages) > MAX_MESSAGES
                or any(not isinstance(m, dict) or m.get("role") not in ("user", "assistant")
                       or not isinstance(m.get("content"), str) for m in messages)
                or messages[-1]["role"] != "user"):
            return HTTPStatus.BAD_REQUEST, {"error": "messages must be user/assistant turns ending with a user turn"}
        session = ChatSession(backend=backend, guard=SafetyGuard() if self.safety else None)
        session.history = [{"role": m["role"], "content": m["content"]} for m in messages[:-1]]
        try:
            with self.locks[backend.id]:
                reply = session.reply(messages[-1]["content"])
        except BackendError as e:
            return HTTPStatus.BAD_GATEWAY, {"error": str(e)}
        return HTTPStatus.OK, {"reply": reply, "engine": {"id": backend.id, "label": backend.label,
                                                          "is_thor": backend.is_thor},
                               "safety": session.last_safety}


def make_handler(app: App) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "THOR"

        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy",
                             "default-src 'none'; script-src 'self' 'unsafe-inline'; style-src 'unsafe-inline'; "
                             "connect-src 'self'; img-src data:; base-uri 'none'; form-action 'none'")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, data: dict | list) -> None:
            self._send(status, json.dumps(data).encode(), "application/json")

        def do_GET(self) -> None:  # noqa: N802
            if self.path in ("/", "/index.html"):
                self._send(HTTPStatus.OK, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/engines":
                self._json(HTTPStatus.OK, {"engines": app.describe(), "safety": app.safety})
            elif self.path == "/health":
                self._json(HTTPStatus.OK, {"ok": True})
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/chat":
                return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY:
                return self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "request too large"})
            try:
                payload = json.loads(self.rfile.read(length))
            except json.JSONDecodeError:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid JSON"})
            status, body = app.chat(payload if isinstance(payload, dict) else {})
            self._json(status, body)

        def log_message(self, fmt: str, *args) -> None:  # don't log request contents
            pass

    return Handler


def build_backends(args: argparse.Namespace) -> dict[str, ThorBackend | ExternalBackend]:
    backends: dict[str, ThorBackend | ExternalBackend] = {}
    if args.thor_ckpt:
        from .tokenizer import Tokenizer
        from .train import load_model

        if not args.thor_tokenizer:
            raise SystemExit("--thor-ckpt needs --thor-tokenizer")
        tok = Tokenizer.load(args.thor_tokenizer)
        model, ckpt = load_model(args.thor_ckpt, args.device)
        if ckpt["tokenizer_sha256"] != tok.fingerprint():
            raise SystemExit("The tokenizer does not match the checkpoint")
        b = ThorBackend(model, tok, args.max_new_tokens, args.temperature, device=args.device, stage=ckpt.get("stage"))
        backends[b.id] = b
    for spec in args.provider or []:
        name, _, model = spec.partition(":")
        b = ExternalBackend(name, model or None, temperature=args.temperature, max_tokens=args.max_new_tokens * 4)
        backends[b.id] = b
    if not backends:
        raise SystemExit("Give at least one engine: --thor-ckpt (THOR) and/or --provider gemini[:model]")
    return backends


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--thor-ckpt")
    ap.add_argument("--thor-tokenizer")
    ap.add_argument("--provider", action="append", metavar="NAME[:MODEL]",
                    help=f"External engine, repeatable. Providers: {', '.join(PROVIDERS)}")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-safety", action="store_true", help="Disable the safety layer (research only)")
    args = ap.parse_args(argv)

    app = App(build_backends(args), safety=not args.no_safety)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"Warning: listening on {args.host}. Anyone who can reach it can use your engines and API quota.")
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    for e in app.describe():
        print(f"  engine: {e['label']}{'' if e['available'] else ' (unavailable)'}")
    print(f"THOR Chat on http://{args.host}:{args.port}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
