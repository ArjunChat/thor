"""Chat engines ("backends") behind THOR Chat.

- ThorBackend: a THOR checkpoint, running locally. The only backend that is a THOR model.
- ExternalBackend: a third-party model through an OpenAI-compatible HTTP API
  (Google Gemini, Groq, OpenRouter, ...). For testing the product before a THOR
  chat model is trained. Every reply from it is labelled as external.

External providers have their own terms. Gemini's, for example, forbid using the
service to develop competing models, and on the free tier Google uses what you
send to improve its products (human reviewers may read it). So THOR never trains
on external outputs, and ChatSession strips personal data from messages before
they leave the machine.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

import torch

from .chat_format import END, Message, render, require_chat
from .model import ThorModel
from .tokenizer import Tokenizer


class BackendError(RuntimeError):
    """A failure the user should see in plain words (missing key, rate limit, ...)."""


class ThorBackend:
    is_thor = True

    def __init__(self, model: ThorModel, tok: Tokenizer, max_new_tokens: int = 256, temperature: float = 0.7,
                 top_k: int | None = 50, seed: int = 0, device: str = "cpu", stage: str | None = None) -> None:
        require_chat(tok)
        self.model, self.tok = model, tok
        self.max_new_tokens, self.temperature, self.top_k = max_new_tokens, temperature, top_k
        self.device = device
        self.stage = stage
        self.generator = torch.Generator(device=device).manual_seed(seed)

    @property
    def id(self) -> str:
        return f"thor:{self.model.cfg.name}"

    @property
    def label(self) -> str:
        tuned = "instruction-tuned" if self.stage == "sft" else "base model, not instruction-tuned"
        return f"{self.model.cfg.name} · THOR model · {self.model.num_params() / 1e6:.1f}M parameters · {tuned}"

    def _prompt_ids(self, messages: list[Message]) -> list[int]:
        """Render the conversation, dropping the oldest turns until it fits with room to reply."""
        budget = self.model.cfg.max_seq_len - min(self.max_new_tokens, self.model.cfg.max_seq_len // 2)
        system = [m for m in messages[:1] if m["role"] == "system"]
        turns = messages[len(system):]
        while True:
            ids, _ = render(self.tok, system + turns, add_generation_prompt=True)
            if len(ids) <= budget or len(turns) <= 1:
                return ids[-budget:]
            turns = turns[2:] if len(turns) > 2 else turns[1:]  # drop the oldest user/assistant pair

    def complete(self, messages: list[Message]) -> str:
        prompt = self._prompt_ids(messages)
        idx = torch.tensor([prompt], dtype=torch.long, device=self.device)
        stop = [self.tok.special_tokens[END], self.tok.eot_id]
        out = self.model.generate(idx, self.max_new_tokens, self.temperature, self.top_k,
                                  eos_id=stop, generator=self.generator)[0, len(prompt):].tolist()
        out = [t for t in out if t not in stop and t not in self.tok.special_ids]
        return self.tok.decode(out).strip()


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    key_env: str
    default_model: str
    signup: str


PROVIDERS: dict[str, Provider] = {
    "gemini": Provider("Google Gemini", "https://generativelanguage.googleapis.com/v1beta/openai",
                       "GEMINI_API_KEY", "gemini-2.5-flash", "https://aistudio.google.com/apikey"),
    "groq": Provider("Groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY",
                     "llama-3.1-8b-instant", "https://console.groq.com/keys"),
    "openrouter": Provider("OpenRouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY",
                           "meta-llama/llama-3.3-70b-instruct:free", "https://openrouter.ai/keys"),
}

# Sent as the first system message so an external model doesn't present itself as THOR.
EXTERNAL_SYSTEM_PROMPT = (
    "You are a general-purpose AI assistant being tested inside THOR's chat interface. "
    "You are not a THOR model. If asked who you are, say which model and provider you are. "
    "Be concise and helpful, and decline requests that could cause harm."
)

Transport = Callable[[str, dict[str, str], bytes | None, float], tuple[int, bytes]]


def _urllib_transport(url: str, headers: dict[str, str], body: bytes | None, timeout: float) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except urllib.error.URLError as e:
        raise BackendError(f"Could not reach the provider: {e.reason}") from None


class ExternalBackend:
    is_thor = False

    def __init__(self, provider: str, model: str | None = None, api_key: str | None = None,
                 temperature: float = 0.7, max_tokens: int = 1024, timeout: float = 60.0,
                 transport: Transport | None = None) -> None:
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown provider {provider!r}. Choose from: {', '.join(PROVIDERS)}")
        self.provider = PROVIDERS[provider]
        self.provider_id = provider
        self.model = model or self.provider.default_model
        # Behind a credential-injecting proxy any placeholder works; locally, set the env var.
        self.api_key = api_key or os.environ.get(self.provider.key_env)
        self.temperature, self.max_tokens, self.timeout = temperature, max_tokens, timeout
        self.transport = transport or _urllib_transport

    @property
    def id(self) -> str:
        return f"{self.provider_id}:{self.model}"

    @property
    def label(self) -> str:
        return f"{self.model} via {self.provider.name} · external model, not THOR"

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise BackendError(
                f"No API key for {self.provider.name}. Get a free key at {self.provider.signup} "
                f"and set {self.provider.key_env}."
            )
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _request(self, path: str, payload: dict | None) -> dict:
        body = json.dumps(payload).encode() if payload is not None else None
        status, raw = self.transport(f"{self.provider.base_url}/{path}", self._headers(), body, self.timeout)
        try:
            data = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            data = {}
        if status in (401, 403):
            raise BackendError(f"{self.provider.name} rejected the API key (HTTP {status}). Check {self.provider.key_env}.")
        if status == 429:
            raise BackendError(f"{self.provider.name} rate limit reached (free tier). Wait a minute and try again.")
        if status >= 400:
            err = data.get("error") if isinstance(data, dict) else None
            msg = err.get("message") if isinstance(err, dict) else (err if isinstance(err, str) else raw[:200])
            if isinstance(data, list) and data and isinstance(data[0], dict):
                msg = data[0].get("error", {}).get("message", msg)
            raise BackendError(f"{self.provider.name} error (HTTP {status}): {msg}")
        return data

    def complete(self, messages: list[Message]) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": EXTERNAL_SYSTEM_PROMPT}] + [dict(m) for m in messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        data = self._request("chat/completions", payload)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise BackendError(f"Unexpected response from {self.provider.name}") from None
        return (content or "").strip()

    def list_models(self) -> list[str]:
        data = self._request("models", None)
        return sorted(m.get("id", "") for m in data.get("data", []) if isinstance(m, dict))


def main(argv: list[str] | None = None) -> None:
    """List the models a provider offers: python -m thor.providers --provider gemini"""
    import argparse

    ap = argparse.ArgumentParser(description="List models available from an external provider")
    ap.add_argument("--provider", choices=sorted(PROVIDERS), required=True)
    args = ap.parse_args(argv)
    try:
        for name in ExternalBackend(args.provider).list_models():
            print(name)
    except BackendError as e:
        raise SystemExit(str(e)) from None


if __name__ == "__main__":
    main()
