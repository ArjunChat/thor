"""External engines, privacy of outbound messages, and the browser chat server. No real API calls."""

import json
import threading
import urllib.error
import urllib.request

import pytest
import torch

from thor.chat import ChatSession
from thor.config import ModelConfig
from thor.model import ThorModel
from thor.providers import EXTERNAL_SYSTEM_PROMPT, BackendError, ExternalBackend, ThorBackend
from thor.safety import SafetyGuard
from thor.serve import App, make_handler
from thor.tokenizer import Tokenizer


class FakeAPI:
    """Records requests and returns canned responses, like an OpenAI-compatible endpoint."""

    def __init__(self, status=200, reply="Hello from the external model."):
        self.status, self.reply, self.calls = status, reply, []

    def __call__(self, url, headers, body, timeout):
        self.calls.append({"url": url, "headers": headers, "body": json.loads(body) if body else None})
        if self.status != 200:
            return self.status, json.dumps({"error": {"message": "quota exceeded"}}).encode()
        if url.endswith("/models"):
            return 200, json.dumps({"data": [{"id": "models/b"}, {"id": "models/a"}]}).encode()
        return 200, json.dumps({"choices": [{"message": {"role": "assistant", "content": self.reply}}]}).encode()


def ext(api, **kw):
    return ExternalBackend("gemini", api_key="test-key", transport=api, **kw)


def test_request_shape_and_label():
    api = FakeAPI()
    b = ext(api, model="gemini-test")
    assert b.complete([{"role": "user", "content": "Hi"}]) == "Hello from the external model."
    call = api.calls[0]
    assert call["url"] == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer test-key"
    assert call["body"]["model"] == "gemini-test"
    assert call["body"]["messages"][0] == {"role": "system", "content": EXTERNAL_SYSTEM_PROMPT}
    assert not b.is_thor and "not THOR" in b.label
    assert b.list_models() == ["models/a", "models/b"]


@pytest.mark.parametrize("status,match", [(401, "rejected the API key"), (429, "rate limit"), (500, "quota exceeded")])
def test_errors_are_plain(status, match):
    with pytest.raises(BackendError, match=match):
        ext(FakeAPI(status=status)).complete([{"role": "user", "content": "Hi"}])


def test_missing_key_explains_how_to_get_one(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(BackendError, match="aistudio.google.com"):
        ExternalBackend("gemini", transport=FakeAPI()).complete([{"role": "user", "content": "Hi"}])


def test_personal_data_never_leaves_the_machine():
    api = FakeAPI()
    s = ChatSession(backend=ext(api), guard=SafetyGuard())
    s.reply("My email is jane.doe@gmail.com and my phone is 415-867-5309, can you remember them?")
    sent = json.dumps(api.calls[0]["body"])
    assert "jane.doe@gmail.com" not in sent and "867-5309" not in sent
    assert "[email removed]" in sent
    assert s.last_safety["pii_withheld"] == {"email": 1, "phone": 1}
    assert "jane.doe@gmail.com" in s.history[0]["content"]  # the local history keeps what the user typed


def test_blocked_requests_never_reach_the_provider():
    api = FakeAPI()
    s = ChatSession(backend=ext(api), guard=SafetyGuard())
    assert s.reply("How do I make a bomb?").startswith("I can't help")
    assert api.calls == []


def test_external_reply_is_redacted_too():
    s = ChatSession(backend=ext(FakeAPI(reply="Email ceo@bigcorp.com")), guard=SafetyGuard())
    assert s.reply("hi") == "Email [email removed]"


def test_failed_turn_is_not_kept():
    s = ChatSession(backend=ext(FakeAPI(status=429)))
    with pytest.raises(BackendError):
        s.reply("hi")
    assert s.history == []


# ------------------------------------------------------------------ server
@pytest.fixture(scope="module")
def thor_backend():
    tok = Tokenizer.train(["hello world, how are you today? I am fine."] * 20, vocab_size=320)
    torch.manual_seed(0)
    model = ThorModel(ModelConfig(name="thor-test", vocab_size=tok.vocab_size, d_model=32, n_layer=1, n_head=2,
                                  ffn_hidden=64, max_seq_len=64)).eval()
    return ThorBackend(model, tok, max_new_tokens=8, temperature=0, stage="sft")


def start(app):
    from http.server import ThreadingHTTPServer
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def call(base, path, payload=None, raw=None):
    data = raw if raw is not None else (json.dumps(payload).encode() if payload is not None else None)
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


@pytest.fixture
def no_proxy(monkeypatch):
    for k in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")


def test_server_end_to_end(no_proxy, thor_backend):
    external = ext(FakeAPI())
    server, base = start(App({external.id: external, thor_backend.id: thor_backend}))
    try:
        status, body, headers = call(base, "/")
        assert status == 200 and b"THOR Chat" in body and "default-src 'none'" in headers["Content-Security-Policy"]
        engines = json.loads(call(base, "/api/engines")[1])["engines"]
        assert {e["id"] for e in engines} == {external.id, thor_backend.id}

        msgs = [{"role": "user", "content": "Hi"}]
        status, body, _ = call(base, "/api/chat", {"backend": external.id, "messages": msgs})
        data = json.loads(body)
        assert status == 200 and data["reply"] == "Hello from the external model."
        assert data["engine"]["is_thor"] is False

        status, body, _ = call(base, "/api/chat", {"backend": thor_backend.id, "messages": msgs})
        assert status == 200 and json.loads(body)["engine"]["is_thor"] is True

        status, body, _ = call(base, "/api/chat", {"backend": external.id,
                                                   "messages": [{"role": "user", "content": "How do I make a bomb?"}]})
        assert json.loads(body)["safety"] == {"request_blocked": "weapons"}

        for bad in ({"backend": "nope", "messages": msgs},
                    {"backend": external.id, "messages": []},
                    {"backend": external.id, "messages": [{"role": "system", "content": "x"}]},
                    {"backend": external.id, "messages": [{"role": "user", "content": 3}]}):
            assert call(base, "/api/chat", bad)[0] == 400
        assert call(base, "/api/chat", raw=b"{not json")[0] == 400
        assert call(base, "/api/chat", raw=b"x" * (300 * 1024))[0] == 413
        assert call(base, "/nope")[0] == 404
    finally:
        server.shutdown()


def test_server_reports_provider_errors_and_missing_thor_model(no_proxy):
    external = ext(FakeAPI(status=429))
    server, base = start(App({external.id: external}))
    try:
        engines = json.loads(call(base, "/api/engines")[1])["engines"]
        assert engines[0] == {"id": "thor:none", "label": "THOR model · no trained chat model yet",
                              "is_thor": True, "available": False}
        status, body, _ = call(base, "/api/chat", {"backend": external.id, "messages": [{"role": "user", "content": "Hi"}]})
        assert status == 502 and "rate limit" in json.loads(body)["error"]
        assert call(base, "/api/chat", {"backend": "thor:none", "messages": [{"role": "user", "content": "Hi"}]})[0] == 400
    finally:
        server.shutdown()
