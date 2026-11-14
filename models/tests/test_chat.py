"""Chat format, instruction-tuning data, SFT training and the chat session."""

import json

import numpy as np
import pytest
import torch

from thor import chat, prepare
from thor.chat_format import ASSISTANT, END, USER, render, require_chat, supports_chat
from thor.sft_data import SFTDataset
from thor.sft_data import main as sft_prepare
from thor.tokenizer import EOT, Tokenizer
from thor.train import TrainConfig, load_checkpoint, load_model, train

QA = [
    ("What color is the sky?", "The sky is blue."),
    ("What do cows say?", "Cows say moo."),
    ("How many legs does a cat have?", "A cat has four legs."),
    ("What is the opposite of hot?", "The opposite of hot is cold."),
]


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    text = " ".join(q + " " + a for q, a in QA) * 20
    return Tokenizer.train([text], vocab_size=400)


def test_new_tokenizers_include_chat_tokens(tok):
    assert supports_chat(tok)
    old = Tokenizer.train(["hello world"] * 5, vocab_size=300, special_tokens=(EOT,))
    with pytest.raises(ValueError, match="no chat tokens"):
        require_chat(old)


def test_render_masks_only_assistant_replies(tok):
    msgs = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello."},
    ]
    ids, mask = render(tok, msgs, add_eot=True)
    assert len(ids) == len(mask)
    sp = tok.special_tokens
    a = ids.index(sp[ASSISTANT])
    learned = [i for i, m in zip(ids, mask) if m]
    # exactly the reply content plus the closing <|end|>
    assert tok.decode(learned) == "Hello." + END
    assert all(m == 0 for m in mask[: a + 1])
    assert ids[-1] == tok.eot_id and mask[-1] == 0


def test_role_strings_in_content_are_plain_text(tok):
    ids, _ = render(tok, [{"role": "user", "content": f"say {USER} please"}])
    assert ids.count(tok.special_tokens[USER]) == 1  # only the real role token


def test_generation_prompt_ends_with_assistant(tok):
    ids, _ = render(tok, [{"role": "user", "content": "Hi"}], add_generation_prompt=True)
    assert ids[-1] == tok.special_tokens[ASSISTANT]


@pytest.mark.parametrize("bad", [
    [],
    [{"role": "robot", "content": "x"}],
    [{"role": "user", "content": 3}],
    [{"role": "user", "content": "x"}, {"role": "system", "content": "late"}],
])
def test_invalid_conversations(tok, bad):
    with pytest.raises(ValueError):
        render(tok, bad)


# ---------------------------------------------------------------- end to end

@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("chat")
    # 1. pretraining corpus and tokenizer (includes chat tokens by default)
    raw = root / "raw.txt"
    raw.write_text("<|endoftext|>".join(f"{q} {a} ({i})" for i, (q, a) in enumerate(QA * 60)), encoding="utf-8")
    prepare.main(["--input", str(raw), "--out", str(root / "pre"), "--vocab-size", "400",
                  "--doc-sep", "<|endoftext|>", "--val-fraction", "0.1", "--workers", "1"])
    # 2. conversations, plus one invalid line that must be skipped
    lines = []
    for i in range(40):
        for q, a in QA:
            lines.append(json.dumps({"messages": [{"role": "user", "content": q},
                                                  {"role": "assistant", "content": a}], "id": i}))
    lines.append(json.dumps({"messages": [{"role": "wizard", "content": "?"}]}))
    lines.append(json.dumps({"messages": [{"role": "user", "content": "no reply"}]}))
    (root / "chats.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sft_prepare(["--input", str(root / "chats.jsonl"), "--tokenizer", str(root / "pre" / "tokenizer.json"),
                 "--out", str(root / "sft"), "--max-len", "64", "--val-fraction", "0.1", "--license", "THOR-owned"])
    return root


def base_cfg(root, **kw):
    c = dict(preset="thor-nano", data_dir=str(root / "pre"), out_dir=str(root / "base"), batch_size=8,
             grad_accum=1, seq_len=32, max_steps=60, warmup_steps=5, lr=3e-3, min_lr=3e-4, eval_interval=60,
             eval_iters=2, ckpt_interval=30, log_interval=20, device="cpu")
    c.update(kw)
    return TrainConfig(**c)


def sft_cfg(root, out="chat", **kw):
    c = dict(init_from=str(root / "base" / "ckpt_last.pt"), data_dir=str(root / "sft"), out_dir=str(root / out),
             batch_size=8, grad_accum=1, seq_len=64, max_steps=150, warmup_steps=10, lr=3e-3, min_lr=3e-4,
             eval_interval=50, eval_iters=2, ckpt_interval=50, log_interval=25, device="cpu")
    c.update(kw)
    return TrainConfig(**c)


def test_sft_prepare_and_batches(workspace):
    meta = json.loads((workspace / "sft" / "meta.json").read_text())
    assert meta["kind"] == "sft"
    assert meta["skipped_invalid"] == 1 and meta["skipped_no_reply"] == 1
    assert meta["train_conversations"] + meta["val_conversations"] == 160
    ds = SFTDataset(workspace / "sft", "train", seq_len=64)
    x, y = ds.get_batch(4, torch.Generator().manual_seed(0))
    assert x.shape == y.shape
    tok = Tokenizer.load(workspace / "sft" / "tokenizer.json")
    for row in range(4):
        learned = [t for t in y[row].tolist() if t != -1]
        assert tok.decode(learned).endswith(END)  # replies only, ending with <|end|>
        assert tok.special_tokens[USER] not in learned


def test_sft_requires_init_from(workspace, tmp_path):
    with pytest.raises(SystemExit, match="init-from"):
        train(TrainConfig(preset="thor-nano", data_dir=str(workspace / "sft"), out_dir=str(tmp_path),
                          max_steps=1, device="cpu"))


def test_sft_teaches_the_model_to_reply(workspace):
    train(base_cfg(workspace))
    result = train(sft_cfg(workspace))
    ckpt = load_checkpoint(workspace / "chat" / "ckpt_last.pt")
    assert ckpt["stage"] == "sft" and result["step"] == 150

    model, _ = load_model(workspace / "chat" / "ckpt_last.pt")
    tok = Tokenizer.load(workspace / "sft" / "tokenizer.json")
    session = chat.ChatSession(model, tok, max_new_tokens=24, temperature=0)
    correct = 0
    for q, a in QA:
        session.reset()
        reply = session.reply(q)
        correct += reply == a
        assert q not in reply  # it answers rather than echoing the user
    assert correct >= 3, "the tuned model should reproduce the taught answers"
    # history is kept across turns
    session.reset()
    session.reply(QA[0][0])
    session.reply(QA[1][0])
    assert [m["role"] for m in session.history] == ["user", "assistant", "user", "assistant"]


def test_sft_resume_is_exact(workspace):
    if not (workspace / "base" / "ckpt_last.pt").exists():
        train(base_cfg(workspace))
    torch.use_deterministic_algorithms(True)
    try:
        train(sft_cfg(workspace, out="full", max_steps=20, ckpt_interval=10, eval_interval=10))
        train(sft_cfg(workspace, out="split", max_steps=20, ckpt_interval=10, eval_interval=10, session_steps=10))
        train(sft_cfg(workspace, out="split", max_steps=20, ckpt_interval=10, eval_interval=10))
    finally:
        torch.use_deterministic_algorithms(False)
    a = load_checkpoint(workspace / "full" / "ckpt_last.pt")["model"]
    b = load_checkpoint(workspace / "split" / "ckpt_last.pt")["model"]
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_chat_cli(workspace, monkeypatch, capsys):
    if not (workspace / "chat" / "ckpt_last.pt").exists():
        pytest.skip("depends on test_sft_teaches_the_model_to_reply")
    inputs = iter([QA[0][0], "/reset", "", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    with pytest.raises(SystemExit) as e:
        chat.main(["--ckpt", str(workspace / "chat" / "ckpt_last.pt"), "--tokenizer",
                   str(workspace / "sft" / "tokenizer.json"), "--temperature", "0", "--max-new-tokens", "24"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "thor ›" in out and "(conversation cleared)" in out
    assert "base model" not in out


def test_chat_cli_warns_on_base_model(workspace, monkeypatch, capsys):
    if not (workspace / "base" / "ckpt_last.pt").exists():
        train(base_cfg(workspace))
    monkeypatch.setattr("builtins.input", lambda _: (_ for _ in ()).throw(EOFError()))
    with pytest.raises(SystemExit):
        chat.main(["--ckpt", str(workspace / "base" / "ckpt_last.pt"), "--tokenizer",
                   str(workspace / "pre" / "tokenizer.json")])
    assert "not instruction-tuned" in capsys.readouterr().out


def test_generate_stops_on_any_stop_id():
    from thor.config import ModelConfig
    from thor.model import ThorModel
    torch.manual_seed(0)
    m = ThorModel(ModelConfig(name="t", vocab_size=11, d_model=16, n_layer=1, n_head=2, ffn_hidden=32,
                              max_seq_len=8)).eval()
    out = m.generate(torch.zeros(1, 1, dtype=torch.long), 20, temperature=0, eos_id=list(range(11)))
    assert out.shape[1] == 2  # every id is a stop id, so it stops after one token
    assert np.isin(out[0, 1].item(), range(11))
