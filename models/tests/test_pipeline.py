"""End-to-end: prepare data -> train -> checkpoint -> resume -> evaluate -> sample."""

import json

import pytest
import torch

from thor import prepare, sample
from thor.evaluate import evaluate
from thor.train import TrainConfig, load_checkpoint, lr_at, parse_args, train

STORY = (
    "Once upon a time there was a small robot named Bit. Bit liked to count the stars. "
    "Every night Bit counted one more star than the night before. "
)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("corpus")
    raw = root / "raw.txt"
    docs = [f"{STORY} Story number {i}. The end." for i in range(300)]
    raw.write_text("<|endoftext|>".join(docs), encoding="utf-8")
    out = root / "prepared"
    prepare.main(["--input", str(raw), "--out", str(out), "--vocab-size", "300",
                  "--doc-sep", "<|endoftext|>", "--val-fraction", "0.1", "--workers", "2"])
    return out


def cfg(data_dir, out_dir, **kw) -> TrainConfig:
    base = dict(preset="thor-nano", data_dir=str(data_dir), out_dir=str(out_dir), batch_size=4,
                grad_accum=2, seq_len=32, max_steps=12, warmup_steps=2, lr=1e-3, min_lr=1e-4,
                eval_interval=6, eval_iters=2, ckpt_interval=6, log_interval=3, device="cpu", dropout=0.1)
    base.update(kw)
    return TrainConfig(**base)


def test_prepare_outputs(data_dir):
    meta = json.loads((data_dir / "meta.json").read_text())
    assert meta["train_tokens"] > 0 and meta["val_tokens"] > 0
    assert meta["train_documents"] + meta["val_documents"] == 300
    with pytest.raises(SystemExit):  # refuses to overwrite without --overwrite
        prepare.main(["--input", str(data_dir / "meta.json"), "--out", str(data_dir)])


def test_resume_is_exact(data_dir, tmp_path):
    torch.use_deterministic_algorithms(True)
    try:
        full = train(cfg(data_dir, tmp_path / "full"))
        paused = train(cfg(data_dir, tmp_path / "split", session_steps=6))
        resumed = train(cfg(data_dir, tmp_path / "split"))
    finally:
        torch.use_deterministic_algorithms(False)
    assert paused["step"] == 6
    a = load_checkpoint(tmp_path / "full" / "ckpt_last.pt")
    b = load_checkpoint(tmp_path / "split" / "ckpt_last.pt")
    assert a["step"] == b["step"] == 12
    for k in a["model"]:
        assert torch.equal(a["model"][k], b["model"][k]), k
    assert full["loss"] == resumed["loss"]


def test_loss_decreases(data_dir, tmp_path):
    train(cfg(data_dir, tmp_path / "run", max_steps=60, eval_interval=60, ckpt_interval=60, lr=3e-3))
    lines = [json.loads(l) for l in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    losses = [l["loss"] for l in lines if "loss" in l]
    assert losses[-1] < losses[0] * 0.6


def test_refuses_existing_run_when_resume_never(data_dir, tmp_path):
    train(cfg(data_dir, tmp_path / "run", max_steps=2))
    with pytest.raises(SystemExit):
        train(cfg(data_dir, tmp_path / "run", max_steps=4, resume="never"))


def test_refuses_mismatched_model(data_dir, tmp_path):
    train(cfg(data_dir, tmp_path / "run", max_steps=2))
    with pytest.raises(SystemExit):
        train(cfg(data_dir, tmp_path / "run", max_steps=4, dropout=0.0))


def test_evaluate_and_sample(data_dir, tmp_path, capsys):
    train(cfg(data_dir, tmp_path / "run"))
    result = evaluate(str(tmp_path / "run" / "model_best.pt"), str(data_dir))
    assert result["tokens"] > 0 and result["perplexity"] > 1 and result["bits_per_byte"] > 0
    sample.main(["--ckpt", str(tmp_path / "run" / "model_best.pt"), "--tokenizer",
                 str(data_dir / "tokenizer.json"), "--prompt", "Once upon", "--max-new-tokens", "10"])
    assert "Once upon" in capsys.readouterr().out


def test_lr_schedule():
    c = TrainConfig(lr=1.0, min_lr=0.1, warmup_steps=10, max_steps=110)
    assert lr_at(0, c) == pytest.approx(0.1)
    assert lr_at(9, c) == pytest.approx(1.0)
    assert lr_at(60, c) == pytest.approx(0.55)
    assert lr_at(110, c) == pytest.approx(0.1)


def test_cli_parsing(tmp_path):
    conf = tmp_path / "c.json"
    conf.write_text(json.dumps({"preset": "thor-100m", "lr": 0.001}))
    c = parse_args(["--config", str(conf), "--lr", "0.002", "--compile", "true", "--max-steps", "5"])
    assert (c.preset, c.lr, c.compile, c.max_steps) == ("thor-100m", 0.002, True, 5)


def test_iter_documents_streams_across_block_boundaries(tmp_path):
    sep = "<|endoftext|>"
    docs = [f"document {i} " + "x" * (i * 7 % 23) for i in range(50)]
    path = tmp_path / "docs.txt"
    path.write_text(("\n" + sep + "\n").join(docs) + "\n" + sep + "\n\n", encoding="utf-8")
    for block in (3, 7, 13, 1 << 20):  # tiny blocks split the separator itself
        assert list(prepare.iter_documents([path], sep, block_chars=block)) == [d.strip() for d in docs]
    assert list(prepare.iter_documents([path], None)) == [path.read_text().strip()]


def test_rerunning_a_finished_run_is_a_noop(data_dir, tmp_path):
    first = train(cfg(data_dir, tmp_path / "run", max_steps=3))
    again = train(cfg(data_dir, tmp_path / "run", max_steps=3))
    assert again["step"] == first["step"] == 3
    assert again["loss"] is None
    json.dumps(again, allow_nan=False)  # valid JSON
