"""Train a THOR model, with checkpointing and exact resume.

Example:
    python -m thor.train --preset thor-30m --data-dir data/tinystories --out-dir runs/thor-30m

Every option can be set from a JSON file (--config) and overridden on the
command line. Re-running the same command resumes from runs/<name>/ckpt_last.pt.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import signal
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import ModelConfig, get_preset
from .data import TokenDataset, load_meta
from .sft_data import SFTDataset
from .model import ThorModel
from .tokenizer import Tokenizer

CHECKPOINT_FORMAT = 1


@dataclass
class TrainConfig:
    preset: str = "thor-30m"
    # Start from this checkpoint's weights and shape (preset is then ignored). Required for
    # instruction tuning on data prepared by thor.sft_data.
    init_from: str = ""
    data_dir: str = "data/prepared"
    out_dir: str = "runs/thor-30m"
    # optimization
    batch_size: int = 16
    grad_accum: int = 4
    seq_len: int = 0  # 0 = model max_seq_len
    max_steps: int = 10_000
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 500
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    dropout: float = 0.0
    # bookkeeping
    eval_interval: int = 250
    eval_iters: int = 50
    ckpt_interval: int = 250
    log_interval: int = 10
    keep_every: int = 0  # also keep ckpt_step{N}.pt every N steps (0 = off)
    seed: int = 1337
    # runtime
    device: str = "auto"  # auto | cpu | cuda | mps
    dtype: str = "auto"  # auto | float32 | bfloat16 | float16
    compile: bool = False
    resume: str = "auto"  # auto: resume if a checkpoint exists | never: refuse to touch an existing run
    session_steps: int = 0  # stop (with a checkpoint) after this many steps in this run; 0 = no limit


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup, then cosine decay to min_lr at max_steps."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (cfg.lr - cfg.min_lr)


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        if device.type != "cuda":
            return torch.float32
        # Native bf16 needs Ampere (compute capability 8.0) or newer. Older GPUs such as
        # the T4 can only emulate it slowly, so they use fp16 with gradient scaling.
        if torch.cuda.get_device_capability(device)[0] >= 8:
            return torch.bfloat16
        return torch.float16
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def _rng_state(sampler: torch.Generator) -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "sampler": sampler.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict[str, Any], sampler: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    sampler.set_state(state["sampler"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_save(obj: Any, path: Path) -> None:
    """Write to a temp file and rename, so a crash never leaves a half-written checkpoint."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if ckpt.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path}: unsupported checkpoint format {ckpt.get('format')}")
    return ckpt


def load_model(path: str | Path, device: str | torch.device = "cpu") -> tuple[ThorModel, dict[str, Any]]:
    ckpt = load_checkpoint(path, map_location=device)
    model = ThorModel(ModelConfig.from_dict(ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


@torch.no_grad()
def estimate_loss(model: ThorModel, datasets: dict[str, TokenDataset], cfg: TrainConfig,
                  device: torch.device, autocast) -> dict[str, float]:
    """Mean loss over a fixed set of batches (same batches every eval)."""
    model.eval()
    out = {}
    for split, ds in datasets.items():
        gen = torch.Generator().manual_seed(cfg.seed + 1)
        losses = []
        for _ in range(cfg.eval_iters):
            x, y = ds.get_batch(cfg.batch_size, gen, device)
            with autocast():
                _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = float(np.mean(losses))
    model.train()
    return out


def train(cfg: TrainConfig) -> dict[str, Any]:
    out_dir = Path(cfg.out_dir)
    last_path = out_dir / "ckpt_last.pt"
    if last_path.exists() and cfg.resume == "never":
        raise SystemExit(f"{last_path} exists and resume='never'; choose a new --out-dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_meta(cfg.data_dir)
    tokenizer = Tokenizer.load(Path(cfg.data_dir) / "tokenizer.json")
    if tokenizer.fingerprint() != meta["tokenizer_sha256"]:
        raise SystemExit("tokenizer.json does not match the tokenizer recorded in meta.json")

    stage = "sft" if meta.get("kind") == "sft" else "pretrain"
    base = None
    if cfg.init_from:
        base = load_checkpoint(cfg.init_from, map_location="cpu")
        if base["tokenizer_sha256"] != tokenizer.fingerprint():
            raise SystemExit("--init-from checkpoint was trained with a different tokenizer than this data")
        model_cfg = dataclasses.replace(ModelConfig.from_dict(base["model_config"]), dropout=cfg.dropout)
        print(f"Initializing from {cfg.init_from} (step {base.get('step')}); preset is ignored")
    elif stage == "sft":
        raise SystemExit("Instruction tuning starts from a pretrained model: pass --init-from <checkpoint>")
    else:
        model_cfg = dataclasses.replace(get_preset(cfg.preset).with_vocab_size(meta["vocab_size"]),
                                        dropout=cfg.dropout)
    seq_len = cfg.seq_len or model_cfg.max_seq_len
    if seq_len > model_cfg.max_seq_len:
        raise SystemExit(f"seq_len {seq_len} > model max_seq_len {model_cfg.max_seq_len}")

    device = resolve_device(cfg.device)
    dtype = resolve_dtype(cfg.dtype, device)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    sampler = torch.Generator().manual_seed(cfg.seed)

    if stage == "sft":
        datasets = {split: SFTDataset(cfg.data_dir, split, seq_len) for split in ("train", "val")}
    else:
        datasets = {
            "train": TokenDataset(Path(cfg.data_dir) / "train.bin", seq_len),
            "val": TokenDataset(Path(cfg.data_dir) / "val.bin", seq_len),
        }
    model = ThorModel(model_cfg).to(device)
    if base is not None and not last_path.exists():
        model.load_state_dict(base["model"])
    del base
    optimizer = model.configure_optimizer(cfg.lr, cfg.weight_decay, (cfg.beta1, cfg.beta2), device.type)
    scaler = torch.amp.GradScaler(device.type, enabled=dtype == torch.float16)

    def autocast():
        return torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32)

    step, best_val, tokens_seen = 0, float("inf"), 0
    if last_path.exists():
        ckpt = load_checkpoint(last_path, map_location=device)
        if ckpt["model_config"] != model_cfg.to_dict():
            raise SystemExit("Checkpoint model config differs from the requested one; use a new --out-dir")
        if ckpt["tokenizer_sha256"] != tokenizer.fingerprint():
            raise SystemExit("Checkpoint was trained with a different tokenizer")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        step, best_val, tokens_seen = ckpt["step"], ckpt["best_val_loss"], ckpt["tokens_seen"]
        _set_rng_state(ckpt["rng"], sampler)
        changed = {k: (v, getattr(cfg, k)) for k, v in ckpt["train_config"].items()
                   if getattr(cfg, k, v) != v and k not in ("resume", "session_steps")}
        print(f"Resumed from {last_path} at step {step}")
        if changed:
            print(f"  note: training options changed since checkpoint: {changed}")

    run_model = torch.compile(model) if cfg.compile else model
    n_params = model.num_params()
    tokens_per_step = cfg.batch_size * cfg.grad_accum * seq_len
    print(f"Model {model_cfg.name} ({stage}): {n_params:,} parameters | device {device} | dtype {dtype}")
    if stage == "sft":
        print(f"Up to {tokens_per_step:,} positions/step | train conversations {len(datasets['train']):,}")
    else:
        print(f"Tokens/step {tokens_per_step:,} | train tokens {len(datasets['train']):,}")
    (out_dir / "config.json").write_text(json.dumps(
        {"train": dataclasses.asdict(cfg), "model": model_cfg.to_dict(), "params": n_params}, indent=2))

    def save(path: Path, extra: dict | None = None) -> None:
        atomic_save({
            "format": CHECKPOINT_FORMAT,
            "stage": stage,
            "model_config": model_cfg.to_dict(),
            "train_config": dataclasses.asdict(cfg),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "best_val_loss": best_val,
            "tokens_seen": tokens_seen,
            "rng": _rng_state(sampler),
            "tokenizer_sha256": tokenizer.fingerprint(),
            "saved_at": datetime.now(timezone.utc).isoformat(),
            **(extra or {}),
        }, path)

    stop = {"requested": False}

    def request_stop(signum, _frame):
        print(f"\nSignal {signum} received — saving a checkpoint after this step.")
        stop["requested"] = True

    previous_handlers = {s: signal.signal(s, request_stop) for s in (signal.SIGINT, signal.SIGTERM)}
    log = open(out_dir / "metrics.jsonl", "a", encoding="utf-8")
    last_loss = float("nan")
    try:
        model.train()
        t0 = time.time()
        session_end = step + cfg.session_steps if cfg.session_steps else cfg.max_steps
        if step >= cfg.max_steps:
            print(f"Run already complete at step {step} (max_steps {cfg.max_steps}); nothing to do.")
        while step < min(cfg.max_steps, session_end):
            lr = lr_at(step, cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr
            loss_sum = 0.0
            for _ in range(cfg.grad_accum):
                x, y = datasets["train"].get_batch(cfg.batch_size, sampler, device)
                with autocast():
                    _, loss = run_model(x, y)
                loss_sum += loss.item()
                scaler.scale(loss / cfg.grad_accum).backward()
            last_loss = loss_sum / cfg.grad_accum
            if not math.isfinite(last_loss):
                raise SystemExit(f"Non-finite loss at step {step}; last good checkpoint kept at {last_path}")
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            tokens_seen += tokens_per_step

            if step % cfg.log_interval == 0 or step == 1:
                dt = time.time() - t0
                t0 = time.time()
                steps_in_window = cfg.log_interval if step != 1 else 1
                record = {"step": step, "loss": round(last_loss, 4), "lr": lr, "grad_norm": round(grad_norm, 3),
                          "tokens": tokens_seen, "tok_per_s": round(tokens_per_step * steps_in_window / max(dt, 1e-9))}
                print(json.dumps(record))
                log.write(json.dumps(record) + "\n")
                log.flush()

            if step % cfg.eval_interval == 0 or step == cfg.max_steps:
                losses = estimate_loss(run_model, datasets, cfg, device, autocast)
                record = {"step": step, "eval_train_loss": round(losses["train"], 4),
                          "eval_val_loss": round(losses["val"], 4),
                          "val_perplexity": round(math.exp(losses["val"]), 2)}
                print(json.dumps(record))
                log.write(json.dumps(record) + "\n")
                log.flush()
                if losses["val"] < best_val:
                    best_val = losses["val"]
                    atomic_save({"format": CHECKPOINT_FORMAT, "stage": stage, "model_config": model_cfg.to_dict(),
                                 "model": model.state_dict(), "step": step, "val_loss": best_val,
                                 "tokenizer_sha256": tokenizer.fingerprint()}, out_dir / "model_best.pt")

            session_done = step >= session_end and step < cfg.max_steps
            if session_done:
                stop["requested"] = True
            if step % cfg.ckpt_interval == 0 or step == cfg.max_steps or stop["requested"]:
                save(last_path)
                if cfg.keep_every and step % cfg.keep_every == 0:
                    save(out_dir / f"ckpt_step{step}.pt")
            if stop["requested"]:
                print(f"Stopped at step {step}. Re-run the same command to resume.")
                break
    finally:
        log.close()
        for s, h in previous_handlers.items():
            signal.signal(s, h)
    return {"step": step, "loss": last_loss if math.isfinite(last_loss) else None,
            "best_val_loss": best_val if math.isfinite(best_val) else None, "params": n_params}


def _parse_bool(value: str) -> bool:
    if value.lower() in ("1", "true", "yes", "on"):
        return True
    if value.lower() in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {value!r}")


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="JSON file with TrainConfig fields")
    types = {"bool": _parse_bool, "int": int, "float": float, "str": str}
    for f in dataclasses.fields(TrainConfig):
        ap.add_argument("--" + f.name.replace("_", "-"), dest=f.name, type=types[str(f.type)], default=None)
    args = ap.parse_args(argv)
    values: dict[str, Any] = {}
    if args.config:
        values.update(json.loads(Path(args.config).read_text(encoding="utf-8")))
    values.update({k: v for k, v in vars(args).items() if k != "config" and v is not None})
    return TrainConfig(**values)


def main(argv: list[str] | None = None) -> None:
    result = train(parse_args(argv))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
