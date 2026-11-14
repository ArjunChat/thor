import math

import pytest
import torch

from thor.config import PRESETS, ModelConfig, get_preset
from thor.model import ThorModel

TINY = ModelConfig(name="tiny", vocab_size=97, d_model=32, n_layer=2, n_head=4, ffn_hidden=64, max_seq_len=16)


@pytest.mark.parametrize("name", list(PRESETS))
def test_param_count_matches_module(name):
    cfg = get_preset(name)
    with torch.device("meta"):
        model = ThorModel(cfg)
    assert model.num_params() == cfg.num_params()


@pytest.mark.parametrize("name,target", [("thor-30m", 30e6), ("thor-100m", 100e6), ("thor-150m", 150e6)])
def test_presets_match_their_names(name, target):
    # Names are specifications: actual size must be within 10% of the label.
    assert abs(get_preset(name).num_params() - target) / target < 0.10


def test_untied_counts_head():
    cfg = ModelConfig(**{**TINY.to_dict(), "tie_embeddings": False})
    assert ThorModel(cfg).num_params() == cfg.num_params() == TINY.num_params() + 97 * 32


def test_forward_shapes_and_initial_loss():
    torch.manual_seed(0)
    model = ThorModel(TINY)
    x = torch.randint(0, 97, (3, 16))
    y = torch.randint(0, 97, (3, 16))
    logits, loss = model(x, y)
    assert logits.shape == (3, 16, 97)
    # A fresh model should be close to uniform: loss ~ ln(vocab)
    assert abs(loss.item() - math.log(97)) < 0.5


def test_causal_masking():
    torch.manual_seed(0)
    model = ThorModel(TINY).eval()
    x = torch.randint(0, 97, (1, 16))
    y = x.clone()
    y[0, 10:] = (y[0, 10:] + 1) % 97  # change only the future
    with torch.no_grad():
        a, _ = model(x)
        b, _ = model(y)
    assert torch.allclose(a[0, :10], b[0, :10], atol=1e-5)
    assert not torch.allclose(a[0, 10:], b[0, 10:], atol=1e-5)


def test_tied_embeddings_share_storage():
    model = ThorModel(TINY)
    assert model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()


def test_can_overfit_one_batch():
    torch.manual_seed(0)
    model = ThorModel(TINY)
    opt = model.configure_optimizer(3e-3, 0.0, (0.9, 0.95), "cpu")
    x = torch.randint(0, 97, (4, 16))
    for _ in range(200):
        _, loss = model(x[:, :-1], x[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1


def test_generate():
    torch.manual_seed(0)
    model = ThorModel(TINY).eval()
    out = model.generate(torch.zeros(2, 3, dtype=torch.long), max_new_tokens=20, top_k=5)
    assert out.shape == (2, 23)  # can exceed max_seq_len: context is cropped
    greedy1 = model.generate(torch.zeros(1, 1, dtype=torch.long), 5, temperature=0)
    greedy2 = model.generate(torch.zeros(1, 1, dtype=torch.long), 5, temperature=0)
    assert torch.equal(greedy1, greedy2)


def test_rejects_too_long_input():
    with pytest.raises(ValueError):
        ThorModel(TINY)(torch.zeros(1, 17, dtype=torch.long))


def test_invalid_config():
    with pytest.raises(ValueError):
        ModelConfig(d_model=30, n_head=4)
    with pytest.raises(ValueError):
        ModelConfig.from_dict({**TINY.to_dict(), "bogus": 1})
