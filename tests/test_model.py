import pytest
import torch
import torch.nn as nn

from lfm_my.model import GecTagger


class TinyEncoder(nn.Module):
    """Stand-in trunk: embeddings + one linear, exposes get_input_embeddings()."""

    def __init__(self, vocab_size=64, hidden=16):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden)
        self.proj = nn.Linear(hidden, hidden)

    def get_input_embeddings(self):
        return self.emb

    def forward(self, input_ids=None, attention_mask=None, **kw):
        h = self.proj(self.emb(input_ids))
        if attention_mask is not None:
            h = h * attention_mask.unsqueeze(-1)

        class Out:
            last_hidden_state = h

        return Out


@pytest.fixture(scope="module")
def tagger():
    m = GecTagger(TinyEncoder(64, 16), vocab_size=64, hidden_size=16)
    m.eval()
    return m


def test_submodule_names_match_inference_class():
    expected = {"encoder", "dropout", "detect_head", "base_head", "replace_proj",
                "append_proj", "replace_bias", "append_bias"}
    m = GecTagger(TinyEncoder(), vocab_size=64, hidden_size=16)
    # replace_bias/append_bias are nn.Parameter in GecTaggerForGEC, not submodules, so
    # named_children() cannot see them -- compare top-level state_dict names instead.
    named = {n for n, _ in m.named_children()} | {n for n, _ in m.named_parameters(recurse=False)}
    assert named == expected


def test_state_dict_matches_real_inference_class(monkeypatch):
    """The compatibility linchpin: our trained state_dict must load 1:1 into Liquid's published
    GecTaggerForGEC. Built against the real class (backbone patched out -- it wants network)."""
    from lfm_my import modeling_gectagger as mg

    monkeypatch.setattr(mg, "_build_backbone", lambda name: TinyEncoder(64, 16))
    cfg = mg.GecTaggerConfig(hidden_size=16, num_tags=2 + 2 * 64, tie_replace=True, use_swap=False)
    ref = mg.GecTaggerForGEC(cfg)
    ours = GecTagger(TinyEncoder(64, 16), vocab_size=64, hidden_size=16)

    strip = lambda sd: {k for k in sd if not k.startswith("encoder.")}  # noqa: E731
    assert strip(ours.state_dict()) == strip(ref.state_dict())
    for k in strip(ours.state_dict()):
        assert ours.state_dict()[k].shape == ref.state_dict()[k].shape, k
    assert ours.vocab_size == ref.vocab_size


def test_state_dict_keys_legal_for_inference_class():
    # every key we produce must sit under a GecTaggerForGEC submodule name
    m = GecTagger(TinyEncoder(), vocab_size=64, hidden_size=16)
    legal = ("encoder.", "detect_head.", "base_head.", "replace_proj.",
             "append_proj.", "replace_bias", "append_bias")
    for k in m.state_dict():
        assert any(k.startswith(p) for p in legal), k


def test_forward_shapes(tagger):
    ids = torch.randint(0, 64, (2, 7))
    out = tagger(ids, torch.ones_like(ids))
    assert out["label_logits"].shape == (2, 7, 2 + 2 * 64)
    assert out["detect_logits"].shape == (2, 7, 2)


def test_label_logits_finite(tagger):
    ids = torch.tensor([[3, 5]])
    out = tagger(ids, torch.ones_like(ids))
    assert torch.isfinite(out["label_logits"]).all()


def test_detect_and_label_heads_independent(tagger):
    ids = torch.tensor([[3, 5]])
    mask = torch.ones_like(ids)
    out1 = tagger(ids, mask)
    with torch.no_grad():
        tagger.detect_head.weight.add_(1.0)
    out2 = tagger(ids, mask)
    assert not torch.allclose(out1["detect_logits"], out2["detect_logits"])
    assert torch.allclose(out1["label_logits"], out2["label_logits"])
