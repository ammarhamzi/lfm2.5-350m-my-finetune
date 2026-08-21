import json

import pytest
import torch

from lfm_my.export import export_model_dir
from lfm_my.model import GecTagger
from lfm_my.modeling_gectagger import GecTaggerConfig
from tests.test_dataset import CharTokenizer
from tests.test_model import TinyEncoder


@pytest.fixture(scope="module")
def export_dir(tmp_path_factory):
    torch.manual_seed(0)
    tagger = GecTagger(TinyEncoder(64, 16), vocab_size=64, hidden_size=16)
    return export_model_dir(tagger, CharTokenizer(), tmp_path_factory.mktemp("export"),
                            encoder_name="fake/encoder")


def test_dir_contents(export_dir):
    for f in ("model.safetensors", "config.json", "modeling_gectagger.py"):
        assert (export_dir / f).is_file(), f


def test_config_matches_liquid_layout(export_dir):
    cfg = json.loads((export_dir / "config.json").read_text())
    assert cfg["model_type"] == "gec_tagger"
    assert cfg["architectures"] == ["GecTaggerForGEC"]
    assert cfg["auto_map"]["AutoModel"] == "modeling_gectagger.GecTaggerForGEC"
    assert cfg["auto_map"]["AutoConfig"] == "modeling_gectagger.GecTaggerConfig"
    assert cfg["tie_replace"] is True
    assert cfg["use_swap"] is False
    assert cfg["aux_loss_weight"] == 0.5
    assert cfg["num_tags"] == 2 + 2 * 64            # from the model's vocab_size (64 in tests)
    assert "num_labels" not in cfg                  # reserved property — must not be serialized


def test_safetensors_keys_match_training_state_dict(export_dir):
    from safetensors import safe_open
    tagger = GecTagger(TinyEncoder(64, 16), vocab_size=64, hidden_size=16)
    with safe_open(export_dir / "model.safetensors", framework="pt") as f:
        keys = set(f.keys())
    assert keys == set(tagger.state_dict().keys())


def test_config_class_roundtrip(export_dir):
    cfg = GecTaggerConfig.from_pretrained(str(export_dir))
    assert cfg.num_tags == 130
    assert cfg.hidden_size == 16


import torch.nn as nn
from safetensors import safe_open


class TiedEncoder(nn.Module):
    """Real encoders commonly tie the input embedding to an output head, so state_dict()
    returns two keys backed by ONE storage -- which safetensors refuses to write."""

    def __init__(self, vocab=64, hidden=16):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.lm_head.weight = self.emb.weight

    def get_input_embeddings(self):
        return self.emb

    def forward(self, input_ids=None, attention_mask=None, **kw):
        class Out:
            last_hidden_state = self.emb(input_ids)

        return Out


def test_export_handles_tied_encoder_weights(tmp_path):
    m = GecTagger(TiedEncoder(), vocab_size=64, hidden_size=16)
    sd = m.state_dict()
    assert sd["encoder.emb.weight"].data_ptr() == sd["encoder.lm_head.weight"].data_ptr()

    out = export_model_dir(m, CharTokenizer(), tmp_path / "tied", encoder_name="fake/encoder")
    with safe_open(out / "model.safetensors", framework="pt") as f:
        keys = set(f.keys())
        assert torch.equal(f.get_tensor("encoder.emb.weight"),
                           f.get_tensor("encoder.lm_head.weight"))
    assert keys == set(sd)


def test_verify_export_catches_a_partial_file(tmp_path):
    from safetensors.torch import save_file

    from lfm_my.export import verify_export

    m = GecTagger(TinyEncoder(64, 16), vocab_size=64, hidden_size=16)
    out = export_model_dir(m, CharTokenizer(), tmp_path / "ok", encoder_name="fake/encoder")
    sd = {k: v.contiguous() for k, v in m.state_dict().items()}
    sd.pop("detect_head.weight")
    save_file(sd, out / "model.safetensors")
    with pytest.raises(RuntimeError, match="missing"):
        verify_export(out, m)
