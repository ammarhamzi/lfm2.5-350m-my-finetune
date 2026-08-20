"""Training-side two-head GECToR tagger.

Submodule names exactly mirror lfm_my.modeling_gectagger.GecTaggerForGEC (tie_replace=True
branch) so the trained state_dict loads 1:1 into Liquid's inference class on export.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from lfm_my import ENCODER_NAME
from lfm_my.modeling_gectagger import _last_hidden


class GecTagger(nn.Module):
    def __init__(self, encoder: nn.Module, vocab_size: int, hidden_size: int,
                 dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.detect_head = nn.Linear(hidden_size, 2)
        self.base_head = nn.Linear(hidden_size, 2)          # $KEEP / $DELETE
        self.replace_proj = nn.Linear(hidden_size, hidden_size)
        self.append_proj = nn.Linear(hidden_size, hidden_size)
        self.replace_bias = nn.Parameter(torch.zeros(vocab_size))
        self.append_bias = nn.Parameter(torch.zeros(vocab_size))

    def _label_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        E = self.encoder.get_input_embeddings().weight[:self.vocab_size]   # [V, H] tied
        base = self.base_head(hidden)                                       # [..., 2]
        rep = self.replace_proj(hidden) @ E.t() + self.replace_bias         # [..., V]
        app = self.append_proj(hidden) @ E.t() + self.append_bias           # [..., V]
        return torch.cat([base, rep, app], dim=-1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        hidden = self.dropout(_last_hidden(self.encoder, input_ids, attention_mask))
        return {"label_logits": self._label_logits(hidden),
                "detect_logits": self.detect_head(hidden)}


def build_tagger(encoder_name: str = ENCODER_NAME) -> GecTagger:
    """Load the pretrained bidirectional LFM2.5 encoder and attach tagger heads (needs network)."""
    from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

    cfg = AutoConfig.from_pretrained(encoder_name, trust_remote_code=True)
    enc = AutoModelForMaskedLM.from_pretrained(encoder_name, trust_remote_code=True)
    backbone = enc.base_model
    hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "d_model", None)
    if hidden is None:
        raise RuntimeError(f"cannot resolve hidden size from {encoder_name} config")
    tok = AutoTokenizer.from_pretrained(encoder_name, trust_remote_code=True)
    return GecTagger(backbone, vocab_size=len(tok.get_vocab()), hidden_size=hidden)
