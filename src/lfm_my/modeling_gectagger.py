"""Self-contained `transformers` modeling code for the LFM2.5 subword GEC tagger.

This file ships INSIDE the Hugging Face model repo and is loaded via `trust_remote_code=True`. It must
NOT import the `spellchecker` package (end users won't have it) — the architecture, the algorithmic
subword tag space, and the iterative decode loop are all inlined here so the published model is usable
with nothing but `transformers`:

    from transformers import AutoModel
    model = AutoModel.from_pretrained("LiquidAI/LFM2.5-Spellchecker-350M", trust_remote_code=True)
    print(model.correct(["She go to school every day ."]))
    # -> ["She goes to school every day ."]

The tagger is a GECToR-style two-head model on a bidirectional MLM encoder. Tags are predicted per BPE
piece; the label space is algorithmic (no vocab file):

    0            = $KEEP                  leave this piece
    1            = $DELETE                drop this piece
    2            = $SWAP                  (only when use_swap) swap this piece with the next one
    base..base+V = $REPLACE_<piece_id>    replace this piece with BPE piece <piece_id>
    base+V..     = $APPEND_<piece_id>     keep this piece, insert BPE piece <piece_id> after it
    base = 3 if use_swap else 2 ;  num_labels = base + 2*V ;  V = tokenizer vocab size

Multi-piece / multi-word corrections emerge over the iterative passes (`correct` re-runs the tagger
until the text stops changing or `max_iter` is hit). The sentence-initial anchor is the tokenizer BOS,
prepended to every sequence; an $APPEND on it inserts at sentence start.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForMaskedLM, PretrainedConfig, PreTrainedModel

# The self-contained reranker companion ships in the SAME repo (export_hub.py copies it next to this
# file). transformers' trust_remote_code resolver scans the source for a relative import and ALSO copies
# that sibling into the dynamic-module cache when the repo is loaded by id — without it the companion is
# never materialised and the lazy load below would ModuleNotFoundError. The import FORM matters: the
# resolver (dynamic_module_utils.get_relative_imports) only matches `from .<name> import ...` or
# `import .<name>` — `from . import <name>` is NOT matched. So we import a sentinel from the companion
# (which forces it into sys.modules) and grab the module object via sys.modules. Best effort: a model
# with no reranker bundled (older export) still loads tagger-only.
try:
    from .reranker_gectagger import rerank as _rerank_fn       # noqa: F401  (resolver hint)
    import sys as _sys
    _reranker_mod = _sys.modules[__name__.rsplit(".", 1)[0] + ".reranker_gectagger"] \
        if "." in __name__ else _sys.modules["reranker_gectagger"]
except Exception:                                              # standalone / no companion -> tagger-only
    _reranker_mod = None

# --------------------------------------------------------------------------- tag space (algorithmic)

KEEP_ID, DELETE_ID, SWAP_ID = 0, 1, 2
INCORRECT = 1                 # detection-head class id gated by min_error_prob
REPLACE, APPEND, SWAP = "$REPLACE_", "$APPEND_", "$SWAP"


def _rep_base(use_swap: bool) -> int:
    return 3 if use_swap else 2


def id_to_tag(idx: int, vocab_size: int, use_swap: bool = False) -> str:
    if idx == KEEP_ID:
        return "$KEEP"
    if idx == DELETE_ID:
        return "$DELETE"
    if use_swap and idx == SWAP_ID:
        return SWAP
    base = _rep_base(use_swap)
    if idx < base + vocab_size:
        return f"{REPLACE}{idx - base}"
    return f"{APPEND}{idx - base - vocab_size}"


def apply_tags(pieces: List[int], tags: List[str]) -> List[int]:
    """Apply per-piece tags, returning the new piece-id list. pieces[0] is the BOS anchor and is never
    emitted (an $APPEND on it inserts at sentence start). $SWAP emits the next piece then this one and
    consumes both; it never fires on the BOS anchor."""
    out: List[int] = []
    i, n = 0, len(pieces)
    while i < n:
        p, t = pieces[i], tags[i]
        is_start = i == 0
        if t == SWAP and not is_start and i + 1 < n:
            out.append(pieces[i + 1]); out.append(p); i += 2; continue
        if t == "$KEEP" or t == SWAP:                  # SWAP with no valid neighbour -> safe keep
            if not is_start:
                out.append(p)
        elif t == "$DELETE":
            pass
        elif t.startswith(REPLACE):
            out.append(int(t[len(REPLACE):]))
        elif t.startswith(APPEND):
            if not is_start:
                out.append(p)
            out.append(int(t[len(APPEND):]))
        else:                                          # unknown -> safe keep
            if not is_start:
                out.append(p)
        i += 1
    return out


# --------------------------------------------------------------------------- config

class GecTaggerConfig(PretrainedConfig):
    model_type = "gec_tagger"

    # NB: the field is `num_tags`, NOT `num_labels` — `num_labels` is a reserved PretrainedConfig
    # property that auto-builds an id2label dict (here that would be 128802 entries) and breaks loading.
    def __init__(self, encoder_name: str = "LiquidAI/mlm_phase2_bidir2_step140800", num_tags: int = 128802,
                 hidden_size: int = 1024, tie_replace: bool = True, multi_head: bool = False,
                 aux_loss_weight: float = 0.5, use_swap: bool = False, qat_applied: bool = False,
                 qat_group_size: int = 32, dropout: float = 0.1, **kwargs):
        self.encoder_name = encoder_name
        self.num_tags = num_tags
        self.hidden_size = hidden_size
        self.tie_replace = tie_replace
        self.multi_head = multi_head
        self.aux_loss_weight = aux_loss_weight
        self.use_swap = use_swap
        self.qat_applied = qat_applied
        self.qat_group_size = qat_group_size
        self.dropout = dropout
        super().__init__(**kwargs)


# --------------------------------------------------------------------------- model

def _last_hidden(backbone, input_ids, attention_mask) -> torch.Tensor:
    out = backbone(input_ids=input_ids, attention_mask=attention_mask)
    hs = getattr(out, "last_hidden_state", None)
    if hs is None and getattr(out, "hidden_states", None) is not None:
        hs = out.hidden_states[-1]
    if hs is None:
        hs = out[0]
    return hs


def _build_backbone(encoder_name: str):
    """Build the bidirectional-LFM2 trunk shared across the whole encoder family (embedding / ColBERT /
    encoder-MLM / token-classification / this tagger). Weights come from THIS repo's safetensors, so the
    trunk is always built from config — no second encoder download.

    Prefers the in-library ``transformers.Lfm2BidirectionalModel`` once the bidirectional-LFM2 family PR
    lands: it is version-stable (no ``trust_remote_code`` for the trunk) and numerically identical to the
    encoder repo's remote-code MLM base (verified 0.0 CPU / <1e-5 GPU by the family integration), so the
    trained state_dict still loads 1:1 under ``encoder.*``. Until the class is exposed by ``transformers``
    this transparently falls back to the encoder repo's own remote-code MLM class with the head stripped —
    byte-identical to the original behaviour.
    """
    import transformers
    native = getattr(transformers, "Lfm2BidirectionalModel", None)
    if native is not None:                                    # native foundation (post family-PR)
        cfg = AutoConfig.from_pretrained(encoder_name)        # model_type resolves natively, no remote code
        try:
            return native._from_config(cfg)
        except AttributeError:
            return native(cfg)
    enc_cfg = AutoConfig.from_pretrained(encoder_name, trust_remote_code=True)   # fallback: encoder remote code
    return AutoModelForMaskedLM.from_config(enc_cfg, trust_remote_code=True).base_model


class GecTaggerForGEC(PreTrainedModel):
    """GECToR two-head tagger. Submodule names match the training-time `spellchecker.model.GecTagger`
    so the trained state_dict loads 1:1. With `tie_replace` the $REPLACE/$APPEND blocks are tied to the
    encoder input embeddings (logit = proj(h) · embedding_i) rather than a free Linear(hidden, 2+2V)."""

    config_class = GecTaggerConfig
    base_model_prefix = "encoder"

    def __init__(self, config: GecTaggerConfig):
        super().__init__(config)
        self._base = _rep_base(config.use_swap)
        # Bidirectional-LFM2 trunk, shared with the rest of the encoder family. Weights come from this
        # repo's safetensors (built from config, no second encoder download). Rides the in-library
        # Lfm2BidirectionalModel once the family PR lands; falls back to the encoder's remote code today.
        self.encoder = _build_backbone(config.encoder_name)
        if config.qat_applied:
            # dynamic import (not a top-level `from torchao...`): transformers statically scans this file
            # for import lines and would otherwise REQUIRE torchao even for non-QAT models that never hit
            # this branch. importlib keeps the dependency truly optional.
            import importlib
            qat = importlib.import_module("torchao.quantization.qat")
            self.encoder = qat.Int4WeightOnlyQATQuantizer(groupsize=config.qat_group_size).prepare(self.encoder)
        hidden = config.hidden_size
        self.dropout = nn.Dropout(config.dropout)
        self.detect_head = nn.Linear(hidden, 2)
        if config.multi_head:
            self.del_head = nn.Linear(hidden, 2)
            self.ins_head = nn.Linear(hidden, 2)
            self.sub_head = nn.Linear(hidden, 2)
        if config.tie_replace:
            V = (config.num_tags - self._base) // 2
            assert self._base + 2 * V == config.num_tags, "tie_replace requires num_tags=base+2V"
            self.vocab_size = V
            self.base_head = nn.Linear(hidden, self._base)
            self.replace_proj = nn.Linear(hidden, hidden)
            self.append_proj = nn.Linear(hidden, hidden)
            self.replace_bias = nn.Parameter(torch.zeros(V))
            self.append_bias = nn.Parameter(torch.zeros(V))
        else:
            self.vocab_size = (config.num_tags - self._base) // 2
            self.label_head = nn.Linear(hidden, config.num_tags)
        self._tok = None                                # lazily-built tokenizer for .correct()
        self._scorer = None                             # lazily-loaded reranker (reranker/ subfolder)
        self._rerank_op = None                          # cached (keep_confidence, tau) operating point
        self.post_init()                                # transformers 5.x: registers tied-weight keys etc.

    def _label_logits(self, hidden):
        if not self.config.tie_replace:
            return self.label_head(hidden)
        E = self.encoder.get_input_embeddings().weight[:self.vocab_size]      # [V, hidden] tied
        base = self.base_head(hidden)
        rep = self.replace_proj(hidden) @ E.t() + self.replace_bias
        app = self.append_proj(hidden) @ E.t() + self.append_bias
        return torch.cat([base, rep, app], dim=-1)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        hidden = self.dropout(_last_hidden(self.encoder, input_ids, attention_mask))
        out = {"label_logits": self._label_logits(hidden), "detect_logits": self.detect_head(hidden)}
        if self.config.multi_head:
            out["del_logits"] = self.del_head(hidden)
            out["ins_logits"] = self.ins_head(hidden)
            out["sub_logits"] = self.sub_head(hidden)
        return out

    # ---- inference ----------------------------------------------------------
    def _tokenizer(self, tokenizer=None):
        if tokenizer is not None:
            return tokenizer
        if self._tok is None:
            from transformers import AutoTokenizer
            # tokenizer files are saved into this repo, so name_or_path resolves locally; fall back to encoder
            src = self.name_or_path or self.config.encoder_name
            try:
                self._tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
            except Exception:
                self._tok = AutoTokenizer.from_pretrained(self.config.encoder_name, trust_remote_code=True)
        return self._tok

    @torch.no_grad()
    def _step(self, seqs: List[List[int]], pad_id: int, batch_size: int, keep_confidence: float,
              min_error_prob: float) -> List[List[int]]:
        V, use_swap = self.vocab_size, self.config.use_swap
        device = self.device
        new_seqs: List[List[int]] = []
        for i in range(0, len(seqs), batch_size):
            chunk = seqs[i:i + batch_size]
            maxlen = max(len(s) for s in chunk)
            ids = torch.full((len(chunk), maxlen), pad_id, dtype=torch.long)
            mask = torch.zeros((len(chunk), maxlen), dtype=torch.long)
            for b, s in enumerate(chunk):
                ids[b, :len(s)] = torch.tensor(s)
                mask[b, :len(s)] = 1
            res = self(ids.to(device), mask.to(device))
            label_logits = res["label_logits"]
            label_logits[..., KEEP_ID] += keep_confidence
            best = label_logits.argmax(-1)
            err_prob = res["detect_logits"].softmax(-1)[..., INCORRECT]
            for b, s in enumerate(chunk):
                tags = []
                for pos in range(len(s)):
                    lid = int(best[b, pos])
                    ok = lid != KEEP_ID and float(err_prob[b, pos]) >= min_error_prob
                    tags.append(id_to_tag(lid, V, use_swap) if ok else "$KEEP")
                new_seqs.append(apply_tags(s, tags))
        return new_seqs

    def _tag_correct(self, texts, tok, max_iter, max_len, batch_size, keep_confidence,
                     min_error_prob) -> List[str]:
        """Tagger-only iterative decode (the original `.correct()` body). Returns a corrected string
        per input. `keep_confidence` < 0 over-generates edits (used by the reranker)."""
        bos_id = tok.bos_token_id if tok.bos_token_id is not None else (tok.cls_token_id or 0)
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        cur = [[bos_id] + tok.encode(t, add_special_tokens=False)[:max_len - 1] for t in texts]
        active = list(range(len(cur)))
        for _ in range(max_iter):
            if not active:
                break
            updated = self._step([cur[i] for i in active], pad_id, batch_size,
                                  keep_confidence, min_error_prob)
            still = []
            for idx, new in zip(active, updated):
                new = [bos_id] + new
                if new != cur[idx]:
                    cur[idx] = new
                    still.append(idx)
            active = still
        return [tok.decode(s[1:], skip_special_tokens=True).strip() for s in cur]

    # ---- reranker (lazy, from this repo's reranker/ subfolder) ---------------
    def _reranker(self):
        """Lazily load (scorer, keep_confidence, tau) from the SAME repo's `reranker/` subfolder.
        Returns None if no reranker is bundled (then `.correct()` falls back to tagger-only). The
        scorer + operating point ship next to the model weights; nothing is fetched from elsewhere."""
        if self._scorer is not None:
            return self._scorer, self._rerank_op
        import json
        import os
        rr = _reranker_mod
        if rr is None:                                    # companion not present -> tagger-only
            print("[gectagger] reranker companion module not bundled; using tagger-only .correct()")
            self._scorer, self._rerank_op = False, None
            return False, None
        # locate the reranker/ subfolder: local export dir, or fetch from the hub repo by id
        base = self.name_or_path or ""
        scorer_dir = os.path.join(base, "reranker") if base else ""
        if not (scorer_dir and os.path.isfile(os.path.join(scorer_dir, "scorer_config.json"))):
            try:                                          # not a local dir -> resolve the hub repo
                from huggingface_hub import snapshot_download
                tok_env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
                local = snapshot_download(repo_id=base, allow_patterns=["reranker/*"], token=tok_env)
                scorer_dir = os.path.join(local, "reranker")
            except Exception as e:
                print(f"[gectagger] reranker not available ({e}); using tagger-only .correct()")
                self._scorer, self._rerank_op = False, None
                return False, None
        if not os.path.isfile(os.path.join(scorer_dir, "scorer_config.json")):
            print(f"[gectagger] no reranker/ in {base}; using tagger-only .correct()")
            self._scorer, self._rerank_op = False, None
            return False, None
        kc, tau = -0.25, 0.4                              # shipped default operating point
        op_path = os.path.join(scorer_dir, "operating_point.json")
        if os.path.isfile(op_path):
            op = json.load(open(op_path))
            kc = float(op.get("keep_confidence", kc))
            tau = float(op.get("tau", tau))
        # match the tagger's ACTUAL weight dtype (read from a real parameter, not self.dtype which is
        # unreliable when the from_config encoder carries fp32 buffers) so the shared-shape matmuls in
        # the scorer don't hit a Float-vs-Half mismatch (the Space casts the whole model to fp16).
        try:
            tagger_dtype = next(self.base_head.parameters()).dtype
        except Exception:
            tagger_dtype = self.dtype
        scorer = rr.EditScorer.load(scorer_dir).to(self.device).to(tagger_dtype)
        scorer.eval()
        self._scorer, self._rerank_op = scorer, (kc, tau)
        print(f"[gectagger] reranker loaded from {scorer_dir} (mode={scorer.mode}, "
              f"type_feature={scorer.type_feature}); operating point keep_confidence={kc} tau={tau}")
        return self._scorer, self._rerank_op

    @torch.no_grad()
    def correct(self, texts, tokenizer=None, max_iter: int = 3, max_len: int = 128, batch_size: int = 64,
                keep_confidence: float = 0.0, min_error_prob: float = 0.0, rerank: bool = True) -> List[str]:
        """Correct a list of (whitespace-tokenized) sentences. Iterates tag->apply until the text stops
        changing or `max_iter` is reached. `min_error_prob` / `keep_confidence` are the GECToR precision
        knobs (higher => fewer edits).

        rerank=True (default): run the FULL system — let the tagger over-generate at the bundled
        operating point's `keep_confidence`, then a per-edit scorer (loaded once from this repo's
        `reranker/` subfolder) keeps only edits with P(correct) >= tau and re-applies them to the
        source. This is the published MASTER-composite operating point. If no reranker is bundled, it
        transparently falls back to tagger-only. rerank=False: exact tagger-only behaviour, using the
        `keep_confidence` / `min_error_prob` passed in."""
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        tok = self._tokenizer(tokenizer)
        self.eval()

        if not rerank:
            out = self._tag_correct(texts, tok, max_iter, max_len, batch_size,
                                    keep_confidence, min_error_prob)
            return out[0] if single else out

        scorer, op = self._reranker()
        if not scorer:                                    # no reranker bundled -> tagger-only fallback
            out = self._tag_correct(texts, tok, max_iter, max_len, batch_size,
                                    keep_confidence, min_error_prob)
            return out[0] if single else out
        kc, tau = op
        # over-generate with the tagger at the scorer's training operating point (negative kc), then
        # filter per-edit. min_error_prob stays at 0 here so the scorer — not the detection gate — is
        # the precision lever (this is the configuration the published MASTER was measured at).
        hyps = self._tag_correct(texts, tok, max_iter, max_len, batch_size, kc, 0.0)
        out = _reranker_mod.rerank(scorer, tok, list(texts), hyps, tau, self.device,
                                   batch_size=batch_size)
        return out[0] if single else out
