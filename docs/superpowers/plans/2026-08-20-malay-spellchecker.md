# Bahasa Malaysia Spellchecker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the full pipeline for a Bahasa Malaysia GECToR-style spellchecker: synthetic data generation, training of a two-head tagger on `LiquidAI/LFM2.5-Encoder-350M` (on Colab T4), export compatible with `LiquidAI/LFM2.5-Encoder-350M-Spellchecker`'s inference class, and a FastAPI HF Space demo.

**Architecture:** Rule-based Malay error injection over clean Malay Wikipedia sentences produces `(noisy, corrected)` pairs. A training-side `GecTagger` module mirrors the exact submodule names of Liquid's `GecTaggerForGEC` (`encoder`, `dropout`, `detect_head`, `base_head`, `replace_proj`, `append_proj`, `replace_bias`, `append_bias`) so the trained `state_dict` loads 1:1 into Liquid's published inference class. Tag targets are computed by word-level diff → per-BPE-piece tags, iterated over multiple passes (one insertion piece per anchor per pass — matches iterative `model.correct()` decoding). Export ships Liquid's own `modeling_gectagger.py` with the weights.

**Tech Stack:** Python 3.12 (uv-managed), PyTorch, transformers==5.1.0, datasets, FastAPI, uvicorn, huggingface_hub, safetensors, pytest. Full training runs on Google Colab (free T4); this repo is developed and unit-tested on macOS (CPU).

**Reference files (fetched 2026-08-20, authoritative for compatibility):**
- `LiquidAI/LFM2.5-Encoder-350M-Spellchecker` → `modeling_gectagger.py`, `config.json`
- `LiquidAI/spellchecker` Space → `server.py`, `static/index.html`, `Dockerfile`, `requirements.txt`

## Global Constraints

- **Vocab size V = 64,400** (the real LFM2.5 tokenizer vocab; `num_tags = 2 + 2·V = 128,802` — exactly Liquid's config value; the design spec's "65,536 / 131,074" was an estimate — always compute `num_tags` from the live tokenizer instead of hardcoding).
- **Tag space (use_swap=False):** `0=$KEEP, 1=$DELETE, base=2`; `$REPLACE_<pid> = 2+pid`; `$APPEND_<pid> = 2+V+pid`.
- **BOS anchor:** every model input is `[bos] + encode(text)[:max_len-1]`; `pieces[0]` is never emitted; `$APPEND` on BOS inserts at sentence start.
- **Loss:** `label_loss + 0.5 · detect_loss`, both `CrossEntropyLoss` (ignore_index=-100); logits cast to fp32 for loss.
- **Tied replacement (`tie_replace=True`) is mandatory** (T4 memory): replace/append logits = `proj(h) @ E.t() + bias`, `E = encoder input embeddings[:V]`.
- **Training hyperparameters:** AdamW, LR 2e-5, 10% warmup (cosine), weight decay 0.1, batch 8 × grad-accum 4 (effective 32), bf16 autocast on CUDA / fp32 on CPU, max_len 128, head dropout 0.1, early stopping patience 2 on val loss, checkpoint every 500 steps.
- **Compatibility guarantee:** trained `state_dict` keys must equal `GecTaggerForGEC`'s (`encoder.*`, `detect_head.*`, `base_head.*`, `replace_proj.*`, `append_proj.*`, `replace_bias`, `append_bias`). No other persistent modules allowed in the training model.
- **Text form:** all training text is whitespace-tokenized with punctuation spaced apart (Liquid's form); Malay needs no contraction splitting (drop the English `n't`/`'s` rules from Liquid's server).
- **Package name:** `lfm_my`, sources in `src/lfm_my/`, tests in `tests/` mirroring.
- **Python:** 3.12 via uv (system Python 3.14 has no torch wheels).
- **UI assets are our own** — do NOT copy LiquidAI's `style.css`/`lliquid.gif`; write a minimal original stylesheet with the same functional layout (green inserts, red strikethrough deletes).
- **HF identifiers** are CLI parameters (`--model-repo`, `--space-repo`), never hardcoded.

---

### Task 1: Repo scaffold + vendored inference module

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `README.md`, `src/lfm_my/__init__.py`, `tests/__init__.py` (empty)
- Create: `src/lfm_my/modeling_gectagger.py` (vendored verbatim from HF)
- Test: `tests/test_imports.py`

**Interfaces:**
- Produces: importable package `lfm_my`; `lfm_my.modeling_gectagger.{KEEP_ID, DELETE_ID, SWAP_ID, INCORRECT, id_to_tag, apply_tags, GecTaggerConfig, GecTaggerForGEC}` — used by every later task.

- [ ] **Step 1: Create `.gitignore`**

```gitignore
.venv/
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
dist/
build/
data/*.jsonl
data/*.jsonl.gz
data/*.parquet
data/raw/
checkpoints/
exports/
space_build/
*.safetensors
.DS_Store
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[project]
name = "lfm-my"
version = "0.1.0"
description = "Bahasa Malaysia spellchecker: GECToR-style fine-tune of LFM2.5-Encoder-350M"
requires-python = ">=3.12,<3.13"
dependencies = [
    "torch>=2.6",
    "transformers==5.1.0",
    "safetensors==0.7.0",
    "datasets>=3.0",
    "huggingface_hub>=0.26",
]

[project.optional-dependencies]
app = [
    "fastapi==0.115.6",
    "uvicorn[standard]==0.34.0",
]
dev = [
    "pytest>=8",
    "httpx>=0.27",
    "fastapi==0.115.6",
    "uvicorn[standard]==0.34.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/lfm_my"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "network: tests that need network access / HF hub (deselect with '-m \"not network\"')",
]
```

- [ ] **Step 3: Vendor the inference module verbatim**

Download Liquid's exact modeling file (this exact byte content is what our weights must load into, and what we will ship with our exported model):

```bash
mkdir -p src/lfm_my
curl -fsSL -o src/lfm_my/modeling_gectagger.py \
  https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M-Spellchecker/resolve/main/modeling_gectagger.py
```

Do NOT edit it. The top-level `try: from .reranker_gectagger import rerank` fails harmlessly (caught) when no reranker sibling exists — we ship tagger-only.

- [ ] **Step 4: Create `src/lfm_my/__init__.py` and empty `tests/__init__.py`**

`tests/__init__.py` must exist (empty file) so cross-test imports like `from tests.test_model import TinyEncoder` resolve.

`src/lfm_my/__init__.py`:

```python
"""Bahasa Malaysia spellchecker: GECToR-style tagger on LFM2.5-Encoder-350M."""

ENCODER_NAME = "LiquidAI/LFM2.5-Encoder-350M"
```

- [ ] **Step 5: Create root `README.md`**

```markdown
# lfm-my — Bahasa Malaysia spellchecker

GECToR-style grammatical-error-correction tagger for Bahasa Malaysia, fine-tuned from
[LiquidAI/LFM2.5-Encoder-350M](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M), mirroring
[LiquidAI/LFM2.5-Encoder-350M-Spellchecker](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M-Spellchecker).

- Design spec: `docs/superpowers/specs/2026-08-11-malay-spellchecker-design.md`
- Implementation plan: `docs/superpowers/plans/2026-08-20-malay-spellchecker.md`

## Develop

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
pytest -m "not network"
```
```

- [ ] **Step 6: Create env + install**

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"
```

Expected: installs torch (CPU), transformers 5.1.0, pytest. (First run downloads ~200MB torch wheel.)

- [ ] **Step 7: Write the test** (`tests/test_imports.py`)

```python
import lfm_my
from lfm_my.modeling_gectagger import (
    KEEP_ID, DELETE_ID, INCORRECT, apply_tags, id_to_tag, GecTaggerConfig,
)


def test_constants():
    assert KEEP_ID == 0
    assert DELETE_ID == 1
    assert INCORRECT == 1


def test_config_defaults_match_liquid():
    cfg = GecTaggerConfig(encoder_name=lfm_my.ENCODER_NAME)
    assert cfg.hidden_size == 1024
    assert cfg.tie_replace is True
    assert cfg.use_swap is False
    assert cfg.aux_loss_weight == 0.5


def test_tag_space_ids():
    V = 64400
    assert id_to_tag(0, V) == "$KEEP"
    assert id_to_tag(1, V) == "$DELETE"
    assert id_to_tag(2, V) == "$REPLACE_0"
    assert id_to_tag(2 + V, V) == "$APPEND_0"
    assert id_to_tag(2 + 2 * V - 1, V) == f"$APPEND_{V - 1}"
```

- [ ] **Step 8: Run test to verify it passes**

Run: `uv run pytest tests/test_imports.py -v`
Expected: PASS (vendored module is already implemented; this test verifies the vendor worked).

- [ ] **Step 9: Commit**

```bash
git add .gitignore pyproject.toml README.md src/ tests/
git commit -m "scaffold: project skeleton + vendored Liquid modeling_gectagger"
```

---

### Task 2: Text utilities (Malay tokenize/detok + word diff)

**Files:**
- Create: `src/lfm_my/text.py`
- Test: `tests/test_text.py`

**Interfaces:**
- Produces:
  - `tokenize(text: str) -> str` — spaces punctuation apart, collapses whitespace (Liquid's training form, minus English contractions).
  - `detok(text: str) -> str` — reattaches punctuation for display.
  - `diff_segments(source: str, corrected: str) -> list[dict]` — word-level diff → `[{"text": str, "kind": "keep"|"edit"|"del"}]` (same contract as Liquid's `server.py`, consumed by Task 12's API).

- [ ] **Step 1: Write the failing test** (`tests/test_text.py`)

```python
from lfm_my.text import tokenize, detok, diff_segments


def test_tokenize_spaces_punctuation():
    assert tokenize("Dia pergi ke pasar, bukan?") == "Dia pergi ke pasar , bukan ?"
    assert tokenize("  Sudah   makan. ") == "Sudah makan ."
    assert tokenize("Tiada tanda") == "Tiada tanda"


def test_tokenize_keeps_malay_affixes_joined():
    assert tokenize("Buku itu dibaca oleh Ali.") == "Buku itu dibaca oleh Ali ."


def test_detok_reattaches():
    assert detok("Dia pergi ke pasar , bukan ?") == "Dia pergi ke pasar, bukan?"
    assert detok("Sudah makan .") == "Sudah makan."


def test_roundtrip_natural_malay():
    text = "Saya tidak tahu, dia sudah pergi ke sekolah."
    assert detok(tokenize(text)) == text


def test_diff_segments_kinds():
    seg = diff_segments("dia tidak arah pinjam buku", "dia tidak akan pinjam buku")
    assert seg == [
        {"text": "dia tidak", "kind": "keep"},
        {"text": "arah", "kind": "del"},
        {"text": "akan", "kind": "edit"},
        {"text": "pinjam buku", "kind": "keep"},
    ]


def test_diff_segments_insert_delete():
    seg = diff_segments("saya makan", "saya sudah makan nasi")
    assert {"text": "sudah", "kind": "edit"} in seg
    assert {"text": "nasi", "kind": "edit"} in seg
    assert all(s["kind"] != "del" for s in seg)
    seg = diff_segments("saya sudah makan", "saya makan")
    assert {"text": "sudah", "kind": "del"} in seg


def test_diff_segments_identical():
    assert diff_segments("bersih dan kemas", "bersih dan kemas") == [
        {"text": "bersih dan kemas", "kind": "keep"}
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_text.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lfm_my.text'`

- [ ] **Step 3: Implement** (`src/lfm_my/text.py`)

```python
"""Whitespace tokenization (Liquid's training form) + word-level diff segments.

Malay notes: no contraction splitting (English-only in Liquid's server). Malay prefixes
(di-, ber-, meN-, -kan, -nya ...) stay attached; punctuation is spaced apart.
"""
from __future__ import annotations

import difflib
import re

_PUNCT = re.compile(r'([.,!?;:()\[\]{}"«»…])')
_ATTACH_LEFT = re.compile(r"\s+([.,!?;:%)\]}»…])")
_ATTACH_RIGHT = re.compile(r"([(\[{«])\s+")


def tokenize(text: str) -> str:
    """Natural text -> whitespace-tokenized form (punctuation spaced apart)."""
    text = _PUNCT.sub(r" \1 ", text)
    return re.sub(r"\s+", " ", text).strip()


def detok(text: str) -> str:
    """Whitespace-tokenized form -> natural display text."""
    text = _ATTACH_LEFT.sub(r"\1", text)
    text = _ATTACH_RIGHT.sub(r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def diff_segments(source: str, corrected: str) -> list[dict]:
    """Word-level diff -> [{"text", "kind"}], kind in {keep, edit, del}, for inline rendering:
    `keep` unchanged, `edit` inserted/replaced (green), `del` removed (red strikethrough; NOT part
    of the corrected text). Same contract as Liquid's server.py (language-agnostic)."""
    s, c = source.split(), corrected.split()
    seg: list[dict] = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, s, c, autojunk=False).get_opcodes():
        if op == "equal":
            seg.append({"text": " ".join(c[j1:j2]), "kind": "keep"})
        elif op == "insert":
            seg.append({"text": " ".join(c[j1:j2]), "kind": "edit"})
        elif op == "delete":
            seg.append({"text": " ".join(s[i1:i2]), "kind": "del"})
        elif op == "replace":
            seg.append({"text": " ".join(s[i1:i2]), "kind": "del"})
            seg.append({"text": " ".join(c[j1:j2]), "kind": "edit"})
    return seg or [{"text": corrected, "kind": "keep"}]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_text.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/lfm_my/text.py tests/test_text.py
git commit -m "feat: Malay whitespace tokenizer + word diff segments"
```

---

### Task 3: Tag-space helpers

**Files:**
- Create: `src/lfm_my/tags.py`
- Test: `tests/test_tags.py`

**Interfaces:**
- Produces (used by Tasks 4, 8, 10):
  - `BASE: int = 2` (use_swap=False)
  - `replace_id(piece_id: int) -> int` → `BASE + piece_id`
  - `append_id(piece_id: int, vocab_size: int) -> int` → `BASE + vocab_size + piece_id`

- [ ] **Step 1: Write the failing test** (`tests/test_tags.py`)

```python
from lfm_my.modeling_gectagger import apply_tags, id_to_tag
from lfm_my.tags import BASE, append_id, replace_id

V = 64400


def test_tag_id_layout():
    assert BASE == 2
    assert replace_id(0) == 2
    assert replace_id(17) == 19
    assert append_id(0, V) == 2 + V
    assert append_id(17, V) == 2 + V + 17


def test_ids_roundtrip_with_inference_helpers():
    assert id_to_tag(replace_id(42), V) == "$REPLACE_42"
    assert id_to_tag(append_id(42, V), V) == "$APPEND_42"


def test_apply_tags_semantics():
    BOS = 1
    pieces = [BOS, 100, 101, 102, 103]
    tags = ["$KEEP", "$KEEP", "$DELETE", "$REPLACE_55", "$APPEND_66"]
    assert apply_tags(pieces, tags) == [100, 55, 103, 66]


def test_apply_tags_bos_append_inserts_at_start():
    BOS = 1
    pieces = [BOS, 100]
    tags = ["$APPEND_77", "$KEEP"]
    assert apply_tags(pieces, tags) == [77, 100]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tags.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lfm_my.tags'`

- [ ] **Step 3: Implement** (`src/lfm_my/tags.py`)

```python
"""Algorithmic subword tag-space ids (use_swap=False in v1).

Layout: 0=$KEEP, 1=$DELETE, BASE..BASE+V = $REPLACE_<pid>, BASE+V..BASE+2V = $APPEND_<pid>.
Mirrors lfm_my.modeling_gectagger (Liquid's inference module).
"""
from __future__ import annotations

BASE = 2  # _rep_base(use_swap=False)


def replace_id(piece_id: int) -> int:
    return BASE + piece_id


def append_id(piece_id: int, vocab_size: int) -> int:
    return BASE + vocab_size + piece_id
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_tags.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/lfm_my/tags.py tests/test_tags.py
git commit -m "feat: tag-space id helpers"
```

---

### Task 4: Target conversion — (noisy, corrected) → per-piece tags

This is the highest-risk module (design spec risk table: "Tag-conversion bugs → noisy targets"). Multi-pass targets are required because one `$APPEND` inserts exactly one piece per anchor per pass — matching `model.correct()`'s iterative decode.

**Files:**
- Create: `src/lfm_my/convert.py`
- Test: `tests/test_convert.py`

**Interfaces:**
- Consumes: `lfm_my.tags.{replace_id, append_id}`, `lfm_my.modeling_gectagger.{KEEP_ID, DELETE_ID, apply_tags}`.
- Produces (used by Task 8's dataset):
  - `@dataclass Pass: tags: list[int]; detect: list[int]; bos_append: int | None` — per input piece, **without** BOS.
  - `@dataclass Conversion: inputs: list[list[int]]; passes: list[Pass]` — pass *k* trains on input `inputs[k]` (piece ids, no BOS) with target `passes[k]`.
  - `pieces_of(units, encode) -> list[int]`
  - `one_pass(noisy_units, corrected_units, encode, vocab_size) -> (Pass, out_pieces)`
  - `convert_pair(noisy: str, corrected: str, encode_word: Callable[[str], list[int]], vocab_size: int, max_passes: int = 5) -> Conversion | None` — `None` = failed to converge (drop the example); empty `passes` with equal inputs = clean pair (collate emits all-$KEEP).

- [ ] **Step 1: Write the failing test** (`tests/test_convert.py`)

```python
from lfm_my.convert import Conversion, convert_pair, one_pass, pieces_of
from lfm_my.modeling_gectagger import KEEP_ID, DELETE_ID, apply_tags
from lfm_my.tags import append_id, replace_id

V = 1000
BOS = 1


def char_encode(word: str) -> list[int]:
    """Stub BPE: one piece per character (id = 10 + ordinal)."""
    return [10 + ord(c) for c in word]


def pieces(words: list[str]) -> list[int]:
    return pieces_of(words, char_encode)


def _tag_str(t: int) -> str:
    if t == 0:
        return "$KEEP"
    if t == 1:
        return "$DELETE"
    if t < 2 + V:
        return f"$REPLACE_{t - 2}"
    return f"$APPEND_{t - 2 - V}"


def apply_pass(input_pieces: list[int], p) -> list[int]:
    """Apply a Pass through Liquid's inference-time apply_tags (the ground truth)."""
    bos_tag = "$KEEP" if p.bos_append is None else f"$APPEND_{p.bos_append}"
    tags = [bos_tag] + [_tag_str(t) for t in p.tags]
    return apply_tags([BOS] + input_pieces, tags)


def test_clean_pair_gives_empty_passes():
    conv = convert_pair("saya makan nasi", "saya makan nasi", char_encode, V)
    assert isinstance(conv, Conversion)
    assert conv.passes == []


def test_single_word_replace_one_pass():
    conv = convert_pair("saya makan nasik", "saya makan nasi", char_encode, V)
    assert len(conv.passes) == 1
    p = conv.passes[0]
    # 'nasik' -> 'nasi': n,a,s,i kept; trailing 'k' deleted
    assert p.tags[-1] == DELETE_ID
    assert p.detect[-1] == 1
    assert apply_pass(conv.inputs[0], p) == pieces(["saya", "makan", "nasi"])


def test_word_replace_longer_needs_two_passes():
    # 'nasi' (4 pieces) -> 'nasilemak' (9 pieces): paired prefix + deferred tail
    conv = convert_pair("saya makan nasi", "saya makan nasilemak", char_encode, V)
    assert conv is not None and len(conv.passes) >= 2
    out = pieces(["saya", "makan", "nasi"])
    for inp, p in zip(conv.inputs, conv.passes):
        assert inp == out
        out = apply_pass(inp, p)
    assert out == pieces(["saya", "makan", "nasilemak"])


def test_word_delete():
    conv = convert_pair("saya sudah makan", "saya makan", char_encode, V)
    assert len(conv.passes) == 1
    out = apply_pass(conv.inputs[0], conv.passes[0])
    assert out == pieces(["saya", "makan"])


def test_word_insert_after_anchor():
    conv = convert_pair("saya makan", "saya sudah makan", char_encode, V)
    assert conv is not None
    out = pieces(["saya", "makan"])
    for inp, p in zip(conv.inputs, conv.passes):
        assert inp == out
        out = apply_pass(inp, p)
    assert out == pieces(["saya", "sudah", "makan"])


def test_insert_at_sentence_start_uses_bos():
    conv = convert_pair("saya makan", "tolong saya makan", char_encode, V)
    assert conv is not None
    p0 = conv.passes[0]
    assert p0.bos_append is not None
    out = apply_pass(conv.inputs[0], p0)
    assert out[0] == p0.bos_append


def test_replace_then_insert_defers_correctly():
    # 'dia' replaced AND insertion right after it -> insertion must defer (anchor is $REPLACE)
    conv = convert_pair("dia makan", "mereka sudah makan", char_encode, V)
    assert conv is not None
    out = pieces(["dia", "makan"])
    for inp, p in zip(conv.inputs, conv.passes):
        assert inp == out
        out = apply_pass(inp, p)
    assert out == pieces(["mereka", "sudah", "makan"])


def test_one_pass_detect_flags():
    p, out = one_pass(["saya", "nasik"], ["saya", "nasi"], char_encode, V)
    assert p.detect[: len(pieces(["saya"]))] == [0] * 4
    assert 1 in p.detect


def test_non_convergence_returns_none():
    conv = convert_pair("a b c d e f g h", "z y x w v u t s", char_encode, V, max_passes=1)
    assert conv is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_convert.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lfm_my.convert'`

- [ ] **Step 3: Implement** (`src/lfm_my/convert.py`)

```python
"""(noisy, corrected) -> per-BPE-piece GECToR tags.

Tags are generated pass-by-pass: one $APPEND inserts exactly ONE piece per anchor per pass
(apply_tags semantics), so multi-piece insertions are deferred to later passes. This matches
inference exactly — model.correct() re-runs the tagger until the text stops changing
(max_iter=3), and training pass k>0 sees pass k-1's OUTPUT as input, same as decode.

Pass 1 aligns at WORD level (cleaner whole-word edits); later passes align at piece level.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Callable, List, Optional

from lfm_my.modeling_gectagger import DELETE_ID, KEEP_ID
from lfm_my.tags import append_id, replace_id


@dataclass
class Pass:
    tags: List[int]                # per input piece (no BOS)
    detect: List[int]              # 1 where tag != $KEEP
    bos_append: Optional[int]      # piece id to $APPEND on the BOS anchor, else None


@dataclass
class Conversion:
    inputs: List[List[int]]        # piece ids per pass (no BOS); inputs[k] <-> passes[k]
    passes: List[Pass]             # may be empty: noisy == corrected (clean pair)


def pieces_of(units: List, encode: Callable) -> List[int]:
    out: List[int] = []
    for u in units:
        out.extend(encode(u))
    return out


def one_pass(noisy_units: List, corrected_units: List, encode: Callable, vocab_size: int):
    """One feasible edit pass. Returns (Pass, out_pieces) where out_pieces is exactly what
    apply_tags would emit for these tags (the next pass's input / this pass's target)."""
    tags: List[int] = []
    detect: List[int] = []
    out: List[int] = []
    bos_append: Optional[int] = None
    anchor: Optional[int] = None  # index into tags of the LAST EMITTED piece

    def keep_pieces(pieces: List[int]) -> None:
        nonlocal anchor
        for x in pieces:
            tags.append(KEEP_ID)
            detect.append(0)
            out.append(x)
            anchor = len(tags) - 1

    ops = difflib.SequenceMatcher(None, noisy_units, corrected_units, autojunk=False).get_opcodes()
    for op, i1, i2, j1, j2 in ops:
        if op == "equal":
            for u in noisy_units[i1:i2]:
                keep_pieces(encode(u))
        elif op == "delete":
            for u in noisy_units[i1:i2]:
                for _ in encode(u):
                    tags.append(DELETE_ID)
                    detect.append(1)
        elif op == "insert":
            q = pieces_of(list(corrected_units[j1:j2]), encode)
            if not q:
                continue
            if anchor is None:                       # nothing emitted yet -> BOS anchor
                if bos_append is None:
                    bos_append = q[0]                # one piece per pass; rest defers
            elif tags[anchor] == KEEP_ID:            # a KEEP slot can take the APPEND
                tags[anchor] = append_id(q[0], vocab_size)
                detect[anchor] = 1
                out.append(q[0])
            # anchor is $REPLACE context -> defer whole insertion to next pass
        elif op == "replace":
            p = pieces_of(list(noisy_units[i1:i2]), encode)
            q = pieces_of(list(corrected_units[j1:j2]), encode)
            k = min(len(p), len(q))
            for t in range(k):
                if p[t] == q[t]:
                    tags.append(KEEP_ID)
                    detect.append(0)
                else:
                    tags.append(replace_id(q[t]))
                    detect.append(1)
                out.append(q[t])
                anchor = len(tags) - 1
            for _ in range(k, len(p)):
                tags.append(DELETE_ID)
                detect.append(1)
            # surplus q[k:] defers to the next pass
    return Pass(tags=tags, detect=detect, bos_append=bos_append), out


def convert_pair(noisy: str, corrected: str, encode_word: Callable[[str], List[int]],
                 vocab_size: int, max_passes: int = 5) -> Optional[Conversion]:
    """Convert one whitespace-tokenized pair. Returns None if it does not converge within
    max_passes (caller drops the example). Empty passes == identical strings (clean pair)."""
    noisy_words, corrected_words = noisy.split(), corrected.split()
    corrected_pieces = pieces_of(corrected_words, encode_word)
    identity = lambda x: [x]                                        # noqa: E731  piece-level passes

    inputs: List[List[int]] = []
    passes: List[Pass] = []
    cur_words: Optional[List] = noisy_words
    cur_pieces = pieces_of(noisy_words, encode_word)
    encode = encode_word
    target: List = corrected_words

    for _ in range(max_passes):
        if cur_pieces == corrected_pieces:
            return Conversion(inputs=inputs, passes=passes)
        p, out = one_pass(cur_words, target, encode, vocab_size)
        inputs.append(cur_pieces)
        passes.append(p)
        cur_pieces = out
        cur_words, encode, target = cur_pieces, identity, corrected_pieces

    return Conversion(inputs=inputs, passes=passes) if cur_pieces == corrected_pieces else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_convert.py -v`
Expected: PASS (all 9 tests)

- [ ] **Step 5: Commit**

```bash
git add src/lfm_my/convert.py tests/test_convert.py
git commit -m "feat: multi-pass (noisy, corrected) -> per-piece tag conversion"
```

---

### Task 5: Malay error injection

**Files:**
- Create: `src/lfm_my/errors.py`
- Test: `tests/test_errors.py`

**Interfaces:**
- Produces (used by Task 6):
  - `@dataclass InjectorConfig: seed=0, p_clean=0.17, min_errors=1, max_errors=3`
  - `inject_errors(sentence: str, rng: random.Random, cfg: InjectorConfig = InjectorConfig()) -> str` — takes/returns whitespace-tokenized text; with probability `p_clean` returns the input unchanged.

Operator catalogue (each returns the modified word list or `None` if inapplicable; the injector retries up to 4 times per requested error):
1. typo char substitution (QWERTY neighbours), deletion, insertion, transposition, doubling — on a random word of length ≥ 3
2. Malay affix mangling: `meN-` allomorph swaps (`meng-`/`mem-`/`men-`/`meny-`/`me-`/`menge-`), restore-dropped-root-letter (`menulis`→`mentulis`), prefix stripping, `ber-`→`be-`
3. `di-` prefix/space confusion: `dimakan`→`di makan`, `di makan`→`dimakan`
4. suffix drop: `-kan`, `-i`, `-lah`, `-nya`
5. SMS abbreviation (yang→yg, dengan→dgn, tidak→tak, …)
6. word duplication (`buku`→`buku buku`)
7. adjacent word merge / random word split
8. stray comma insertion, comma deletion
9. sentence-initial lowercasing

- [ ] **Step 1: Write the failing test** (`tests/test_errors.py`)

```python
import random

from lfm_my.errors import InjectorConfig, inject_errors


def _clean_fraction(n=400, seed=0):
    clean = 0
    for i in range(n):
        rng = random.Random(seed + i)
        if inject_errors("dia pergi ke pasar raya", rng, InjectorConfig(seed=seed + i)) == "dia pergi ke pasar raya":
            clean += 1
    return clean / n


def test_clean_fraction_within_band():
    frac = _clean_fraction()
    assert 0.05 <= frac <= 0.30          # spec: ~15–20% kept clean


def test_output_stays_nonempty_str():
    for i in range(200):
        out = inject_errors("saya tidak tahu akan perkara itu", random.Random(i))
        assert isinstance(out, str) and out.strip()


def test_abbreviation_applies():
    applied = False
    for i in range(200):
        rng = random.Random(i)
        cfg = InjectorConfig(seed=i, p_clean=0.0, min_errors=1, max_errors=1)
        if "dgn" in inject_errors("saya pergi dengan kawan", rng, cfg).split():
            applied = True
            break
    assert applied


def test_affix_mangling_applies():
    seen = set()
    for i in range(400):
        rng = random.Random(i)
        cfg = InjectorConfig(seed=i, p_clean=0.0, min_errors=1, max_errors=1)
        out = inject_errors("dia menulis surat dan membaca buku", rng, cfg)
        for w in out.split():
            if w not in {"dia", "menulis", "surat", "dan", "membaca", "buku"}:
                seen.add(w)
    assert seen  # some operator mutated at least one word


def test_di_split_or_join_applies():
    seen = False
    for i in range(400):
        rng = random.Random(i)
        cfg = InjectorConfig(seed=i, p_clean=0.0, min_errors=1, max_errors=1)
        if "di" in inject_errors("buku itu dimakan oleh anjing", rng, cfg).split():
            seen = True
            break
    assert seen


def test_deterministic_for_same_seed():
    s = "mereka sudah pergi ke sekolah pagi tadi"
    assert inject_errors(s, random.Random(7)) == inject_errors(s, random.Random(7))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_errors.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lfm_my.errors'`

- [ ] **Step 3: Implement** (`src/lfm_my/errors.py`)

```python
"""Rule-based Malay error injection mirroring real learner/informal errors.

Input and output are whitespace-tokenized strings (punctuation spaced apart).
~p_clean of sentences are returned untouched so the model learns $KEEP on clean text.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

ABBREV = {
    "yang": "yg", "dengan": "dgn", "untuk": "utk", "tidak": "tak", "saya": "sy",
    "sudah": "dah", "hendak": "nak", "macam": "mc", "akan": "akn",
    "bagaimana": "gimana", "begitu": "gitu", "sekarang": "skrg",
}

# Malay QWERTY-ish neighbours (lowercase a-z)
NEIGHBOURS = {
    "a": "sqwz", "b": "vghn", "c": "xdfv", "d": "serfcx", "e": "wsdfr", "f": "drtgvc",
    "g": "ftyhbv", "h": "gyujbn", "i": "ujklo", "j": "huiknm", "k": "jiolm", "l": "kop",
    "m": "njk", "n": "bhjm", "o": "iklp", "p": "ol", "q": "wa", "r": "edft", "s": "awdexz",
    "t": "rfgy", "u": "yhjki", "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu", "z": "asx",
}

# meN- allomorphs (ordered longest-first) ; _RESTORE: nasal -> dropped root-initial letter
_ME_PREFIXES = ["meny", "meng", "mem", "men", "me"]
_RESTORE = {"meng": "k", "men": "t", "meny": "s", "mem": "p"}
_PUNCT = set(".,!?;:")


@dataclass
class InjectorConfig:
    seed: int = 0
    p_clean: float = 0.17
    min_errors: int = 1
    max_errors: int = 3


# ----------------------------------------------------------------------------- operators
# each: (words: list[str], rng) -> list[str] | None   (mutates nothing; returns new list)

def op_typo_sub(words, rng):
    idx = [i for i, w in enumerate(words) if len(w) >= 3 and w.isalpha() and w.islower()]
    if not idx:
        return None
    i = rng.choice(idx)
    w = list(words[i])
    j = rng.randrange(len(w))
    alt = NEIGHBOURS.get(w[j])
    if not alt:
        return None
    w[j] = rng.choice(alt)
    words[i] = "".join(w)
    return words


def op_typo_delete(words, rng):
    idx = [i for i, w in enumerate(words) if len(w) >= 4 and w.isalpha()]
    if not idx:
        return None
    i = rng.choice(idx)
    j = rng.randrange(1, len(words[i]))
    words[i] = words[i][:j] + words[i][j + 1:]
    return words


def op_typo_insert(words, rng):
    idx = [i for i, w in enumerate(words) if len(w) >= 3 and w.isalpha()]
    if not idx:
        return None
    i = rng.choice(idx)
    j = rng.randrange(1, len(words[i]))
    words[i] = words[i][:j] + rng.choice("aeioubkmnt") + words[i][j:]
    return words


def op_typo_swap(words, rng):
    idx = [i for i, w in enumerate(words) if len(w) >= 4 and w.isalpha()]
    if not idx:
        return None
    i = rng.choice(idx)
    j = rng.randrange(1, len(words[i]) - 1)
    w = list(words[i])
    w[j], w[j + 1] = w[j + 1], w[j]
    if "".join(w) == words[i]:
        return None
    words[i] = "".join(w)
    return words


def op_typo_double(words, rng):
    idx = [i for i, w in enumerate(words) if len(w) >= 3 and w.isalpha()]
    if not idx:
        return None
    i = rng.choice(idx)
    j = rng.randrange(len(words[i]))
    words[i] = words[i][:j] + words[i][j] + words[i][j:]
    return words


def op_affix_allomorph(words, rng):
    idx = [i for i, w in enumerate(words) if any(w.startswith(p) and len(w) > len(p) + 2
                                                 for p in _ME_PREFIXES)]
    if not idx:
        return None
    i = rng.choice(idx)
    w = words[i]
    pref = next(p for p in _ME_PREFIXES if w.startswith(p) and len(w) > len(p) + 2)
    root = w[len(pref):]
    cands = [p + root for p in _ME_PREFIXES if p != pref and p != "me" and len(p + root) > 3]
    if pref == "meng":
        cands.append("menge" + root)
    elif pref == "menge":
        cands.append("meng" + root)
    if not cands:
        return None
    words[i] = rng.choice(cands)
    return words


def op_affix_restore_letter(words, rng):
    # menulis -> mentulis, mengirim -> mengkirim, menyapu -> mensapu, memakai -> mempakai
    idx = [i for i, w in enumerate(words) if any(w.startswith(p) and len(w) > len(p) + 2
                                                 for p in ("meng", "meny", "men", "mem"))]
    if not idx:
        return None
    i = rng.choice(idx)
    w = words[i]
    for pref in ("meng", "meny", "men", "mem"):
        if w.startswith(pref) and len(w) > len(pref) + 2 and pref in _RESTORE:
            words[i] = pref + _RESTORE[pref] + w[len(pref):]
            return words
    return None


def op_affix_strip(words, rng):
    prefs = _ME_PREFIXES + ["ber", "be", "di", "ke"]
    idx = [i for i, w in enumerate(words) if any(w.startswith(p) and len(w) > len(p) + 3
                                                 for p in prefs)]
    if not idx:
        return None
    i = rng.choice(idx)
    w = words[i]
    for pref in prefs:
        if w.startswith(pref) and len(w) > len(pref) + 3:
            words[i] = w[len(pref):]
            return words
    return None


def op_ber_be(words, rng):
    idx = [i for i, w in enumerate(words) if w.startswith("ber") and len(w) > 5]
    if not idx:
        return None
    i = rng.choice(idx)
    words[i] = "be" + words[i][3:]
    return words


def op_di_space(words, rng):
    # joined passive -> split preposition, OR split 'di' -> joined
    idx = [i for i, w in enumerate(words) if w.startswith("di") and len(w) > 4
           and w[2] not in _PUNCT and w[2].islower()]
    if idx:
        i = rng.choice(idx)
        words[i:i + 1] = ["di", words[i][2:]]
        return words
    idx = [i for i in range(len(words) - 1) if words[i] == "di" and words[i + 1].islower()
           and len(words[i + 1]) >= 3]
    if not idx:
        return None
    i = rng.choice(idx)
    words[i:i + 2] = ["di" + words[i + 1]]
    return words


def op_suffix_drop(words, rng):
    idx = [i for i, w in enumerate(words) if any(w.endswith(s) and len(w) > len(s) + 3
                                                 for s in ("kan", "nya", "lah", "i"))]
    if not idx:
        return None
    i = rng.choice(idx)
    w = words[i]
    for s in ("kan", "nya", "lah", "i"):
        if w.endswith(s) and len(w) > len(s) + 3:
            words[i] = w[:-len(s)]
            return words
    return None


def op_abbrev(words, rng):
    idx = [i for i, w in enumerate(words) if w in ABBREV]
    if not idx:
        return None
    i = rng.choice(idx)
    words[i] = ABBREV[words[i]]
    return words


def op_duplicate_word(words, rng):
    idx = [i for i, w in enumerate(words) if w.isalpha() and len(words) < 30]
    if not idx:
        return None
    i = rng.choice(idx)
    words.insert(i + 1, words[i])
    return words


def op_merge_words(words, rng):
    idx = [i for i in range(len(words) - 1) if words[i].isalpha() and words[i + 1].isalpha()
           and len(words[i]) + len(words[i + 1]) <= 14]
    if not idx:
        return None
    i = rng.choice(idx)
    words[i:i + 2] = [words[i] + words[i + 1]]
    return words


def op_split_word(words, rng):
    idx = [i for i, w in enumerate(words) if len(w) >= 6 and w.isalpha()]
    if not idx:
        return None
    i = rng.choice(idx)
    j = rng.randrange(2, len(words[i]) - 2)
    words[i:i + 1] = [words[i][:j], words[i][j:]]
    return words


def op_stray_comma(words, rng):
    idx = [i for i in range(1, len(words) - 1) if words[i].isalpha()]
    if not idx:
        return None
    words.insert(rng.choice(idx) + 1, ",")
    return words


def op_comma_drop(words, rng):
    idx = [i for i, w in enumerate(words) if w == ","]
    if not idx:
        return None
    words.pop(rng.choice(idx))
    return words


def op_case_first(words, rng):
    if not words or not words[0][:1].isupper():
        return None
    words[0] = words[0][0].lower() + words[0][1:]
    return words


OPS = [
    op_typo_sub, op_typo_delete, op_typo_insert, op_typo_swap, op_typo_double,
    op_affix_allomorph, op_affix_restore_letter, op_affix_strip, op_ber_be, op_di_space,
    op_suffix_drop, op_abbrev, op_duplicate_word, op_merge_words, op_split_word,
    op_stray_comma, op_comma_drop, op_case_first,
]


def inject_errors(sentence: str, rng: random.Random, cfg: InjectorConfig = InjectorConfig()) -> str:
    """Whitespace-tokenized sentence -> noisy version (or unchanged with probability p_clean)."""
    if rng.random() < cfg.p_clean:
        return sentence
    words = sentence.split()
    for _ in range(rng.randint(cfg.min_errors, cfg.max_errors)):
        for _ in range(4):                     # retry loop for inapplicable operators
            out = rng.choice(OPS)(list(words), rng)
            if out and out != words and all(out):
                words = out
                break
    return " ".join(words)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_errors.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/lfm_my/errors.py tests/test_errors.py
git commit -m "feat: rule-based Malay error injection"
```

---

### Task 6: Dataset builder (Malay corpus → synthetic pairs)

**Files:**
- Create: `src/lfm_my/data_build.py`
- Create: `tests/fixtures/malay_corpus.txt` (12 hand-written Malay sentences)
- Test: `tests/test_data_build.py`

**Interfaces:**
- Consumes: `lfm_my.text.tokenize`, `lfm_my.errors.{inject_errors, InjectorConfig}`.
- Produces (runs on Colab at full scale; Task 10 notebook):
  - `clean_sentences(lines: Iterable[str], min_words=5, max_words=30) -> Iterator[str]` — whitespace-tokenized, filtered.
  - `build_pairs(sentences, cfg=InjectorConfig(), pairs_per_sentence=1) -> Iterator[dict]` — yields `{"noisy": str, "correct": str}`.
  - `build_from_lines(lines, out_path, cfg, pairs_per_sentence=1, val_size=1000, seed=0) -> dict` — writes `<out_path>-train.jsonl` / `<out_path>-val.jsonl`, returns stats dict.
  - `load_wiki_sentences(lang="ms", limit=None) -> Iterator[str]` — Malay Wikipedia via `datasets` streaming (Colab only; not covered by offline tests).

JSONL record format (both files): `{"noisy": "...", "correct": "..."}` — whitespace-tokenized text.

- [ ] **Step 1: Create the fixture** (`tests/fixtures/malay_corpus.txt`)

```text
Bahasa Melayu ialah bahasa utama di Malaysia dan dituturkan oleh hampir seluruh penduduk negara ini.
Sekolah itu terletak berhampiran dengan pasar raya yang baharu dibuka tahun lepas.
Ali gemar membaca buku sejarah di perpustakaan pada hujung minggu.
Cuaca hari ini panas terik sejak pagi lagi dan dijangka hujan pada petang nanti.
Kerajaan telah mengumumkan projek pembinaan lebuh raya baharu di pantai timur.
Emak sedang memasak nasi lemak dan ayam goreng di dapur untuk sarapan pagi ini.
Pelajar itu tidak hadir ke sekolah kerana demam sejak dua hari lalu.
Kami bercadang untuk melancong ke Pulau Pinang pada cuti sekolah akan datang.
Sungai itu dahulu kotor tetapi kini sudah bersih selepas projek pemulihan dijalankan.
Pemain muda itu menunjukkan prestasi yang memberangsangkan dalam perlawanan akhir semalam.
Perpustakaan negara menyimpan ribuan naskhah manuskrip lama yang bernilai tinggi.
Kedai makan di tepi jalan itu terkenal dengan masakan asam pedas yang sedap.
```

- [ ] **Step 2: Write the failing test** (`tests/test_data_build.py`)

```python
import json
from pathlib import Path

from lfm_my.data_build import build_from_lines, build_pairs, clean_sentences
from lfm_my.errors import InjectorConfig

FIXTURE = Path(__file__).parent / "fixtures" / "malay_corpus.txt"


def _lines():
    return FIXTURE.read_text(encoding="utf-8").splitlines()


def test_clean_sentences_filters_and_tokenizes():
    out = list(clean_sentences([
        "Dia pergi.",
        "Pendek.",
        "Satu dua tiga empat lima enam tujuh lapan sembilan sepuluh sebelas "
        "dua belas tiga belas empat belas lima belas enam belas tujuh belas "
        "lapan belas sembilan belas dua puluh dua puluh satu",
    ]))
    assert out == ["Dia pergi ."]


def test_clean_sentences_range():
    out = list(clean_sentences(_lines(), min_words=5, max_words=30))
    assert len(out) >= 8
    for s in out:
        assert 5 <= len(s.split()) <= 30


def test_build_pairs_shape():
    sents = list(clean_sentences(_lines()))
    pairs = list(build_pairs(sents, InjectorConfig(seed=1), pairs_per_sentence=2))
    assert len(pairs) == 2 * len(sents)
    for p in pairs[:20]:
        assert set(p) == {"noisy", "correct"}
        assert p["correct"].split() and p["noisy"].split()
        assert p["correct"] in sents


def test_build_from_lines_writes_jsonl(tmp_path):
    stats = build_from_lines(_lines(), tmp_path / "ms", InjectorConfig(seed=2),
                             pairs_per_sentence=2, val_size=3, seed=2)
    train = [json.loads(l) for l in (tmp_path / "ms-train.jsonl").read_text().splitlines()]
    val = [json.loads(l) for l in (tmp_path / "ms-val.jsonl").read_text().splitlines()]
    assert len(val) == 3
    assert len(train) == stats["train_pairs"]
    assert stats["clean_kept"] > 0
    assert all(set(p) == {"noisy", "correct"} for p in train + val)


def test_noisy_differs_from_correct_sometimes():
    sents = list(clean_sentences(_lines()))
    pairs = list(build_pairs(sents, InjectorConfig(seed=3, p_clean=0.0), pairs_per_sentence=1))
    assert any(p["noisy"] != p["correct"] for p in pairs)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_data_build.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lfm_my.data_build'`

- [ ] **Step 4: Implement** (`src/lfm_my/data_build.py`)

```python
"""Synthetic Malay GEC data: clean corpus sentences + rule-based error injection.

Runs on Colab (wiki download via `datasets`) and offline (tests use the fixture corpus).
Output: JSONL {"noisy": ..., "correct": ...} in whitespace-tokenized form.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Iterable, Iterator

from lfm_my.errors import InjectorConfig, inject_errors
from lfm_my.text import tokenize

_SKIP = re.compile(r"https?://|\{\{|\[\[|^\s*[#*|>=]")


def clean_sentences(lines: Iterable[str], min_words: int = 5, max_words: int = 30) -> Iterator[str]:
    """Raw corpus lines -> whitespace-tokenized sentences within the word-length band."""
    for line in lines:
        line = line.strip()
        if not line or _SKIP.search(line):
            continue
        for sent in re.split(r"(?<=[.!?])\s+", line):
            sent = tokenize(sent)
            words = sent.split()
            if min_words <= len(words) <= max_words and any(c.isalpha() for c in sent):
                yield sent


def build_pairs(sentences, cfg: InjectorConfig = InjectorConfig(),
                pairs_per_sentence: int = 1) -> Iterator[dict]:
    """Each clean sentence -> pairs_per_sentence (noisy, correct) pairs."""
    rng = random.Random(cfg.seed)
    for sent in sentences:
        for _ in range(pairs_per_sentence):
            yield {"noisy": inject_errors(sent, rng, cfg), "correct": sent}


def build_from_lines(lines, out_path, cfg: InjectorConfig, pairs_per_sentence: int = 1,
                     val_size: int = 1000, seed: int = 0) -> dict:
    """Write <out_path>-train.jsonl / <out_path>-val.jsonl; return stats."""
    sents = list(clean_sentences(lines))
    rng = random.Random(seed)
    rng.shuffle(sents)
    val_sents, train_sents = sents[:val_size], sents[val_size:]
    stats = {"clean_sentences": len(sents), "val_sentences": len(val_sents),
             "clean_kept": 0, "train_pairs": 0}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for split, chunk in (("train", train_sents), ("val", val_sents)):
        n = 0
        with open(out_path.with_name(f"{out_path.name}-{split}.jsonl"), "w", encoding="utf-8") as f:
            for p in build_pairs(chunk, cfg, pairs_per_sentence):
                if p["noisy"] == p["correct"]:
                    stats["clean_kept"] += 1
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
                n += 1
        if split == "train":
            stats["train_pairs"] = n
    return stats


def load_wiki_sentences(lang: str = "ms", limit: int | None = None) -> Iterator[str]:
    """Malay Wikipedia article lines via HF `datasets` (Colab). Yields raw lines; filter
    with clean_sentences()."""
    from datasets import load_dataset
    ds = load_dataset("wikipedia", f"20230601.{lang}", split="train", streaming=True)
    n = 0
    for row in ds:
        for line in (row.get("text") or "").splitlines():
            yield line
            n += 1
            if limit and n >= limit:
                return
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_data_build.py -v`
Expected: PASS

- [ ] **Step 6: Smoke-run the builder on the fixture**

Run:
```bash
uv run python - <<'EOF'
from pathlib import Path
from lfm_my.data_build import build_from_lines
from lfm_my.errors import InjectorConfig
stats = build_from_lines(
    Path("tests/fixtures/malay_corpus.txt").read_text().splitlines(),
    Path("data/ms"), InjectorConfig(seed=0), pairs_per_sentence=2, val_size=2, seed=0)
print(stats)
EOF
head -3 data/ms-train.jsonl
```
Expected: stats dict with `train_pairs` ≥ 16 and `clean_kept` > 0; JSONL lines show plausible Malay corruptions. (`data/*.jsonl` is gitignored.)

- [ ] **Step 7: Commit**

```bash
git add src/lfm_my/data_build.py tests/test_data_build.py tests/fixtures/
git commit -m "feat: synthetic Malay pair builder (corpus filter + injection)"
```

---

### Task 7: Training-side tagger model

**Files:**
- Create: `src/lfm_my/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `lfm_my.modeling_gectagger._last_hidden` (reused); `lfm_my.ENCODER_NAME`.
- Produces:
  - `class GecTagger(nn.Module)` — constructor `GecTagger(encoder: nn.Module, vocab_size: int, hidden_size: int, dropout: float = 0.1)`. Submodules exactly `encoder`, `dropout`, `detect_head`, `base_head`, `replace_proj`, `append_proj`, `replace_bias`, `append_bias` (1:1 with `GecTaggerForGEC`'s tie_replace=True branch). Attribute `self.vocab_size`.
  - `forward(input_ids, attention_mask) -> {"label_logits": [B,T,2+2V], "detect_logits": [B,T,2]}`
  - `build_tagger(encoder_name=ENCODER_NAME) -> GecTagger` — loads the pretrained bidirectional encoder + tokenizer (needs network; used on Colab).

- [ ] **Step 1: Write the failing test** (`tests/test_model.py`)

```python
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
    assert {name for name, _ in m.named_children()} == expected


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lfm_my.model'`

- [ ] **Step 3: Implement** (`src/lfm_my/model.py`)

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_model.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/lfm_my/model.py tests/test_model.py
git commit -m "feat: training-side GecTagger mirroring Liquid inference class"
```

---

### Task 8: Dataset + collate (pieces → padded batches)

**Files:**
- Create: `src/lfm_my/dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `lfm_my.convert.convert_pair`, `lfm_my.modeling_gectagger.KEEP_ID`, `lfm_my.tags.append_id`.
- Produces (used by Task 10 notebook):
  - `class GecDataset(torch.utils.data.Dataset)` — `GecDataset(pairs: list[dict], tokenizer, vocab_size: int, max_len: int = 128)`; each pair is one `{"noisy", "correct"}` record; multi-pass conversions expand to multiple samples. Attributes: `self.samples`, `self.skipped` (count of non-converging pairs). `__getitem__` returns `{"input_ids": list[int], "label_targets": list[int], "detect_targets": list[int]}` (all include BOS at position 0).
  - `collate_fn(batch: list[dict], pad_id: int) -> dict` — returns `{"input_ids": [B,T] long, "attention_mask": [B,T] long, "label_targets": [B,T] long (-100 padded), "detect_targets": [B,T] long (-100 padded)}`.

Sample layout: `[bos] + pieces[:max_len-1]`; BOS label target = `append_id(bos_append, V)` when set else `KEEP_ID`; BOS detect target = 1 when set else 0.

- [ ] **Step 1: Write the failing test** (`tests/test_dataset.py`)

```python
import torch

from lfm_my.dataset import GecDataset, collate_fn
from lfm_my.tags import append_id, replace_id

V = 1000
BOS = 1
PAD = 0


class CharTokenizer:
    """Stub: one piece per char (id = 10 + ord), whitespace pre-tokenization."""
    pad_token_id, bos_token_id = PAD, BOS

    def encode(self, text, add_special_tokens=False):
        ids = []
        for w in text.split():
            ids.extend(10 + ord(c) for c in w)
        return ids

    def get_vocab(self):
        return {f"c{i}": i for i in range(V)}


PAIRS = [
    {"noisy": "saya makan nasik", "correct": "saya makan nasi"},
    {"noisy": "dia pergi ke pasar", "correct": "dia pergi ke pasar"},      # clean
    {"noisy": "saya makan", "correct": "saya sudah makan"},                # insert
]


def test_dataset_sizes_and_shapes():
    ds = GecDataset(PAIRS, CharTokenizer(), V, max_len=128)
    assert ds.skipped == 0
    assert len(ds) == 3                     # all single-pass
    s = ds[0]
    assert s["input_ids"][0] == BOS
    assert len(s["label_targets"]) == len(s["input_ids"])
    assert len(s["detect_targets"]) == len(s["input_ids"])


def test_clean_pair_all_keep():
    ds = GecDataset(PAIRS, CharTokenizer(), V)
    s = ds[1]
    assert all(t == 0 for t in s["label_targets"])
    assert all(t == 0 for t in s["detect_targets"])


def test_replace_target_ids():
    ds = GecDataset([{"noisy": "ab", "correct": "ac"}], CharTokenizer(), V)
    s = ds[0]
    # pieces: a=107 b=108 -> BOS kept, 'a' kept, 'b' replaced by 'c'(109)
    assert s["label_targets"][-1] == replace_id(10 + ord("c"))
    assert s["detect_targets"][-1] == 1
    assert s["detect_targets"][1] == 0          # 'a' kept


def test_bos_insert_target():
    ds = GecDataset([{"noisy": "ab", "correct": "z ab"}], CharTokenizer(), V)
    s = ds[0]
    assert s["label_targets"][0] == append_id(10 + ord("z"), V)
    assert s["detect_targets"][0] == 1


def test_multi_pass_expands_samples():
    pairs = [{"noisy": "ab", "correct": "ab xyzw"}]        # 4-piece insert after short anchor
    ds = GecDataset(pairs, CharTokenizer(), V)
    assert len(ds) >= 2


def test_collate_padding():
    ds = GecDataset(PAIRS, CharTokenizer(), V)
    batch = collate_fn([ds[0], ds[1]], PAD)
    assert batch["input_ids"].shape == batch["attention_mask"].shape \
        == batch["label_targets"].shape == batch["detect_targets"].shape
    lens = [len(ds[0]["input_ids"]), len(ds[1]["input_ids"])]
    row = 0 if lens[0] < lens[1] else 1
    shorter = min(lens)
    assert (batch["attention_mask"][row, shorter:] == 0).all()
    assert (batch["label_targets"][row, shorter:] == -100).all()
    assert (batch["input_ids"][row, shorter:] == PAD).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lfm_my.dataset'`

- [ ] **Step 3: Implement** (`src/lfm_my/dataset.py`)

```python
"""Torch dataset: JSONL pairs -> per-pass padded tagger batches.

Sample layout: [bos] + pieces[:max_len-1]. Targets align position-for-position with the
input; padding positions are -100 (ignored by CrossEntropyLoss).
"""
from __future__ import annotations

from typing import List

import torch
from torch.utils.data import Dataset

from lfm_my.convert import convert_pair
from lfm_my.modeling_gectagger import KEEP_ID
from lfm_my.tags import append_id


class GecDataset(Dataset):
    """pairs: [{"noisy": str, "correct": str}] (whitespace-tokenized). Multi-pass conversions
    expand into multiple training samples (pass k trains on pass k-1's output)."""

    def __init__(self, pairs: List[dict], tokenizer, vocab_size: int, max_len: int = 128):
        self.max_len = max_len
        self.skipped = 0
        self.samples = []
        bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1

        def enc_word(w):
            return tokenizer.encode(w, add_special_tokens=False)

        for rec in pairs:
            conv = convert_pair(rec["noisy"], rec["correct"], enc_word, vocab_size)
            if conv is None:
                self.skipped += 1
                continue
            if not conv.passes:                                  # clean pair -> all-$KEEP pass
                pieces = tokenizer.encode(rec["correct"], add_special_tokens=False)
                pieces = pieces[:max_len - 1]
                self.samples.append({
                    "input_ids": [bos] + pieces,
                    "label_targets": [KEEP_ID] * (len(pieces) + 1),
                    "detect_targets": [0] * (len(pieces) + 1),
                })
                continue
            for inp, p in zip(conv.inputs, conv.passes):
                pieces = inp[:max_len - 1]
                tags = p.tags[:max_len - 1]
                det = p.detect[:max_len - 1]
                bos_label = append_id(p.bos_append, vocab_size) if p.bos_append is not None else KEEP_ID
                bos_detect = 1 if p.bos_append is not None else 0
                self.samples.append({
                    "input_ids": [bos] + pieces,
                    "label_targets": [bos_label] + tags,
                    "detect_targets": [bos_detect] + det,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def collate_fn(batch: List[dict], pad_id: int) -> dict:
    T = max(len(s["input_ids"]) for s in batch)
    input_ids = torch.full((len(batch), T), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), T), dtype=torch.long)
    label_targets = torch.full((len(batch), T), -100, dtype=torch.long)
    detect_targets = torch.full((len(batch), T), -100, dtype=torch.long)
    for b, s in enumerate(batch):
        n = len(s["input_ids"])
        input_ids[b, :n] = torch.tensor(s["input_ids"], dtype=torch.long)
        attention_mask[b, :n] = 1
        label_targets[b, :n] = torch.tensor(s["label_targets"], dtype=torch.long)
        detect_targets[b, :n] = torch.tensor(s["detect_targets"], dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask,
            "label_targets": label_targets, "detect_targets": detect_targets}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_dataset.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/lfm_my/dataset.py tests/test_dataset.py
git commit -m "feat: GEC dataset + collate (BOS anchor, multi-pass samples, -100 padding)"
```

---

### Task 9: Loss, metrics, checkpointing

**Files:**
- Create: `src/lfm_my/train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `lfm_my.modeling_gectagger.KEEP_ID`.
- Produces (used by Task 10 notebook):
  - `compute_loss(out: dict, label_targets, detect_targets, aux_loss_weight: float = 0.5) -> (loss, label_loss, detect_loss)` — fp32 CE, ignore_index=-100.
  - `metrics_for(out, label_targets, detect_targets) -> dict` — `{"label_acc_nokeep": float, "detect_acc": float, "n_edits": int}` (accuracy over non-padding positions; label acc restricted to positions where target ≠ KEEP).
  - `save_checkpoint(path, model, optimizer, step: int, best_val: float) -> None`
  - `load_checkpoint(path, model, optimizer=None) -> {"step": int, "best_val": float}`

- [ ] **Step 1: Write the failing test** (`tests/test_train.py`)

```python
import torch

from lfm_my.model import GecTagger
from lfm_my.train import compute_loss, load_checkpoint, metrics_for, save_checkpoint
from tests.test_model import TinyEncoder


def _tiny():
    torch.manual_seed(0)
    return GecTagger(TinyEncoder(64, 16), vocab_size=64, hidden_size=16)


def test_compute_loss_finite_and_decreases():
    m = _tiny()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    ids = torch.randint(0, 64, (2, 6))
    label_t = torch.randint(0, 2, (2, 6))
    det_t = torch.randint(0, 2, (2, 6))
    first = None
    for _ in range(30):
        out = m(ids, torch.ones_like(ids))
        loss, ll, dl = compute_loss(out, label_t, det_t)
        if first is None:
            first = float(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert torch.isfinite(loss).all()
    assert float(loss) < first


def test_loss_respects_ignore_index():
    m = _tiny()
    ids = torch.randint(0, 64, (2, 6))
    out = m(ids, torch.ones_like(ids))
    label_t = torch.randint(0, 2, (2, 6))
    det_t = torch.randint(0, 2, (2, 6))
    loss_full, _, _ = compute_loss(out, label_t, det_t)
    partial = label_t.clone()
    partial[:, 3:] = -100
    det_partial = det_t.clone()
    det_partial[:, 3:] = -100
    loss_part, _, _ = compute_loss(out, partial, det_partial)
    assert loss_part != loss_full


def test_metrics_keys_and_ranges():
    m = _tiny()
    ids = torch.randint(0, 64, (2, 6))
    out = m(ids, torch.ones_like(ids))
    label_t = torch.full((2, 6), 0)
    label_t[0, 0] = 3
    det_t = torch.zeros((2, 6), dtype=torch.long)
    met = metrics_for(out, label_t, det_t)
    assert set(met) == {"label_acc_nokeep", "detect_acc", "n_edits"}
    assert met["n_edits"] == 1
    assert 0.0 <= met["label_acc_nokeep"] <= 1.0
    assert 0.0 <= met["detect_acc"] <= 1.0


def test_checkpoint_roundtrip(tmp_path):
    m = _tiny()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    out = m(torch.tensor([[1, 2, 3]]), torch.ones(1, 3, dtype=torch.long))
    loss, _, _ = compute_loss(out, torch.tensor([[0, 1, 2]]), torch.tensor([[0, 1, 0]]))
    loss.backward()
    opt.step()
    p = tmp_path / "ckpt.pt"
    save_checkpoint(p, m, opt, step=42, best_val=1.23)
    m2 = _tiny()
    meta = load_checkpoint(p, m2)
    assert meta["step"] == 42 and meta["best_val"] == 1.23
    for (k1, v1), (k2, v2) in zip(m.state_dict().items(), m2.state_dict().items()):
        assert k1 == k2 and torch.equal(v1, v2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lfm_my.train'`

- [ ] **Step 3: Implement** (`src/lfm_my/train.py`)

```python
"""Loss, metrics, checkpointing for the GEC tagger (training loop lives in the notebook)."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from lfm_my.modeling_gectagger import KEEP_ID


def compute_loss(out: dict, label_targets: torch.Tensor, detect_targets: torch.Tensor,
                 aux_loss_weight: float = 0.5):
    """label_loss + aux_loss_weight * detect_loss, fp32 CE, ignore_index=-100."""
    label_logits = out["label_logits"].float()
    detect_logits = out["detect_logits"].float()
    label_loss = F.cross_entropy(label_logits.reshape(-1, label_logits.size(-1)),
                                 label_targets.reshape(-1), ignore_index=-100)
    detect_loss = F.cross_entropy(detect_logits.reshape(-1, 2), detect_targets.reshape(-1),
                                  ignore_index=-100)
    return label_loss + aux_loss_weight * detect_loss, label_loss, detect_loss


@torch.no_grad()
def metrics_for(out: dict, label_targets: torch.Tensor, detect_targets: torch.Tensor) -> dict:
    """label accuracy over non-KEEP targets (excl. padding), detect accuracy over all non-padding."""
    label_pred = out["label_logits"].argmax(-1)
    detect_pred = out["detect_logits"].argmax(-1)
    valid = label_targets != -100
    edit = valid & (label_targets != KEEP_ID)
    n_edits = int(edit.sum())
    label_acc = float((label_pred[edit] == label_targets[edit]).float().mean()) if n_edits else 1.0
    n_valid = int(valid.sum())
    detect_acc = float((detect_pred[valid] == detect_targets[valid]).float().mean()) if n_valid else 1.0
    return {"label_acc_nokeep": label_acc, "detect_acc": detect_acc, "n_edits": n_edits}


def save_checkpoint(path, model, optimizer, step: int, best_val: float) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step, "best_val": best_val}, path)


def load_checkpoint(path, model, optimizer=None) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return {"step": ckpt.get("step", 0), "best_val": ckpt.get("best_val", float("inf"))}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_train.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/lfm_my/train.py tests/test_train.py
git commit -m "feat: loss, metrics, checkpoint save/load"
```

---

### Task 10: Export (safetensors + HF model repo layout)

**Files:**
- Create: `src/lfm_my/export.py`
- Test: `tests/test_export.py`

**Interfaces:**
- Consumes: `lfm_my.model.GecTagger` (attribute `vocab_size`, submodule `detect_head.in_features`), `src/lfm_my/modeling_gectagger.py` (shipped as the repo's remote code).
- Produces:
  - `export_model_dir(tagger, tokenizer, out_dir, encoder_name=ENCODER_NAME) -> Path` — writes a loadable HF model dir: `model.safetensors` (full `GecTagger` state dict), `config.json` (Liquid-compatible layout + `architectures` + `auto_map`), `modeling_gectagger.py` (verbatim copy of the vendored file), tokenizer files via `tokenizer.save_pretrained` (best-effort; stub tokenizers in tests are skipped with a warning).
  - `push_model_repo(local_dir, repo_id, private=True) -> str` — `HfApi().create_repo(..., exist_ok=True)` + `upload_folder` (needs `HF_TOKEN`; manual step).

The exported `config.json` mirrors Liquid's field-for-field (`num_tags` computed as `2 + 2·tagger.vocab_size`, `torch_dtype: "float32"`); must NOT contain `num_labels` (reserved `PretrainedConfig` property that breaks loading).

- [ ] **Step 1: Write the failing test** (`tests/test_export.py`)

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_export.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lfm_my.export'`

- [ ] **Step 3: Implement** (`src/lfm_my/export.py`)

```python
"""Package a trained GecTagger as a HF model repo dir compatible with Liquid's inference class."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from lfm_my import ENCODER_NAME

_VENDORED_MODELING = Path(__file__).parent / "modeling_gectagger.py"


def export_model_dir(tagger, tokenizer, out_dir, encoder_name: str = ENCODER_NAME) -> Path:
    """Write model.safetensors + config.json + modeling_gectagger.py + tokenizer files."""
    from safetensors.torch import save_file

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sd = {k: v.contiguous() for k, v in tagger.state_dict().items()}
    save_file(sd, out_dir / "model.safetensors")

    V = tagger.vocab_size
    cfg = {
        "architectures": ["GecTaggerForGEC"],
        "model_type": "gec_tagger",
        "auto_map": {
            "AutoConfig": "modeling_gectagger.GecTaggerConfig",
            "AutoModel": "modeling_gectagger.GecTaggerForGEC",
        },
        "encoder_name": encoder_name,
        "num_tags": 2 + 2 * V,
        "hidden_size": tagger.detect_head.in_features,
        "tie_replace": True,
        "multi_head": False,
        "aux_loss_weight": 0.5,
        "use_swap": False,
        "qat_applied": False,
        "qat_group_size": 32,
        "dropout": 0.1,
        "torch_dtype": "float32",
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))

    shutil.copyfile(_VENDORED_MODELING, out_dir / "modeling_gectagger.py")

    try:
        tokenizer.save_pretrained(out_dir)
    except Exception as e:                                  # stub tokenizers in tests
        print(f"[export] tokenizer.save_pretrained skipped: {e}")
    return out_dir


def push_model_repo(local_dir, repo_id: str, private: bool = True) -> str:
    """Create the HF model repo and upload the export dir (requires HF_TOKEN in env)."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(local_dir), repo_id=repo_id, repo_type="model")
    return f"https://huggingface.co/{repo_id}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_export.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/lfm_my/export.py tests/test_export.py
git commit -m "feat: export tagger to HF model repo layout (safetensors + remote code)"
```

---

### Task 11: Colab training notebook

**Files:**
- Create: `notebooks/train_malay_tagger.ipynb`

**Interfaces:**
- Consumes: everything from Tasks 1–10. No unit tests of its own; correctness = the Task 1–10 test suite plus the notebook's code cells parsing and running on Colab.

The notebook is the user-facing training driver (Google Colab free T4). All heavy code lives in `lfm_my`; notebook cells stay thin. Write it as valid `.ipynb` JSON (`nbformat` 4, `kernel: python3`).

- [ ] **Step 1: Write the notebook cells**

Cell sequence (each item = one notebook cell; code cells shown exactly):

1. **MD — title:** `# Bahasa Malaysia spellchecker — GECToR fine-tune of LFM2.5-Encoder-350M` and one paragraph: drives the `lfm_my` package end-to-end (data → train → probe → export) on a free Colab T4; checkpoints persist on Google Drive.

2. **Code — setup:**

```python
from google.colab import drive
drive.mount('/content/drive')

import os
if not os.path.isdir('/content/lfm-my'):
    !git clone https://github.com/GITHUB_USER/lfm-my.git /content/lfm-my
else:
    !(cd /content/lfm-my && git pull)
!pip install -q -e /content/lfm-my
```

(`GITHUB_USER` is the only user-specific value; replace before first run.)

3. **Code — config block (all knobs in one place):**

```python
import json, math, random, time
from pathlib import Path
import torch

ENCODER = "LiquidAI/LFM2.5-Encoder-350M"
DATA_DIR = Path("/content/drive/MyDrive/lfm-my/data")     # persists across sessions
CKPT_DIR = Path("/content/drive/MyDrive/lfm-my/ckpts")
EXPORT_DIR = Path("/content/drive/MyDrive/lfm-my/export")
MAX_LEN, BATCH, ACCUM = 128, 8, 4
LR, WARMUP, WD = 2e-5, 0.1, 0.1
EPOCHS, PATIENCE = 7, 2
CKPT_EVERY = 500                                            # optimizer steps
SEED = 0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"
print(f"device={DEVICE} amp={USE_AMP}")
torch.manual_seed(SEED); random.seed(SEED)
```

4. **MD — data section:** "Generate ~15–20K train + ~1K val pairs from Malay Wikipedia (first run only; cached on Drive)."

5. **Code — data build (skips if files exist):**

```python
from lfm_my.data_build import build_from_lines, load_wiki_sentences
from lfm_my.errors import InjectorConfig

if not (DATA_DIR / "ms-train.jsonl").exists():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    lines = list(load_wiki_sentences("ms"))
    print(f"wiki lines: {len(lines)}")
    stats = build_from_lines(lines, DATA_DIR / "ms",
                             InjectorConfig(seed=SEED), pairs_per_sentence=1,
                             val_size=1000, seed=SEED)
    print(stats)
else:
    print("data already on Drive, skipping build")
```

6. **Code — inspect samples:**

```python
import itertools
for line in itertools.islice(open(DATA_DIR / "ms-train.jsonl"), 5):
    p = json.loads(line)
    print("NOISY   :", p["noisy"])
    print("CORRECT :", p["correct"], "\n")
```

7. **Code — load tokenizer + model:**

```python
from transformers import AutoTokenizer
from lfm_my.model import build_tagger

tok = AutoTokenizer.from_pretrained(ENCODER, trust_remote_code=True)
V = len(tok.get_vocab())
print("vocab:", V)

model = build_tagger(ENCODER).to(DEVICE)
print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
```

8. **Code — datasets + loaders:**

```python
from functools import partial
from torch.utils.data import DataLoader
from lfm_my.dataset import GecDataset, collate_fn

def load_pairs(name):
    return [json.loads(l) for l in open(DATA_DIR / f"ms-{name}.jsonl")]

train_ds = GecDataset(load_pairs("train"), tok, V, max_len=MAX_LEN)
val_ds = GecDataset(load_pairs("val"), tok, V, max_len=MAX_LEN)
print(f"train samples: {len(train_ds)} (skipped {train_ds.skipped}), val: {len(val_ds)}")

pad = tok.pad_token_id if tok.pad_token_id is not None else 0
train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                      collate_fn=partial(collate_fn, pad_id=pad))
val_dl = DataLoader(val_ds, batch_size=BATCH, collate_fn=partial(collate_fn, pad_id=pad))
```

9. **Code — optimizer + scheduler:**

```python
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from lfm_my.train import save_checkpoint, load_checkpoint

opt = AdamW(model.parameters(), lr=LR, weight_decay=WD)
total_steps = max(1, (len(train_dl) // ACCUM) * EPOCHS)
warm = int(total_steps * WARMUP)

def lr_at(step):
    if step < warm:
        return max(1e-2, step / max(1, warm))
    prog = (step - warm) / max(1, total_steps - warm)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

sched = LambdaLR(opt, lr_at)
scaler = torch.amp.GradScaler(DEVICE, enabled=USE_AMP)
print(f"total_steps={total_steps} warmup={warm}")
```

10. **Code — run_epoch (train/eval, checkpoint every CKPT_EVERY optimizer steps):**

```python
from lfm_my.train import compute_loss, metrics_for

STATE = {"step": 0, "best_val": float("inf")}

def run_epoch(dl, train: bool):
    model.train(train)
    tot = {"loss": 0.0, "lab_acc": 0.0, "det_acc": 0.0, "n": 0}
    opt.zero_grad()
    for i, batch in enumerate(dl):
        b = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=USE_AMP):
            out = model(b["input_ids"], b["attention_mask"])
            loss, ll, dl = compute_loss(out, b["label_targets"], b["detect_targets"])
        if train:
            scaler.scale(loss / ACCUM).backward()
            if (i + 1) % ACCUM == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad()
                STATE["step"] += 1
                if STATE["step"] % CKPT_EVERY == 0:        # Colab-timeout safety
                    save_checkpoint(CKPT_DIR / "ckpt.pt", model, opt,
                                    STATE["step"], STATE["best_val"])
                    print(f"  [ckpt] step {STATE['step']} saved", flush=True)
        m = metrics_for(out, b["label_targets"], b["detect_targets"])
        bs = b["input_ids"].size(0)
        tot["loss"] += float(loss) * bs; tot["lab_acc"] += m["label_acc_nokeep"] * bs
        tot["det_acc"] += m["detect_acc"] * bs; tot["n"] += bs
        if train and (i + 1) % 50 == 0:
            print(f"  step {i+1}/{len(dl)} loss={tot['loss']/tot['n']:.4f} "
                  f"lab_acc={tot['lab_acc']/tot['n']:.3f}", flush=True)
    n = max(1, tot["n"])
    return {"loss": tot["loss"] / n, "label_acc_nokeep": tot["lab_acc"] / n,
            "detect_acc": tot["det_acc"] / n}
```

11. **Code — main loop (resume-safe, early stopping):**

```python
CKPT_DIR.mkdir(parents=True, exist_ok=True)
if (CKPT_DIR / "ckpt.pt").exists():
    meta = load_checkpoint(CKPT_DIR / "ckpt.pt", model, opt)
    STATE.update(step=meta["step"], best_val=meta["best_val"])
    print(f"resumed from step {STATE['step']}, best_val={STATE['best_val']:.4f}")

bad = 0
for epoch in range(EPOCHS):
    t0 = time.time()
    tr = run_epoch(train_dl, train=True)
    va = run_epoch(val_dl, train=False)
    print(f"epoch {epoch+1}: train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f} "
          f"val_lab_acc={va['label_acc_nokeep']:.3f} val_det_acc={va['detect_acc']:.3f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    if va["loss"] < STATE["best_val"] - 1e-4:
        STATE["best_val"], bad = va["loss"], 0
        save_checkpoint(CKPT_DIR / "best.pt", model, opt, STATE["step"], STATE["best_val"])
        print("  new best — saved best.pt")
    else:
        bad += 1
        if bad >= PATIENCE:
            print("early stopping"); break
```

12. **Code — probe sentences (post-training validation, spec §3):**

```python
from lfm_my.text import tokenize

load_checkpoint(CKPT_DIR / "best.pt", model)
model.eval()
PROBES = [
    "saya sudah makan nasik di kedai itu semalam",
    "dia tidak arah pinjam buku",
    "buku itu di baca oleh ali",
    "dia pergi ke sekolah setiap hari",
]
bos = tok.bos_token_id if tok.bos_token_id is not None else 1
with torch.no_grad():
    for p in PROBES:
        src = tokenize(p)
        ids = torch.tensor([[bos] + tok.encode(src, add_special_tokens=False)[:MAX_LEN - 1]]).to(DEVICE)
        out = model(ids, torch.ones_like(ids))
        preds = out["label_logits"].argmax(-1)[0]
        err = out["detect_logits"].softmax(-1)[0][:, 1]
        edits = [int(x) for x in (preds != 0).nonzero().flatten()]
        print("IN   :", p)
        print("tags :", [(int(i), int(preds[i]), round(float(err[i]), 3)) for i in edits] or "KEEP all")
        print()
```

13. **Code — export:**

```python
from lfm_my.export import export_model_dir
export_model_dir(model, tok, EXPORT_DIR, encoder_name=ENCODER)
print(sorted(p.name for p in EXPORT_DIR.iterdir()))
```

14. **MD — next steps:** "Download `EXPORT_DIR` (or `from lfm_my.export import push_model_repo; push_model_repo(EXPORT_DIR, 'USER/lfm-malay-spellchecker')`), then verify the app locally (Task 12) and publish the Space (Task 13)."

- [ ] **Step 2: Verify notebook JSON is valid**

Run: `uv run python -c "import json,pathlib; nb=json.loads(pathlib.Path('notebooks/train_malay_tagger.ipynb').read_text()); print(len(nb['cells']), 'cells'); assert all(c['cell_type'] in ('code','markdown') for c in nb['cells'])"`
Expected: prints cell count, no assertion error.

- [ ] **Step 3: Syntax-check every code cell**

Run: `uv run python -c "import ast,json,pathlib; [ast.parse(''.join(c['source'])) for c in json.loads(pathlib.Path('notebooks/train_malay_tagger.ipynb').read_text())['cells'] if c['cell_type']=='code']; print('all code cells parse')"`
Expected: `all code cells parse`

- [ ] **Step 4: Commit**

```bash
git add notebooks/
git commit -m "feat: Colab training notebook (data build -> train -> probe -> export)"
```

---

### Task 12: Demo app (FastAPI + static UI, Malay-adapted)

**Files:**
- Create: `app/server.py`, `app/static/index.html`, `app/static/style.css`, `app/requirements.txt`, `app/Dockerfile`
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `lfm_my.text.{tokenize, detok, diff_segments}` (re-implemented inline in `server.py` to keep the Space self-contained — same code as `src/lfm_my/text.py`); model object exposing `correct(texts, min_error_prob=..., max_iter=...) -> list[str]`, `.parameters()`, `.buffers()`, `.float()`, `.eval()` (the exported model's API).
- Produces: same HTTP contract as Liquid's Space — `GET /api/health`, `POST /api/correct {text, min_error_prob, max_iter} -> {corrected, segments, changed}`, `GET /` serves `static/index.html`.

Malay adaptations vs Liquid's `server.py`: drop all English contraction regexes (`_NT`, `_CONTR`, `_REJOIN_*`); Malay probe sentence; model id from `SPELLCHECKER_MODEL` env; keep `_effective_cpus` + `torch.set_num_threads` logic.

- [ ] **Step 1: Write the failing test** (`tests/test_server.py`)

```python
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def client():
    # stub transformers.AutoModel before importing server (no model download in tests)
    fake_tf = types.ModuleType("transformers_fake_stub")

    class FakeModel:
        def correct(self, texts, min_error_prob=0.0, max_iter=3, **kw):
            return [t.replace("nasik", "nasi") for t in texts]

        def parameters(self):
            return iter([__import__("torch").zeros(8)])

        def buffers(self):
            return iter([])

        def float(self):
            return self

        def eval(self):
            return self

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(*a, **k):
            return FakeModel()

    fake_tf.AutoModel = FakeAutoModel
    sys.modules["transformers"] = fake_tf

    sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
    sys.modules.pop("server", None)
    import server                                    # noqa: E402
    from fastapi.testclient import TestClient
    return TestClient(server.app)


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "self_test" in body


def test_correct_diff(client):
    r = client.post("/api/correct", json={"text": "Saya makan nasik."})
    assert r.status_code == 200
    body = r.json()
    assert body["corrected"] == "Saya makan nasi."
    assert body["changed"] is True
    kinds = {(s["text"], s["kind"]) for s in body["segments"]}
    assert ("nasik", "del") in kinds and ("nasi", "edit") in kinds


def test_correct_empty(client):
    assert client.post("/api/correct", json={"text": "  "}).json() == \
        {"corrected": "", "segments": [], "changed": False}


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Bahasa" in r.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_server.py -v`
Expected: FAIL (no `app/server.py`)

- [ ] **Step 3: Implement `app/server.py`**

Adapted from Liquid's (same structure; contraction handling removed; Malay probe):

```python
"""FastAPI backend for the Bahasa Malaysia spellchecker demo (Docker Space).

    uvicorn server:app --host 0.0.0.0 --port 7860
"""
import difflib
import os
import re

import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoModel


def _effective_cpus():
    """Cores actually granted to this container (cgroup quota), not the host core count."""
    try:                                                          # cgroup v2
        raw = open("/sys/fs/cgroup/cpu.max").read().split()
        if raw and raw[0] != "max":
            return max(1, round(int(raw[0]) / int(raw[1])))
    except (OSError, ValueError):
        pass
    try:                                                          # cgroup v1
        q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
        p = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        if q > 0:
            return max(1, q // p)
    except (OSError, ValueError):
        pass
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return os.cpu_count() or 1


_CPUS = _effective_cpus()
torch.set_num_threads(_CPUS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

MODEL_ID = os.environ.get("SPELLCHECKER_MODEL", "USER/lfm-malay-spellchecker")
MODEL_REV = os.environ.get("SPELLCHECKER_REVISION", "main")
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

print(f"[server] loading {MODEL_ID}@{MODEL_REV} on {_CPUS} CPU thread(s) ...", flush=True)
_model = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REV, trust_remote_code=True,
                                   token=os.environ.get("HF_TOKEN")).float().eval()
_mem_bytes = sum(t.numel() * t.element_size() for t in (*_model.parameters(), *_model.buffers()))
_mem_human = (f"{_mem_bytes / 1024**3:.2f} GB" if _mem_bytes >= 1024**3
              else f"{_mem_bytes / 1024**2:.0f} MB")
print(f"[server] model ready ({_mem_human} in memory)", flush=True)


# Malay training form: punctuation spaced apart. No contraction splitting (English-only).
_PUNCT = re.compile(r'([.,!?;:()\[\]{}"«»…])')
_ATTACH_LEFT = re.compile(r"\s+([.,!?;:%)\]}»…])")
_ATTACH_RIGHT = re.compile(r"([(\[{«])\s+")


def tokenize(text: str) -> str:
    text = _PUNCT.sub(r" \1 ", text)
    return re.sub(r"\s+", " ", text).strip()


def detok(text: str) -> str:
    text = _ATTACH_LEFT.sub(r"\1", text)
    text = _ATTACH_RIGHT.sub(r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


_PROBE_IN = "Dia pergi ke sekolah setiap hari."
try:
    _PROBE_OUT = detok(_model.correct([tokenize(_PROBE_IN)], max_iter=3)[0])
    _PROBE_OK = (_PROBE_OUT == _PROBE_IN)          # clean Malay sentence should stay untouched
except Exception as e:                                           # pragma: no cover
    _PROBE_OUT, _PROBE_OK = f"ERROR: {e}", None
print(f"[server] self-test ok={_PROBE_OK}: {_PROBE_IN!r} -> {_PROBE_OUT!r}", flush=True)


def diff_segments(source: str, corrected: str):
    """Word-level diff -> [{text, kind}] segments, kind in {keep, edit, del}."""
    s, c = source.split(), corrected.split()
    seg = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, s, c, autojunk=False).get_opcodes():
        if op == "equal":
            seg.append({"text": " ".join(c[j1:j2]), "kind": "keep"})
        elif op == "insert":
            seg.append({"text": " ".join(c[j1:j2]), "kind": "edit"})
        elif op == "delete":
            seg.append({"text": " ".join(s[i1:i2]), "kind": "del"})
        elif op == "replace":
            seg.append({"text": " ".join(s[i1:i2]), "kind": "del"})
            seg.append({"text": " ".join(c[j1:j2]), "kind": "edit"})
    return seg or [{"text": corrected, "kind": "keep"}]


class CorrectRequest(BaseModel):
    text: str
    min_error_prob: float = 0.0
    max_iter: int = 3


app = FastAPI(title="Bahasa Malaysia spellchecker")


@app.get("/api/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "revision": MODEL_REV,
            "mem_bytes": _mem_bytes, "mem_human": _mem_human,
            "self_test": {"in": _PROBE_IN, "out": _PROBE_OUT, "ok": _PROBE_OK}}


@app.post("/api/correct")
@torch.no_grad()
def correct(req: CorrectRequest):
    text = (req.text or "").strip()
    if not text:
        return {"corrected": "", "segments": [], "changed": False}
    src = tokenize(text)
    out = _model.correct([src], min_error_prob=float(req.min_error_prob),
                         max_iter=int(req.max_iter))[0]
    return {"corrected": detok(out), "segments": diff_segments(src, out), "changed": out != src}


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


app.mount("/", StaticFiles(directory=STATIC), name="static")
```

- [ ] **Step 4: Implement `app/static/index.html`**

Original minimal UI, same API contract. Functional requirements: textarea input; debounced (650 ms) POST to `/api/correct`; renders `segments` (`edit` → `<mark>`, `del` → `<s class="deleted">`, `keep` → plain text; reattach `.,!?;:` left when spacing output); sliders for `min_error_prob` (0–1, step 0.05) and `max_iter` (1–5); Malay example buttons; status line fed by `/api/health`; copy button. No external assets/fonts. Reference implementation (keep under ~200 lines):

```html
<!doctype html>
<html lang="ms">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bahasa Malaysia spellchecker</title>
<link rel="stylesheet" href="style.css">
</head>
<body>
<div class="wrap">
  <header>
    <h1>Bahasa Malaysia <em>spellchecker</em></h1>
    <p class="sub">Betulkan ejaan dan tatabahasa dengan LFM2.5. <span id="status">Memuatkan model…</span></p>
  </header>

  <main class="grid">
    <section class="card">
      <h2>Input</h2>
      <textarea id="input" spellcheck="false" placeholder="Tulis atau tampal teks di sini…"></textarea>
      <div class="meta"><span id="count">0 aksara</span><span>semakan selepas 650 ms</span></div>
      <details>
        <summary>Tetapan lanjut</summary>
        <label>Ambang keyakinan <output id="mepv">0.00</output>
          <input type="range" id="mep" min="0" max="1" step="0.05" value="0"></label>
        <label>Pass pembetulan <output id="mitv">3</output>
          <input type="range" id="mit" min="1" max="5" step="1" value="3"></label>
      </details>
    </section>

    <section class="card">
      <h2>Dibetulkan <button id="copy" disabled>Salin</button></h2>
      <div id="output" class="output" aria-live="polite"></div>
      <div class="legend"><span><i class="sw edit"></i>Diubah</span>
        <span><i class="sw del"></i>Dibuang</span></div>
      <div class="stats"><span id="activity">Sedia</span>
        <span>Edit: <strong id="stat-edits">—</strong></span>
        <span>Latensi: <strong id="stat-latency">—</strong></span></div>
    </section>
  </main>

  <section class="examples">
    <span>Contoh</span><div id="examples"></div>
  </section>
</div>

<script>
const EXAMPLES = [
  { label:"Ejaan", text:"Saya sudah makan nasik di kedai itu." },
  { label:"Imbuhan", text:"Dia men tulis surat kepada kawannya." },
  { label:"Huruf kecil", text:"ali pergi ke sekolah setiap hari." },
  { label:"Teks bersih", text:"Dia pergi ke sekolah setiap hari." },
];
const $ = id => document.getElementById(id);
const ATTACH_LEFT = /^[.,!?;:%)\]}»…]+$/;
const OPEN = /^[(\[{«]+$/;
const input = $("input"), output = $("output");
let timer = null, seq = 0, lastCorrected = "";

function updateCount(){ $("count").textContent = `${input.value.length} aksara`; }
function schedule(d = 650){ clearTimeout(timer); timer = setTimeout(correct, d); }

function render(segments){
  output.innerHTML = "";
  let prev = null, edits = 0;
  for (const s of segments){
    for (const tok of s.text.split(/\s+/).filter(Boolean)){
      if (prev !== null && !ATTACH_LEFT.test(tok) && !OPEN.test(prev))
        output.append(document.createTextNode(" "));
      let node;
      if (s.kind === "edit"){ node = document.createElement("mark"); edits++; }
      else if (s.kind === "del"){ node = document.createElement("s"); node.className = "deleted"; edits++; }
      else node = document.createTextNode(tok);
      if (node.nodeType === 1) node.textContent = tok;
      output.append(node);
      prev = tok;
    }
  }
  return edits;
}

async function correct(){
  const text = input.value.replace(/\s+/g, " ").trim();
  if (!text){ output.innerHTML = ""; $("activity").textContent = "Sedia"; return; }
  const my = ++seq, t0 = performance.now();
  $("activity").textContent = "Menyemak…";
  try {
    const r = await fetch("/api/correct", { method:"POST",
      headers:{ "Content-Type":"application/json" },
      body: JSON.stringify({ text, min_error_prob:+$("mep").value, max_iter:+$("mit").value }) });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    if (my !== seq) return;
    const edits = render(data.segments);
    lastCorrected = data.corrected;
    $("copy").disabled = !lastCorrected;
    $("activity").textContent = data.changed ? "Dibetulkan" : "Nampak betul";
    $("stat-edits").textContent = String(edits);
    $("stat-latency").textContent = `${Math.round(performance.now() - t0)} ms`;
  } catch (e){ if (my === seq) $("activity").textContent = "Ralat"; console.error(e); }
}

input.addEventListener("input", () => { updateCount(); schedule(); });
for (const r of [$("mep"), $("mit")]) r.addEventListener("input", e => {
  $(e.target.id + "v").textContent = e.target.id === "mep"
    ? (+e.target.value).toFixed(2) : e.target.value;
  schedule(0);
});
for (const ex of EXAMPLES){
  const b = document.createElement("button");
  b.textContent = ex.label;
  b.onclick = () => { input.value = ex.text; updateCount(); schedule(0); };
  $("examples").append(b);
}
$("copy").onclick = async () => {
  await navigator.clipboard.writeText(lastCorrected);
  $("copy").textContent = "Disalin";
  setTimeout(() => $("copy").textContent = "Salin", 1200);
};
fetch("/api/health").then(r => r.json())
  .then(d => { $("status").textContent = `Model sedia (${d.mem_human ?? ""})`; })
  .catch(() => { $("status").textContent = "Model tidak tersedia"; });
updateCount();
</script>
</body>
</html>
```

- [ ] **Step 5: Implement `app/static/style.css`** (original, minimal — functional layout only):

```css
:root { color-scheme: light; --bg:#faf9f7; --card:#fff; --ink:#1c1917; --mut:#78716c;
        --line:#e7e5e4; --green:#dcfce7; --green-ink:#166534; --red:#fee2e2; --red-ink:#991b1b; }
* { box-sizing: border-box; }
body { margin:0; font: 16px/1.6 -apple-system, "Segoe UI", sans-serif; background:var(--bg); color:var(--ink); }
.wrap { max-width: 980px; margin: 0 auto; padding: 32px 20px 64px; }
h1 { font-size: 30px; margin: 0 0 4px; } h1 em { font-style: italic; }
.sub { color: var(--mut); margin: 0 0 24px; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
@media (max-width: 760px) { .grid { grid-template-columns: 1fr; } }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 16px; }
.card h2 { font-size: 14px; text-transform: uppercase; letter-spacing: .06em; color: var(--mut);
           margin: 0 0 10px; display: flex; justify-content: space-between; align-items: center; }
textarea { width: 100%; min-height: 220px; border: 1px solid var(--line); border-radius: 8px;
           padding: 12px; font: inherit; resize: vertical; }
.meta { display: flex; justify-content: space-between; color: var(--mut); font-size: 13px; margin-top: 6px; }
details { margin-top: 10px; font-size: 14px; color: var(--mut); }
details label { display: block; margin: 8px 0; }
details input[type=range] { width: 100%; }
.output { min-height: 220px; border: 1px solid var(--line); border-radius: 8px; padding: 12px; }
mark { background: var(--green); color: var(--green-ink); border-radius: 3px; padding: 0 2px; text-decoration: none; }
s.deleted { color: var(--red-ink); background: var(--red); border-radius: 3px; padding: 0 2px; }
.legend { display: flex; gap: 16px; color: var(--mut); font-size: 13px; margin-top: 8px; }
.sw { display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 4px; }
.sw.edit { background: var(--green); } .sw.del { background: var(--red); }
.stats { display: flex; gap: 16px; color: var(--mut); font-size: 13px; margin-top: 8px; }
.examples { margin-top: 24px; color: var(--mut); font-size: 14px; }
.examples div { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }
.examples button, #copy { border: 1px solid var(--line); background: var(--card); border-radius: 999px;
           padding: 6px 14px; font: inherit; font-size: 13px; cursor: pointer; }
#copy { border-radius: 6px; padding: 4px 12px; }
```

- [ ] **Step 6: Implement `app/requirements.txt`**

```text
transformers==5.1.0
safetensors==0.7.0
fastapi==0.115.6
uvicorn[standard]==0.34.0
huggingface_hub>=0.26
```

- [ ] **Step 7: Implement `app/Dockerfile`**

```dockerfile
# Docker Space: FastAPI backend + static frontend (no Gradio). CPU-only torch.
FROM python:3.12-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1

WORKDIR /home/user/app

RUN pip install --no-cache-dir --user torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

EXPOSE 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
```

- [ ] **Step 8: Run test to verify it passes**

Run: `uv run pytest tests/test_server.py -v`
Expected: PASS (4 tests)

- [ ] **Step 9: Commit**

```bash
git add app/ tests/test_server.py
git commit -m "feat: Malay demo app (FastAPI + inline-diff UI, Dockerfile)"
```

---

### Task 13: Space publish script + root Docker artifacts

**Files:**
- Create: `export/push_space.py`
- Create: root `Dockerfile`, root `requirements.txt` (copies of `app/Dockerfile`, `app/requirements.txt`, kept at root to match the design-spec layout)
- Test: `tests/test_push_space.py`

**Interfaces:**
- Produces:
  - `build_space_dir(out_dir, model_repo: str) -> Path` — assembles the flat Space repo: `server.py` (from `app/`, with the `MODEL_ID` default rewritten to `model_repo`), `static/`, `Dockerfile`, `requirements.txt`, `README.md` (HF front-matter: `title: Bahasa Malaysia spellchecker`, `sdk: docker`, `app_port: 7860`).
  - CLI: `python export/push_space.py --space-repo USER/lfm-my-spellchecker --model-repo USER/lfm-malay-spellchecker [--private]` — builds into `space_build/`, uploads via `HfApi`, prints Space URL. (Space env var `SPELLCHECKER_MODEL` can override at runtime; the rewritten default covers the common case.)

- [ ] **Step 1: Write the failing test** (`tests/test_push_space.py`)

```python
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "export"))

from push_space import build_space_dir


def test_build_space_dir_layout(tmp_path):
    out = build_space_dir(tmp_path / "space", model_repo="someone/lfm-malay-spellchecker")
    for f in ("server.py", "Dockerfile", "requirements.txt", "README.md"):
        assert (out / f).is_file(), f
    assert (out / "static" / "index.html").is_file()
    assert (out / "static" / "style.css").is_file()


def test_model_repo_injected(tmp_path):
    out = build_space_dir(tmp_path / "space", model_repo="someone/lfm-malay-spellchecker")
    src = (out / "server.py").read_text()
    assert '"someone/lfm-malay-spellchecker"' in src


def test_readme_frontmatter(tmp_path):
    out = build_space_dir(tmp_path / "space", model_repo="someone/lfm-malay-spellchecker")
    readme = (out / "README.md").read_text()
    assert readme.startswith("---")
    assert "title: Bahasa Malaysia spellchecker" in readme
    assert "sdk: docker" in readme
    assert "app_port: 7860" in readme
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_push_space.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'push_space'`

- [ ] **Step 3: Implement** (`export/push_space.py`)

```python
"""Assemble the flat HF Space repo from app/ and push it.

Usage:
    python export/push_space.py --space-repo USER/lfm-my-spellchecker \
        --model-repo USER/lfm-malay-spellchecker [--private]
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app"

README_TEMPLATE = """---
title: Bahasa Malaysia spellchecker
emoji: "\U0001F4DD"
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

Bahasa Malaysia spelling & grammar checker — GECToR-style tagger fine-tuned from
LiquidAI/LFM2.5-Encoder-350M. Model: [{model_repo}](https://huggingface.co/{model_repo}).
"""


def build_space_dir(out_dir, model_repo: str) -> Path:
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    shutil.copytree(APP / "static", out_dir / "static")
    shutil.copyfile(APP / "Dockerfile", out_dir / "Dockerfile")
    shutil.copyfile(APP / "requirements.txt", out_dir / "requirements.txt")

    server_src = (APP / "server.py").read_text()
    server_src = re.sub(
        r'(MODEL_ID = os\.environ\.get\("SPELLCHECKER_MODEL", )"[^"]*"',
        rf'\1"{model_repo}"', server_src, count=1)
    assert model_repo in server_src, "MODEL_ID default rewrite failed"
    (out_dir / "server.py").write_text(server_src)

    (out_dir / "README.md").write_text(README_TEMPLATE.format(model_repo=model_repo))
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space-repo", required=True)
    ap.add_argument("--model-repo", required=True)
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    out = build_space_dir(ROOT / "space_build", args.model_repo)
    print(f"built {out}:")
    for p in sorted(out.rglob("*")):
        if p.is_file():
            print("  ", p.relative_to(out))

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.space_repo, repo_type="space", space_sdk="docker",
                    private=args.private, exist_ok=True)
    api.upload_folder(folder_path=str(out), repo_id=args.space_repo, repo_type="space")
    print(f"https://huggingface.co/spaces/{args.space_repo}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Copy root Docker artifacts**

```bash
cp app/Dockerfile Dockerfile
cp app/requirements.txt requirements.txt
```

(These live at repo root per the design-spec layout; the Space build uses the `app/` copies.)

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_push_space.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest -q`
Expected: all tests pass (~40 tests).

- [ ] **Step 7: Commit**

```bash
git add export/ tests/test_push_space.py Dockerfile requirements.txt
git commit -m "feat: HF Space build + publish script"
```

---

## Manual steps after this plan (need user accounts / Colab)

1. **Train on Colab:** open `notebooks/train_malay_tagger.ipynb` in Colab (T4 runtime), replace `GITHUB_USER`, run all cells. Expect a few hours; resume-safe via Drive checkpoints.
2. **Publish model:** `push_model_repo(EXPORT_DIR, "<user>/lfm-malay-spellchecker")` (needs `HF_TOKEN`).
3. **Verify app locally against the trained model:** `SPELLCHECKER_MODEL=<user>/lfm-malay-spellchecker uv run uvicorn server:app --port 7860 --app-dir app`, then `curl` probes (spec §4 verification).
4. **Publish Space:** `python export/push_space.py --space-repo <user>/lfm-my-spellchecker --model-repo <user>/lfm-malay-spellchecker`, watch it boot, run the same probes through the public URL.

## Design decisions vs the spec (recorded for reviewers)

1. **V = 64,400 / num_tags = 128,802** — the spec estimated V = 65,536; Liquid's actual config (128,802 tags) and the tokenizer give V = 64,400. `num_tags` is always computed from the live tokenizer, never hardcoded.
2. **Multi-pass tag targets** — one `$APPEND` inserts one piece per anchor per pass, so the target conversion emits one training sample per decode pass (pass k trains on pass k−1's output), exactly mirroring `model.correct()`'s iterative decode. The spec's single mapping ("word inserted → $APPEND_<piece>") is the pass-1 special case of this.
3. **Original UI assets** — `app/static` is written from scratch (same functional contract); Liquid's `style.css`/`lliquid.gif` are not copied.
4. **Space assembly** — dev repo keeps `app/` nested; `export/push_space.py` flattens it into the Space repo layout (server.py at root), matching Liquid's Space structure.
5. **Training-side model is a separate `nn.Module`** (`GecTagger`) rather than instantiating `GecTaggerForGEC` for training — same submodule names, no backbone rebuild-from-config step; the exported state dict still loads 1:1 into `GecTaggerForGEC` (verified by key-set tests + Colab export-load probe).
