# Bahasa Malaysia Spellchecker — Design Spec

**Date:** 2026-08-11
**Status:** Approved
**Approach:** Full GECToR-style fine-tune of LFM2.5-Encoder-350M for Bahasa Malaysia, published as an HF Space (mirror of `LiquidAI/spellchecker`).

---

## Background

- **Reference:** `LiquidAI/spellchecker` HF Space — FastAPI backend + static inline-diff UI, loading `LiquidAI/LFM2.5-Encoder-350M-Spellchecker`, a subword-level GECToR-style grammatical-error-correction (GEC) tagger fine-tuned on the LFM2.5 bidirectional encoder. Covers spelling, grammar, punctuation, casing.
- **Reference backend:** `server.py` loads model with `trust_remote_code=True`, exposes `/api/health` and `/api/correct` (word-level diff → `{keep, edit, del}` segments), serves a static `index.html` with green-insert / red-strikethrough rendering. Serves on port 7860 via Docker.
- **Model architecture** (from leaked `modeling_gectagger.py`): two-head GECToR tagger on bidirectional MLM encoder. Tag space is algorithmic per BPE piece: `$KEEP`, `$DELETE`, `$REPLACE_<piece_id>`, `$APPEND_<piece_id>` (optional `$SWAP`). With `tie_replace=True` the replace/append logits are tied to the encoder's input embeddings (`logit = proj(h) · E_i`) rather than a free `Linear(hidden, 2+2V)`. `num_tags = base + 2V`, `base = 3 if use_swap else 2`. Iterative decode via `model.correct(texts, max_iter=3, min_error_prob=0.0)`.
- **Liquid's training recipe is not public** — only the inference class ships. We reconstruct the training side from the published model file.

## Key constraints

- **Malay is out-of-distribution** for the base encoder's 15-language vocab (English, German, Spanish, French, Italian, Dutch, Polish, Portuguese, Arabic, Hindi, Japanese, Russian, Turkish, Vietnamese, Chinese). Fine-tuning will adapt it, but expect a lower quality ceiling than Liquid's English model.
- **Compute:** Google Colab free tier (T4, 16GB VRAM). PyTorch, bf16 autocast.
- **Demo runtime:** published as an HF Space (Docker SDK), same layout as LiquidAI's.

## Deliverables

1. Trained Malay GECToR model (`model.correct()` compatible)
2. Notebook-based training pipeline
3. FastAPI + static demo app, deployed to HF Spaces

---

## 1. Data synthesis pipeline

### Sources
Pull on Colab via `datasets`:
- **Primary:** Malay Wikipedia (e.g. `wikipedia` with `ms` language config, or `wikimedia/wikipedia` filtered to `ms`)
- **Augmentation:** search HF for Malay (`ms`) text datasets; if none suitable, use Indonesian (`id`) corpora (closely related script/orthography) — final dataset choice is an implementation-time decision, logged in the notebook

Filter to sentences ~5–30 words, strip boilerplate. Target **~15–20K pairs** train, **~1K** held-out val.

### Error injection (rule-based, mirrors real Malay learner errors)
For each clean sentence, inject 1–3 errors:
- **Typos:** keyboard-neighbor substitution, single-char delete/insert/swap, doubled/omitted letters (e.g. `di`/`de`, `rahsia`/`rahasia`)
- **Affix errors** (Malay's core difficulty): `meN-` allomorphs (`mem-`/`men-`/`meng-`/`me-`), `ber-`/`be-`, `di-`/`ke-`, dropped/misused affixes (`-kan`, `-i`, `-lah`, `-nya`)
- **Punctuation:** missing/extra spaces, split or joined words, stray commas
- **Casing:** lowercased sentence-initial or proper nouns
- Keep a fraction (~15–20%) uncorrupted so the model learns `$KEEP` on clean text

### Output format
`(noisy_text, corrected_text)` pairs in Liquid's whitespace-tokenized form (punctuation spaced apart).

---

## 2. Model & tagger head

### Base
`LiquidAI/LFM2.5-Encoder-350M`, `trust_remote_code=True`, bf16 autocast on T4.

### Tagger head (reconstructed from `modeling_gectagger.py`)
- Two outputs per BPE token: **detect head** (2 classes) and **label head** (tag space)
- Tag space: `num_tags = 2 + 2·V`, `V = 65,536` → **131,074 tags**
- Copy reference architecture structurally (same `encoder.*` param names) so trained weights load 1:1 into Liquid's published inference class on export → `model.correct([...])` works out of the box

### Target conversion (noisy, corrected) → per-piece tags
1. Whitespace-tokenize both sides
2. `difflib.SequenceMatcher` (reuse `server.py`'s `diff_segments` logic) → word-level keep/replace/delete/insert ops
3. Map word ops → per-BPE-piece tags:
   - word kept → `$KEEP` per piece
   - word replaced → `$DELETE` + `$REPLACE_<pieces>`
   - word inserted → `$APPEND_<piece>`
   - word deleted → `$DELETE`
4. `detect = 1` on any piece whose word was touched, else `0`

### Loss
`label_loss + 0.5 · detect_loss` (matches `aux_loss_weight=0.5`), `CrossEntropyLoss`.

### Memory: tied replacement
Use `tie_replace=True` — replace/append logits computed as `proj(h) @ Embedding.t()` instead of a dense 131K-class linear. Non-negotiable for T4.

### Export
Checkpoint → safetensors + `config.json` mirroring `GecTaggerConfig` + modeling file → push to HF repo (public, consistent with the public Space so no token is needed at load time).

---

## 3. Training recipe

### Hyperparameters (Colab T4)
| Param | Value |
|---|---|
| Optimizer | AdamW |
| LR | 2e-5 |
| Warmup | 0.1 |
| Weight decay | 0.1 |
| Batch | 8 (grad accum → effective 32) |
| Precision | bf16 autocast |
| Epochs | 5–7, early stopping (patience 2) on val loss |
| Head dropout | 0.1 |
| max_len | 128 tokens/sentence |

### Feasibility
350M params bf16 ≈ ~2.5GB VRAM with grad accum; ~3–8K optimizer steps total; a few hours worst-case, within Colab free daily limits. Fallback: 3 epochs with `early_stopping_patience=1`.

### Monitoring
Log `label/acc` (excl. `$KEEP`), `detect/acc`, val loss; checkpoint every 500 steps (Colab timeout safety).

### Post-training validation (before demo)
- Self-test on Malay probe sentences with planted errors (e.g. `dia tidak arah pinjam buku` → `dia tidak akan pinjam buku`)
- Holdout ~1K pairs → label accuracy + edit-level precision/recall on `$REPLACE`/`$APPEND`/`$DELETE`
- Tune `min_error_prob` on holdout
- **On-device check:** run `model.correct()` on CPU in the notebook against same probes to confirm weights load into Liquid's inference class 1:1

---

## 4. Demo app & HF Space

### Repo layout
```
lfm-my/
├── data/          # synthetic Malay pairs + tokenizer artifacts (gitignored data files)
├── src/           # tagger model, tag-conversion, training utilities
├── notebooks/     # main Colab training notebook (mirrors src/)
├── app/
│   ├── server.py      # FastAPI: /api/health, /api/correct (Malay-adapted)
│   └── static/        # index.html diff UI (green=inserted, strikethrough-red=deleted)
├── export/        # scripts to package trained weights for HF model repo
├── Dockerfile     # uvicorn server:app, port 7860
├── requirements.txt
└── README.md
```

### Malay adaptations to `server.py`
- Keep punctuation-spacing tokenizer; **drop English contraction rules** (`n't`, `'s`); add Malay-specific ones if needed (e.g. `di-` prefix spacing)
- `SPELLCHECKER_MODEL` env → user's HF model repo
- Same `diff_segments` for inline highlight rendering (language-agnostic)

### HF Space setup
- Model repo: `YOUR_USER/lfm-malay-spellchecker` — safetensors + `config.json` + modeling file
- Space: Docker SDK, `app_port: 7860`, CPU basic tier, `private: false`
- README front-matter copied from LiquidAI's, title `Bahasa Malaysia spellchecker`

### Verification before shipping
1. Notebook probes pass
2. `uvicorn server:app` locally on Mac, `curl /api/correct` returns sensible Malay diffs
3. Push to HF Space, watch it boot on space CPU, run same curl probes through public URL

---

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Malay vocab OOD → lower quality ceiling | Larger training set; monitor val metrics; consider warm-start from Liquid's English checkpoint as fallback |
| Colab free tier limits / timeouts | Checkpoint every 500 steps; keep total steps ≤ 8K; grad accum |
| Training recipe divergence from Liquid's private one | Validate via held-out metrics; iterate |
| Tag-conversion bugs → noisy targets | Unit-test conversion on hand-crafted examples before training |
| HF Space CPU too slow | 350M is fast on CPU (verified by LiquidAI's space); fallback: quantized export |

## Out of scope (v1)
- Reranker (Liquid ships one; we skip → tagger-only `correct()`)
- Indonesian/Bahasa Indonesia variants (Malay only)
- Publishing the model repo as public org artifact (private user repo first)
- QAT/quantization
