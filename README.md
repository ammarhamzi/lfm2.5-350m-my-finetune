# lfm-my — Bahasa Malaysia spellchecker

A GECToR-style grammatical-error-correction tagger for Bahasa Malaysia, fine-tuned from
[LiquidAI/LFM2.5-Encoder-350M](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M).

The trained weights load **1:1** into Liquid's published inference class,
[LFM2.5-Encoder-350M-Spellchecker](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M-Spellchecker) —
`AutoModel.from_pretrained(..., trust_remote_code=True)` then `model.correct(...)`, no bridging code.

All 13 tasks of the implementation plan are code-complete, **84 tests passing offline**. No model
has been trained yet: the Colab run, model publish, and Space publish are still ahead.

---

# The method

## Correction as tagging, not generation

A seq2seq corrector generates the fixed sentence token by token. That is slow, needs a decoder,
and is free to hallucinate text that was never in the input. GECToR reframes correction as
**token-level classification**: the encoder reads the sentence once, and every subword piece is
assigned one *edit tag*. Applying the tags rewrites the sentence in a single pass — no
autoregression, no beam search, one encoder forward per iteration.

The tradeoff is that a single tagging pass can only express a limited edit. So decoding
**iterates**: tag, apply, re-tag the result, until the text stops changing. That iteration is the
single fact that shapes almost everything below.

## The tag space

Four edit operations, encoded into one flat label space of size `2 + 2V` where `V` is the
tokenizer's vocabulary:

| id range | tag | effect at position *i* |
|---|---|---|
| `0` | `$KEEP` | emit piece *i* unchanged |
| `1` | `$DELETE` | emit nothing |
| `2 … 2+V` | `$REPLACE_<piece>` | emit `<piece>` instead of piece *i* |
| `2+V … 2+2V` | `$APPEND_<piece>` | emit piece *i*, then `<piece>` |

With `V = 64,400` that is `num_tags = 128,802`. `src/lfm_my/tags.py` does the id arithmetic and
`modeling_gectagger.id_to_tag` inverts it; both derive `V` from the live tokenizer, never a
hardcoded constant.

**Position 0 is the BOS anchor.** It is never emitted, so an `$APPEND` on it is how the model
inserts text at the very start of a sentence — the only way to prepend, since there is no
position to the left of the first word.

## Two heads, and why the second one exists

```
                    ┌─> detect_head  ──> [B,T,2]        "is this position wrong?"
encoder ──> hidden ─┤
                    └─> label_logits ──> [B,T,2+2V]     "which edit?"
```

The label head alone would do the job, but the tag distribution is brutally imbalanced — the vast
majority of positions in any sentence are `$KEEP`. The **detection head** is a binary auxiliary
task on exactly that question, and at decode time it acts as a *gate*: an edit is applied only if
the label head proposes a non-`$KEEP` tag **and** the detection head's error probability clears
`min_error_prob`. One knob, tunable after training, trading precision against recall without
retraining anything.

Training optimises both, fp32 cross-entropy, padding at `ignore_index=-100`:

```
loss = CE(label_logits, label_targets) + 0.5 · CE(detect_logits, detect_targets)
```

## The tied output head

A `[hidden=1024 → 128,802]` classifier is ~132M parameters — a third of the encoder, bolted on as
a randomly-initialised head. Instead, `$REPLACE` and `$APPEND` logits are computed against the
encoder's own **input embedding matrix** `E`:

```python
E   = encoder.get_input_embeddings().weight[:V]       # [V, H], tied — not a copy
rep = replace_proj(hidden) @ E.t() + replace_bias     # [..., V]
app = append_proj(hidden) @ E.t() + append_bias       # [..., V]
return cat([base_head(hidden), rep, app], dim=-1)     # [..., 2 + 2V]
```

Two `[H,H]` projections and two `[V]` bias vectors replace the giant matrix. Beyond the parameter
saving, this gives the head a real prior: "replace this with piece *x*" is scored as a dot product
against *x*'s pretrained embedding, so the head starts out knowing which pieces are similar
instead of learning 64,400 output directions from scratch.

`base_head` stays a plain `[H → 2]` for `$KEEP`/`$DELETE`, which have no piece to point at.

## Building the targets — the hard part

This is `src/lfm_my/convert.py`, and it is where the design earns its keep.

### Word-level diff, piece-level tags

Given a `(noisy, corrected)` pair, pass 1 aligns at the **word** level with
`difflib.SequenceMatcher`. Word-level alignment produces clean whole-word edits; aligning
subwords directly yields noisy, arbitrary-looking splices. Each opcode then expands to per-piece
tags:

- `equal` → `$KEEP` on every piece
- `delete` → `$DELETE` on every piece
- `replace` → pair up pieces positionally: `$KEEP` where they already match, `$REPLACE_<q>`
  where they differ, `$DELETE` for any surplus input pieces
- `insert` → `$APPEND_<q>` on the **anchor**, the last piece emitted so far

`"saya makan nasik"` → `"saya makan nasi"` is a one-pass replace of `nasik`(5 pieces) by
`nasi`(4): four `$KEEP`, one trailing `$DELETE`.

### One insertion piece per anchor per pass

Here is the constraint everything bends around: `apply_tags` lets a position emit **at most one**
appended piece. A position's tag is either `$REPLACE` or `$APPEND` — never both. So a multi-piece
insertion cannot be expressed in one pass, and neither can the surplus of a replacement that grew
longer.

The conversion does not paper over this. It emits what one pass can actually do, **applies the
tags**, and re-diffs the result against the target — producing one training sample per pass:

```
"saya makan"  ->  "saya sudah makan"          ("sudah" = pieces ▁su, dah)

pass 1   input:  [BOS] ▁saya ▁makan
         tags:    KEEP  APPEND_▁su  KEEP
         output:       ▁saya ▁su ▁makan

pass 2   input:  [BOS] ▁saya ▁su ▁makan       <- pass 1's OUTPUT
         tags:    KEEP  KEEP  APPEND_dah  KEEP
         output:       ▁saya ▁su dah ▁makan   == target, converged
```

Two passes, two training samples. `Conversion.inputs[k]` pairs with `Conversion.passes[k]`, and
`GecDataset` expands each conversion into that many rows.

Two deferral rules fall out of the same constraint:

- an insertion whose anchor already carries `$REPLACE` cannot also append — the whole insertion
  waits for the next pass
- surplus pieces `q[k:]` from a lengthening replacement wait too

Pairs that do not converge within `max_passes` (default 5) return `None` and are **dropped** from
training. A pair needing more passes than decode will ever run is not a training signal; it is
noise.

### Why this matters

Inference runs `model.correct()`, which re-tags its own output until it reaches a fixpoint (capped
at `max_iter`, default 3). Training pass *k* takes pass *k−1*'s output as its input — **the same
distribution the model faces at iteration *k* during decode.** Targets that assumed a single
omnipotent pass would train the model to attempt edits that `apply_tags` cannot execute, and would
never show it the partially-corrected text that iterations 2 and 3 actually receive.

The invariant is pinned by a seeded fuzz test that round-trips every generated pass through
Liquid's own `apply_tags`, asserting `inputs[k+1] == apply_tags(inputs[k], passes[k])` and that
the final pass lands exactly on the corrected pieces.

## Training data: synthetic Malay errors

There is no large annotated Malay GEC corpus, so the training signal is manufactured: take clean
Malay Wikipedia sentences as the *targets*, and corrupt them to make the *inputs*.

`src/lfm_my/errors.py` implements **18 operators**, applied 1–3 at a time to each sentence:

| family | examples |
|---|---|
| typos | QWERTY-neighbour substitution, deletion, insertion, transposition, doubling |
| `meN-` affixes | allomorph swaps (`mengirim`→`mengkirim`), dropped-root-letter restoration (`menulis`→`mentulis`), prefix stripping |
| `ber-` | `berlari` → `belari` |
| `di-` | prefix/space confusion both ways: `dimakan` ↔ `di makan` |
| suffixes | dropping `-kan`, `-i`, `-lah`, `-nya` |
| SMS register | `dengan`→`dgn`, `tidak`→`tak`, `sudah`→`dah` |
| word-level | duplication, adjacent merge, random split |
| punctuation & case | stray/dropped commas, sentence-initial lowercasing |

The affix and `di-` operators are the point of doing this for Malay specifically rather than
reusing an English error model: `meN-` allomorphy and the `di-` prefix/preposition distinction are
where real Malay writing goes wrong, and neither has an English analogue.

**~17% of sentences are left untouched** (`p_clean`). Without clean examples the model learns that
every sentence contains an error and starts inventing edits in correct text.

Sentences are filtered to 5–30 words, and the train/val split holds out **whole sentences** — all
of a sentence's pairs share one target, so splitting mid-sentence would put a validation target in
the training set.

## The training loop

`src/lfm_my/train.py` holds loss, metrics, and checkpointing; the loop itself lives in
`notebooks/train_malay_tagger.ipynb` so the knobs are visible in one cell.

| | |
|---|---|
| optimiser | AdamW, lr `2e-5`, weight decay `0.1` |
| schedule | linear warmup 10% → cosine decay |
| batch | 8 × 4 gradient accumulation = **effective 32** |
| sequence | `max_len` 128 pieces |
| epochs | 7, early stopping on val loss, patience 2 |
| grad clip | 1.0 |
| precision | autocast; dtype chosen from the hardware (see below) |

**Precision is picked at runtime, not hardcoded.** The free Colab GPU is a T4 — Turing, `sm_75`,
which has **no bfloat16**; that needs Ampere or newer. The notebook selects `bfloat16` only if
`torch.cuda.is_bf16_supported()`, otherwise `float16` with `GradScaler` (bf16 does not need loss
scaling, fp16 does).

**Checkpoints survive preemption.** A free-tier runtime can be reclaimed mid-write, so
`save_checkpoint` writes to a temp file and `os.replace`s it — atomic, so a killed write cannot
destroy the last good checkpoint. It fires every 500 optimizer steps, plus a separate `best.pt` on
each val-loss improvement, all on Drive so a reconnect resumes rather than restarts.

### Reading the metrics

Reporting plain accuracy over all positions would be meaningless — a model that predicted `$KEEP`
everywhere would score in the high 90s. So `metrics_for` reports **`label_acc_nokeep`: accuracy
restricted to positions whose target is not `$KEEP`.** That is the number that says whether the
model can actually edit.

That restriction has a sharp edge. A clean pair produces an all-`$KEEP` sample with *no* non-KEEP
positions, so the metric is undefined for it. Returning `1.0` there — scoring a vacuous 100% on an
empty set — inflates the average by exactly the clean fraction: a model with 0.0 real accuracy
reports **0.170**. `metrics_for` returns `NaN` instead, and `aggregate_metrics` pools batches
weighted by how many positions each was actually computed over.

## Decoding

```python
label_logits[..., KEEP_ID] += keep_confidence     # bias toward leaving text alone
best      = label_logits.argmax(-1)
err_prob  = detect_logits.softmax(-1)[..., INCORRECT]
apply     = (best != KEEP_ID) and err_prob >= min_error_prob
```

Both knobs are decode-time only. `min_error_prob` raises the bar for acting; `keep_confidence`
shifts the whole label distribution toward `$KEEP` (negative values over-generate edits, which is
how the optional reranker feeds on candidates). Sequences drop out of the batch as soon as they
stop changing, so a fully-corrected sentence costs one iteration, not `max_iter`.

## Staying loadable by Liquid's class

The training module `GecTagger` is a plain `nn.Module`, not `GecTaggerForGEC` — no config plumbing
or backbone rebuild during training. What makes the export work is that its submodules are named
**one-for-one** with the published class's `tie_replace=True` branch: `encoder`, `dropout`,
`detect_head`, `base_head`, `replace_proj`, `append_proj`, `replace_bias`, `append_bias`.

This is enforced, not documented-and-hoped. `tests/test_model.py` instantiates the **real**
`GecTaggerForGEC` (backbone patched out, since it wants network) and asserts exact `state_dict`
key-set and per-key shape equality against ours. Rename a head and the suite fails immediately
rather than at export time after a training run.

`src/lfm_my/modeling_gectagger.py` is Liquid's file, vendored unmodified. It is the compatibility
ground truth: tests check against it, and `export_model_dir` copies it into the exported repo as
the remote code.

---

# Running it

## Install

```bash
uv sync --all-extras
uv run pytest
```

Python 3.12, `torch>=2.6`, `transformers==5.1.0`. All 84 tests run offline — nothing downloads a
model. The network-only paths (`build_tagger`, `load_wiki_sentences`, `push_model_repo`, the real
`GecTaggerForGEC` load) are stubbed in tests and first execute for real on Colab.

## Train

Open `notebooks/train_malay_tagger.ipynb` in Colab on a **T4**, replace `GITHUB_USER` in the setup
cell, run all. Data, checkpoints, and the export live on Drive, so a disconnected runtime resumes.

> **Read the tokenization-consistency cell's output before letting training proceed** — see
> [Known risk](#known-risk-tokenization-skew).

## Export and publish

The final cell calls `export_model_dir`, writing `model.safetensors`, `config.json`,
`modeling_gectagger.py`, and the tokenizer files. `verify_export` re-reads the result and checks
the key set, shapes, `num_tags`, and the absence of `num_labels` (a reserved `PretrainedConfig`
property that silently breaks loading). Tensors sharing storage — common when an encoder ties its
input embedding to an output head — are cloned, since `safetensors` refuses to write them.

```python
from lfm_my.export import push_model_repo
push_model_repo(EXPORT_DIR, "<user>/lfm-malay-spellchecker")     # needs HF_TOKEN
```

```bash
# verify locally against the trained model
SPELLCHECKER_MODEL=<user>/lfm-malay-spellchecker uv run uvicorn server:app --port 7860 --app-dir app

# publish the Space
uv run python export/push_space.py --space-repo <user>/lfm-my-spellchecker --model-repo <user>/lfm-malay-spellchecker
```

`push_space.py` flattens `app/` into the Space layout, rewrites the `SPELLCHECKER_MODEL` default,
and writes the HF front-matter README.

## Demo app

`GET /api/health`, `POST /api/correct {text, min_error_prob, max_iter}` →
`{corrected, segments, changed}`, `GET /` for the UI — the same HTTP contract as Liquid's Space,
with the English contraction handling removed. Every request field is bounded
(`SPELLCHECKER_MAX_CHARS`, `max_iter` 1–10, `min_error_prob` 0–1); the endpoint is public and the
model runs on CPU, so an unbounded request would tie up the container.

## Layout

```
src/lfm_my/
  text.py         Malay tokenize/detok + word-level diff segments
  tags.py         tag-space id arithmetic
  convert.py      (noisy, corrected) -> per-piece tags, multi-pass   <- highest-risk module
  errors.py       18 Malay error-injection operators
  data_build.py   corpus lines -> synthetic JSONL pairs, train/val split
  model.py        GecTagger: training module, names mirror GecTaggerForGEC
  dataset.py      GecDataset + collate_fn (pieces -> padded batches)
  train.py        loss, metrics, checkpointing
  export.py       safetensors + HF model repo layout
  modeling_gectagger.py    vendored verbatim from Liquid; ships with the export

notebooks/train_malay_tagger.ipynb   Colab driver (data -> train -> probe -> export)
app/                                 FastAPI demo + static UI
export/push_space.py                 assembles and pushes the HF Space
```

---

# Known risk: tokenization skew

**Unresolved, and worth one minute before a multi-hour training run.**

Target construction aligns edits at the *word* level, so `convert_pair` and `GecDataset` encode
one word at a time. At inference, `model.correct()` encodes the *whole sentence* in one call. For
tokenizers that mark word boundaries with a leading space (GPT-2 style),
`encode("a b") != encode("a") + encode("b")` — and the model would be served inputs shaped unlike
anything it trained on.

Whether `LFM2.5-Encoder-350M`'s tokenizer behaves this way cannot be checked offline, so the
notebook has a cell that tests it against the real tokenizer and prints the fix if it warns
(encode words with a leading space in `GecDataset`, or align `convert_pair` on whole-sentence
pieces). Read its output before proceeding.

# Notes on the plan

`docs/superpowers/plans/2026-08-20-malay-spellchecker.md` carries a reference implementation per
task. Those were treated as drafts, not spec: most tasks contained a real defect, and the
accompanying tests were consistently too loose to catch the adjacent bug class. Fixes are recorded
in the commit messages — among them the export crashing on tied encoder weights, validation
metrics scoring an empty edit set as 100%, `val_size` producing twice the requested validation
records, and a shadowed loop variable that would have crashed training at step 50.

Where the plan's tests only asserted that *something* happened, the suite now pins the actual
invariant: the `apply_tags` round-trip fuzz test, operator reachability, train/val leakage, the
`state_dict` cross-check against the real inference class, and an end-to-end pipeline smoke test
covering the seams between tasks.

- Design spec: `docs/superpowers/specs/2026-08-11-malay-spellchecker-design.md`
- Implementation plan: `docs/superpowers/plans/2026-08-20-malay-spellchecker.md`
