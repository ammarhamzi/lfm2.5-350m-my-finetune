# lfm-my — Bahasa Malaysia spellchecker

A GECToR-style grammatical-error-correction tagger for Bahasa Malaysia, fine-tuned from
[LiquidAI/LFM2.5-Encoder-350M](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M).

The trained weights load **1:1** into Liquid's published inference class,
[LFM2.5-Encoder-350M-Spellchecker](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M-Spellchecker) —
so the exported model works with `AutoModel.from_pretrained(..., trust_remote_code=True)` and
`model.correct(...)` with no bridging code.

Training data is synthetic: rule-based Malay error injection over clean Malay Wikipedia
sentences produces `(noisy, corrected)` pairs. Training runs on a free Colab T4; the repo is
developed and tested on macOS CPU.

## Status

All 13 tasks of the implementation plan are code-complete, **84 tests passing offline**.

| Stage | State |
|---|---|
| Data generation, tag conversion, model, training loop, export | done, tested |
| Colab notebook, demo app, Space publish script | done, tested |
| Train the model on Colab | **not run yet** — needs your Colab session |
| Publish model + Space to Hugging Face | **not run yet** — needs your HF account |

No model has been trained yet. Everything below the line is machinery waiting for a training run.

## Install

```bash
uv sync --all-extras
```

Python 3.12, `torch>=2.6`, `transformers==5.1.0`. Run the suite:

```bash
uv run pytest
```

All 84 tests run offline — nothing downloads a model. Anything needing the network
(`build_tagger`, `load_wiki_sentences`, `push_model_repo`, the real `GecTaggerForGEC` load path)
is stubbed in tests and first executes for real on Colab.

## How it works

The model tags every BPE piece with one of `$KEEP`, `$DELETE`, `$REPLACE_<piece>`, or
`$APPEND_<piece>`, plus a binary "is this position wrong" detection head. Applying the tags
rewrites the sentence; inference re-runs the tagger until the text stops changing.

Two details drive most of the code:

**Targets are multi-pass.** One `$APPEND` inserts exactly one piece per anchor per pass — that is
what `apply_tags` does, so it is what the targets must encode. Turning `"nasi"` into
`"nasi lemak"` therefore takes several passes, and `convert_pair` emits **one training sample per
pass**, where pass *k* trains on pass *k−1*'s output. This mirrors `model.correct()`'s iterative
decode exactly, instead of teaching the model a one-shot edit it can never perform. Pairs that
do not converge within `max_passes` are dropped.

**The BOS anchor is real.** Position 0 is never emitted, so an `$APPEND` on it inserts at
sentence start. `Pass.bos_append` carries it, and `GecDataset` splices it back onto row 0.

**Weight compatibility is enforced by a test, not by convention.** `GecTagger` (training) mirrors
`GecTaggerForGEC`'s submodule names one-for-one — `encoder`, `dropout`, `detect_head`,
`base_head`, `replace_proj`, `append_proj`, `replace_bias`, `append_bias`.
`tests/test_model.py` instantiates the *real* published class with its backbone patched out and
asserts exact `state_dict` key-set and shape equality. If Liquid's class drifts, the suite fails.

## Layout

```
src/lfm_my/
  text.py         Malay tokenize/detok + word-level diff segments
  tags.py         tag-space id helpers ($REPLACE/$APPEND encoding)
  convert.py      (noisy, corrected) -> per-piece tags, multi-pass    <- highest-risk module
  errors.py       18 Malay error-injection operators
  data_build.py   corpus lines -> synthetic JSONL pairs, train/val split
  model.py        GecTagger: training-side module, names mirror GecTaggerForGEC
  dataset.py      GecDataset + collate_fn (pieces -> padded batches)
  train.py        loss, metrics, checkpointing
  export.py       safetensors + HF model repo layout
  modeling_gectagger.py   vendored verbatim from Liquid; ships with the export

notebooks/train_malay_tagger.ipynb   Colab driver (data -> train -> probe -> export)
app/                                 FastAPI demo + static UI
export/push_space.py                 assembles and pushes the HF Space
```

`src/lfm_my/modeling_gectagger.py` is Liquid's file, unmodified. It is the compatibility ground
truth: tests check against it, and `export_model_dir` copies it into the exported repo as the
remote code.

### Error injection

`errors.py` implements 18 operators across the families in the design spec: QWERTY-neighbour
typos (substitute/delete/insert/transpose/double), `meN-` allomorph swaps and dropped-root-letter
restoration (`menulis` → `mentulis`), prefix stripping, `ber-` → `be-`, `di-` prefix/space
confusion (`dimakan` ↔ `di makan`), suffix drops (`-kan`, `-i`, `-lah`, `-nya`), SMS abbreviations
(`dengan` → `dgn`), word duplication, adjacent merge, word split, comma noise, and
sentence-initial lowercasing.

About 17% of sentences are left clean (`p_clean`), so the model learns to leave correct Malay
alone. `tests/test_errors.py` asserts every operator is individually reachable — a dead operator
would silently narrow error diversity across the entire corpus without failing anything else.

## Train (Colab)

1. Open `notebooks/train_malay_tagger.ipynb` in Colab on a **T4** runtime.
2. Replace `GITHUB_USER` in the setup cell.
3. Run all cells.

Data, checkpoints, and the export land on Google Drive, so a disconnected runtime resumes where
it left off. Checkpoints are written atomically every 500 optimizer steps — a preempted free-tier
runtime cannot corrupt the last good one.

> **Read the tokenization-consistency cell before letting training run.** See
> [Known risk](#known-risk-tokenization-skew) below. It prints one line; if it warns, stop.

## Export and publish

The notebook's last cell calls `export_model_dir`, which writes a loadable HF model repo:
`model.safetensors`, `config.json`, `modeling_gectagger.py`, and the tokenizer files.
`verify_export` then re-reads the file and checks the key set, shapes, `num_tags`, and that
`config.json` carries no `num_labels` (a reserved `PretrainedConfig` property that breaks
loading).

Push the model:

```python
from lfm_my.export import push_model_repo
push_model_repo(EXPORT_DIR, "<user>/lfm-malay-spellchecker")   # needs HF_TOKEN
```

Verify the app against it locally:

```bash
SPELLCHECKER_MODEL=<user>/lfm-malay-spellchecker uv run uvicorn server:app --port 7860 --app-dir app
```

Then publish the Space:

```bash
uv run python export/push_space.py --space-repo <user>/lfm-my-spellchecker --model-repo <user>/lfm-malay-spellchecker
```

`push_space.py` flattens `app/` into the Space layout, rewrites the `SPELLCHECKER_MODEL` default
to your model repo, and writes the HF front-matter README.

## Demo app

`GET /api/health`, `POST /api/correct {text, min_error_prob, max_iter}` →
`{corrected, segments, changed}`, and `GET /` serving the UI — the same HTTP contract as Liquid's
Space, with the English contraction handling removed. Every request field is bounded
(`SPELLCHECKER_MAX_CHARS`, `max_iter` 1–10, `min_error_prob` 0–1): the endpoint is public and the
model runs on CPU, so an unbounded request would tie up the container.

## Known risk: tokenization skew

**Unresolved, and worth one minute before a multi-hour training run.**

Training aligns edits at the *word* level, so `convert_pair` and `GecDataset` encode one word at a
time. At inference, `model.correct()` encodes the *whole sentence* in one call. For tokenizers
that mark word boundaries with a leading space (GPT-2 style), those two disagree, and the model
would be served inputs shaped unlike anything it trained on.

Whether `LFM2.5-Encoder-350M`'s tokenizer has this property cannot be checked offline, so the
notebook has a cell that checks it against the real tokenizer and prints a warning with the fix
(encode words with a leading space in `GecDataset`, or align `convert_pair` on whole-sentence
pieces). Read its output before proceeding.

## Notes on the plan

`docs/superpowers/plans/2026-08-20-malay-spellchecker.md` carries a reference implementation per
task. Those were treated as drafts, not spec: most tasks contained a real defect, and the
accompanying tests were consistently too loose to catch the adjacent bug class. Fixes are
recorded in the commit messages — among them the export crashing on tied encoder weights,
validation metrics scoring an empty edit set as 100%, `val_size` producing twice the requested
validation records, and a shadowed loop variable that would have crashed training at step 50.

Where the plan's tests only asserted that *something* happened, the suite now pins the actual
invariant: a fuzz test round-tripping every conversion pass through Liquid's own `apply_tags`,
operator reachability, train/val leakage, a `state_dict` cross-check against the real inference
class, and an end-to-end pipeline smoke test covering the seams between tasks.

- Design spec: `docs/superpowers/specs/2026-08-11-malay-spellchecker-design.md`
- Implementation plan: `docs/superpowers/plans/2026-08-20-malay-spellchecker.md`
