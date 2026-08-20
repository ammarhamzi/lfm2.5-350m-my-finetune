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
    # Hold out whole SENTENCES (a clean sentence must never straddle the split -- its pairs
    # share the same target), but size the split in PAIRS, which is what val_size names.
    val_sent_n = -(-val_size // max(pairs_per_sentence, 1))
    val_sents, train_sents = sents[:val_sent_n], sents[val_sent_n:]
    stats = {"clean_sentences": len(sents), "val_sentences": len(val_sents),
             "clean_kept": 0, "train_pairs": 0, "val_pairs": 0}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for split, chunk in (("train", train_sents), ("val", val_sents)):
        n = 0
        with open(out_path.with_name(f"{out_path.name}-{split}.jsonl"), "w", encoding="utf-8") as f:
            for p in build_pairs(chunk, cfg, pairs_per_sentence):
                if split == "val" and n >= val_size:
                    break
                if p["noisy"] == p["correct"]:
                    stats["clean_kept"] += 1
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
                n += 1
        stats["train_pairs" if split == "train" else "val_pairs"] = n
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
