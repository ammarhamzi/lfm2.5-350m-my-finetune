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
