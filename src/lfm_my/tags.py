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
