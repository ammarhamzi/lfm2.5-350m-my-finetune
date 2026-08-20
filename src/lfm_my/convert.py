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
    if bos_append is not None:                       # BOS anchor emits before every other piece
        out.insert(0, bos_append)
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
