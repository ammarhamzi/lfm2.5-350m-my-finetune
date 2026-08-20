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
