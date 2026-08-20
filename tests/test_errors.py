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


# Sentences chosen so every operator has something to bite on: a capitalized first word,
# words >= 6 chars to split, an existing comma to drop, meN-/ber-/di- forms, and abbreviables.
_COVERAGE_CORPUS = [
    "Dia menulis surat , kemudian membaca buku kesejahteraan itu",
    "Buku itu dimakan oleh anjing yang berlari dengan pantas",
    "Mereka tidak bacakan pengumuman , dan pergilah ke sekolah",
]


def test_every_operator_is_reachable():
    """A dead operator silently narrows error diversity in the whole training corpus."""
    from lfm_my.errors import OPS

    unfired = []
    for op in OPS:
        if not any(
            op(s.split(), random.Random(i)) not in (None, s.split())
            for s in _COVERAGE_CORPUS
            for i in range(200)
        ):
            unfired.append(op.__name__)
    assert not unfired, f"operators never fired: {unfired}"


def test_operators_never_emit_empty_words():
    from lfm_my.errors import OPS

    for op in OPS:
        for s in _COVERAGE_CORPUS:
            for i in range(100):
                out = op(s.split(), random.Random(i))
                assert out is None or (isinstance(out, list) and all(out)), op.__name__


def test_error_count_respects_bounds():
    """max_errors=1 must not mangle the sentence beyond one operator's worth of change."""
    src = "mereka sudah pergi ke sekolah pagi tadi"
    for i in range(200):
        cfg = InjectorConfig(p_clean=0.0, min_errors=1, max_errors=1)
        out = inject_errors(src, random.Random(i), cfg).split()
        assert abs(len(out) - len(src.split())) <= 1
