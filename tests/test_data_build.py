import json
from pathlib import Path

from lfm_my.data_build import build_from_lines, build_pairs, clean_sentences
from lfm_my.errors import InjectorConfig

FIXTURE = Path(__file__).parent / "fixtures" / "malay_corpus.txt"


def _lines():
    return FIXTURE.read_text(encoding="utf-8").splitlines()


def test_clean_sentences_filters_and_tokenizes():
    # min_words=3 so the short-but-valid sentence survives: this case is about tokenization
    # ("Dia pergi." -> "Dia pergi .") and the two length filters, not the production band.
    out = list(clean_sentences([
        "Dia pergi.",
        "Pendek.",
        "Satu dua tiga empat lima enam tujuh lapan sembilan sepuluh sebelas "
        "dua belas tiga belas empat belas lima belas enam belas tujuh belas "
        "lapan belas sembilan belas dua puluh dua puluh satu",
    ], min_words=3))
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


def test_splits_do_not_leak_sentences(tmp_path):
    """Both pairs of one clean sentence share a target -- if a sentence straddles the split,
    validation is scored on text the model trained on."""
    build_from_lines(_lines(), tmp_path / "ms", InjectorConfig(seed=4),
                     pairs_per_sentence=2, val_size=4, seed=4)
    train = {json.loads(l)["correct"] for l in (tmp_path / "ms-train.jsonl").read_text().splitlines()}
    val = {json.loads(l)["correct"] for l in (tmp_path / "ms-val.jsonl").read_text().splitlines()}
    assert val and train and not (train & val)


def test_val_size_larger_than_corpus_is_safe(tmp_path):
    stats = build_from_lines(_lines(), tmp_path / "ms", InjectorConfig(seed=5),
                             pairs_per_sentence=1, val_size=10_000, seed=5)
    assert stats["train_pairs"] == 0
    assert stats["val_pairs"] == stats["clean_sentences"]


def test_build_from_lines_is_deterministic(tmp_path):
    def run(d):
        build_from_lines(_lines(), d / "ms", InjectorConfig(seed=6), pairs_per_sentence=2,
                         val_size=4, seed=6)
        return (d / "ms-train.jsonl").read_text(), (d / "ms-val.jsonl").read_text()

    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    assert run(tmp_path / "a") == run(tmp_path / "b")
