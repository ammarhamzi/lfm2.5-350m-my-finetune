from lfm_my.convert import Conversion, convert_pair, one_pass, pieces_of
from lfm_my.modeling_gectagger import KEEP_ID, DELETE_ID, apply_tags
from lfm_my.tags import append_id, replace_id

V = 1000
BOS = 1


def char_encode(word: str) -> list[int]:
    """Stub BPE: one piece per character (id = 10 + ordinal).

    NB: one piece per *character* is far more granular than the real tokenizer, and an $APPEND
    inserts exactly one piece per anchor per pass — so inserting the word 'sudah' here costs 5
    passes, where real BPE would spend 1-2. Insertion tests below therefore pass an explicit
    max_passes; the production default (5) is sized for real pieces, not for this stub.
    """
    return [10 + ord(c) for c in word]


def pieces(words: list[str]) -> list[int]:
    return pieces_of(words, char_encode)


def _tag_str(t: int) -> str:
    if t == 0:
        return "$KEEP"
    if t == 1:
        return "$DELETE"
    if t < 2 + V:
        return f"$REPLACE_{t - 2}"
    return f"$APPEND_{t - 2 - V}"


def apply_pass(input_pieces: list[int], p) -> list[int]:
    """Apply a Pass through Liquid's inference-time apply_tags (the ground truth)."""
    bos_tag = "$KEEP" if p.bos_append is None else f"$APPEND_{p.bos_append}"
    tags = [bos_tag] + [_tag_str(t) for t in p.tags]
    return apply_tags([BOS] + input_pieces, tags)


def test_clean_pair_gives_empty_passes():
    conv = convert_pair("saya makan nasi", "saya makan nasi", char_encode, V)
    assert isinstance(conv, Conversion)
    assert conv.passes == []


def test_single_word_replace_one_pass():
    conv = convert_pair("saya makan nasik", "saya makan nasi", char_encode, V)
    assert len(conv.passes) == 1
    p = conv.passes[0]
    # 'nasik' -> 'nasi': n,a,s,i kept; trailing 'k' deleted
    assert p.tags[-1] == DELETE_ID
    assert p.detect[-1] == 1
    assert apply_pass(conv.inputs[0], p) == pieces(["saya", "makan", "nasi"])


def test_word_replace_longer_needs_two_passes():
    # 'nasi' (4 pieces) -> 'nasilemak' (9 pieces): paired prefix + deferred tail
    conv = convert_pair("saya makan nasi", "saya makan nasilemak", char_encode, V, max_passes=6)
    assert conv is not None and len(conv.passes) >= 2
    out = pieces(["saya", "makan", "nasi"])
    for inp, p in zip(conv.inputs, conv.passes):
        assert inp == out
        out = apply_pass(inp, p)
    assert out == pieces(["saya", "makan", "nasilemak"])


def test_word_delete():
    conv = convert_pair("saya sudah makan", "saya makan", char_encode, V)
    assert len(conv.passes) == 1
    out = apply_pass(conv.inputs[0], conv.passes[0])
    assert out == pieces(["saya", "makan"])


def test_word_insert_after_anchor():
    conv = convert_pair("saya makan", "saya sudah makan", char_encode, V, max_passes=5)
    assert conv is not None
    out = pieces(["saya", "makan"])
    for inp, p in zip(conv.inputs, conv.passes):
        assert inp == out
        out = apply_pass(inp, p)
    assert out == pieces(["saya", "sudah", "makan"])


def test_insert_at_sentence_start_uses_bos():
    conv = convert_pair("saya makan", "tolong saya makan", char_encode, V, max_passes=6)
    assert conv is not None
    p0 = conv.passes[0]
    assert p0.bos_append is not None
    out = apply_pass(conv.inputs[0], p0)
    assert out[0] == p0.bos_append


def test_replace_then_insert_defers_correctly():
    # 'dia' replaced AND insertion right after it -> insertion must defer (anchor is $REPLACE)
    conv = convert_pair("dia makan", "mereka sudah makan", char_encode, V, max_passes=9)
    assert conv is not None
    out = pieces(["dia", "makan"])
    for inp, p in zip(conv.inputs, conv.passes):
        assert inp == out
        out = apply_pass(inp, p)
    assert out == pieces(["mereka", "sudah", "makan"])


def test_one_pass_detect_flags():
    p, out = one_pass(["saya", "nasik"], ["saya", "nasi"], char_encode, V)
    assert p.detect[: len(pieces(["saya"]))] == [0] * 4
    assert 1 in p.detect


def test_non_convergence_returns_none():
    # Inserting 'sudah' needs one pass per piece; 2 passes cannot finish it -> drop the example.
    conv = convert_pair("saya makan", "saya sudah makan", char_encode, V, max_passes=2)
    assert conv is None


def test_equal_length_replace_converges_in_one_pass():
    # Every word swapped 1:1 at piece level -> a single $REPLACE pass suffices.
    conv = convert_pair("a b c d e f g h", "z y x w v u t s", char_encode, V, max_passes=1)
    assert conv is not None and len(conv.passes) == 1
    assert apply_pass(conv.inputs[0], conv.passes[0]) == pieces(list("zyxwvuts"))


def test_fuzz_every_pass_round_trips_through_apply_tags():
    """The invariant that matters: inputs[k+1] == apply_tags(inputs[k], passes[k]), and the last
    pass lands exactly on the corrected pieces. Checked against Liquid's own apply_tags."""
    import random

    words = ["saya", "makan", "nasi", "dia", "sudah", "tolong", "lemak", "ayam", "pergi", "ke"]
    rng = random.Random(7)
    for _ in range(300):
        noisy_words = [rng.choice(words) for _ in range(rng.randint(1, 6))]
        corrected_words = list(noisy_words)
        for _ in range(rng.randint(1, 3)):
            op = rng.choice(["insert", "delete", "replace"])
            if op == "insert":
                corrected_words.insert(rng.randint(0, len(corrected_words)), rng.choice(words))
            elif op == "delete" and len(corrected_words) > 1:
                corrected_words.pop(rng.randrange(len(corrected_words)))
            else:
                corrected_words[rng.randrange(len(corrected_words))] = rng.choice(words)
        noisy, corrected = " ".join(noisy_words), " ".join(corrected_words)

        # generous budget: the char stub spends one pass per inserted character (see char_encode)
        conv = convert_pair(noisy, corrected, char_encode, V, max_passes=40)
        assert conv is not None, f"{noisy!r} -> {corrected!r} did not converge"

        cur = pieces(noisy_words)
        for k, (inp, p) in enumerate(zip(conv.inputs, conv.passes)):
            assert inp == cur, f"pass {k} input mismatch for {noisy!r} -> {corrected!r}"
            assert len(p.tags) == len(p.detect) == len(inp), f"pass {k} length mismatch"
            assert all((t != KEEP_ID) == bool(d) for t, d in zip(p.tags, p.detect))
            cur = apply_pass(inp, p)
        assert cur == pieces(corrected_words), f"{noisy!r} -> {corrected!r} landed wrong"
