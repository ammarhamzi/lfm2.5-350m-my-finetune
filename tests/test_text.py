from lfm_my.text import tokenize, detok, diff_segments


def test_tokenize_spaces_punctuation():
    assert tokenize("Dia pergi ke pasar, bukan?") == "Dia pergi ke pasar , bukan ?"
    assert tokenize("  Sudah   makan. ") == "Sudah makan ."
    assert tokenize("Tiada tanda") == "Tiada tanda"


def test_tokenize_keeps_malay_affixes_joined():
    assert tokenize("Buku itu dibaca oleh Ali.") == "Buku itu dibaca oleh Ali ."


def test_detok_reattaches():
    assert detok("Dia pergi ke pasar , bukan ?") == "Dia pergi ke pasar, bukan?"
    assert detok("Sudah makan .") == "Sudah makan."


def test_roundtrip_natural_malay():
    text = "Saya tidak tahu, dia sudah pergi ke sekolah."
    assert detok(tokenize(text)) == text


def test_diff_segments_kinds():
    seg = diff_segments("dia tidak arah pinjam buku", "dia tidak akan pinjam buku")
    assert seg == [
        {"text": "dia tidak", "kind": "keep"},
        {"text": "arah", "kind": "del"},
        {"text": "akan", "kind": "edit"},
        {"text": "pinjam buku", "kind": "keep"},
    ]


def test_diff_segments_insert_delete():
    seg = diff_segments("saya makan", "saya sudah makan nasi")
    assert {"text": "sudah", "kind": "edit"} in seg
    assert {"text": "nasi", "kind": "edit"} in seg
    assert all(s["kind"] != "del" for s in seg)
    seg = diff_segments("saya sudah makan", "saya makan")
    assert {"text": "sudah", "kind": "del"} in seg


def test_diff_segments_identical():
    assert diff_segments("bersih dan kemas", "bersih dan kemas") == [
        {"text": "bersih dan kemas", "kind": "keep"}
    ]
