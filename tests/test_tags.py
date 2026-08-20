from lfm_my.modeling_gectagger import apply_tags, id_to_tag
from lfm_my.tags import BASE, append_id, replace_id

V = 64400


def test_tag_id_layout():
    assert BASE == 2
    assert replace_id(0) == 2
    assert replace_id(17) == 19
    assert append_id(0, V) == 2 + V
    assert append_id(17, V) == 2 + V + 17


def test_ids_roundtrip_with_inference_helpers():
    assert id_to_tag(replace_id(42), V) == "$REPLACE_42"
    assert id_to_tag(append_id(42, V), V) == "$APPEND_42"


def test_apply_tags_semantics():
    BOS = 1
    pieces = [BOS, 100, 101, 102, 103]
    tags = ["$KEEP", "$KEEP", "$DELETE", "$REPLACE_55", "$APPEND_66"]
    assert apply_tags(pieces, tags) == [100, 55, 103, 66]


def test_apply_tags_bos_append_inserts_at_start():
    BOS = 1
    pieces = [BOS, 100]
    tags = ["$APPEND_77", "$KEEP"]
    assert apply_tags(pieces, tags) == [77, 100]
