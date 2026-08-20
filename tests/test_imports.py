import lfm_my
from lfm_my.modeling_gectagger import (
    KEEP_ID, DELETE_ID, INCORRECT, apply_tags, id_to_tag, GecTaggerConfig,
)


def test_constants():
    assert KEEP_ID == 0
    assert DELETE_ID == 1
    assert INCORRECT == 1


def test_config_defaults_match_liquid():
    cfg = GecTaggerConfig(encoder_name=lfm_my.ENCODER_NAME)
    assert cfg.hidden_size == 1024
    assert cfg.tie_replace is True
    assert cfg.use_swap is False
    assert cfg.aux_loss_weight == 0.5


def test_tag_space_ids():
    V = 64400
    assert id_to_tag(0, V) == "$KEEP"
    assert id_to_tag(1, V) == "$DELETE"
    assert id_to_tag(2, V) == "$REPLACE_0"
    assert id_to_tag(2 + V, V) == "$APPEND_0"
    assert id_to_tag(2 + 2 * V - 1, V) == f"$APPEND_{V - 1}"
