import torch

from lfm_my.dataset import GecDataset, collate_fn
from lfm_my.tags import append_id, replace_id

V = 1000
BOS = 1
PAD = 0


class CharTokenizer:
    """Stub: one piece per char (id = 10 + ord), whitespace pre-tokenization."""
    pad_token_id, bos_token_id = PAD, BOS

    def encode(self, text, add_special_tokens=False):
        ids = []
        for w in text.split():
            ids.extend(10 + ord(c) for c in w)
        return ids

    def get_vocab(self):
        return {f"c{i}": i for i in range(V)}


PAIRS = [
    {"noisy": "saya makan nasik", "correct": "saya makan nasi"},
    {"noisy": "dia pergi ke pasar", "correct": "dia pergi ke pasar"},      # clean
    {"noisy": "saya makan", "correct": "saya sudah makan"},                # insert
]


def test_dataset_sizes_and_shapes():
    ds = GecDataset(PAIRS, CharTokenizer(), V, max_len=128)
    assert ds.skipped == 0
    # 1 (single-pass replace) + 1 (clean -> one all-$KEEP sample) + 5: inserting 'sudah' is a
    # 5-piece insert under the char stub, and one $APPEND lands one piece per pass.
    assert len(ds) == 1 + 1 + 5
    s = ds[0]
    assert s["input_ids"][0] == BOS
    assert len(s["label_targets"]) == len(s["input_ids"])
    assert len(s["detect_targets"]) == len(s["input_ids"])


def test_clean_pair_all_keep():
    ds = GecDataset(PAIRS, CharTokenizer(), V)
    s = ds[1]
    assert all(t == 0 for t in s["label_targets"])
    assert all(t == 0 for t in s["detect_targets"])


def test_replace_target_ids():
    ds = GecDataset([{"noisy": "ab", "correct": "ac"}], CharTokenizer(), V)
    s = ds[0]
    # pieces: a=107 b=108 -> BOS kept, 'a' kept, 'b' replaced by 'c'(109)
    assert s["label_targets"][-1] == replace_id(10 + ord("c"))
    assert s["detect_targets"][-1] == 1
    assert s["detect_targets"][1] == 0          # 'a' kept


def test_bos_insert_target():
    ds = GecDataset([{"noisy": "ab", "correct": "z ab"}], CharTokenizer(), V)
    s = ds[0]
    assert s["label_targets"][0] == append_id(10 + ord("z"), V)
    assert s["detect_targets"][0] == 1


def test_multi_pass_expands_samples():
    pairs = [{"noisy": "ab", "correct": "ab xyzw"}]        # 4-piece insert after short anchor
    ds = GecDataset(pairs, CharTokenizer(), V)
    assert len(ds) >= 2


def test_collate_padding():
    ds = GecDataset(PAIRS, CharTokenizer(), V)
    batch = collate_fn([ds[0], ds[1]], PAD)
    assert batch["input_ids"].shape == batch["attention_mask"].shape \
        == batch["label_targets"].shape == batch["detect_targets"].shape
    lens = [len(ds[0]["input_ids"]), len(ds[1]["input_ids"])]
    row = 0 if lens[0] < lens[1] else 1
    shorter = min(lens)
    assert (batch["attention_mask"][row, shorter:] == 0).all()
    assert (batch["label_targets"][row, shorter:] == -100).all()
    assert (batch["input_ids"][row, shorter:] == PAD).all()


class ContextTokenizer(CharTokenizer):
    """GPT-2-style: a word following another IN THE SAME CALL gets a space marker, so
    encode('ab cd') != encode('ab') + encode('cd')."""

    def encode(self, text, add_special_tokens=False):
        ids = []
        for i, w in enumerate(text.split()):
            if i:
                ids.append(900)
            ids.extend(10 + ord(c) for c in w)
        return ids


def test_clean_and_edited_pairs_share_one_encoding():
    """A clean pair and an edited pair must tokenize the same sentence identically -- otherwise
    the ~17% of examples kept clean train the model on a different tokenization."""
    ds = GecDataset([{"noisy": "ab cd", "correct": "ab cd"},
                     {"noisy": "ab cx", "correct": "ab cd"}], ContextTokenizer(), V)
    clean, edited = ds[0]["input_ids"], ds[1]["input_ids"]
    assert len(clean) == len(edited)
    assert clean[:-1] == edited[:-1]           # differ only where the injected error sits


def test_sample_count_matches_conversion_passes():
    from lfm_my.convert import convert_pair

    tok = CharTokenizer()
    enc = lambda w: tok.encode(w)              # noqa: E731
    for pair in PAIRS:
        ds = GecDataset([pair], tok, V)
        conv = convert_pair(pair["noisy"], pair["correct"], enc, V)
        assert len(ds) == max(len(conv.passes), 1)


def test_targets_align_with_inputs_for_every_sample():
    ds = GecDataset(PAIRS, CharTokenizer(), V)
    for i in range(len(ds)):
        s = ds[i]
        assert s["input_ids"][0] == BOS
        assert len(s["input_ids"]) == len(s["label_targets"]) == len(s["detect_targets"])
        assert all((t != 0) == bool(d) for t, d in zip(s["label_targets"], s["detect_targets"]))
