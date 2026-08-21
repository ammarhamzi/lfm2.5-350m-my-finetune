import torch

from lfm_my.model import GecTagger
from lfm_my.train import compute_loss, load_checkpoint, metrics_for, save_checkpoint
from tests.test_model import TinyEncoder


def _tiny():
    torch.manual_seed(0)
    return GecTagger(TinyEncoder(64, 16), vocab_size=64, hidden_size=16)


def test_compute_loss_finite_and_decreases():
    m = _tiny()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    ids = torch.randint(0, 64, (2, 6))
    label_t = torch.randint(0, 2, (2, 6))
    det_t = torch.randint(0, 2, (2, 6))
    first = None
    for _ in range(30):
        out = m(ids, torch.ones_like(ids))
        loss, ll, dl = compute_loss(out, label_t, det_t)
        if first is None:
            first = float(loss.detach())
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert torch.isfinite(loss).all()
    assert float(loss.detach()) < first


def test_loss_respects_ignore_index():
    m = _tiny()
    ids = torch.randint(0, 64, (2, 6))
    out = m(ids, torch.ones_like(ids))
    label_t = torch.randint(0, 2, (2, 6))
    det_t = torch.randint(0, 2, (2, 6))
    loss_full, _, _ = compute_loss(out, label_t, det_t)
    partial = label_t.clone()
    partial[:, 3:] = -100
    det_partial = det_t.clone()
    det_partial[:, 3:] = -100
    loss_part, _, _ = compute_loss(out, partial, det_partial)
    assert loss_part != loss_full


def test_metrics_keys_and_ranges():
    m = _tiny()
    ids = torch.randint(0, 64, (2, 6))
    out = m(ids, torch.ones_like(ids))
    label_t = torch.full((2, 6), 0)
    label_t[0, 0] = 3
    det_t = torch.zeros((2, 6), dtype=torch.long)
    met = metrics_for(out, label_t, det_t)
    # n_valid extends the plan's interface: aggregate_metrics needs both counts to weight
    # the two accuracies correctly.
    assert set(met) == {"label_acc_nokeep", "detect_acc", "n_edits", "n_valid"}
    assert met["n_edits"] == 1
    assert 0.0 <= met["label_acc_nokeep"] <= 1.0
    assert 0.0 <= met["detect_acc"] <= 1.0


def test_checkpoint_roundtrip(tmp_path):
    m = _tiny()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    out = m(torch.tensor([[1, 2, 3]]), torch.ones(1, 3, dtype=torch.long))
    loss, _, _ = compute_loss(out, torch.tensor([[0, 1, 2]]), torch.tensor([[0, 1, 0]]))
    loss.backward()
    opt.step()
    p = tmp_path / "ckpt.pt"
    save_checkpoint(p, m, opt, step=42, best_val=1.23)
    m2 = _tiny()
    meta = load_checkpoint(p, m2)
    assert meta["step"] == 42 and meta["best_val"] == 1.23
    for (k1, v1), (k2, v2) in zip(m.state_dict().items(), m2.state_dict().items()):
        assert k1 == k2 and torch.equal(v1, v2)


def test_empty_edit_set_is_not_scored_as_perfect():
    """Every clean pair yields an all-$KEEP sample; scoring that 1.0 inflates validation."""
    import math

    m = _tiny()
    ids = torch.randint(0, 64, (2, 6))
    out = m(ids, torch.ones_like(ids))
    met = metrics_for(out, torch.zeros((2, 6), dtype=torch.long),
                      torch.zeros((2, 6), dtype=torch.long))
    assert met["n_edits"] == 0
    assert math.isnan(met["label_acc_nokeep"])


def test_aggregate_metrics_weights_by_count():
    from lfm_my.train import aggregate_metrics

    batches = [
        {"label_acc_nokeep": float("nan"), "detect_acc": 1.0, "n_edits": 0, "n_valid": 10},
        {"label_acc_nokeep": 1.0, "detect_acc": 1.0, "n_edits": 1, "n_valid": 10},
        {"label_acc_nokeep": 0.0, "detect_acc": 0.0, "n_edits": 3, "n_valid": 10},
    ]
    agg = aggregate_metrics(batches)
    assert agg["n_edits"] == 4
    assert agg["label_acc_nokeep"] == 0.25          # 1 correct of 4 edit positions, not 0.5
    assert abs(agg["detect_acc"] - 2 / 3) < 1e-9


def test_checkpoint_survives_interrupted_write(tmp_path):
    m, opt = _tiny(), None
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    p = tmp_path / "ckpt.pt"
    save_checkpoint(p, m, opt, step=1, best_val=0.5)
    good = p.read_bytes()
    (p.with_name(p.name + ".tmp")).write_bytes(b"truncated garbage")   # a killed write
    assert p.read_bytes() == good
    assert load_checkpoint(p, _tiny())["step"] == 1
