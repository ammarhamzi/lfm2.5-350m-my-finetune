"""End-to-end smoke: corpus -> pairs -> dataset -> batches -> train step -> checkpoint -> export.

Mirrors the notebook's cell sequence on CPU with a tiny model. Each task is unit-tested in
isolation; this guards the seams between them, which is where a Colab run would actually break.
"""
import json
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lfm_my.data_build import build_from_lines
from lfm_my.dataset import GecDataset, collate_fn
from lfm_my.errors import InjectorConfig
from lfm_my.export import export_model_dir
from lfm_my.model import GecTagger
from lfm_my.train import (aggregate_metrics, compute_loss, load_checkpoint, metrics_for,
                          save_checkpoint)
from tests.test_dataset import CharTokenizer
from tests.test_model import TinyEncoder

FIXTURE = Path(__file__).parent / "fixtures" / "malay_corpus.txt"
V = 1000


def test_full_pipeline_runs_and_loss_decreases(tmp_path):
    # 1. corpus -> JSONL pairs
    stats = build_from_lines(FIXTURE.read_text(encoding="utf-8").splitlines(), tmp_path / "ms",
                             InjectorConfig(seed=0, p_clean=0.2), pairs_per_sentence=3,
                             val_size=4, seed=0)
    assert stats["train_pairs"] > 0 and stats["val_pairs"] == 4

    # 2. JSONL -> dataset -> padded batches
    tok = CharTokenizer()
    pairs = [json.loads(l) for l in (tmp_path / "ms-train.jsonl").read_text().splitlines()]
    ds = GecDataset(pairs, tok, V, max_len=64)
    assert len(ds) > 0
    dl = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=partial(collate_fn, pad_id=0))

    # 3. train a few steps
    torch.manual_seed(0)
    model = GecTagger(TinyEncoder(V, 16), vocab_size=V, hidden_size=16)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    first, last, mets = None, None, []
    for _ in range(3):
        for batch in dl:
            out = model(batch["input_ids"], batch["attention_mask"])
            loss, _, _ = compute_loss(out, batch["label_targets"], batch["detect_targets"])
            opt.zero_grad(); loss.backward(); opt.step()
            mets.append(metrics_for(out, batch["label_targets"], batch["detect_targets"]))
            last = float(loss.detach())
            first = first if first is not None else last
    assert torch.isfinite(torch.tensor(last)) and last < first

    # 4. metrics pool without NaN leaking out of the all-KEEP batches
    agg = aggregate_metrics(mets)
    assert agg["n_edits"] > 0
    assert 0.0 <= agg["label_acc_nokeep"] <= 1.0
    assert 0.0 <= agg["detect_acc"] <= 1.0

    # 5. checkpoint round-trip
    save_checkpoint(tmp_path / "ck.pt", model, opt, step=7, best_val=last)
    fresh = GecTagger(TinyEncoder(V, 16), vocab_size=V, hidden_size=16)
    assert load_checkpoint(tmp_path / "ck.pt", fresh)["step"] == 7
    for (k1, v1), (k2, v2) in zip(model.state_dict().items(), fresh.state_dict().items()):
        assert k1 == k2 and torch.equal(v1, v2)

    # 6. export the trained weights
    out_dir = export_model_dir(fresh, tok, tmp_path / "export", encoder_name="fake/encoder")
    assert (out_dir / "model.safetensors").exists()
    assert (out_dir / "modeling_gectagger.py").exists()
    cfg = json.loads((out_dir / "config.json").read_text())
    assert cfg["num_tags"] == 2 + 2 * V and "num_labels" not in cfg
