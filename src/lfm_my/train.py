"""Loss, metrics, checkpointing for the GEC tagger (training loop lives in the notebook)."""
from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn.functional as F

from lfm_my.modeling_gectagger import KEEP_ID


def compute_loss(out: dict, label_targets: torch.Tensor, detect_targets: torch.Tensor,
                 aux_loss_weight: float = 0.5):
    """label_loss + aux_loss_weight * detect_loss, fp32 CE, ignore_index=-100."""
    label_logits = out["label_logits"].float()
    detect_logits = out["detect_logits"].float()
    label_loss = F.cross_entropy(label_logits.reshape(-1, label_logits.size(-1)),
                                 label_targets.reshape(-1), ignore_index=-100)
    detect_loss = F.cross_entropy(detect_logits.reshape(-1, 2), detect_targets.reshape(-1),
                                  ignore_index=-100)
    return label_loss + aux_loss_weight * detect_loss, label_loss, detect_loss


@torch.no_grad()
def metrics_for(out: dict, label_targets: torch.Tensor, detect_targets: torch.Tensor) -> dict:
    """label accuracy over non-KEEP targets (excl. padding), detect accuracy over all non-padding."""
    label_pred = out["label_logits"].argmax(-1)
    detect_pred = out["detect_logits"].argmax(-1)
    valid = label_targets != -100
    edit = valid & (label_targets != KEEP_ID)
    n_edits = int(edit.sum())
    n_valid = int(valid.sum())
    # NaN, not 1.0, when the set is empty: an all-$KEEP batch (every clean pair produces one)
    # has no non-KEEP target to score, and calling that 100% inflates any naive average.
    # Aggregate with aggregate_metrics(), which weights by n_edits.
    nan = float("nan")
    label_acc = float((label_pred[edit] == label_targets[edit]).float().mean()) if n_edits else nan
    detect_acc = float((detect_pred[valid] == detect_targets[valid]).float().mean()) if n_valid else nan
    return {"label_acc_nokeep": label_acc, "detect_acc": detect_acc,
            "n_edits": n_edits, "n_valid": n_valid}


def aggregate_metrics(batch_metrics: list) -> dict:
    """Correctly pool per-batch metrics_for() dicts: each accuracy is weighted by the number of
    positions it was computed over, so empty batches contribute nothing instead of 1.0."""
    edits = sum(m["n_edits"] for m in batch_metrics)
    valid = sum(m["n_valid"] for m in batch_metrics)
    label = sum(m["label_acc_nokeep"] * m["n_edits"] for m in batch_metrics if m["n_edits"])
    detect = sum(m["detect_acc"] * m["n_valid"] for m in batch_metrics if m["n_valid"])
    return {"label_acc_nokeep": (label / edits) if edits else float("nan"),
            "detect_acc": (detect / valid) if valid else float("nan"),
            "n_edits": edits, "n_valid": valid}


def save_checkpoint(path, model, optimizer, step: int, best_val: float) -> None:
    """Written via a temp file + atomic replace: a free-tier Colab runtime can be preempted
    mid-write, and overwriting the checkpoint in place would destroy the last good one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step, "best_val": best_val}, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model, optimizer=None) -> dict:
    # weights_only=True is the safe default and suffices here: a checkpoint holds only
    # tensors and plain scalars, never arbitrary pickled objects.
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return {"step": ckpt.get("step", 0), "best_val": ckpt.get("best_val", float("inf"))}
