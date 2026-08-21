"""Package a trained GecTagger as a HF model repo dir compatible with Liquid's inference class."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from lfm_my import ENCODER_NAME

_VENDORED_MODELING = Path(__file__).parent / "modeling_gectagger.py"


def export_model_dir(tagger, tokenizer, out_dir, encoder_name: str = ENCODER_NAME) -> Path:
    """Write model.safetensors + config.json + modeling_gectagger.py + tokenizer files."""
    from safetensors.torch import save_file

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # safetensors refuses to write two keys backed by one storage, and real encoders commonly
    # tie their input embedding to an output head. Clone the duplicates so every key lands in
    # the file independently -- the foreign loader then needs no tying logic of its own.
    sd, seen, shared = {}, {}, []
    for k, v in tagger.state_dict().items():
        v = v.contiguous()
        if v.data_ptr() in seen:
            shared.append((seen[v.data_ptr()], k))
            v = v.clone()
        else:
            seen[v.data_ptr()] = k
        sd[k] = v
    if shared:
        print(f"[export] untied {len(shared)} shared tensor(s): "
              + ", ".join(f"{a} <-> {b}" for a, b in shared))
    save_file(sd, out_dir / "model.safetensors")

    V = tagger.vocab_size
    cfg = {
        "architectures": ["GecTaggerForGEC"],
        "model_type": "gec_tagger",
        "auto_map": {
            "AutoConfig": "modeling_gectagger.GecTaggerConfig",
            "AutoModel": "modeling_gectagger.GecTaggerForGEC",
        },
        "encoder_name": encoder_name,
        "num_tags": 2 + 2 * V,
        "hidden_size": tagger.detect_head.in_features,
        "tie_replace": True,
        "multi_head": False,
        "aux_loss_weight": 0.5,
        "use_swap": False,
        "qat_applied": False,
        "qat_group_size": 32,
        "dropout": 0.1,
        "torch_dtype": "float32",
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))

    shutil.copyfile(_VENDORED_MODELING, out_dir / "modeling_gectagger.py")

    try:
        tokenizer.save_pretrained(out_dir)
    except Exception as e:                                  # stub tokenizers in tests
        print(f"[export] tokenizer.save_pretrained skipped: {e}")

    verify_export(out_dir, tagger)
    return out_dir


def verify_export(out_dir, tagger) -> None:
    """Re-read what was written: every state_dict key must be present with the right shape, and
    config.json must not carry `num_labels`. A silently partial export only surfaces at load."""
    from safetensors import safe_open

    out_dir = Path(out_dir)
    with safe_open(out_dir / "model.safetensors", framework="pt") as f:
        written = {k: list(f.get_slice(k).get_shape()) for k in f.keys()}
    expected = {k: list(v.shape) for k, v in tagger.state_dict().items()}
    missing = sorted(set(expected) - set(written))
    extra = sorted(set(written) - set(expected))
    if missing or extra:
        raise RuntimeError(f"export key mismatch: missing={missing} extra={extra}")
    bad = {k: (expected[k], written[k]) for k in expected if written[k] != expected[k]}
    if bad:
        raise RuntimeError(f"export shape mismatch: {bad}")

    cfg = json.loads((out_dir / "config.json").read_text())
    if "num_labels" in cfg:
        raise RuntimeError("config.json must not contain num_labels (breaks PretrainedConfig)")
    if cfg["num_tags"] != 2 + 2 * tagger.vocab_size:
        raise RuntimeError(f"num_tags {cfg['num_tags']} != 2 + 2*{tagger.vocab_size}")


def push_model_repo(local_dir, repo_id: str, private: bool = True) -> str:
    """Create the HF model repo and upload the export dir (requires HF_TOKEN in env)."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(local_dir), repo_id=repo_id, repo_type="model")
    return f"https://huggingface.co/{repo_id}"
