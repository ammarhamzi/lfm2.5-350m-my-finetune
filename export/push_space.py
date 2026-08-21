"""Assemble the flat HF Space repo from app/ and push it.

Usage:
    python export/push_space.py --space-repo USER/lfm-my-spellchecker \
        --model-repo USER/lfm-malay-spellchecker [--private]
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app"

README_TEMPLATE = """---
title: Bahasa Malaysia spellchecker
emoji: "\U0001F4DD"
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

Bahasa Malaysia spelling & grammar checker — GECToR-style tagger fine-tuned from
LiquidAI/LFM2.5-Encoder-350M. Model: [{model_repo}](https://huggingface.co/{model_repo}).
"""


MARKER = ".space_build"          # written by us; proof a dir is ours to delete


def build_space_dir(out_dir, model_repo: str) -> Path:
    out_dir = Path(out_dir)
    if out_dir.exists():
        # Only ever delete a directory this function created. Pointing it at a real directory
        # by mistake would otherwise wipe it without warning.
        if any(out_dir.iterdir()) and not (out_dir / MARKER).exists():
            raise RuntimeError(
                f"{out_dir} is not empty and was not built by build_space_dir "
                f"(no {MARKER}); refusing to delete it")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    (out_dir / MARKER).write_text("")
    shutil.copytree(APP / "static", out_dir / "static")
    shutil.copyfile(APP / "Dockerfile", out_dir / "Dockerfile")
    shutil.copyfile(APP / "requirements.txt", out_dir / "requirements.txt")

    server_src = (APP / "server.py").read_text()
    server_src, n = re.subn(
        r'(MODEL_ID = os\.environ\.get\("SPELLCHECKER_MODEL", )"[^"]*"',
        rf'\1"{model_repo}"', server_src, count=1)
    # Check the DEFAULT specifically -- `model_repo in server_src` would also be satisfied by
    # the string turning up in a comment while the real default stayed pointed at USER/...
    if n != 1:
        raise RuntimeError("MODEL_ID default rewrite did not match app/server.py")
    found = re.search(r'MODEL_ID = os\.environ\.get\("SPELLCHECKER_MODEL", "([^"]*)"', server_src)
    if not found or found.group(1) != model_repo:
        raise RuntimeError(f"MODEL_ID default is {found and found.group(1)!r}, want {model_repo!r}")
    (out_dir / "server.py").write_text(server_src)

    (out_dir / "README.md").write_text(README_TEMPLATE.format(model_repo=model_repo))
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space-repo", required=True)
    ap.add_argument("--model-repo", required=True)
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    out = build_space_dir(ROOT / "space_build", args.model_repo)
    print(f"built {out}:")
    for p in sorted(out.rglob("*")):
        if p.is_file():
            print("  ", p.relative_to(out))

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.space_repo, repo_type="space", space_sdk="docker",
                    private=args.private, exist_ok=True)
    api.upload_folder(folder_path=str(out), repo_id=args.space_repo, repo_type="space")
    print(f"https://huggingface.co/spaces/{args.space_repo}")


if __name__ == "__main__":
    main()
