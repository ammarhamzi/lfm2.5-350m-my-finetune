"""FastAPI backend for the Bahasa Malaysia spellchecker demo (Docker Space).

    uvicorn server:app --host 0.0.0.0 --port 7860
"""
import difflib
import os
import re

import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import AutoModel


def _effective_cpus():
    """Cores actually granted to this container (cgroup quota), not the host core count."""
    try:                                                          # cgroup v2
        raw = open("/sys/fs/cgroup/cpu.max").read().split()
        if raw and raw[0] != "max":
            return max(1, round(int(raw[0]) / int(raw[1])))
    except (OSError, ValueError):
        pass
    try:                                                          # cgroup v1
        q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
        p = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        if q > 0:
            return max(1, q // p)
    except (OSError, ValueError):
        pass
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return os.cpu_count() or 1


_CPUS = _effective_cpus()
torch.set_num_threads(_CPUS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

MODEL_ID = os.environ.get("SPELLCHECKER_MODEL", "USER/lfm-malay-spellchecker")
MODEL_REV = os.environ.get("SPELLCHECKER_REVISION", "main")
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

print(f"[server] loading {MODEL_ID}@{MODEL_REV} on {_CPUS} CPU thread(s) ...", flush=True)
_model = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REV, trust_remote_code=True,
                                   token=os.environ.get("HF_TOKEN")).float().eval()
_mem_bytes = sum(t.numel() * t.element_size() for t in (*_model.parameters(), *_model.buffers()))
_mem_human = (f"{_mem_bytes / 1024**3:.2f} GB" if _mem_bytes >= 1024**3
              else f"{_mem_bytes / 1024**2:.0f} MB")
print(f"[server] model ready ({_mem_human} in memory)", flush=True)


# Malay training form: punctuation spaced apart. No contraction splitting (English-only).
_PUNCT = re.compile(r'([.,!?;:()\[\]{}"«»…])')
_ATTACH_LEFT = re.compile(r"\s+([.,!?;:%)\]}»…])")
_ATTACH_RIGHT = re.compile(r"([(\[{«])\s+")


def tokenize(text: str) -> str:
    text = _PUNCT.sub(r" \1 ", text)
    return re.sub(r"\s+", " ", text).strip()


def detok(text: str) -> str:
    text = _ATTACH_LEFT.sub(r"\1", text)
    text = _ATTACH_RIGHT.sub(r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


_PROBE_IN = "Dia pergi ke sekolah setiap hari."
try:
    _PROBE_OUT = detok(_model.correct([tokenize(_PROBE_IN)], max_iter=3)[0])
    _PROBE_OK = (_PROBE_OUT == _PROBE_IN)          # clean Malay sentence should stay untouched
except Exception as e:                                           # pragma: no cover
    _PROBE_OUT, _PROBE_OK = f"ERROR: {e}", None
print(f"[server] self-test ok={_PROBE_OK}: {_PROBE_IN!r} -> {_PROBE_OUT!r}", flush=True)


def diff_segments(source: str, corrected: str):
    """Word-level diff -> [{text, kind}] segments, kind in {keep, edit, del}."""
    s, c = source.split(), corrected.split()
    seg = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, s, c, autojunk=False).get_opcodes():
        if op == "equal":
            seg.append({"text": " ".join(c[j1:j2]), "kind": "keep"})
        elif op == "insert":
            seg.append({"text": " ".join(c[j1:j2]), "kind": "edit"})
        elif op == "delete":
            seg.append({"text": " ".join(s[i1:i2]), "kind": "del"})
        elif op == "replace":
            seg.append({"text": " ".join(s[i1:i2]), "kind": "del"})
            seg.append({"text": " ".join(c[j1:j2]), "kind": "edit"})
    return seg or [{"text": corrected, "kind": "keep"}]


MAX_CHARS = int(os.environ.get("SPELLCHECKER_MAX_CHARS", "2000"))


class CorrectRequest(BaseModel):
    """Bounded on every field: this endpoint is public on the Space, and the model runs on CPU.
    Unbounded text or max_iter lets one request occupy the whole container indefinitely."""

    text: str = Field(default="", max_length=MAX_CHARS)
    min_error_prob: float = Field(default=0.0, ge=0.0, le=1.0)
    max_iter: int = Field(default=3, ge=1, le=10)


app = FastAPI(title="Bahasa Malaysia spellchecker")


@app.get("/api/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "revision": MODEL_REV,
            "mem_bytes": _mem_bytes, "mem_human": _mem_human,
            "self_test": {"in": _PROBE_IN, "out": _PROBE_OUT, "ok": _PROBE_OK}}


@app.post("/api/correct")
@torch.no_grad()
def correct(req: CorrectRequest):
    text = (req.text or "").strip()
    if not text:
        return {"corrected": "", "segments": [], "changed": False}
    src = tokenize(text)
    out = _model.correct([src], min_error_prob=float(req.min_error_prob),
                         max_iter=int(req.max_iter))[0]
    return {"corrected": detok(out), "segments": diff_segments(src, out), "changed": out != src}


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


app.mount("/", StaticFiles(directory=STATIC), name="static")
