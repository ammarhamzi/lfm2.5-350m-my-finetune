# lfm-my — Bahasa Malaysia spellchecker

GECToR-style grammatical-error-correction tagger for Bahasa Malaysia, fine-tuned from
[LiquidAI/LFM2.5-Encoder-350M](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M), mirroring
[LiquidAI/LFM2.5-Encoder-350M-Spellchecker](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M-Spellchecker).

- Design spec: `docs/superpowers/specs/2026-08-11-malay-spellchecker-design.md`
- Implementation plan: `docs/superpowers/plans/2026-08-20-malay-spellchecker.md`

## Develop

    uv venv --python 3.12
    uv pip install -e ".[dev]"
    pytest -m "not network"
