import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "export"))

from push_space import build_space_dir


def test_build_space_dir_layout(tmp_path):
    out = build_space_dir(tmp_path / "space", model_repo="someone/lfm-malay-spellchecker")
    for f in ("server.py", "Dockerfile", "requirements.txt", "README.md"):
        assert (out / f).is_file(), f
    assert (out / "static" / "index.html").is_file()
    assert (out / "static" / "style.css").is_file()


def test_model_repo_injected(tmp_path):
    out = build_space_dir(tmp_path / "space", model_repo="someone/lfm-malay-spellchecker")
    src = (out / "server.py").read_text()
    assert '"someone/lfm-malay-spellchecker"' in src


def test_readme_frontmatter(tmp_path):
    out = build_space_dir(tmp_path / "space", model_repo="someone/lfm-malay-spellchecker")
    readme = (out / "README.md").read_text()
    assert readme.startswith("---")
    assert "title: Bahasa Malaysia spellchecker" in readme
    assert "sdk: docker" in readme
    assert "app_port: 7860" in readme


import pytest


def test_model_id_default_is_actually_rewritten(tmp_path):
    """Not just 'the string appears somewhere' -- the env default itself must point at the repo."""
    import re

    out = build_space_dir(tmp_path / "space", model_repo="someone/lfm-malay-spellchecker")
    src = (out / "server.py").read_text()
    m = re.search(r'MODEL_ID = os\.environ\.get\("SPELLCHECKER_MODEL", "([^"]*)"', src)
    assert m and m.group(1) == "someone/lfm-malay-spellchecker"
    assert "USER/lfm-malay-spellchecker" not in src


def test_refuses_to_wipe_a_directory_it_did_not_build(tmp_path):
    victim = tmp_path / "not_a_build_dir"
    victim.mkdir()
    (victim / "important.txt").write_text("do not delete")
    with pytest.raises(RuntimeError, match="refusing to delete"):
        build_space_dir(victim, model_repo="someone/lfm-malay-spellchecker")
    assert (victim / "important.txt").read_text() == "do not delete"


def test_rebuild_over_its_own_output_is_allowed(tmp_path):
    out = build_space_dir(tmp_path / "space", model_repo="someone/x")
    again = build_space_dir(tmp_path / "space", model_repo="someone/y")
    assert (again / "server.py").is_file()
    assert '"someone/y"' in (again / "server.py").read_text()
