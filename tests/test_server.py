import sys
import types
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def client():
    # stub transformers.AutoModel before importing server (no model download in tests)
    fake_tf = types.ModuleType("transformers_fake_stub")

    class FakeModel:
        def correct(self, texts, min_error_prob=0.0, max_iter=3, **kw):
            return [t.replace("nasik", "nasi") for t in texts]

        def parameters(self):
            return iter([__import__("torch").zeros(8)])

        def buffers(self):
            return iter([])

        def float(self):
            return self

        def eval(self):
            return self

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(*a, **k):
            return FakeModel()

    fake_tf.AutoModel = FakeAutoModel
    sys.modules["transformers"] = fake_tf

    sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
    sys.modules.pop("server", None)
    import server                                    # noqa: E402
    from fastapi.testclient import TestClient
    return TestClient(server.app)


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "self_test" in body


def test_correct_diff(client):
    r = client.post("/api/correct", json={"text": "Saya makan nasik."})
    assert r.status_code == 200
    body = r.json()
    assert body["corrected"] == "Saya makan nasi."
    assert body["changed"] is True
    kinds = {(s["text"], s["kind"]) for s in body["segments"]}
    assert ("nasik", "del") in kinds and ("nasi", "edit") in kinds


def test_correct_empty(client):
    assert client.post("/api/correct", json={"text": "  "}).json() == \
        {"corrected": "", "segments": [], "changed": False}


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Bahasa" in r.text


def test_oversized_text_is_rejected(client):
    """The Space runs the model on CPU; an unbounded body would tie up the container."""
    import server

    r = client.post("/api/correct", json={"text": "a " * server.MAX_CHARS})
    assert r.status_code == 422


def test_out_of_range_knobs_are_rejected(client):
    assert client.post("/api/correct",
                       json={"text": "saya", "max_iter": 10_000}).status_code == 422
    assert client.post("/api/correct",
                       json={"text": "saya", "max_iter": 0}).status_code == 422
    assert client.post("/api/correct",
                       json={"text": "saya", "min_error_prob": 5.0}).status_code == 422
    assert client.post("/api/correct",
                       json={"text": "saya", "min_error_prob": -1.0}).status_code == 422


def test_in_range_knobs_are_accepted(client):
    r = client.post("/api/correct",
                    json={"text": "Saya makan nasik.", "max_iter": 10, "min_error_prob": 1.0})
    assert r.status_code == 200
