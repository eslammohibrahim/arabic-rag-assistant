"""API tests. The language model is replaced by a stub, so these run without Ollama.

Keyword search (BM25) is used for speed; the real service loads the retriever chosen in the evaluation.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import app.main as main  # noqa: E402
import llm  # noqa: E402
from data import load  # noqa: E402
from rag import RAG  # noqa: E402
from retrieval import BM25Retriever  # noqa: E402

PASSAGES, QUESTIONS = load()


@pytest.fixture(scope="module")
def client():
    main.state["rag"] = RAG(BM25Retriever(PASSAGES["text"].tolist()), PASSAGES, k=3)
    with TestClient(main.app) as c:
        yield c
    main.state.clear()


def test_search_finds_the_right_paragraph(client):
    q = QUESTIONS.iloc[0]
    r = client.get("/search", params={"q": q.question, "k": 5})
    assert r.status_code == 200
    assert q.passage_id in [s["passage_id"] for s in r.json()]


def test_ask_returns_answer_and_sources(client, monkeypatch):
    seen = {}

    def fake_generate(prompt, system="", **kwargs):
        seen["prompt"] = prompt
        return "إجابة تجريبية"

    monkeypatch.setattr(llm, "generate", fake_generate)
    q = QUESTIONS.iloc[0]
    r = client.post("/ask", json={"question": q.question})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "إجابة تجريبية"
    assert len(body["sources"]) == 3
    # The retrieved paragraphs and the question are passed to the model
    assert q.question in seen["prompt"]
    assert body["sources"][0]["text"] in seen["prompt"]


def test_ask_rejects_too_short_question(client):
    assert client.post("/ask", json={"question": "؟"}).status_code == 422


def test_ask_returns_503_when_model_is_down(client, monkeypatch):
    def broken(*args, **kwargs):
        raise ConnectionError("down")

    monkeypatch.setattr(llm, "generate", broken)
    r = client.post("/ask", json={"question": "ما هي عاصمة مصر؟"})
    assert r.status_code == 503


def test_index_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "مساعد الأسئلة العربي" in r.text
