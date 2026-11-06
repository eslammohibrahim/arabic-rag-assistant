"""FastAPI service: ask a question in Arabic, get an answer with the paragraphs it came from.

Run (Ollama must be running with the model pulled: `ollama pull qwen2.5:1.5b`):
    uvicorn app.main:app
Then open http://127.0.0.1:8000
"""

import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import llm  # noqa: E402
from data import load  # noqa: E402
from eval_generation import build_chosen_retriever  # noqa: E402
from rag import RAG  # noqa: E402

state: dict = {}


def build_rag() -> RAG:
    passages, _ = load()
    choice = json.loads((ROOT / "reports" / "retrieval_results.json").read_text(encoding="utf-8"))
    return RAG(build_chosen_retriever(passages, choice), passages, k=3)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if "rag" not in state:
        state["rag"] = build_rag()
    yield


app = FastAPI(
    title="Arabic RAG Assistant",
    description="Answers Arabic questions from a collection of Arabic Wikipedia paragraphs, with sources.",
    version="1.0.0",
    lifespan=lifespan,
)


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500, examples=["كم يبلغ طول نهر النيل؟"])


class Source(BaseModel):
    passage_id: int
    title: str
    text: str
    score: float


class AnswerResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "retriever_loaded": "rag" in state, "llm_available": llm.is_available()}


@app.get("/search", response_model=list[Source])
def search(q: str, k: int = 3):
    """Retrieval only: the most relevant paragraphs, no language model needed."""
    if not q.strip():
        raise HTTPException(status_code=400, detail="Empty query.")
    state["rag"].k = max(1, min(k, 10))
    sources = state["rag"].retrieve(q)
    state["rag"].k = 3
    return sources


@app.post("/ask", response_model=AnswerResponse)
def ask(body: Question):
    try:
        result = state["rag"].answer(body.question.strip())
    except Exception as exc:  # the language model server is down or failed
        raise HTTPException(status_code=503, detail=f"Language model unavailable: {exc.__class__.__name__}")
    return AnswerResponse(answer=result.answer, sources=result.sources)
