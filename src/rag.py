"""The RAG pipeline: retrieve paragraphs, then ask the language model to answer from them only."""

from dataclasses import dataclass

import pandas as pd

import llm

SYSTEM_RAG = (
    "أنت مساعد يجيب عن الأسئلة باللغة العربية اعتمادا على النصوص المعطاة فقط. "
    "أجب بإجابة قصيرة ومباشرة (كلمات قليلة أو جملة واحدة). "
    "إذا لم تكن الإجابة موجودة في النصوص فقل: لا أعرف."
)
SYSTEM_CLOSED_BOOK = "أجب عن السؤال باللغة العربية بإجابة قصيرة ومباشرة (كلمات قليلة أو جملة واحدة)."


@dataclass
class Answer:
    answer: str
    sources: list[dict]


def build_prompt(question: str, passages: list[str]) -> str:
    context = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
    return f"النصوص:\n{context}\n\nالسؤال: {question}\nالإجابة:"


class RAG:
    def __init__(self, retriever, passages: pd.DataFrame, k: int = 3):
        self.retriever = retriever
        self.passages = passages
        self.k = k

    def retrieve(self, question: str) -> list[dict]:
        hits = self.retriever.search(question, self.k)
        rows = self.passages.set_index("passage_id")
        return [
            {"passage_id": pid, "title": rows.at[pid, "title"], "text": rows.at[pid, "text"],
             "score": round(score, 4)}
            for pid, score in hits
        ]

    def answer(self, question: str) -> Answer:
        sources = self.retrieve(question)
        text = llm.generate(build_prompt(question, [s["text"] for s in sources]), system=SYSTEM_RAG)
        return Answer(answer=text, sources=sources)


def answer_closed_book(question: str) -> str:
    return llm.generate(f"السؤال: {question}\nالإجابة:", system=SYSTEM_CLOSED_BOOK)


def answer_with_passages(question: str, passages: list[str]) -> str:
    return llm.generate(build_prompt(question, passages), system=SYSTEM_RAG)
