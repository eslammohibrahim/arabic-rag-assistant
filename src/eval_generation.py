"""End-to-end evaluation: does retrieval actually improve the answers?

Three settings, same model, same test questions:
- closed book: the model answers from what it memorized, no paragraphs
- RAG: the model gets the top 3 paragraphs from the chosen retriever
- oracle: the model gets the correct paragraph (an upper bound for the reader)
"""

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import llm
from data import load
from metrics import contains, exact_match, f1, says_unknown
from rag import RAG, answer_closed_book, answer_with_passages
from retrieval import DENSE_MODELS, BM25Retriever, DenseRetriever, HybridRetriever

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "reports" / "figures"
K = 3
SAMPLE = 300  # random test questions; about 15 s each on a laptop CPU (three model calls)
SEED = 42


def build_chosen_retriever(passages, choice: dict):
    texts = passages["text"].tolist()

    def make(name):
        if name.startswith("BM25"):
            return BM25Retriever(texts, stem="stemming" in name)
        model = next(m for m in DENSE_MODELS if m.endswith(name.removeprefix("Dense: ")))
        return DenseRetriever(texts, model, **DENSE_MODELS[model])

    if choice["chosen"].startswith("Hybrid"):
        return HybridRetriever([make(choice["best_bm25"]), make(choice["best_dense"])])
    return make(choice["chosen"])


def score(rows: list[dict], key: str) -> dict:
    n = len(rows)
    return {
        "exact_match": round(sum(exact_match(r[key], r["reference"]) for r in rows) / n, 4),
        "f1": round(sum(f1(r[key], r["reference"]) for r in rows) / n, 4),
        "contains_answer": round(sum(contains(r[key], r["reference"]) for r in rows) / n, 4),
        "says_unknown": round(sum(says_unknown(r[key]) for r in rows) / n, 4),
    }


def plot(summary: dict) -> None:
    settings = ["closed_book", "rag", "oracle"]
    labels = ["No retrieval\n(closed book)", f"RAG\n(top {K} retrieved)", "Correct paragraph given\n(upper bound)"]
    metrics = [("f1", "Word-overlap F1"), ("contains_answer", "Answer contained")]
    colors = {"f1": "#2563eb", "contains_answer": "#93c5fd"}
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    width = 0.36
    for j, (m, label) in enumerate(metrics):
        xs = [i + (j - 0.5) * width for i in range(len(settings))]
        bars = ax.bar(xs, [summary[s][m] for s in settings], width, label=label, color=colors[m])
        fmt = "{:.2f}" if m == "f1" else "{:.0%}"
        ax.bar_label(bars, labels=[fmt.format(summary[s][m]) for s in settings], padding=2, fontsize=9)
    ax.set_xticks(range(len(settings)), labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score on test questions")
    ax.set_title("Answer quality with and without retrieval (qwen2.5 1.5B, local)", loc="left")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "answers.png", dpi=120)


def main(limit: int | None = None) -> None:
    if not llm.is_available():
        raise SystemExit(f"Ollama model '{llm.MODEL}' not found. Run: ollama pull {llm.MODEL}")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    passages, questions = load()
    test = questions[questions.split == "test"].sample(n=SAMPLE, random_state=SEED)
    if limit:
        test = test.head(limit)
    choice = json.loads((ROOT / "reports" / "retrieval_results.json").read_text(encoding="utf-8"))
    rag = RAG(build_chosen_retriever(passages, choice), passages, k=K)
    gold_text = passages.set_index("passage_id")["text"]
    print(f"Retriever: {choice['chosen']}  |  model: {llm.MODEL}  |  {len(test)} test questions")

    rows, times = [], {"closed_book": 0.0, "rag": 0.0, "oracle": 0.0}
    for n, q in enumerate(test.itertuples(), 1):
        t = time.time(); closed = answer_closed_book(q.question); times["closed_book"] += time.time() - t
        t = time.time(); result = rag.answer(q.question); times["rag"] += time.time() - t
        t = time.time(); oracle = answer_with_passages(q.question, [gold_text[q.passage_id]]); times["oracle"] += time.time() - t
        rows.append({
            "question": q.question, "reference": q.answer, "title": q.title,
            "closed_book": closed, "rag": result.answer, "oracle": oracle,
            "retrieved_correct": q.passage_id in [s["passage_id"] for s in result.sources],
        })
        if n % 50 == 0:
            print(f"  {n}/{len(test)} done", flush=True)

    summary = {key: score(rows, key) for key in ["closed_book", "rag", "oracle"]}
    for key in summary:
        summary[key]["seconds_per_question"] = round(times[key] / len(rows), 2)
    # Split RAG results by whether retrieval found the right paragraph
    found = [r for r in rows if r["retrieved_correct"]]
    missed = [r for r in rows if not r["retrieved_correct"]]
    summary["rag_when_retrieval_found_paragraph"] = score(found, "rag") | {"questions": len(found)}
    summary["rag_when_retrieval_missed"] = score(missed, "rag") | {"questions": len(missed)} if missed else {}

    for key, value in summary.items():
        print(f"{key:<36} {value}")
    plot(summary)
    (ROOT / "reports" / "generation_results.json").write_text(
        json.dumps({"model": llm.MODEL, "retriever": choice["chosen"], "k": K, "summary": summary},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    (ROOT / "reports" / "generation_answers.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    print("Saved reports/generation_results.json and generation_answers.jsonl")


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
