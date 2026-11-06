"""Why does retrieval miss?

1. Questions that name their subject vs those that don't. ARCD questions were written while
   looking at the paragraph, so some depend on it: "بماذا اشتهر؟" doesn't say who "he" is.
2. Overlapping paragraphs. Some ARCD paragraphs are longer versions of others, so returning
   the longer one counts as a miss even though it contains the answer. A lenient metric,
   "answer text appears in the retrieved paragraphs", shows how much this matters.
"""

import json
from pathlib import Path

from data import load
from retrieval import DENSE_MODELS, BM25Retriever, DenseRetriever, HybridRetriever
from text import normalize, tokenize

ROOT = Path(__file__).resolve().parent.parent
E5 = "intfloat/multilingual-e5-small"


def main() -> None:
    passages, questions = load()
    test = questions[questions.split == "test"].copy()
    texts = passages["text"].tolist()
    normalized = [normalize(t) for t in texts]
    bm25 = BM25Retriever(texts, stem=True)
    dense = DenseRetriever(texts, E5, **DENSE_MODELS[E5])
    hybrid = HybridRetriever([bm25, dense])

    contained = sum(
        1 for i, a in enumerate(normalized) for j, b in enumerate(normalized)
        if i != j and len(a) < len(b) and a in b
    )

    def answer_hit(retriever, k):
        hits = 0
        for q, ans in zip(test["question"], test["answer"]):
            a = normalize(ans)
            # An answer that is empty after normalization (only symbols) can't be matched.
            hits += bool(a) and any(a in normalized[d] for d, _ in retriever.search(q, k))
        return round(hits / len(test), 4)

    lenient = {
        name: {f"answer_in_top{k}": answer_hit(r, k) for k in (1, 3, 5)}
        for name, r in [("BM25 + light stemming", bm25), ("Dense: multilingual-e5-small", dense), ("Hybrid", hybrid)]
    }

    test["found_in_top3"] = [
        gold in [doc for doc, _ in hybrid.search(q, 3)]
        for q, gold in zip(test["question"], test["passage_id"])
    ]
    title_words = {pid: set(tokenize(t)) for pid, t in zip(passages["passage_id"], passages["title"])}
    test["names_subject"] = [
        bool(set(tokenize(q)) & title_words[pid]) for q, pid in zip(test["question"], test["passage_id"])
    ]
    test["short"] = test["question"].map(lambda q: len(tokenize(q)) <= 2)

    def summary(column):
        return {str(flag): {"n": int(len(g)), "hit3": round(float(g["found_in_top3"].mean()), 4)}
                for flag, g in test.groupby(column)}

    out = {
        "overall_hit3": round(float(test["found_in_top3"].mean()), 4),
        "mentions_subject": summary("names_subject"),
        "short_questions": summary("short"),
        "paragraphs_contained_in_a_longer_one": contained,
        "answer_text_in_retrieved_paragraphs": lenient,
    }
    print(json.dumps(out, indent=2))
    (ROOT / "reports" / "retrieval_misses.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
