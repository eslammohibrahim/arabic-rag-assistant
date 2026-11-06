"""Three ways to find the paragraphs that answer a question."""

import numpy as np
from rank_bm25 import BM25Okapi

from text import tokenize


class BM25Retriever:
    """Keyword search: scores paragraphs by shared (normalized, lightly stemmed) words."""

    name = "BM25 (keywords)"

    def __init__(self, texts: list[str], stem: bool = True):
        self.stem = stem
        self.bm25 = BM25Okapi([tokenize(t, stem) for t in texts])

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query, self.stem))
        top = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top]


class DenseRetriever:
    """Semantic search: compares meaning with a multilingual embedding model."""

    def __init__(self, texts: list[str], model_name: str, query_prefix: str = "", passage_prefix: str = ""):
        from sentence_transformers import SentenceTransformer

        self.name = model_name.split("/")[-1]
        self.model = SentenceTransformer(model_name, device="cpu")
        self.query_prefix = query_prefix
        self.embeddings = self.model.encode(
            [passage_prefix + t for t in texts], batch_size=32, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        q = self.model.encode([self.query_prefix + query], normalize_embeddings=True, convert_to_numpy=True)[0]
        scores = self.embeddings @ q
        top = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top]


class HybridRetriever:
    """Reciprocal rank fusion: combines rankings, so a paragraph ranked well by either method rises."""

    def __init__(self, retrievers: list, k_rrf: int = 60, depth: int = 50):
        self.retrievers = retrievers
        self.k_rrf = k_rrf
        self.depth = depth
        self.name = "Hybrid: " + " + ".join(r.name for r in retrievers)

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        fused: dict[int, float] = {}
        for r in self.retrievers:
            for rank, (doc, _) in enumerate(r.search(query, self.depth)):
                fused[doc] = fused.get(doc, 0.0) + 1.0 / (self.k_rrf + rank + 1)
        return sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]


# Embedding models compared (all multilingual, all small enough for a laptop CPU).
DENSE_MODELS = {
    "intfloat/multilingual-e5-small": {"query_prefix": "query: ", "passage_prefix": "passage: "},
    "intfloat/multilingual-e5-base": {"query_prefix": "query: ", "passage_prefix": "passage: "},
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": {},
}


def recall_mrr(retriever, questions, ks=(1, 3, 5, 10)) -> dict:
    hits = {k: 0 for k in ks}
    rr = 0.0
    for q, gold in zip(questions["question"], questions["passage_id"]):
        ranked = [doc for doc, _ in retriever.search(q, max(ks))]
        for k in ks:
            hits[k] += gold in ranked[:k]
        rr += 1.0 / (ranked.index(gold) + 1) if gold in ranked else 0.0
    n = len(questions)
    out = {f"recall@{k}": round(hits[k] / n, 4) for k in ks}
    out["mrr@10"] = round(rr / n, 4)
    return out
