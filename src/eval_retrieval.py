"""Compare retrieval methods. The best one is chosen on the dev questions, then confirmed on test."""

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import load
from retrieval import DENSE_MODELS, BM25Retriever, DenseRetriever, HybridRetriever, recall_mrr

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "reports" / "figures"


def plot(results: dict, split: str) -> None:
    names = list(results)
    ks = ["recall@1", "recall@3", "recall@5"]
    width = 0.8 / len(names)
    colors = ["#94a3b8", "#cbd5e1", "#f59e0b", "#fbbf24", "#fcd34d", "#2563eb"]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for i, name in enumerate(names):
        xs = np.arange(len(ks)) + (i - (len(names) - 1) / 2) * width
        bars = ax.bar(xs, [results[name][split][k] for k in ks], width, label=name,
                      color=colors[i % len(colors)])
        if name.startswith("Hybrid"):
            ax.bar_label(bars, labels=[f"{results[name][split][k]:.0%}" for k in ks], fontsize=8, padding=2)
    ax.set_xticks(range(len(ks)), ["Correct paragraph in top 1", "in top 3", "in top 5"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Share of questions")
    ax.set_title(f"Retrieval quality on {split} questions (460 paragraphs)", loc="left")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "retrieval.png", dpi=120)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    passages, questions = load()
    texts = passages["text"].tolist()
    dev, test = questions[questions.split == "dev"], questions[questions.split == "test"]
    print(f"{len(passages)} paragraphs, {len(dev)} dev questions, {len(test)} test questions\n")

    retrievers = {
        "BM25, no stemming": BM25Retriever(texts, stem=False),
        "BM25 + light stemming": BM25Retriever(texts, stem=True),
    }
    for model_name, prefixes in DENSE_MODELS.items():
        start = time.time()
        r = DenseRetriever(texts, model_name, **prefixes)
        retrievers[f"Dense: {r.name}"] = r
        print(f"Encoded paragraphs with {r.name} in {time.time() - start:.0f}s")

    results = {}
    for name, r in retrievers.items():
        results[name] = {"dev": recall_mrr(r, dev)}
        print(f"  {name:<52} dev {results[name]['dev']}")

    # Hybrid of the best keyword method and the best dense model, both picked on dev
    best_bm25 = max([n for n in results if n.startswith("BM25")], key=lambda n: results[n]["dev"]["mrr@10"])
    best_dense = max([n for n in results if n.startswith("Dense")], key=lambda n: results[n]["dev"]["mrr@10"])
    hybrid_name = f"Hybrid: {best_bm25} + {best_dense.removeprefix('Dense: ')}"
    retrievers[hybrid_name] = HybridRetriever([retrievers[best_bm25], retrievers[best_dense]])
    results[hybrid_name] = {"dev": recall_mrr(retrievers[hybrid_name], dev)}
    print(f"  {hybrid_name:<52} dev {results[hybrid_name]['dev']}")

    chosen = max(results, key=lambda n: results[n]["dev"]["mrr@10"])
    print(f"\nChosen on dev (by MRR@10): {chosen}\n")

    for name, r in retrievers.items():
        results[name]["test"] = recall_mrr(r, test)
        print(f"  {name:<52} test {results[name]['test']}")

    plot(results, "test")
    out = {"chosen": chosen, "best_bm25": best_bm25, "best_dense": best_dense, "results": results}
    (ROOT / "reports" / "retrieval_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSaved reports/retrieval_results.json and figures/retrieval.png")


if __name__ == "__main__":
    main()
