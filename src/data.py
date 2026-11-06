"""ARCD (Arabic Reading Comprehension Dataset): questions about Arabic Wikipedia paragraphs.

The paragraphs become the searchable document collection; each question's own paragraph is
the "correct" document for retrieval, and its answer span is the reference answer.
"""

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
URL = "https://huggingface.co/api/datasets/hsseinmz/arcd/parquet/plain_text/{split}/0.parquet"


def download(split: str) -> Path:
    path = RAW_DIR / f"arcd_{split}.parquet"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(URL.format(split=split), path)
    return path


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (passages, questions).

    passages:  passage_id, title, text
    questions: question, answer, passage_id, split ('dev' = ARCD train, 'test' = ARCD validation)
    """
    frames = []
    for split, name in [("train", "dev"), ("validation", "test")]:
        df = pd.read_parquet(download(split))
        df["split"] = name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    passages = df[["title", "context"]].drop_duplicates("context").reset_index(drop=True)
    passages["passage_id"] = range(len(passages))
    passages = passages.rename(columns={"context": "text"})[["passage_id", "title", "text"]]

    df = df.merge(passages[["passage_id", "text"]], left_on="context", right_on="text")
    df["answer"] = df["answers"].map(lambda a: str(a["text"][0]))
    df["question"] = df["question"].str.strip().str.lstrip("-").str.strip()
    questions = (
        df.drop_duplicates("question")[["question", "answer", "passage_id", "split", "title"]]
        .reset_index(drop=True)
    )
    return passages, questions
