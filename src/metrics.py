"""Answer scoring, SQuAD-style, adapted to Arabic.

- exact match: the normalized prediction equals the normalized reference
- F1: word overlap between prediction and reference
- contains: the reference answer appears inside the prediction (useful because
  language models often answer with a full sentence around the correct words)
"""

from collections import Counter

from text import normalize, tokenize


def answer_tokens(text: str) -> list[str]:
    return tokenize(text, stem=True)


def exact_match(prediction: str, reference: str) -> float:
    return float(answer_tokens(prediction) == answer_tokens(reference))


def f1(prediction: str, reference: str) -> float:
    pred, ref = answer_tokens(prediction), answer_tokens(reference)
    if not pred or not ref:
        return float(pred == ref)
    common = sum((Counter(pred) & Counter(ref)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def contains(prediction: str, reference: str) -> float:
    ref = normalize(reference)
    return float(bool(ref) and ref in normalize(prediction))


def says_unknown(prediction: str) -> bool:
    return "لا اعرف" in normalize(prediction)
