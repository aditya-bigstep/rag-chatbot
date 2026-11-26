"""A small BM25 keyword scorer, used for the keyword half of hybrid retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter

_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def bm25_scores(query: str, texts: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    docs = [tokenize(text) for text in texts]
    if not docs:
        return []
    average = sum(len(doc) for doc in docs) / len(docs) or 1.0
    doc_freq: Counter[str] = Counter()
    for doc in docs:
        doc_freq.update(set(doc))

    scores = []
    for doc in docs:
        counts = Counter(doc)
        score = 0.0
        for term in set(tokenize(query)):
            if term not in counts:
                continue
            idf = math.log(1 + (len(docs) - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            tf = counts[term]
            score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(doc) / average))
        scores.append(score)
    return scores
