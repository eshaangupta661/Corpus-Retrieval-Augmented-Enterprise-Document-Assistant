"""Rank-based retrieval metrics.

All functions take ``gains``: the relevance grade of each retrieved result in rank
order (0 = irrelevant, 1 = relevant, higher = more relevant).
"""

import math
from collections.abc import Sequence


def reciprocal_rank(gains: Sequence[float]) -> float:
    """1 / rank of the first relevant result, or 0.0 if there is none."""
    for rank, gain in enumerate(gains, start=1):
        if gain > 0:
            return 1.0 / rank
    return 0.0


def hit_at_k(gains: Sequence[float], k: int) -> float:
    """1.0 if any relevant result appears in the top k."""
    return 1.0 if any(g > 0 for g in gains[:k]) else 0.0


def recall_at_k(gains: Sequence[float], k: int, num_relevant: int) -> float:
    """Fraction of the labelled relevant items found in the top k."""
    if num_relevant <= 0:
        return 0.0
    found = sum(1 for g in gains[:k] if g > 0)
    return min(found / num_relevant, 1.0)


def dcg_at_k(gains: Sequence[float], k: int) -> float:
    """Discounted cumulative gain with the (2^g - 1) / log2(rank + 1) formulation."""
    return sum(
        (2**gain - 1) / math.log2(rank + 1)
        for rank, gain in enumerate(gains[:k], start=1)
    )


def ndcg_at_k(gains: Sequence[float], k: int, ideal_grades: Sequence[float]) -> float:
    """
    nDCG@k. ``ideal_grades`` are the grades of *all* labelled relevant items, so the
    ideal ranking is computed from the labels rather than from what was retrieved.
    """
    ideal = dcg_at_k(sorted(ideal_grades, reverse=True), k)
    if ideal == 0:
        return 0.0
    return dcg_at_k(gains, k) / ideal


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
