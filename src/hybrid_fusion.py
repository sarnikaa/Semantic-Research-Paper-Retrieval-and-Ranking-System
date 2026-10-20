"""
Hybrid Retrieval Fusion algorithms.
Implements Reciprocal Rank Fusion (RRF) and Normalized Weighted Score Combination.
"""

from typing import Dict, List, Optional
import numpy as np


def reciprocal_rank_fusion(
    runs: List[Dict[str, Dict[str, float]]],
    k: int = 60,
    top_k: int = 100
) -> Dict[str, Dict[str, float]]:
    """
    Combines multiple retrieval runs using Reciprocal Rank Fusion (RRF).

    Args:
        runs: List of runs, each mapping query_id -> {doc_id: score}
        k: Smoothing constant (standard default = 60)
        top_k: Number of final merged candidates to retain

    Returns:
        fused_run: Dict mapping query_id -> {doc_id: rrf_score}
    """
    # Gather all unique query IDs across all runs
    all_query_ids = set()
    for run in runs:
        all_query_ids.update(run.keys())

    fused_run: Dict[str, Dict[str, float]] = {}

    for q_id in all_query_ids:
        rrf_scores: Dict[str, float] = {}

        for run in runs:
            if q_id not in run:
                continue
            # Sort documents by score descending to get ranks (1-indexed)
            sorted_docs = sorted(run[q_id].items(), key=lambda x: x[1], reverse=True)
            for rank, (doc_id, _) in enumerate(sorted_docs, start=1):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))

        # Sort combined results descending by RRF score
        sorted_fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        fused_run[q_id] = {doc_id: score for doc_id, score in sorted_fused}

    return fused_run


def min_max_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalizes a dictionary of scores to [0.0, 1.0]."""
    if not scores:
        return {}
    vals = list(scores.values())
    min_v, max_v = min(vals), max(vals)
    if max_v == min_v:
        return {doc_id: 1.0 for doc_id in scores}
    return {doc_id: (v - min_v) / (max_v - min_v) for doc_id, v in scores.items()}


def weighted_linear_fusion(
    run_a: Dict[str, Dict[str, float]],
    run_b: Dict[str, Dict[str, float]],
    weight_a: float = 0.5,
    top_k: int = 100
) -> Dict[str, Dict[str, float]]:
    """
    Merges two runs using min-max normalized weighted linear interpolation.
    """
    weight_b = 1.0 - weight_a
    all_queries = set(run_a.keys()).union(set(run_b.keys()))
    fused_run: Dict[str, Dict[str, float]] = {}

    for q_id in all_queries:
        norm_a = min_max_normalize(run_a.get(q_id, {}))
        norm_b = min_max_normalize(run_b.get(q_id, {}))

        combined: Dict[str, float] = {}
        all_docs = set(norm_a.keys()).union(set(norm_b.keys()))
        for doc_id in all_docs:
            s_a = norm_a.get(doc_id, 0.0)
            s_b = norm_b.get(doc_id, 0.0)
            combined[doc_id] = (weight_a * s_a) + (weight_b * s_b)

        sorted_docs = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
        fused_run[q_id] = {doc_id: s for doc_id, s in sorted_docs}

    return fused_run
