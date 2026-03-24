"""
Evaluation Metrics — ranking quality + fairness.

Primary:   nDCG@K (positional utility with graded relevance)
Secondary: MRR (how fast first relevant result appears)
           Precision@K (fraction relevant in top-K)
           Spearman rho (rank correlation)
           Impact ratio (NYC LL144 bias audit)
"""

import json
import math
from typing import List, Dict


def dcg_at_k(scores: List[float], k: int) -> float:
    return sum(s / math.log2(i + 2) for i, s in enumerate(scores[:k]))


def ndcg_at_k(scores: List[float], k: int) -> float:
    actual = dcg_at_k(scores, k)
    ideal = dcg_at_k(sorted(scores, reverse=True), k)
    return round(actual / ideal, 4) if ideal > 0 else 0.0


def mrr(scores: List[float], threshold: float = 0.5) -> float:
    for i, s in enumerate(scores):
        if s >= threshold:
            return round(1.0 / (i + 1), 4)
    return 0.0


def precision_at_k(scores: List[float], k: int, threshold: float = 0.5) -> float:
    top = scores[:k]
    return round(sum(1 for s in top if s >= threshold) / max(k, 1), 4)


def spearman_rho(pred_ranks: List[int], true_ranks: List[int]) -> float:
    n = len(pred_ranks)
    if n < 2:
        return 1.0
    d_sq = sum((p - t) ** 2 for p, t in zip(pred_ranks, true_ranks))
    return round(1 - (6 * d_sq) / (n * (n ** 2 - 1)), 4)


def impact_ratio(scores_by_group: Dict[str, List[float]], threshold: float = 0.5) -> Dict[str, float]:
    """NYC LL144 bias audit: selection rate per group / max selection rate."""
    if not scores_by_group:
        return {}
    rates = {}
    for group, scores in scores_by_group.items():
        if scores:
            rates[group] = sum(1 for s in scores if s >= threshold) / len(scores)
        else:
            rates[group] = 0.0
    max_rate = max(rates.values()) if rates else 1.0
    if max_rate == 0:
        return {g: 0.0 for g in rates}
    return {g: round(r / max_rate, 4) for g, r in rates.items()}


def load_golden_dataset(path: str) -> Dict[str, Dict[str, float]]:
    with open(path, 'r') as f:
        return json.load(f)


def evaluate_full(results: list, golden_labels: Dict[str, float]) -> dict:
    """Complete evaluation suite against golden labels."""
    relevances = [golden_labels.get(r.resume_id, 0.0) for r in results]

    metrics = {
        "ndcg@3": ndcg_at_k(relevances, 3),
        "ndcg@5": ndcg_at_k(relevances, 5),
        "ndcg@10": ndcg_at_k(relevances, 10),
        "mrr": mrr(relevances),
        "precision@3": precision_at_k(relevances, 3),
        "precision@5": precision_at_k(relevances, 5),
        "num_resumes": len(results),
    }

    # Spearman correlation
    id_to_pred = {r.resume_id: r.rank for r in results}
    ideal_order = sorted(golden_labels.keys(), key=lambda k: golden_labels[k], reverse=True)
    pred_ranks = [id_to_pred.get(rid, len(results)) for rid in ideal_order]
    true_ranks = list(range(1, len(ideal_order) + 1))
    metrics["spearman_rho"] = spearman_rho(pred_ranks, true_ranks)

    # Score distribution by label
    import numpy as np
    for label_val, label_name in [(1.0, "good"), (0.5, "partial"), (0.0, "poor")]:
        group = [r.final_score for r in results if golden_labels.get(r.resume_id) == label_val]
        if group:
            metrics[f"avg_{label_name}"] = round(float(np.mean(group)), 4)

    # Separation gaps
    good = [r.final_score for r in results if golden_labels.get(r.resume_id, 0) == 1.0]
    partial = [r.final_score for r in results if golden_labels.get(r.resume_id, 0) == 0.5]
    poor = [r.final_score for r in results if golden_labels.get(r.resume_id, 0) == 0.0]
    if good and partial:
        metrics["gap_good_partial"] = round(min(good) - max(partial), 4)
    if partial and poor:
        metrics["gap_partial_poor"] = round(min(partial) - max(poor), 4)

    return metrics
