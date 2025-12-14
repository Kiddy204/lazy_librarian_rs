"""Ranking metrics for recommendation evaluation.

Primary metric: MAP@10 (Mean Average Precision at 10)
Secondary metrics: NDCG@K, Hit Rate@K, MRR, Precision@K, Recall@K
"""
import numpy as np
from typing import List, Dict, Set, Union


def precision_at_k(ranked_items: np.ndarray, relevant_items: Set[int], k: int) -> float:
    """
    Precision@K: fraction of top-K items that are relevant.
    
    Args:
        ranked_items: Array of item IDs sorted by predicted relevance (descending)
        relevant_items: Set of ground truth relevant item IDs
        k: Number of top items to consider
        
    Returns:
        Precision score in [0, 1]
    """
    if k <= 0:
        return 0.0
    top_k = set(ranked_items[:k])
    hits = len(top_k & relevant_items)
    return hits / k


def recall_at_k(ranked_items: np.ndarray, relevant_items: Set[int], k: int) -> float:
    """
    Recall@K: fraction of relevant items that appear in top-K.
    
    Args:
        ranked_items: Array of item IDs sorted by predicted relevance (descending)
        relevant_items: Set of ground truth relevant item IDs
        k: Number of top items to consider
        
    Returns:
        Recall score in [0, 1]
    """
    if len(relevant_items) == 0:
        return 0.0
    if k <= 0:
        return 0.0
    top_k = set(ranked_items[:k])
    hits = len(top_k & relevant_items)
    return hits / len(relevant_items)


def average_precision_at_k(ranked_items: np.ndarray, relevant_items: Set[int], k: int) -> float:
    """
    AP@K: Average Precision at K.
    
    Computes the average of precision values at each position where a relevant 
    item is found, up to position K. Rewards relevant items appearing earlier.
    
    Formula: AP@K = (1/min(K, R)) × Σᵢ (Precision@i × rel(i))
    where R = number of relevant items, rel(i) = 1 if item at position i is relevant
    
    Args:
        ranked_items: Array of item IDs sorted by predicted relevance (descending)
        relevant_items: Set of ground truth relevant item IDs
        k: Number of top items to consider
        
    Returns:
        Average precision score in [0, 1]
    """
    if len(relevant_items) == 0:
        return 0.0
    if k <= 0:
        return 0.0
    
    hits = 0
    sum_precision = 0.0
    
    for i, item in enumerate(ranked_items[:k]):
        if item in relevant_items:
            hits += 1
            precision_at_i = hits / (i + 1)
            sum_precision += precision_at_i
    
    # Normalize by min(K, number of relevant items)
    return sum_precision / min(k, len(relevant_items))


def ndcg_at_k(ranked_items: np.ndarray, relevant_items: Set[int], k: int) -> float:
    """
    NDCG@K: Normalized Discounted Cumulative Gain at K.
    
    Uses binary relevance (1 if relevant, 0 otherwise).
    Discounts relevance by log2(position + 1).
    
    Formula: NDCG@K = DCG@K / IDCG@K
    where DCG@K = Σᵢ rel(i) / log2(i + 2)
    and IDCG@K is the ideal DCG (all relevant items at top)
    
    Args:
        ranked_items: Array of item IDs sorted by predicted relevance (descending)
        relevant_items: Set of ground truth relevant item IDs
        k: Number of top items to consider
        
    Returns:
        NDCG score in [0, 1]
    """
    if len(relevant_items) == 0:
        return 0.0
    if k <= 0:
        return 0.0
    
    # Compute DCG
    dcg = 0.0
    for i, item in enumerate(ranked_items[:k]):
        if item in relevant_items:
            # Position is 0-indexed, so we use log2(i + 2) for 1-indexed formula
            dcg += 1.0 / np.log2(i + 2)
    
    # Compute Ideal DCG (all relevant items at the top positions)
    ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant_items))))
    
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def hit_rate_at_k(ranked_items: np.ndarray, relevant_items: Set[int], k: int) -> float:
    """
    Hit Rate@K (HR@K): Binary indicator if any relevant item appears in top-K.
    
    Also known as Recall@K with binary output.
    
    Args:
        ranked_items: Array of item IDs sorted by predicted relevance (descending)
        relevant_items: Set of ground truth relevant item IDs
        k: Number of top items to consider
        
    Returns:
        1.0 if any relevant item in top-K, else 0.0
    """
    if len(relevant_items) == 0:
        return 0.0
    if k <= 0:
        return 0.0
    top_k = set(ranked_items[:k])
    return 1.0 if len(top_k & relevant_items) > 0 else 0.0


def mrr(ranked_items: np.ndarray, relevant_items: Set[int], max_k: int = None) -> float:
    """
    MRR: Mean Reciprocal Rank.
    
    Returns the reciprocal of the rank of the first relevant item.
    If no relevant item is found, returns 0.
    
    Args:
        ranked_items: Array of item IDs sorted by predicted relevance (descending)
        relevant_items: Set of ground truth relevant item IDs
        max_k: Optional maximum rank to consider (None = no limit)
        
    Returns:
        Reciprocal rank score in [0, 1]
    """
    if len(relevant_items) == 0:
        return 0.0
    
    search_items = ranked_items[:max_k] if max_k else ranked_items
    
    for i, item in enumerate(search_items):
        if item in relevant_items:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_user(
    ranked_items: np.ndarray,
    relevant_items: Set[int],
    k_values: List[int] = [5, 10, 20]
) -> Dict[str, float]:
    """
    Compute all metrics for a single user.
    
    Args:
        ranked_items: Array of item IDs sorted by predicted relevance (descending)
        relevant_items: Set of ground truth relevant item IDs
        k_values: List of K values to compute metrics for
        
    Returns:
        Dictionary with all metric values
    """
    results = {}
    
    for k in k_values:
        results[f"precision@{k}"] = precision_at_k(ranked_items, relevant_items, k)
        results[f"recall@{k}"] = recall_at_k(ranked_items, relevant_items, k)
        results[f"map@{k}"] = average_precision_at_k(ranked_items, relevant_items, k)
        results[f"ndcg@{k}"] = ndcg_at_k(ranked_items, relevant_items, k)
        results[f"hr@{k}"] = hit_rate_at_k(ranked_items, relevant_items, k)
    
    results["mrr"] = mrr(ranked_items, relevant_items)
    
    return results


def compute_metrics_summary(
    all_user_metrics: List[Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics across all users.
    
    Args:
        all_user_metrics: List of metric dictionaries from evaluate_user()
        
    Returns:
        Dictionary with mean and std for each metric
    """
    if not all_user_metrics:
        return {}
    
    # Get all metric names from first user
    metric_names = list(all_user_metrics[0].keys())
    
    summary = {}
    for metric in metric_names:
        values = [m[metric] for m in all_user_metrics]
        summary[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    
    return summary
