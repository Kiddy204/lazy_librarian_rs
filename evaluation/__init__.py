"""Evaluation framework for recommendation models."""
from .metrics import (
    precision_at_k,
    recall_at_k,
    average_precision_at_k,
    ndcg_at_k,
    hit_rate_at_k,
    mrr,
    evaluate_user,
)
from .baselines import (
    RandomRecommender,
    PopularityRecommender,
    UserKNNRecommender,
    DenseUserCF,
    DenseItemCF,
)
from .evaluate import evaluate_model, evaluate_all_baselines

__all__ = [
    # Metrics
    "precision_at_k",
    "recall_at_k", 
    "average_precision_at_k",
    "ndcg_at_k",
    "hit_rate_at_k",
    "mrr",
    "evaluate_user",
    # Baselines
    "RandomRecommender",
    "PopularityRecommender",
    "UserKNNRecommender",
    # Evaluation
    "evaluate_model",
    "evaluate_all_baselines",
]




