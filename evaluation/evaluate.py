"""Evaluation runner for recommendation models.

This module provides functions to evaluate recommendation models
using ranking metrics like MAP@K, NDCG@K, Hit Rate@K, and MRR.
"""
import numpy as np
import pandas as pd
from scipy import sparse
from typing import Callable, Dict, List, Any, Optional, Union
from pathlib import Path
import json
import time
import sys

# Handle both direct execution and module import
if __name__ == "__main__":
    # Add parent directory to path for direct execution
    sys.path.insert(0, str(Path(__file__).parent.parent))
    sys.path.insert(0, str(Path(__file__).parent))
    from metrics import evaluate_user, compute_metrics_summary
    from baselines import (
        RandomRecommender, 
        PopularityRecommender, 
        UserKNNRecommender,
        ItemKNNRecommender,
        DenseUserCF,
        DenseItemCF,
    )
else:
    from .metrics import evaluate_user, compute_metrics_summary
    from .baselines import (
        RandomRecommender, 
        PopularityRecommender, 
        UserKNNRecommender,
        ItemKNNRecommender,
        DenseUserCF,
        DenseItemCF,
    )

from tqdm import tqdm
from data_preprocessing.config import OUTPUT_DIR, OUTPUT_FILES


def load_split_matrices():
    """
    Load train/val/test sparse matrices from disk.
    
    Returns:
        Tuple of (train_matrix, val_matrix, test_matrix)
    """
    train_matrix = sparse.load_npz(OUTPUT_DIR / "train_matrix.npz")
    val_matrix = sparse.load_npz(OUTPUT_DIR / "val_matrix.npz")
    test_matrix = sparse.load_npz(OUTPUT_DIR / "test_matrix.npz")
    
    return train_matrix, val_matrix, test_matrix


def evaluate_model(
    model,
    train_matrix: sparse.csr_matrix,
    test_matrix: sparse.csr_matrix,
    k_values: List[int] = [5, 10, 20],
    n_candidates: Optional[int] = None,
    exclude_train: bool = True,
    show_progress: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluate a recommendation model on test data.
    
    Args:
        model: Recommender model with fit() and predict() methods
        train_matrix: Training interaction matrix (for exclusion)
        test_matrix: Test interaction matrix (ground truth)
        k_values: List of K values for metrics
        n_candidates: Number of candidate items to rank (None = all items)
        exclude_train: Whether to exclude training items from candidates
        show_progress: Whether to show progress bar
        seed: Random seed for candidate sampling
        
    Returns:
        Dictionary with aggregated metrics and timing info
    """
    rng = np.random.RandomState(seed)
    n_users, n_items = train_matrix.shape
    all_items = np.arange(n_items)
    
    # Get users with test interactions
    test_users = np.unique(test_matrix.nonzero()[0])
    
    all_user_metrics = []
    
    iterator = tqdm(test_users, desc="Evaluating", disable=not show_progress)
    
    start_time = time.time()
    
    for user_id in iterator:
        # Get ground truth relevant items
        relevant_items = set(test_matrix[user_id].nonzero()[1])
        if len(relevant_items) == 0:
            continue
        
        # Get candidate items
        if exclude_train:
            train_items = set(train_matrix[user_id].nonzero()[1])
            candidate_items = np.array([i for i in all_items if i not in train_items], dtype=np.int64)
        else:
            candidate_items = all_items.copy().astype(np.int64)
        
        # Optionally sample candidates for efficiency
        if n_candidates and len(candidate_items) > n_candidates:
            # Always keep relevant items in candidates
            relevant_in_candidates = np.array([i for i in candidate_items if i in relevant_items], dtype=np.int64)
            other_candidates = np.array([i for i in candidate_items if i not in relevant_items], dtype=np.int64)
            
            n_others = min(n_candidates - len(relevant_in_candidates), len(other_candidates))
            if n_others > 0:
                sampled_others = rng.choice(other_candidates, n_others, replace=False)
                candidate_items = np.concatenate([relevant_in_candidates, sampled_others]).astype(np.int64)
            else:
                candidate_items = relevant_in_candidates
        
        # Ensure integer type for indexing
        candidate_items = candidate_items.astype(np.int64)
        
        # Get model predictions
        scores = model.predict(user_id, candidate_items)
        
        # Rank items by score (descending)
        ranked_indices = np.argsort(-scores)
        ranked_items = candidate_items[ranked_indices]
        
        # Compute metrics
        user_metrics = evaluate_user(ranked_items, relevant_items, k_values)
        all_user_metrics.append(user_metrics)
    
    eval_time = time.time() - start_time
    
    # Aggregate metrics
    if all_user_metrics:
        summary = compute_metrics_summary(all_user_metrics)
    else:
        summary = {}
    
    results = {
        "metrics": summary,
        "n_users_evaluated": len(all_user_metrics),
        "n_users_total": len(test_users),
        "evaluation_time_seconds": eval_time,
        "model": repr(model),
        "k_values": k_values,
    }
    
    return results


def print_evaluation_results(results: Dict[str, Any], primary_metric: str = "map@10"):
    """
    Pretty-print evaluation results.
    
    Args:
        results: Results dictionary from evaluate_model()
        primary_metric: Metric to highlight as primary
    """
    print("\n" + "="*60)
    print(f"Model: {results['model']}")
    print("="*60)
    print(f"Users evaluated: {results['n_users_evaluated']}/{results['n_users_total']}")
    print(f"Evaluation time: {results['evaluation_time_seconds']:.2f}s")
    print("-"*60)
    
    metrics = results["metrics"]
    
    # Group metrics by type
    metric_groups = {
        "MAP (Primary)": [k for k in metrics.keys() if k.startswith("map@")],
        "NDCG": [k for k in metrics.keys() if k.startswith("ndcg@")],
        "Hit Rate": [k for k in metrics.keys() if k.startswith("hr@")],
        "Precision": [k for k in metrics.keys() if k.startswith("precision@")],
        "Recall": [k for k in metrics.keys() if k.startswith("recall@")],
        "Other": [k for k in metrics.keys() if not any(k.startswith(p) for p in ["map@", "ndcg@", "hr@", "precision@", "recall@"])],
    }
    
    for group_name, metric_names in metric_groups.items():
        if not metric_names:
            continue
        print(f"\n{group_name}:")
        for name in sorted(metric_names):
            mean = metrics[name]["mean"]
            std = metrics[name]["std"]
            highlight = "  <<<" if name == primary_metric else ""
            print(f"  {name:15s}: {mean:.4f} ± {std:.4f}{highlight}")
    
    print("\n" + "="*60)


def evaluate_all_baselines(
    train_matrix: sparse.csr_matrix,
    test_matrix: sparse.csr_matrix,
    k_values: List[int] = [5, 10, 20],
    n_candidates: Optional[int] = 1000,
    show_progress: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate all baseline models and return comparison.
    
    Args:
        train_matrix: Training interaction matrix
        test_matrix: Test interaction matrix (ground truth)
        k_values: List of K values for metrics
        n_candidates: Number of candidate items to rank
        show_progress: Whether to show progress bar
        
    Returns:
        Dictionary mapping model names to their results
    """
    baselines = {
        # "Random": RandomRecommender(seed=42),
        # "Popularity": PopularityRecommender(),
        # "UserKNN-50": UserKNNRecommender(n_neighbors=50),
        # "UserKNN-100": UserKNNRecommender(n_neighbors=100),
        # "ItemKNN-50": ItemKNNRecommender(n_neighbors=50),
        "DenseUserCF": DenseUserCF(),
        "DenseItemCF": DenseItemCF(),
    }
    
    all_results = {}
    
    for name, model in baselines.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print("="*60)
        
        # Fit model
        print("Fitting model...")
        fit_start = time.time()
        model.fit(train_matrix)
        fit_time = time.time() - fit_start
        print(f"Fit time: {fit_time:.2f}s")
        
        # Evaluate
        results = evaluate_model(
            model=model,
            train_matrix=train_matrix,
            test_matrix=test_matrix,
            k_values=k_values,
            n_candidates=n_candidates,
            show_progress=show_progress,
        )
        
        results["fit_time_seconds"] = fit_time
        all_results[name] = results
        
        # Print results
        print_evaluation_results(results)
    
    return all_results


def create_comparison_table(all_results: Dict[str, Dict[str, Any]], metrics_to_show: List[str] = None) -> pd.DataFrame:
    """
    Create a comparison table of model results.
    
    Args:
        all_results: Dictionary from evaluate_all_baselines()
        metrics_to_show: List of metric names to include (None = all)
        
    Returns:
        DataFrame with models as rows and metrics as columns
    """
    if not metrics_to_show:
        # Default: show key metrics
        metrics_to_show = ["map@10", "ndcg@10", "hr@10", "mrr", "precision@10", "recall@10"]
    
    rows = []
    for model_name, results in all_results.items():
        row = {"Model": model_name}
        for metric_name in metrics_to_show:
            if metric_name in results["metrics"]:
                row[metric_name] = results["metrics"][metric_name]["mean"]
            else:
                row[metric_name] = None
        row["Eval Time (s)"] = results["evaluation_time_seconds"]
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.set_index("Model")
    
    return df


def save_results(results: Dict[str, Any], path: Path):
    """Save evaluation results to JSON."""
    # Convert numpy types to Python types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    with open(path, "w") as f:
        json.dump(convert(results), f, indent=2)


def main():
    """Run baseline evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate recommendation baselines")
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 20], help="K values for metrics")
    parser.add_argument("--n-candidates", type=int, default=1000, help="Number of candidates to rank")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--use-val", action="store_true", help="Use validation set instead of test")
    args = parser.parse_args()
    
    print("Loading data...")
    train_matrix, val_matrix, test_matrix = load_split_matrices()
    print(f"Train matrix: {train_matrix.shape}, nnz={train_matrix.nnz}")
    print(f"Test matrix: {test_matrix.shape}, nnz={test_matrix.nnz}")
    
    eval_matrix = val_matrix if args.use_val else test_matrix
    
    # Run evaluation
    all_results = evaluate_all_baselines(
        train_matrix=train_matrix,
        test_matrix=eval_matrix,
        k_values=args.k,
        n_candidates=args.n_candidates,
    )
    
    # Create comparison table
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    comparison_df = create_comparison_table(all_results)
    print(comparison_df.to_string())
    
    # Save results
    output_path = args.output or OUTPUT_DIR / "baseline_results.json"
    save_results(all_results, output_path)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
