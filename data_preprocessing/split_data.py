"""Train/validation/test splitting with temporal awareness."""
import pandas as pd
import numpy as np
from scipy import sparse
from pathlib import Path
from typing import Tuple, Dict, Any
import json

from config import (
    OUTPUT_DIR, OUTPUT_FILES, 
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO
)
from load_data import load_interactions
from clean_interactions import clean_interactions


def temporal_split(
    interactions: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split interactions by timestamp.
    
    Args:
        interactions: DataFrame with 't' (timestamp) column
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        
    Returns:
        (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Sort by time
    df = interactions.sort_values("t").reset_index(drop=True)
    n = len(df)
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    print(f"Temporal split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    return train_df, val_df, test_df


def filter_cold_start(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    filter_users: bool = True,
    filter_items: bool = True
) -> pd.DataFrame:
    """
    Remove users/items from eval set that don't appear in training.
    This simulates real-world: can't evaluate on truly unseen users/items.
    
    Args:
        train_df: Training interactions
        eval_df: Evaluation interactions (val or test)
        filter_users: Remove interactions with unseen users
        filter_items: Remove interactions with unseen items
        
    Returns:
        Filtered evaluation DataFrame
    """
    n_before = len(eval_df)
    filtered = eval_df.copy()
    
    if filter_users:
        train_users = set(train_df["u"].unique())
        filtered = filtered[filtered["u"].isin(train_users)]
    
    if filter_items:
        train_items = set(train_df["i"].unique())
        filtered = filtered[filtered["i"].isin(train_items)]
    
    n_removed = n_before - len(filtered)
    if n_removed > 0:
        print(f"  Filtered {n_removed} cold-start interactions ({n_removed/n_before*100:.1f}%)")
    
    return filtered


def create_interaction_matrix(
    df: pd.DataFrame,
    n_users: int,
    n_items: int,
    weight_by_count: bool = True
) -> sparse.csr_matrix:
    """
    Create user-item interaction matrix.
    
    Args:
        df: Interactions DataFrame with 'u' and 'i' columns
        n_users: Total number of users
        n_items: Total number of items
        weight_by_count: If True, value = interaction count; else binary
        
    Returns:
        Sparse CSR matrix of shape (n_users, n_items)
    """
    if weight_by_count:
        # Count interactions per (u, i) pair
        counts = df.groupby(["u", "i"]).size().reset_index(name="count")
        matrix = sparse.csr_matrix(
            (counts["count"], (counts["u"], counts["i"])),
            shape=(n_users, n_items)
        )
    else:
        # Binary: 1 if any interaction
        pairs = df[["u", "i"]].drop_duplicates()
        matrix = sparse.csr_matrix(
            (np.ones(len(pairs)), (pairs["u"], pairs["i"])),
            shape=(n_users, n_items)
        )
    
    return matrix


def get_split_stats(
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame, 
    test_df: pd.DataFrame
) -> Dict[str, Dict[str, Any]]:
    """Compute statistics for validation."""
    
    def stats_for_split(df: pd.DataFrame, name: str) -> Dict[str, Any]:
        return {
            "n_interactions": len(df),
            "n_unique_pairs": df[["u", "i"]].drop_duplicates().shape[0],
            "n_users": df["u"].nunique(),
            "n_items": df["i"].nunique(),
            "date_start": str(df["datetime"].min()),
            "date_end": str(df["datetime"].max()),
        }
    
    return {
        "train": stats_for_split(train_df, "train"),
        "val": stats_for_split(val_df, "val"),
        "test": stats_for_split(test_df, "test"),
    }


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    save_matrices: bool = True
) -> None:
    """Save split DataFrames and optionally interaction matrices."""
    
    # Save DataFrames
    train_df.to_parquet(OUTPUT_FILES["train"])
    val_df.to_parquet(OUTPUT_FILES["val"])
    test_df.to_parquet(OUTPUT_FILES["test"])
    
    print(f"Saved splits to {OUTPUT_DIR}")
    
    if save_matrices:
        # Determine dimensions from full dataset
        n_users = max(train_df["u"].max(), val_df["u"].max(), test_df["u"].max()) + 1
        n_items = max(train_df["i"].max(), val_df["i"].max(), test_df["i"].max()) + 1
        
        # Create and save matrices
        train_matrix = create_interaction_matrix(train_df, n_users, n_items, weight_by_count=True)
        sparse.save_npz(OUTPUT_DIR / "train_matrix.npz", train_matrix)
        
        # For evaluation, we typically want binary matrices
        val_matrix = create_interaction_matrix(val_df, n_users, n_items, weight_by_count=False)
        sparse.save_npz(OUTPUT_DIR / "val_matrix.npz", val_matrix)
        
        test_matrix = create_interaction_matrix(test_df, n_users, n_items, weight_by_count=False)
        sparse.save_npz(OUTPUT_DIR / "test_matrix.npz", test_matrix)
        
        print(f"Saved interaction matrices: train={train_matrix.shape}, density={train_matrix.nnz/np.prod(train_matrix.shape):.6f}")


def load_or_create_clean_interactions() -> pd.DataFrame:
    """
    Load cleaned interactions from parquet if available,
    otherwise clean from raw CSV.
    """
    if OUTPUT_FILES["interactions_clean"].exists():
        print(f"Loading cleaned interactions from {OUTPUT_FILES['interactions_clean']}")
        return pd.read_parquet(OUTPUT_FILES["interactions_clean"])
    else:
        print("Cleaned interactions not found, processing from raw data...")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        raw_interactions = load_interactions()
        clean_df = clean_interactions(raw_interactions)
        # Save for future use
        clean_df.to_parquet(OUTPUT_FILES["interactions_clean"])
        print(f"Saved cleaned interactions to {OUTPUT_FILES['interactions_clean']}")
        return clean_df


def main():
    """Run the splitting pipeline."""
    print("\n" + "="*50)
    print("Creating train/val/test splits")
    print("="*50)
    
    # Load or create cleaned interactions
    interactions = load_or_create_clean_interactions()
    print(f"Loaded {len(interactions)} interactions")
    
    # Create temporal splits
    train_df, val_df, test_df = temporal_split(interactions)
    
    # Filter cold-start from eval sets
    print("\nFiltering cold-start from validation set:")
    val_df = filter_cold_start(train_df, val_df)
    
    print("Filtering cold-start from test set:")
    test_df = filter_cold_start(train_df, test_df)
    
    # Print stats
    stats = get_split_stats(train_df, val_df, test_df)
    print("\n" + "-"*50)
    for split_name, split_stats in stats.items():
        print(f"\n{split_name.upper()}:")
        for k, v in split_stats.items():
            print(f"  {k}: {v}")
    
    # Save everything
    print("\n" + "-"*50)
    save_splits(train_df, val_df, test_df, save_matrices=True)
    
    # Save stats as JSON
    stats_path = OUTPUT_DIR / "split_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"Saved split statistics to {stats_path}")
    
    print("\n" + "="*50)
    print("Splitting complete!")
    print("="*50)
    
    return stats


if __name__ == "__main__":
    main()
