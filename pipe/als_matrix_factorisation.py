"""
ALS optimized for MAP@10 using implicit feedback.

Assumptions:
- train_data, test_data DataFrames already exist
- Columns:
    - user_idx (0 ... n_users-1)
    - item_idx (0 ... n_items-1)
    - interaction_rank (>=1)
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

from data_preprocessing.clean_interactions import clean_interactions
from data_preprocessing.clean_items import clean_items
from split import create_train_test_split
from eval import compute_map_at_k


# ============================================================
# Sparse Matrix Builder
# ============================================================

def build_interaction_matrix(df, n_users, n_items, weight_col=None):
    data = (
        df[weight_col].astype(np.float32).values
        if weight_col
        else np.ones(len(df), dtype=np.float32)
    )

    return csr_matrix(
        (data, (df.user_idx.values, df.item_idx.values)),
        shape=(n_users, n_items)
    )


# ============================================================
# Mask Seen Items
# ============================================================

def mask_train_items(scores, train_matrix):
    scores = scores.copy()
    users, items = train_matrix.nonzero()
    scores[users, items] = -np.inf
    return scores


# ============================================================
# ALS Training + Evaluation
# ============================================================

def train_and_evaluate_als(
    train_matrix,
    test_matrix,
    factors=128,
    regularization=0.05,
    iterations=40,
    alpha=40.0,
    k=10
):
    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        use_gpu=False,
        random_state=42
    )

    confidence = (train_matrix * alpha).tocsr()
    model.fit(confidence.T)

    # ⚠️ implicit swaps semantics when using transpose
    user_factors = model.item_factors   # (n_users, factors)
    item_factors = model.user_factors   # (n_items, factors)

    scores = user_factors @ item_factors.T
    scores = mask_train_items(scores, train_matrix)

    return compute_map_at_k(scores, test_matrix, k=k)


# ============================================================
# Grid Search
# ============================================================

def als_grid_search(train_matrix, test_matrix):
    param_grid = {
        "factors": [ 192],
        "regularization": [ 0.1],
        "alpha": [80],
        "iterations": [30, 40]
    }

    best_score = 0.0
    best_params = None

    for f in param_grid["factors"]:
        for reg in param_grid["regularization"]:
            for a in param_grid["alpha"]:
                for it in param_grid["iterations"]:
                    score = train_and_evaluate_als(
                        train_matrix=train_matrix,
                        test_matrix=test_matrix,
                        factors=f,
                        regularization=reg,
                        iterations=it,
                        alpha=a
                    )

                    print(
                        f"[ALS] factors={f}, reg={reg}, alpha={a}, it={it} "
                        f"=> MAP@10={score:.5f}"
                    )

                    if score > best_score:
                        best_score = score
                        best_params = (f, reg, a, it)

    return best_score, best_params


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    items_df = pd.read_csv("data/items.csv")
    interactions_df = pd.read_csv("data/interactions.csv")

    items_clean = clean_items(items_df, interactions_df)
    interactions_clean = clean_interactions(interactions_df)

    users = interactions_clean["u"].unique()

    item_id_to_idx = {
        item_id: idx
        for idx, item_id in enumerate(items_clean["i"].values)
    }

    interactions_clean = interactions_clean[
        interactions_clean["i"].isin(item_id_to_idx)
    ].copy()

    interactions_clean["item_idx"] = interactions_clean["i"].map(item_id_to_idx)

    user_id_to_idx = {u: i for i, u in enumerate(users)}
    interactions_clean["user_idx"] = interactions_clean["u"].map(user_id_to_idx)

    n_users = interactions_clean["user_idx"].max() + 1
    n_items = interactions_clean["item_idx"].max() + 1

    train_data, test_data, _ = create_train_test_split(
        interactions_clean,
        test_size=0.2,
        val_size=0.2
    )

    train_matrix = build_interaction_matrix(
        train_data,
        n_users,
        n_items,
        weight_col="interaction_rank"
    )

    test_matrix = build_interaction_matrix(
        test_data,
        n_users,
        n_items
    )

    print("Running ALS hyperparameter search...")
    best_map, best_params = als_grid_search(train_matrix, test_matrix)

    print("\n==============================")
    print("BEST ALS RESULT")
    print("==============================")
    print(f"MAP@10: {best_map:.5f}")
    print(
        f"factors={best_params[0]}, "
        f"regularization={best_params[1]}, "
        f"alpha={best_params[2]}, "
        f"iterations={best_params[3]}"
    )
s