
import sys
import os

from eval import compute_map_at_k_als
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

from data_preprocessing.clean_interactions import clean_interactions
from data_preprocessing.clean_items import clean_items
from split import create_train_test_split

"""
Hybrid Re-ranking: ALS + User-User CF + User-Subject Signal
Optimized for MAP@10 (implicit feedback)

Requirements:
- pandas
- numpy
- scipy
- implicit
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from implicit.als import AlternatingLeastSquares

# ============================================================
# Utilities
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


def compute_map_at_k(scores, test_matrix, k=10):
    average_precisions = []

    for u in range(scores.shape[0]):
        true_items = test_matrix[u].indices
        if len(true_items) == 0:
            continue

        ranked_items = np.argsort(scores[u])[::-1][:k]

        hits = 0
        score = 0.0
        for i, item in enumerate(ranked_items):
            if item in true_items:
                hits += 1
                score += hits / (i + 1)

        average_precisions.append(score / min(len(true_items), k))

    return float(np.mean(average_precisions)) if average_precisions else 0.0


# ============================================================
# User-User CF
# ============================================================

def user_user_scores(train_matrix):
    sim = cosine_similarity(train_matrix)
    np.fill_diagonal(sim, 0.0)
    return sim @ train_matrix


# ============================================================
# User-Subject Matrix
# ============================================================

from scipy.sparse import csr_matrix

# ============================================================
# User-Subject Matrix (LIST-AWARE, ROBUST)
# ============================================================
from scipy.sparse import csr_matrix

# ============================================================
# User-Subject Matrix (LIST-AWARE, ROBUST)
# ============================================================
from scipy.sparse import csr_matrix

# ============================================================
# User-Subject Matrix (LIST-AWARE, ROBUST)
# ============================================================

def build_user_subject_matrix(interactions, items_clean):
    """
    Builds user × subject matrix from item metadata.
    Assumes items_clean['subjects_list'] is a list[str] or NaN.
    """

    # Map item -> list of subjects
    item_subjects = (
        items_clean
        .set_index("i")["subjects_list"]
        .dropna()
    )

    # Build subject vocabulary
    subject_set = set()
    for subs in item_subjects:
        for s in subs:
            subject_set.add(s.strip().lower())

    subject_to_idx = {s: i for i, s in enumerate(sorted(subject_set))}

    rows, cols, data = [], [], []

    for row in interactions.itertuples():
        subjects = item_subjects.get(row.i)
        if subjects is None:
            continue

        for s in subjects:
            s = s.strip().lower()
            idx = subject_to_idx.get(s)
            if idx is not None:
                rows.append(row.user_idx)
                cols.append(idx)
                data.append(1.0)

    return csr_matrix(
        (data, (rows, cols)),
        shape=(
            interactions.user_idx.max() + 1,
            len(subject_to_idx)
        )
    ), subject_to_idx

# ============================================================
# ALS Candidate Generator
# ============================================================

def train_als(train_matrix, factors=192, reg=0.1, alpha=80, iterations=30):
    model = AlternatingLeastSquares(
        factors=factors,
        regularization=reg,
        iterations=iterations,
        random_state=42,
        use_gpu=False
    )
    model.fit((train_matrix * alpha).T)

    user_factors = model.item_factors
    item_factors = model.user_factors

    return user_factors @ item_factors.T


# ============================================================
# Hybrid Re-ranking
# ============================================================

def hybrid_rerank(
    als_scores,
    uu_scores,
    us_matrix,
    items_clean,
    train_data,
    top_n=200,
    w_als=0.6,
    w_uu=0.3,
    w_us=0.1
):
    n_users, n_items = als_scores.shape
    final_scores = np.full_like(als_scores, -np.inf)

    item_subjects = (
        items_clean
        .set_index("i")["subjects_list"]
    )

    for u in range(n_users):
        candidates = np.argsort(als_scores[u])[::-1][:top_n]

        for i in candidates:
            als_s = als_scores[u, i]
            uu_s = uu_scores[u, i]

            us_s = 0.0
            subs = item_subjects.get(i)
            if subs:
                for s in subs:
                    if s in us_matrix[u].indices:
                        us_s += 1.0

            final_scores[u, i] = (
                w_als * als_s +
                w_uu * uu_s +
                w_us * us_s
            )

    return final_scores


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # interactions = pd.read_csv("data/interactions.csv")
    # items_clean = pd.read_csv("data/items.csv")
    items_df = pd.read_csv("data/items.csv")
    interactions_df = pd.read_csv("data/interactions.csv")

    items_clean = clean_items(items_df, interactions_df)
    interactions_clean = clean_interactions(interactions_df)

    users = interactions_clean["u"].unique()
    user_id_to_idx = {u: i for i, u in enumerate(users)}
    interactions_clean["user_idx"] = interactions_clean["u"].map(user_id_to_idx)

    item_id_to_idx = {i: idx for idx, i in enumerate(items_clean["i"])}
    interactions_clean["item_idx"] = interactions_clean["i"].map(item_id_to_idx)

    n_users = interactions_clean["user_idx"].nunique()
    n_items = interactions_clean["item_idx"].nunique()

    from split import create_train_test_split
    train_data, test_data, _ = create_train_test_split(
        interactions_clean, test_size=0.2, val_size=0.2
    )

    train_matrix = build_interaction_matrix(
        train_data, n_users, n_items, weight_col="interaction_rank"
    )
    test_matrix = build_interaction_matrix(
        test_data, n_users, n_items
    )

    print("Training ALS...")
    als_scores = train_als(train_matrix)

    print("Computing User-User CF...")
    uu_scores = user_user_scores(train_matrix)

    print("Building User-Subject matrix...")
    us_matrix, _ = build_user_subject_matrix(train_data, items_clean)

    print("Hybrid re-ranking...")
    final_scores = hybrid_rerank(
        als_scores,
        uu_scores,
        us_matrix,
        items_clean,
        train_data
    )

    map10 = compute_map_at_k(final_scores, test_matrix)
    print(f"\nHybrid MAP@10: {map10:.5f}")
