
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
from submit import generate_submission

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

def build_user_subject_matrix(interactions, items_clean):
    """
    Builds user × subject matrix from item metadata.
    Assumes items_clean['subjects_list'] is list[str] or NaN.
    """

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

    us_matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(
            interactions.user_idx.max() + 1,
            len(subject_to_idx)
        )
    )

    return us_matrix, subject_to_idx

def build_item_subject_index(items_clean, subject_to_idx):
    """
    Build mapping: item_idx -> list of subject indices
    """

    item_subject_indices = {}

    for row in items_clean.itertuples():
        if not isinstance(row.subjects_list, list):
            continue

        idxs = []
        for s in row.subjects_list:
            s = s.strip().lower()
            if s in subject_to_idx:
                idxs.append(subject_to_idx[s])

        if idxs:
            item_subject_indices[row.i] = idxs

    return item_subject_indices

def zscore_per_user(scores, eps=1e-8):
    mean = scores.mean(axis=1, keepdims=True)
    std = scores.std(axis=1, keepdims=True) + eps
    return (scores - mean) / std

def normalize_us_score(us_s, max_overlap):
    return us_s / max_overlap if max_overlap > 0 else 0.0
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
    item_subject_indices,
    top_n=200,
    w_als=0.6,
    w_uu=0.3,
    w_us=0.1
):
    n_users, n_items = als_scores.shape
    final_scores = np.full_like(als_scores, -np.inf)

    us_matrix = us_matrix.tocsr()

    for u in range(n_users):
        user_subjects = set(us_matrix[u].indices)

        if not user_subjects:
            max_overlap = 1.0
        else:
            max_overlap = max(
                (
                    len(user_subjects.intersection(item_subject_indices.get(i, [])))
                    for i in item_subject_indices
                ),
                default=1.0
            )

        candidates = np.argsort(als_scores[u])[::-1][:top_n]

        for i in candidates:
            als_s = als_scores[u, i]
            uu_s = uu_scores[u, i]

            item_subs = item_subject_indices.get(i)
            if item_subs:
                us_raw = len(user_subjects.intersection(item_subs))
                us_s = us_raw / max_overlap
            else:
                us_s = 0.0

            final_scores[u, i] = (
                w_als * als_s +
                w_uu  * uu_s +
                w_us  * us_s
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

    items_clean = clean_items(items_df, interactions_df, remove_subjects=False)
    interactions_clean = clean_interactions(interactions_df)

    users = interactions_clean["u"].unique()
    user_id_to_idx = {u: i for i, u in enumerate(users)}
    interactions_clean["user_idx"] = interactions_clean["u"].map(user_id_to_idx)

    item_id_to_idx = {i: idx for idx, i in enumerate(items_clean["i"])}
    interactions_clean["item_idx"] = interactions_clean["i"].map(item_id_to_idx)

    n_users = interactions_clean["user_idx"].nunique()
    n_items = interactions_clean["item_idx"].nunique()

    from split import create_train_test_split
    train_data, test_data, val_data = create_train_test_split(
        interactions_clean, test_size=0.2, val_size=0.2
    )

    train_matrix = build_interaction_matrix(
        train_data, n_users, n_items, weight_col="interaction_rank"
    )
    test_matrix = build_interaction_matrix(
        test_data, n_users, n_items
    )

    val_matrix = build_interaction_matrix(
        val_data, n_users, n_items
    )

    print("Training ALS...")
    als_scores = train_als(train_matrix)

    print("Computing User-User CF...")
    uu_scores = user_user_scores(train_matrix)


    print("Building User-Subject matrix...")
    us_matrix, subject_to_idx = build_user_subject_matrix(train_data, items_clean)

    item_subject_indices = build_item_subject_index(
        items_clean,
        subject_to_idx
    )

    als_scores = zscore_per_user(als_scores)
    uu_scores = zscore_per_user(uu_scores)


    final_scores = hybrid_rerank(
        als_scores,
        uu_scores,
        us_matrix,
        item_subject_indices
    )


    map10 = compute_map_at_k(final_scores, test_matrix)
    print(f"\nHybrid MAP@10: {map10:.5f}")



    val_scores = hybrid_rerank(
        als_scores,
        uu_scores,
        us_matrix,
        item_subject_indices
    )

    val_map10 = compute_map_at_k(val_scores, val_matrix, k=10)
    print(f"Validation MAP@10: {val_map10:.5f}")
    print(f"Validation scores: {val_scores}")

    print("Generating submission...")
    submission_df = generate_submission(
        predictions=final_scores,
        user_index=user_id_to_idx,
        item_index=item_id_to_idx,
        k=10
    )

    submission_path = "submission_hybrid_ranking.csv"
    submission_df.to_csv(submission_path, index=False)

    best_score = 0
best_weights = None

for w_als in [0.7, 0.80, 0.90]:
    for w_uu in [0.1, 0.15, 0.25, 0.35]:
        for w_us in [0.1, 0.2,0.25]:

            scores = hybrid_rerank(
                als_scores,
                uu_scores,
                us_matrix,
                item_subject_indices,
                w_als=w_als,
                w_uu=w_uu,
                w_us=w_us
            )

            map10 = compute_map_at_k(scores, val_matrix)

            print(
                f"ALS={w_als}, UU={w_uu}, US={w_us} "
                f"=> VAL MAP@10={map10:.5f}"
            )

            if map10 > best_score:
                best_score = map10
                best_weights = (w_als, w_uu, w_us)

# full_train = pd.concat([train_data, val_data])

# full_train_matrix = build_interaction_matrix(
#     full_train, n_users, n_items, weight_col="interaction_rank"
# )

# als_scores = train_als(full_train_matrix)
# uu_scores = user_user_scores(full_train_matrix)
# us_matrix, subject_to_idx = build_user_subject_matrix(full_train, items_clean)
# item_subject_indices = build_item_subject_index(items_clean, subject_to_idx)

# final_scores = hybrid_rerank(
#     als_scores,
#     uu_scores,
#     us_matrix,
#     item_subject_indices,
#     w_als=best_weights[0],
#     w_uu=best_weights[1],
#     w_us=best_weights[2]
# )


