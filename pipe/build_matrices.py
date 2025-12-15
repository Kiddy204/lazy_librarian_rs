import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from split import create_train_test_split
from stats import (
    print_clean_data_stats,
    print_re_indexing_items_stats,
    print_user_item_matrix_stats,
)
from submit import generate_submission

from data_preprocessing.build_features import build_item_features
from data_preprocessing.config import OUTPUT_FILES

# 1- Build all features (clean items + interactions + all encodings)
print("Building features (this may take a while for first run)...")
build_item_features(use_local_embeddings=False)

# Load processed data
items_clean = pd.read_parquet(OUTPUT_FILES["items_clean"])
interactions_clean = pd.read_parquet(OUTPUT_FILES["interactions_clean"])
users_clean = interactions_clean["u"].unique()

print_clean_data_stats(items_clean, interactions_clean, users_clean)

# 2- Re-index items and users
item_id_to_idx = {item_id: idx for idx, item_id in enumerate(items_clean["i"].values)}
interactions_clean = interactions_clean[
    interactions_clean["i"].isin(item_id_to_idx)
].copy()

interactions_clean["item_idx"] = interactions_clean["i"].map(item_id_to_idx)


user_id_to_idx = {user_id: idx for idx, user_id in enumerate(users_clean)}

interactions_clean["user_idx"] = interactions_clean["u"].map(user_id_to_idx)

n_users = len(user_id_to_idx)
# n_users = len(users_clean)
n_items = len(item_id_to_idx)

user_index = user_id_to_idx
item_index = item_id_to_idx

assert interactions_clean["item_idx"].min() == 0
assert interactions_clean["item_idx"].max() == n_items - 1
assert interactions_clean["item_idx"].nunique() == n_items

print_re_indexing_items_stats(interactions_clean, n_users, n_items)

# 3- Split interactions_clean into train, val, test

train_data, test_data, val_data = create_train_test_split(
    interactions_clean, test_size=0.2, val_size=0.2
)

print("=" * 50)
print("Train Test Val Splits")
print("=" * 50)
print(f"train_data shape: {train_data.shape}")
print(f"test_data shape: {test_data.shape}")
print(f"val_data shape: {val_data.shape}")

# 4- Build user item matrixes
### a. Train user item matrix
train_user_item_matrix = np.zeros((n_users, n_items), dtype=np.float32)
rows = train_data["user_idx"].values
cols = train_data["item_idx"].values
train_user_item_matrix[rows, cols] = 1.0

### b. Test user item matrix
test_user_item_matrix = np.zeros((n_users, n_items), dtype=np.float32)
rows = test_data["user_idx"].values
cols = test_data["item_idx"].values
test_user_item_matrix[rows, cols] = 1.0

### c. Val user item matrix
val_user_item_matrix = np.zeros((n_users, n_items), dtype=np.float32)
rows = val_data["user_idx"].values
cols = val_data["item_idx"].values
val_user_item_matrix[rows, cols] = 1.0

### d. Print user item matrix stats
print_user_item_matrix_stats(
    train_user_item_matrix, test_user_item_matrix, val_user_item_matrix
)


# 5- User-based Collaborative Filtering
### a. User-based predict
def user_based_predict(interactions, similarity, epsilon=1e-9):
    """
    Predicts user-item interactions based on user-user similarity.
    Parameters:
        interactions (numpy array): The user-item interaction matrix.
        similarity (numpy array): The user-user similarity matrix.
        epsilon (float): Small constant added to the denominator to avoid division by zero.
    Returns:
        numpy array: The predicted interaction scores for each user-item pair.
    """
    # Calculate the weighted sum of interactions based on user similarity
    pred = similarity.dot(interactions) / (
        np.abs(similarity).sum(axis=1)[:, np.newaxis] + epsilon
    )
    return pred


def compute_map_at_k(predictions, test_matrix, k=10):
    """
    Compute Mean Average Precision at K (MAP@K) for recommendation evaluation.

    Parameters:
        predictions (numpy array): Predicted scores matrix (n_users x n_items).
        test_matrix (numpy array): Ground truth binary matrix (n_users x n_items).
        k (int): Number of top recommendations to consider.

    Returns:
        float: The MAP@K score.
    """
    n_users = predictions.shape[0]
    average_precisions = []

    for user_idx in range(n_users):
        # Get the actual relevant items for this user (items in test set)
        relevant_items = set(np.where(test_matrix[user_idx] == 1)[0])

        # Skip users with no relevant items in test set
        if len(relevant_items) == 0:
            continue

        # Get top-k predicted items (sorted by predicted score, descending)
        top_k_items = np.argsort(predictions[user_idx])[::-1][:k]

        # Compute Average Precision at K
        hits = 0
        precision_sum = 0.0

        for rank, item_idx in enumerate(top_k_items, start=1):
            if item_idx in relevant_items:
                hits += 1
                precision_at_rank = hits / rank
                precision_sum += precision_at_rank

        # Average precision for this user
        # Normalize by min(k, number of relevant items)
        # ap = precision_sum / min(k, len(relevant_items))
        ap = precision_sum / len(relevant_items)
        average_precisions.append(ap)

    # Mean Average Precision across all users
    map_score = np.mean(average_precisions) if average_precisions else 0.0
    return map_score


def user_user_cf(
    train_data_matrix,
    test_data_matrix,
    user_index=None,
    item_index=None,
    mask_seen_items_bool=False,
    set_diagonal_to_zero_bool=False,
):
    user_similarity = cosine_similarity(train_data_matrix)
    # cosine_similarity(train_data_matrix) gives 1.0 on the diagonal.
    # We need to set the diagonal to 0.0 to avoid self-similarity.
    if set_diagonal_to_zero_bool:
        np.fill_diagonal(user_similarity, 0.0)
    user_based_prediction = user_based_predict(train_data_matrix, user_similarity)

    def mask_seen_items(predictions, train_matrix):
        masked = predictions.copy()
        masked[train_matrix > 0] = -np.inf
        return masked

    if mask_seen_items_bool:
        user_based_prediction = mask_seen_items(
            user_based_prediction, train_data_matrix
        )
    user_map_score = compute_map_at_k(user_based_prediction, test_data_matrix, k=10)
    submission_path = "data/submissions/submission_user_based.csv"
    if user_index is not None and item_index is not None:
        user_based_submission = generate_submission(
            user_based_prediction, user_index, item_index, k=10
        )
        user_based_submission.to_csv(submission_path, index=False)

    return {
        "map_score": user_map_score,
        "submission_path": submission_path,
        "predictions": user_based_prediction,
    }


# Predict user item matrix
def test_user_item_matrix_prediction(
    train_user_item_matrix,
    test_user_item_matrix,
    mask_seen_items_bool=False,
    set_diagonal_to_zero_bool=False,
):
    results = {}
    user_predictions = user_user_cf(
        train_user_item_matrix,
        test_user_item_matrix,
        user_index,
        item_index,
        mask_seen_items_bool=mask_seen_items_bool,
        set_diagonal_to_zero_bool=set_diagonal_to_zero_bool,
    )
    results["User-Based CF"] = {
        "map_score": user_predictions["map_score"],
    }
    # print(f"User-Based CF MAP@10: {results['User-Based CF']['map_score']}")
    return user_predictions


print("=" * 50)
print("Testing User Item Matrix Prediction")
print("=" * 50)
print(
    f"User-Based CF MAP@10: {test_user_item_matrix_prediction(train_user_item_matrix, test_user_item_matrix)['map_score']}",
    False,
    False,
)

train_counts = train_data.groupby("user_idx").size()
test_counts = test_data.groupby("user_idx").size()

print("Users with no train data:", (train_counts == 0).sum())
print("Users with no test data:", (test_counts == 0).sum())
#
# build user subject matrix
# build user publisher matrix
# build user author matrix
# matrix factorisation
