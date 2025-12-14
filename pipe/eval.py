import numpy as np

def compute_map_at_k(predictions, test_matrix, k=10):
    n_users = test_matrix.shape[0]
    ap_scores = []

    for user_idx in range(n_users):
        relevant_items = np.where(test_matrix[user_idx] > 0)[0]

        if len(relevant_items) == 0:
            continue

        # Top-k predictions
        top_k_items = np.argsort(predictions[user_idx])[::-1][:k]

        hits = 0
        score = 0.0

        for rank, item_idx in enumerate(top_k_items, start=1):
            if item_idx in relevant_items:
                hits += 1
                score += hits / rank

        ap_scores.append(score / min(len(relevant_items), k))

    return np.mean(ap_scores) if ap_scores else 0.0