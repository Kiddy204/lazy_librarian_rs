"""Generate predictions for submission."""
import numpy as np
import pandas as pd
from scipy import sparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import argparse
import sys

# Handle imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from baselines import (
    RandomRecommender,
    PopularityRecommender,
    UserKNNRecommender,
    ItemKNNRecommender,
    DenseUserCF,
    DenseItemCF,
)
from data_preprocessing.config import OUTPUT_DIR, DATA_DIR


def load_train_matrix() -> sparse.csr_matrix:
    """Load training interaction matrix."""
    return sparse.load_npz(OUTPUT_DIR / "train_matrix.npz")


def load_full_interactions() -> sparse.csr_matrix:
    """
    Load ALL interactions (train + val + test) for final submission.
    This gives the model access to all historical data.
    """
    train = sparse.load_npz(OUTPUT_DIR / "train_matrix.npz")
    val = sparse.load_npz(OUTPUT_DIR / "val_matrix.npz")
    test = sparse.load_npz(OUTPUT_DIR / "test_matrix.npz")
    
    # Combine all interactions
    full_matrix = train + val + test
    return full_matrix.tocsr()


def get_submission_users(sample_path: Path) -> np.ndarray:
    """Get list of users from sample submission."""
    df = pd.read_csv(sample_path)
    return df["user_id"].values


def get_popular_items(interaction_matrix: sparse.csr_matrix, top_k: int = 10) -> list:
    """Get the most popular items as fallback for cold-start users."""
    item_popularity = np.array(interaction_matrix.sum(axis=0)).flatten()
    top_items = np.argsort(-item_popularity)[:top_k]
    return top_items.tolist()


def generate_predictions(
    model,
    interaction_matrix: sparse.csr_matrix,
    users: np.ndarray,
    top_k: int = 10,
    exclude_known: bool = True,
    show_progress: bool = True,
) -> dict:
    """
    Generate top-K recommendations for each user.
    
    Args:
        model: Fitted recommender model
        interaction_matrix: User-item interaction matrix
        users: Array of user IDs to generate predictions for
        top_k: Number of recommendations per user
        exclude_known: Whether to exclude items the user has already interacted with
        show_progress: Whether to show progress bar
        
    Returns:
        Dictionary mapping user_id to list of recommended item_ids
    """
    n_users, n_items = interaction_matrix.shape
    all_items = np.arange(n_items, dtype=np.int64)
    
    # Precompute popular items for cold-start users
    popular_items = get_popular_items(interaction_matrix, top_k * 2)
    
    predictions = {}
    cold_start_count = 0
    
    iterator = tqdm(users, desc="Generating predictions", disable=not show_progress)
    
    for user_id in iterator:
        # Handle cold-start users (user_id not in matrix)
        if user_id >= n_users:
            predictions[user_id] = popular_items[:top_k]
            cold_start_count += 1
            continue
        
        # Get candidate items
        if exclude_known:
            known_items = set(interaction_matrix[user_id].nonzero()[1])
            candidate_items = np.array([i for i in all_items if i not in known_items], dtype=np.int64)
        else:
            candidate_items = all_items.copy()
        
        if len(candidate_items) == 0:
            # User has interacted with all items (unlikely)
            predictions[user_id] = popular_items[:top_k]
            continue
        
        # Get model scores
        scores = model.predict(user_id, candidate_items)
        
        # Get top-K items
        if len(candidate_items) <= top_k:
            top_indices = np.argsort(-scores)
        else:
            # Partial sort for efficiency
            top_indices = np.argpartition(-scores, top_k)[:top_k]
            # Sort the top-K by score
            top_indices = top_indices[np.argsort(-scores[top_indices])]
        
        top_items = candidate_items[top_indices][:top_k]
        predictions[user_id] = top_items.tolist()
    
    if cold_start_count > 0:
        print(f"  Cold-start users (using popular items): {cold_start_count}")
    
    return predictions


def save_submission(
    predictions: dict,
    output_path: Path,
    top_k: int = 10,
):
    """
    Save predictions in submission format.
    
    Format: user_id,recommendation
    Where recommendation is space-separated item IDs
    """
    rows = []
    for user_id in sorted(predictions.keys()):
        items = predictions[user_id]
        # Pad with zeros if needed
        while len(items) < top_k:
            items.append(0)
        recommendation_str = " ".join(map(str, items[:top_k]))
        rows.append({"user_id": user_id, "recommendation": recommendation_str})
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved submission to: {output_path}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate predictions for submission")
    parser.add_argument(
        "--model", 
        type=str, 
        default="userknn",
        choices=["random", "popularity", "userknn", "itemknn", "dense_user", "dense_item"],
        help="Model to use for predictions"
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=50,
        help="Number of neighbors for KNN models"
    )
    parser.add_argument(
        "--use-full-data",
        action="store_true",
        help="Use all interactions (train+val+test) instead of just train"
    )
    parser.add_argument(
        "--exclude-known",
        action="store_true",
        default=True,
        help="Exclude items the user has already interacted with"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for submission file"
    )
    args = parser.parse_args()
    
    # Paths
    sample_path = DATA_DIR / "sample_submission.csv"
    submissions_dir = DATA_DIR / "submissions"
    submissions_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading data...")
    if args.use_full_data:
        print("Using full interaction data (train + val + test)")
        interaction_matrix = load_full_interactions()
    else:
        print("Using training data only")
        interaction_matrix = load_train_matrix()
    
    print(f"Interaction matrix: {interaction_matrix.shape}, nnz={interaction_matrix.nnz}")
    
    # Get users to predict for
    users = get_submission_users(sample_path)
    print(f"Generating predictions for {len(users)} users")
    
    # Create model
    print(f"\nCreating {args.model} model...")
    if args.model == "random":
        model = RandomRecommender(seed=42)
    elif args.model == "popularity":
        model = PopularityRecommender()
    elif args.model == "userknn":
        model = UserKNNRecommender(n_neighbors=args.n_neighbors)
    elif args.model == "itemknn":
        model = ItemKNNRecommender(n_neighbors=args.n_neighbors)
    elif args.model == "dense_user":
        model = DenseUserCF()
    elif args.model == "dense_item":
        model = DenseItemCF()
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    # Fit model
    print("Fitting model...")
    model.fit(interaction_matrix)
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions = generate_predictions(
        model=model,
        interaction_matrix=interaction_matrix,
        users=users,
        top_k=10,
        exclude_known=args.exclude_known,
    )
    
    # Save submission
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = args.model
        if args.model in ["userknn", "itemknn"]:
            model_name = f"{args.model}_{args.n_neighbors}"
        output_path = submissions_dir / f"submission_{model_name}_{timestamp}.csv"
    
    save_submission(predictions, output_path)
    
    # Verify format
    print("\nVerifying submission format...")
    result_df = pd.read_csv(output_path)
    print(f"  Rows: {len(result_df)}")
    print(f"  Columns: {list(result_df.columns)}")
    print(f"  Sample:")
    print(result_df.head())


if __name__ == "__main__":
    main()
