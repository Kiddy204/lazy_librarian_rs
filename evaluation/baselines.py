"""Baseline recommendation models for comparison.

These simple baselines help establish lower bounds for model performance:
- RandomRecommender: Random scores (sanity check)
- PopularityRecommender: Recommend most popular items
- UserKNNRecommender: User-based collaborative filtering
"""
import numpy as np
from scipy import sparse
from typing import Optional, Protocol
import warnings


class Recommender(Protocol):
    """Protocol defining the recommender interface."""
    
    def fit(self, train_matrix: sparse.csr_matrix) -> "Recommender":
        """Fit the model on training data."""
        ...
    
    def predict(self, user_id: int, candidate_items: np.ndarray) -> np.ndarray:
        """Predict scores for candidate items for a given user."""
        ...


class RandomRecommender:
    """
    Random baseline: assigns random scores to items.
    
    Expected performance: MAP@10 ~ 0.001, HR@10 ~ 0.05
    Used as a sanity check - any real model should beat this.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def fit(self, train_matrix: sparse.csr_matrix) -> "RandomRecommender":
        """No fitting needed for random baseline."""
        self.n_users, self.n_items = train_matrix.shape
        return self
    
    def predict(self, user_id: int, candidate_items: np.ndarray) -> np.ndarray:
        """Return random scores for candidate items."""
        return self.rng.random(len(candidate_items))
    
    def __repr__(self) -> str:
        return f"RandomRecommender(seed={self.seed})"


class PopularityRecommender:
    """
    Popularity baseline: recommends most popular items.
    
    Popularity is measured by the number of interactions in the training set.
    This is a strong baseline that often beats simple collaborative filtering.
    
    Expected performance: MAP@10 ~ 0.02-0.05, HR@10 ~ 0.15-0.25
    """
    
    def __init__(self):
        self.item_popularity = None
        self.n_users = None
        self.n_items = None
    
    def fit(self, train_matrix: sparse.csr_matrix) -> "PopularityRecommender":
        """
        Compute item popularity from training interactions.
        
        Args:
            train_matrix: Sparse user-item interaction matrix
        """
        self.n_users, self.n_items = train_matrix.shape
        # Sum interactions per item (column sum)
        self.item_popularity = np.array(train_matrix.sum(axis=0)).flatten()
        return self
    
    def predict(self, user_id: int, candidate_items: np.ndarray) -> np.ndarray:
        """
        Return popularity scores for candidate items.
        
        Args:
            user_id: User ID (not used, but required by interface)
            candidate_items: Array of item IDs to score
            
        Returns:
            Popularity scores (higher = more popular)
        """
        return self.item_popularity[candidate_items]
    
    def __repr__(self) -> str:
        return "PopularityRecommender()"


class UserKNNRecommender:
    """
    User-based k-Nearest Neighbors collaborative filtering.
    
    Recommends items that similar users have interacted with.
    Similarity is computed using cosine similarity.
    
    Expected performance: MAP@10 ~ 0.05-0.10, HR@10 ~ 0.25-0.35
    """
    
    def __init__(self, n_neighbors: int = 50, min_similarity: float = 0.0):
        """
        Args:
            n_neighbors: Number of similar users to consider
            min_similarity: Minimum similarity threshold
        """
        self.n_neighbors = n_neighbors
        self.min_similarity = min_similarity
        self.train_matrix = None
        self.user_norms = None
        self.n_users = None
        self.n_items = None
    
    def fit(self, train_matrix: sparse.csr_matrix) -> "UserKNNRecommender":
        """
        Store training matrix and precompute user norms for cosine similarity.
        
        Args:
            train_matrix: Sparse user-item interaction matrix
        """
        self.train_matrix = train_matrix.tocsr()
        self.n_users, self.n_items = train_matrix.shape
        
        # Precompute user norms for cosine similarity
        # norm = sqrt(sum of squared values per row)
        self.user_norms = np.sqrt(np.array(train_matrix.multiply(train_matrix).sum(axis=1)).flatten())
        # Avoid division by zero
        self.user_norms[self.user_norms == 0] = 1.0
        
        return self
    
    def _compute_user_similarities(self, user_id: int) -> np.ndarray:
        """
        Compute cosine similarities between target user and all other users.
        
        Args:
            user_id: Target user ID
            
        Returns:
            Array of similarity scores for all users
        """
        user_vec = self.train_matrix[user_id]
        
        # Dot product with all users
        similarities = self.train_matrix.dot(user_vec.T).toarray().flatten()
        
        # Normalize by norms (cosine similarity)
        similarities = similarities / (self.user_norms * self.user_norms[user_id])
        
        # Set self-similarity to 0 to exclude
        similarities[user_id] = 0.0
        
        return similarities
    
    def predict(self, user_id: int, candidate_items: np.ndarray) -> np.ndarray:
        """
        Predict scores using weighted average of neighbor preferences.
        
        Args:
            user_id: Target user ID
            candidate_items: Array of item IDs to score
            
        Returns:
            Predicted scores for each candidate item
        """
        # Get similarities to all users
        similarities = self._compute_user_similarities(user_id)
        
        # Apply minimum similarity threshold
        similarities[similarities < self.min_similarity] = 0.0
        
        # Get top-k neighbors
        if self.n_neighbors < self.n_users:
            # Partial sort for efficiency
            neighbor_indices = np.argpartition(-similarities, self.n_neighbors)[:self.n_neighbors]
        else:
            neighbor_indices = np.arange(self.n_users)
        
        neighbor_sims = similarities[neighbor_indices]
        
        # Filter out zero similarities
        valid_mask = neighbor_sims > 0
        neighbor_indices = neighbor_indices[valid_mask]
        neighbor_sims = neighbor_sims[valid_mask]
        
        if len(neighbor_indices) == 0:
            # No similar users found, return zeros
            return np.zeros(len(candidate_items))
        
        # Get neighbor interactions for candidate items
        neighbor_matrix = self.train_matrix[neighbor_indices][:, candidate_items].toarray()
        
        # Weighted sum of neighbor preferences
        scores = neighbor_sims @ neighbor_matrix
        
        # Normalize by sum of weights (optional, helps with varying neighbor counts)
        weight_sum = neighbor_sims.sum()
        if weight_sum > 0:
            scores = scores / weight_sum
        
        return scores
    
    def __repr__(self) -> str:
        return f"UserKNNRecommender(n_neighbors={self.n_neighbors}, min_similarity={self.min_similarity})"


class DenseUserCF:
    """
    Dense User-based Collaborative Filtering.
    
    Uses full similarity matrix and all users (not just top-K).
    This matches the traditional CF formula:
        pred = similarity @ interactions / sum(similarity)
    
    More memory intensive but often more accurate than sparse KNN.
    """
    
    def __init__(self, epsilon: float = 1e-9):
        self.epsilon = epsilon
        self.similarity = None
        self.train_matrix = None
        self.n_users = None
        self.n_items = None
    
    def fit(self, train_matrix: sparse.csr_matrix) -> "DenseUserCF":
        """Compute full user-user similarity matrix."""
        from sklearn.metrics.pairwise import cosine_similarity
        
        self.train_matrix = train_matrix.toarray() if sparse.issparse(train_matrix) else train_matrix
        self.n_users, self.n_items = self.train_matrix.shape
        
        print(f"  Computing user-user similarity matrix ({self.n_users} x {self.n_users})...")
        self.similarity = cosine_similarity(self.train_matrix)
        
        # Zero out self-similarity
        np.fill_diagonal(self.similarity, 0)
        
        return self
    
    def predict(self, user_id: int, candidate_items: np.ndarray) -> np.ndarray:
        """Predict using all users weighted by similarity."""
        user_sim = self.similarity[user_id]  # (n_users,)
        
        # Get interactions for candidate items from all users
        interactions = self.train_matrix[:, candidate_items]  # (n_users, n_candidates)
        
        # Weighted sum: similarity @ interactions
        scores = user_sim @ interactions  # (n_candidates,)
        
        # Normalize by sum of absolute similarities
        norm = np.abs(user_sim).sum() + self.epsilon
        scores = scores / norm
        
        return scores
    
    def predict_all(self) -> np.ndarray:
        """Predict all items for all users at once (batch mode)."""
        pred = self.similarity @ self.train_matrix
        norm = np.abs(self.similarity).sum(axis=1, keepdims=True) + self.epsilon
        return pred / norm
    
    def __repr__(self) -> str:
        return "DenseUserCF()"


class DenseItemCF:
    """
    Dense Item-based Collaborative Filtering.
    
    Uses full item-item similarity matrix.
    Formula: pred = similarity @ interactions.T / sum(similarity)
    """
    
    def __init__(self, epsilon: float = 1e-9):
        self.epsilon = epsilon
        self.similarity = None
        self.train_matrix = None
        self.n_users = None
        self.n_items = None
    
    def fit(self, train_matrix: sparse.csr_matrix) -> "DenseItemCF":
        """Compute full item-item similarity matrix."""
        from sklearn.metrics.pairwise import cosine_similarity
        
        self.train_matrix = train_matrix.toarray() if sparse.issparse(train_matrix) else train_matrix
        self.n_users, self.n_items = self.train_matrix.shape
        
        print(f"  Computing item-item similarity matrix ({self.n_items} x {self.n_items})...")
        self.similarity = cosine_similarity(self.train_matrix.T)
        
        # Zero out self-similarity
        np.fill_diagonal(self.similarity, 0)
        
        return self
    
    def predict(self, user_id: int, candidate_items: np.ndarray) -> np.ndarray:
        """Predict using item-item similarity."""
        user_interactions = self.train_matrix[user_id]  # (n_items,)
        
        # Get similarity for candidate items to all items
        candidate_sim = self.similarity[candidate_items]  # (n_candidates, n_items)
        
        # Weighted sum: each candidate's similarity to items user interacted with
        scores = candidate_sim @ user_interactions  # (n_candidates,)
        
        # Normalize
        norm = candidate_sim.sum(axis=1) + self.epsilon
        scores = scores / norm
        
        return scores
    
    def predict_all(self) -> np.ndarray:
        """Predict all items for all users at once (batch mode)."""
        # pred[u, i] = sum_j(sim[i,j] * interactions[u,j]) / sum_j(sim[i,j])
        pred = self.train_matrix @ self.similarity.T
        norm = self.similarity.sum(axis=1) + self.epsilon
        return pred / norm
    
    def __repr__(self) -> str:
        return "DenseItemCF()"


class ItemKNNRecommender:
    """
    Item-based k-Nearest Neighbors collaborative filtering.
    
    Recommends items similar to those the user has interacted with.
    Often more scalable than user-based KNN for large user bases.
    """
    
    def __init__(self, n_neighbors: int = 50, min_similarity: float = 0.0):
        """
        Args:
            n_neighbors: Number of similar items to consider
            min_similarity: Minimum similarity threshold
        """
        self.n_neighbors = n_neighbors
        self.min_similarity = min_similarity
        self.train_matrix = None
        self.item_norms = None
        self.item_similarity = None
        self.n_users = None
        self.n_items = None
    
    def fit(self, train_matrix: sparse.csr_matrix) -> "ItemKNNRecommender":
        """
        Store training matrix and precompute item similarities.
        
        Args:
            train_matrix: Sparse user-item interaction matrix
        """
        self.train_matrix = train_matrix.tocsr()
        self.n_users, self.n_items = train_matrix.shape
        
        # Transpose for item-based computations
        item_matrix = train_matrix.T.tocsr()
        
        # Compute item norms
        self.item_norms = np.sqrt(np.array(item_matrix.multiply(item_matrix).sum(axis=1)).flatten())
        self.item_norms[self.item_norms == 0] = 1.0
        
        return self
    
    def _get_item_similarities(self, item_id: int) -> np.ndarray:
        """Compute similarities between target item and all other items."""
        item_vec = self.train_matrix[:, item_id]
        
        # Dot product with all items
        similarities = self.train_matrix.T.dot(item_vec).toarray().flatten()
        
        # Normalize
        similarities = similarities / (self.item_norms * self.item_norms[item_id])
        similarities[item_id] = 0.0
        
        return similarities
    
    def predict(self, user_id: int, candidate_items: np.ndarray) -> np.ndarray:
        """
        Predict scores based on similarity to user's historical items.
        
        Args:
            user_id: Target user ID
            candidate_items: Array of item IDs to score
            
        Returns:
            Predicted scores for each candidate item
        """
        # Get user's historical items
        user_items = self.train_matrix[user_id].nonzero()[1]
        
        if len(user_items) == 0:
            return np.zeros(len(candidate_items))
        
        scores = np.zeros(len(candidate_items))
        
        # For each candidate, sum similarities to user's historical items
        for idx, candidate in enumerate(candidate_items):
            item_sims = self._get_item_similarities(candidate)
            
            # Sum similarities to user's items (top-k)
            user_item_sims = item_sims[user_items]
            
            if self.n_neighbors < len(user_item_sims):
                top_k_sims = np.partition(-user_item_sims, self.n_neighbors)[:self.n_neighbors]
                scores[idx] = -top_k_sims.sum()
            else:
                scores[idx] = user_item_sims.sum()
        
        return scores
    
    def __repr__(self) -> str:
        return f"ItemKNNRecommender(n_neighbors={self.n_neighbors}, min_similarity={self.min_similarity})"
