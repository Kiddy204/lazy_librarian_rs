from pathlib import Path
from typing import Optional

import hdbscan
import numpy as np
from sklearn.preprocessing import normalize


def cluster_embeddings_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = 8,
    min_samples: int = 2,
    checkpoint_path: Optional[Path] = None,
    force_regenerate: bool = False,
) -> np.ndarray:
    """
    Cluster embeddings with HDBSCAN with checkpoint support.
    Returns topic_ids per input row; -1 means noise.

    Args:
        embeddings: Array of embeddings to cluster
        min_cluster_size: HDBSCAN min_cluster_size parameter
        min_samples: HDBSCAN min_samples parameter
        checkpoint_path: Path to save/load clustering results
        force_regenerate: If True, ignore existing checkpoint

    Requires:
      - pip install hdbscan scikit-learn
    """

    # Load checkpoint if exists
    if checkpoint_path and checkpoint_path.exists() and not force_regenerate:
        print(f"Loading clustering checkpoint from {checkpoint_path}")
        labels = np.load(checkpoint_path)

        if len(labels) == len(embeddings):
            print(
                f"Using cached clustering results ({len(set(labels)) - (1 if -1 in labels else 0)} clusters)"
            )
            return labels
        else:
            print(
                f"Checkpoint shape mismatch ({len(labels)} vs {len(embeddings)}). Reclustering..."
            )

    X = normalize(embeddings)  # cosine-friendly with euclidean after normalization

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(X)

    # Save checkpoint
    if checkpoint_path:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(checkpoint_path, labels)
        print(f"Saved clustering checkpoint to {checkpoint_path}")

    return labels
