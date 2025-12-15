"""Build final feature matrices combining all encodings."""

import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import numpy as np
import pandas as pd
from scipy import sparse

from data_preprocessing.clean_interactions import (
    clean_interactions,
    get_interaction_stats,
)
from data_preprocessing.clean_items import clean_items
from data_preprocessing.config import (
    OUTPUT_DIR,
    OUTPUT_FILES,
    TOPIC_MIN_CLUSTER_SIZE,
    TOPIC_MIN_SAMPLES,
)
from data_preprocessing.encode_categorical import (
    encode_authors,
    encode_publishers,
    save_encoder,
)
from data_preprocessing.encode_embeddings import generate_title_embeddings
from data_preprocessing.encode_subjects import (
    build_subject_vocabulary,
    encode_subjects_tfidf,
    save_subject_vocab,
)
from data_preprocessing.load_data import load_interactions, load_items
from data_preprocessing.tags import build_subject_topics


def build_item_features(use_local_embeddings: bool = False) -> Dict[str, Any]:
    """
    Build all item features (embeddings, TF-IDF, topics, categorical encodings).
    This should be called BEFORE splitting interactions.

    Args:
        items_clean: Cleaned items DataFrame with 'i' column sorted
        use_local_embeddings: If True, use sentence-transformers instead of OpenAI

    Returns:
        Dictionary with statistics about generated features
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stats = {}

    # ========== 1. Load and clean interactions ==========
    print("\n" + "=" * 50)
    print("Step 1: Processing interactions")
    print("=" * 50)

    interactions_raw = load_interactions()
    interactions_clean = clean_interactions(interactions_raw)
    interactions_clean.to_parquet(OUTPUT_FILES["interactions_clean"])

    stats["interactions"] = get_interaction_stats(interactions_clean)
    print(f"Saved: {OUTPUT_FILES['interactions_clean']}")

    # ========== 2. Load and clean items ==========
    print("\n" + "=" * 50)
    print("Step 2: Processing item metadata")
    print("=" * 50)

    items_raw = load_items()
    items_clean = clean_items(items_raw, interactions_clean)

    stats["items"] = {
        "n_items": len(items_clean),
        "pct_with_subjects": items_clean["has_subjects"].mean() * 100,
    }

    # ========== 3. Encode categorical features ==========
    print("\n" + "=" * 50)
    print("Step 3: Encoding categorical features")
    print("=" * 50)

    author_encoded, author_encoder = encode_authors(items_clean)
    publisher_encoded, publisher_encoder = encode_publishers(items_clean)

    save_encoder(author_encoder, OUTPUT_FILES["author_encoder"])
    save_encoder(publisher_encoder, OUTPUT_FILES["publisher_encoder"])

    stats["authors"] = {"n_categories": author_encoder.n_classes}
    stats["publishers"] = {"n_categories": publisher_encoder.n_classes}

    # ========== 4. Generate title embeddings ==========
    print("\n" + "=" * 50)
    print("Step 4: Generating title embeddings")
    print("=" * 50)

    if use_local_embeddings:
        from data_preprocessing.encode_embeddings import generate_title_embeddings_local

        title_embeddings = generate_title_embeddings_local(
            items_clean, OUTPUT_FILES["title_embeddings"]
        )
    else:
        title_embeddings = generate_title_embeddings(
            items_clean, OUTPUT_FILES["title_embeddings"]
        )

    stats["title_embeddings"] = {"shape": title_embeddings.shape}
    print(f"Saved: {OUTPUT_FILES['title_embeddings']}")
    # ========== End of Step 4 ==========
    final_items = items_clean.copy()

    # ========== 5. Encode subjects (TF-IDF) ==========
    print("\n" + "=" * 50)
    print("Step 5: Encoding subjects (TF-IDF)")
    print("=" * 50)

    subject_vocab = build_subject_vocabulary(items_clean["subjects_list"])
    subject_tfidf = encode_subjects_tfidf(items_clean, subject_vocab)

    sparse.save_npz(OUTPUT_FILES["subject_matrix"], subject_tfidf)
    save_subject_vocab(subject_vocab, OUTPUT_FILES["subject_vocab"])

    stats["subjects"] = {
        "vocab_size": len(subject_vocab),
        "matrix_shape": subject_tfidf.shape,
    }
    print(f"Saved: {OUTPUT_FILES['subject_matrix']}")

    # add subject_vocab to final_items for reference
    final_items.attrs["subject_vocab"] = subject_vocab

    # ========== 5b. Encode topics (clustering) ==========
    print("\n" + "=" * 50)
    print("Step 5b: Clustering subjects into topics")
    print("=" * 50)

    items_with_topics, topic_artifacts = build_subject_topics(
        items_clean,
        subjects_col="subjects_list",  # Use pre-parsed subjects_list instead of raw Subjects
        min_cluster_size=TOPIC_MIN_CLUSTER_SIZE,
        min_samples=TOPIC_MIN_SAMPLES,
        checkpoint_dir=str(OUTPUT_DIR / "checkpoints"),
        force_regenerate=False,
    )

    # save items_with_topics dataframe for reference
    items_with_topics.to_parquet(OUTPUT_FILES["items_with_topics"])

    # Extract topic_multihot vectors and save as sparse matrix
    if topic_artifacts.T > 0:
        topic_multihot_matrix = np.vstack(items_with_topics["topic_multihot"].values)
        topic_multihot_sparse = sparse.csr_matrix(topic_multihot_matrix)
        sparse.save_npz(OUTPUT_FILES["topic_matrix"], topic_multihot_sparse)

        # Save artifacts for inference
        with open(OUTPUT_FILES["topic_artifacts"], "wb") as f:
            pickle.dump(topic_artifacts, f)

        stats["topics"] = {
            "n_topics": topic_artifacts.T,
            "matrix_shape": topic_multihot_sparse.shape,
            "pct_with_topics": items_with_topics["has_topics"].mean() * 100,
        }
        print(f"Saved: {OUTPUT_FILES['topic_matrix']}")
        print(
            f"Topics: {topic_artifacts.T} clusters, {stats['topics']['pct_with_topics']:.1f}% items have topics"
        )
    else:
        topic_multihot_sparse = sparse.csr_matrix((len(items_clean), 0))
        stats["topics"] = {"n_topics": 0, "matrix_shape": (len(items_clean), 0)}
        print("Warning: No topics generated (all subjects classified as noise)")

    # add column topic_multihot from items_with_topics to final_items
    final_items["topic_ids"] = items_with_topics["topic_ids"]
    final_items["topic_multihot"] = items_with_topics["topic_multihot"]

    # Update items_clean with topic metadata (but drop topic_multihot array column for parquet)
    # Keep only metadata columns: has_topics, n_topics
    final_items["has_topics"] = items_with_topics["has_topics"]
    final_items["n_topics"] = items_with_topics["n_topics"]

    # Re-save items_clean with topic columns
    # final_items.to_parquet(OUTPUT_FILES["items_clean"])
    print(f"Updated items_clean with topic metadata: {OUTPUT_FILES['items_clean']}")

    # ========== 6. Build combined feature matrix ==========
    print("\n" + "=" * 50)
    print("Step 6: Building combined feature matrix")
    print("=" * 50)

    # Normalize embeddings
    title_embeddings_norm = title_embeddings / (
        np.linalg.norm(title_embeddings, axis=1, keepdims=True) + 1e-8
    )

    # Convert categorical to one-hot
    n_items = len(items_clean)
    author_onehot = sparse.csr_matrix(
        (np.ones(n_items), (np.arange(n_items), author_encoded)),
        shape=(n_items, author_encoder.n_classes),
    )
    publisher_onehot = sparse.csr_matrix(
        (np.ones(n_items), (np.arange(n_items), publisher_encoded)),
        shape=(n_items, publisher_encoder.n_classes),
    )

    # Combine: [title_emb | subject_tfidf | topic_multihot | author_onehot | publisher_onehot]
    feature_matrix = sparse.hstack(
        [
            sparse.csr_matrix(title_embeddings_norm),
            subject_tfidf,
            topic_multihot_sparse,
            author_onehot,
            publisher_onehot,
        ]
    ).tocsr()

    sparse.save_npz(OUTPUT_FILES["feature_matrix"], feature_matrix)

    stats["combined_features"] = {
        "shape": feature_matrix.shape,
        "density": feature_matrix.nnz / np.prod(feature_matrix.shape),
    }
    print(f"Saved: {OUTPUT_FILES['feature_matrix']}")

    # add column author_onehot, publisher_onehot, subject_tfidf to final_items as one hot / sparse matrix
    # These lines do not work
    # final_items.attrs["author_onehot"] = author_onehot
    # final_items.attrs["publisher_onehot"] = publisher_onehot
    # final_items.attrs["subject_tfidf"] = subject_tfidf
    # final_items.attrs["title_embeddings"] = title_embeddings_norm

    # ========== Summary ==========
    print("\n" + "=" * 50)
    print("Pipeline complete!")
    print("=" * 50)
    print(f"\nOutput files in: {OUTPUT_DIR}")
    for name, path in OUTPUT_FILES.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  {name}: {path.name} ({size_mb:.2f} MB)")

    # ========== Summary ==========
    print("\n" + "=" * 50)
    print("Feature building complete!")
    print("=" * 50)
    print(f"\nOutput files in: {OUTPUT_DIR}")
    for name, path in OUTPUT_FILES.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  {name}: {path.name} ({size_mb:.2f} MB)")

    # Save this output pandas dataframe like clean_items for future reference
    final_items.to_parquet(OUTPUT_FILES["items_clean"])
    return stats
