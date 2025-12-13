"""Build final feature matrices combining all encodings."""
import pandas as pd
import numpy as np
from scipy import sparse
from pathlib import Path
from typing import Dict, Any

from config import OUTPUT_DIR, OUTPUT_FILES
from load_data import load_interactions, load_items
from clean_interactions import clean_interactions, get_interaction_stats
from clean_items import clean_items
from encode_categorical import encode_authors, encode_publishers, save_encoder
from encode_embeddings import generate_title_embeddings
from encode_subjects import build_subject_vocabulary, encode_subjects_tfidf, save_subject_vocab

def build_all_features(use_local_embeddings: bool = False) -> Dict[str, Any]:
    """
    Run complete feature building pipeline.
    
    Args:
        use_local_embeddings: If True, use sentence-transformers instead of OpenAI
        
    Returns:
        Dictionary with statistics and paths to saved artifacts
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stats = {}
    
    # ========== 1. Load and clean interactions ==========
    print("\n" + "="*50)
    print("Step 1: Processing interactions")
    print("="*50)
    
    interactions_raw = load_interactions()
    interactions_clean = clean_interactions(interactions_raw)
    interactions_clean.to_parquet(OUTPUT_FILES["interactions_clean"])
    
    stats["interactions"] = get_interaction_stats(interactions_clean)
    print(f"Saved: {OUTPUT_FILES['interactions_clean']}")
    
    # ========== 2. Load and clean items ==========
    print("\n" + "="*50)
    print("Step 2: Processing item metadata")
    print("="*50)
    
    items_raw = load_items()
    items_clean = clean_items(items_raw, interactions_clean)
    items_clean.to_parquet(OUTPUT_FILES["items_clean"])
    
    stats["items"] = {
        "n_items": len(items_clean),
        "pct_with_subjects": items_clean["has_subjects"].mean() * 100
    }
    print(f"Saved: {OUTPUT_FILES['items_clean']}")
    
    # ========== 3. Encode categorical features ==========
    print("\n" + "="*50)
    print("Step 3: Encoding categorical features")
    print("="*50)
    
    author_encoded, author_encoder = encode_authors(items_clean)
    publisher_encoded, publisher_encoder = encode_publishers(items_clean)
    
    save_encoder(author_encoder, OUTPUT_FILES["author_encoder"])
    save_encoder(publisher_encoder, OUTPUT_FILES["publisher_encoder"])
    
    stats["authors"] = {"n_categories": author_encoder.n_classes}
    stats["publishers"] = {"n_categories": publisher_encoder.n_classes}
    
    # ========== 4. Generate title embeddings ==========
    print("\n" + "="*50)
    print("Step 4: Generating title embeddings")
    print("="*50)
    
    if use_local_embeddings:
        from encode_embeddings import generate_title_embeddings_local
        title_embeddings = generate_title_embeddings_local(
            items_clean, 
            OUTPUT_FILES["title_embeddings"]
        )
    else:
        title_embeddings = generate_title_embeddings(
            items_clean,
            OUTPUT_FILES["title_embeddings"]
        )
    
    stats["title_embeddings"] = {"shape": title_embeddings.shape}
    print(f"Saved: {OUTPUT_FILES['title_embeddings']}")
    
    # ========== 5. Encode subjects ==========
    print("\n" + "="*50)
    print("Step 5: Encoding subjects (TF-IDF)")
    print("="*50)
    
    subject_vocab = build_subject_vocabulary(items_clean["subjects_list"])
    subject_tfidf = encode_subjects_tfidf(items_clean, subject_vocab)
    
    sparse.save_npz(OUTPUT_FILES["subject_matrix"], subject_tfidf)
    save_subject_vocab(subject_vocab, OUTPUT_FILES["subject_vocab"])
    
    stats["subjects"] = {
        "vocab_size": len(subject_vocab),
        "matrix_shape": subject_tfidf.shape
    }
    print(f"Saved: {OUTPUT_FILES['subject_matrix']}")
    
    # ========== 6. Build combined feature matrix ==========
    print("\n" + "="*50)
    print("Step 6: Building combined feature matrix")
    print("="*50)
    
    # Normalize embeddings
    title_embeddings_norm = title_embeddings / (
        np.linalg.norm(title_embeddings, axis=1, keepdims=True) + 1e-8
    )
    
    # Convert categorical to one-hot
    n_items = len(items_clean)
    author_onehot = sparse.csr_matrix(
        (np.ones(n_items), (np.arange(n_items), author_encoded)),
        shape=(n_items, author_encoder.n_classes)
    )
    publisher_onehot = sparse.csr_matrix(
        (np.ones(n_items), (np.arange(n_items), publisher_encoded)),
        shape=(n_items, publisher_encoder.n_classes)
    )
    
    # Combine: [title_emb | subject_tfidf | author_onehot | publisher_onehot]
    feature_matrix = sparse.hstack([
        sparse.csr_matrix(title_embeddings_norm),
        subject_tfidf,
        author_onehot,
        publisher_onehot
    ]).tocsr()
    
    sparse.save_npz(OUTPUT_FILES["feature_matrix"], feature_matrix)
    
    stats["combined_features"] = {
        "shape": feature_matrix.shape,
        "density": feature_matrix.nnz / np.prod(feature_matrix.shape)
    }
    print(f"Saved: {OUTPUT_FILES['feature_matrix']}")
    
    # ========== Summary ==========
    print("\n" + "="*50)
    print("Pipeline complete!")
    print("="*50)
    print(f"\nOutput files in: {OUTPUT_DIR}")
    for name, path in OUTPUT_FILES.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  {name}: {path.name} ({size_mb:.2f} MB)")
    
    return stats

