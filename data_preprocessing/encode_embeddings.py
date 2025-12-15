"""Text embedding generation using OpenAI API."""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from data_preprocessing.clean_items import clean_items
from data_preprocessing.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    OUTPUT_FILES,
)
from data_preprocessing.load_data import load_interactions, load_items


def get_openai_client() -> OpenAI:
    """Initialize OpenAI client. Expects OPENAI_API_KEY in environment."""
    return OpenAI()


def embed_batch(
    client: OpenAI,
    texts: List[str],
    model: str = EMBEDDING_MODEL,
    dimensions: int = EMBEDDING_DIM,
) -> np.ndarray:
    """
    Embed a batch of texts using OpenAI API.
    """
    texts = [t if t.strip() else "[empty]" for t in texts]

    response = client.embeddings.create(input=texts, model=model, dimensions=dimensions)

    embeddings = [item.embedding for item in response.data]
    return np.array(embeddings)


def load_embeddings(path: Path) -> Optional[np.ndarray]:
    """Load embeddings from file if exists."""
    if path.exists():
        print(f"Loading existing embeddings from {path}")
        return np.load(path)
    return None


def generate_title_embeddings(
    df: pd.DataFrame,
    output_path: Path,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    checkpoint_every: int = 1000,
    force_regenerate: bool = False,
) -> np.ndarray:
    """
    Generate embeddings for all titles.

    Args:
        df: DataFrame with 'title_clean' column
        output_path: Where to save embeddings
        batch_size: Texts per API call
        checkpoint_every: Save checkpoint every N items
        force_regenerate: If True, regenerate even if file exists

    Returns:
        Embeddings array of shape (n_items, EMBEDDING_DIM)
    """
    # Check for existing embeddings
    if not force_regenerate:
        existing = load_embeddings(output_path)
        if existing is not None:
            # Validate shape matches current data
            if len(existing) == len(df):
                return existing
            print(
                f"Warning: existing embeddings shape {len(existing)} != data shape {len(df)}. Regenerating..."
            )

    client = get_openai_client()

    df = df.sort_values("i").reset_index(drop=True)
    titles = df["title_clean"].tolist()
    n_items = len(titles)

    # Check for checkpoint (partial progress)
    checkpoint_path = output_path.parent / "embeddings_checkpoint.npy"
    start_idx = 0

    if checkpoint_path.exists():
        embeddings = np.load(checkpoint_path)
        start_idx = (
            np.where(embeddings.sum(axis=1) != 0)[0][-1] + 1
            if embeddings.sum() != 0
            else 0
        )
        print(f"Resuming from checkpoint at index {start_idx}")
    else:
        embeddings = np.zeros((n_items, EMBEDDING_DIM), dtype=np.float32)

    # Process in batches
    for i in tqdm(range(start_idx, n_items, batch_size), desc="Generating embeddings"):
        batch_end = min(i + batch_size, n_items)
        batch_texts = titles[i:batch_end]

        try:
            batch_embeddings = embed_batch(client, batch_texts)
            embeddings[i:batch_end] = batch_embeddings
        except Exception as e:
            print(f"Error at batch {i}: {e}")
            np.save(checkpoint_path, embeddings)
            raise

        if (i + batch_size) % checkpoint_every < batch_size:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(checkpoint_path, embeddings)

        time.sleep(0.1)

    # Save final and cleanup
    np.save(output_path, embeddings)
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print(f"Saved embeddings to {output_path}")
    return embeddings


if __name__ == "__main__":
    items = load_items()
    interactions = load_interactions()
    items_clean = clean_items(items, interactions)
    generate_title_embeddings(items_clean, OUTPUT_FILES["title_embeddings"].resolve())
# # ============================================================
# # ALTERNATIVE: Local embeddings using sentence-transformers
# # ============================================================

# def generate_title_embeddings_local(
#     df: pd.DataFrame,
#     output_path: Path,
#     model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
#     batch_size: int = 64,
#     force_regenerate: bool = False
# ) -> np.ndarray:
#     """
#     Generate embeddings using local sentence-transformers model.
#     """
#     # Check for existing embeddings
#     if not force_regenerate:
#         existing = load_embeddings(output_path)
#         if existing is not None:
#             if len(existing) == len(df):
#                 return existing
#             print(f"Warning: existing embeddings shape {len(existing)} != data shape {len(df)}. Regenerating...")

#     from sentence_transformers import SentenceTransformer

#     model = SentenceTransformer(model_name)

#     df = df.sort_values("i").reset_index(drop=True)
#     titles = df["title_clean"].tolist()

#     print(f"Generating embeddings with {model_name}...")
#     embeddings = model.encode(
#         titles,
#         batch_size=batch_size,
#         show_progress_bar=True,
#         convert_to_numpy=True
#     )

#     np.save(output_path, embeddings)
#     print(f"Saved embeddings to {output_path}")

#     return embeddings
