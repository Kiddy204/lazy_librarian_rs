"""
Docstring for data_preprocessing.tags.embed
"""

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
from openai import OpenAI
from tqdm import tqdm


def embed_texts_openai(
    texts: List[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 50,
    api_key: Optional[str] = None,
    checkpoint_path: Optional[Path] = None,
    checkpoint_every: int = 100,
    force_regenerate: bool = False,
) -> np.ndarray:
    """
    Embed texts using OpenAI embeddings endpoint with checkpoint support.

    Args:
        texts: List of texts to embed
        model: OpenAI embedding model name
        batch_size: Number of texts per API call
        api_key: OpenAI API key (or uses OPENAI_API_KEY env var)
        checkpoint_path: Path to save/load checkpoint file
        checkpoint_every: Save checkpoint every N texts processed
        force_regenerate: If True, ignore existing checkpoints

    Returns:
        Array of embeddings with shape (len(texts), embedding_dim)

    Requires:
      - pip install openai
      - OPENAI_API_KEY in env, or pass api_key.
    """

    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    if client.api_key is None:
        raise ValueError("Missing OPENAI_API_KEY (env var) or api_key argument.")

    n_texts = len(texts)
    start_idx = 0
    embeddings = None

    # Load checkpoint if exists
    if checkpoint_path and checkpoint_path.exists() and not force_regenerate:
        print(f"Loading checkpoint from {checkpoint_path}")
        embeddings = np.load(checkpoint_path)

        if len(embeddings) == n_texts:
            # Find last completed index
            start_idx = (
                np.where(embeddings.sum(axis=1) != 0)[0][-1] + 1
                if embeddings.sum() != 0
                else 0
            )
            if start_idx >= n_texts:
                print("All texts already embedded. Returning cached results.")
                return embeddings
            print(f"Resuming from index {start_idx}")
        else:
            print(
                f"Checkpoint shape mismatch ({len(embeddings)} vs {n_texts}). Starting fresh."
            )
            embeddings = None

    vecs: List[List[float]] = []

    for i in tqdm(range(start_idx, n_texts, batch_size), desc="Embedding texts"):
        batch_end = min(i + batch_size, n_texts)
        batch = texts[i:batch_end]

        try:
            resp = client.embeddings.create(
                model=model,
                input=batch,
                encoding_format="float",
            )
            batch_vecs = [d.embedding for d in resp.data]
            vecs.extend(batch_vecs)

            # Save checkpoint
            if checkpoint_path and (i + batch_size) % checkpoint_every < batch_size:
                if embeddings is None:
                    embedding_dim = len(batch_vecs[0])
                    embeddings = np.zeros((n_texts, embedding_dim), dtype=np.float32)

                embeddings[start_idx:batch_end] = vecs
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(checkpoint_path, embeddings)

        except Exception as e:
            print(f"Error at batch starting index {i}: {e}")
            if checkpoint_path and embeddings is not None:
                np.save(checkpoint_path, embeddings)
            raise

    # Combine with existing embeddings if resuming
    if embeddings is not None:
        embeddings[start_idx:] = vecs
        result = embeddings
    else:
        result = np.asarray(vecs, dtype=np.float32)

    # Save final checkpoint
    if checkpoint_path:
        print(f"Saving final embeddings to {checkpoint_path}")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(checkpoint_path, result)

    return result
