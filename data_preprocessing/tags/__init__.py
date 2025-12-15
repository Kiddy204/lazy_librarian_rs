"""
Clean and prepare item metadata: Subjects -> Topics clustering (OpenAI embeddings + HDBSCAN).

Outputs per item:
- subjects_list: raw subjects split
- subjects_norm: normalized subjects for clustering/matching
- topic_ids: list[int] of topic clusters (noise dropped)
- topic_multihot: np.ndarray binary vector (length = n_topics)
- has_subjects / n_subjects / has_topics / n_topics

Also returns artifacts to reuse at training/inference:
- subject_to_topic
- topic_index (topic_id -> vector index)
- topics_df (subject -> topic_id + counts)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data_preprocessing.tags.clean import normalize_subject, split_subjects
from data_preprocessing.tags.cluster import cluster_embeddings_hdbscan
from data_preprocessing.tags.embed import embed_texts_openai


@dataclass(frozen=True)
class TopicArtifacts:
    subject_to_topic: Dict[str, int]  # normalized subject -> topic_id (-1 possible)
    valid_topics: List[int]  # sorted topic_ids excluding -1
    topic_index: Dict[int, int]  # topic_id -> [0..T-1]
    topics_df: pd.DataFrame  # subject, topic_id, count(optional)
    T: int  # number of valid topics


def build_subject_topics(
    df: pd.DataFrame,
    subjects_col: str = "Subjects",
    openai_model: str = "text-embedding-3-small",
    openai_api_key: Optional[str] = None,
    min_cluster_size: int = 8,
    min_samples: int = 2,
    batch_size: int = 256,
    checkpoint_dir: Optional[str] = None,
    force_regenerate: bool = False,
) -> Tuple[pd.DataFrame, TopicArtifacts]:
    """
    Main entrypoint:
    - parse + normalize subjects
    - embed unique normalized subjects
    - cluster with HDBSCAN
    - map back to items: topic_ids + topic_multihot

    Args:
        subjects_col: Column name containing subjects (can be "Subjects" for raw or "subjects_list" for pre-parsed)
        checkpoint_dir: Directory to save embedding and clustering checkpoints
        force_regenerate: If True, ignore existing checkpoints

    Returns:
      df_out, artifacts
    """
    from pathlib import Path

    df_out = df.copy()

    # 1) subjects_list + subjects_norm
    # Check if subjects_col is already parsed (list) or raw (string)
    if subjects_col in df_out.columns:
        first_val = df_out[subjects_col].iloc[0] if len(df_out) > 0 else None
        if isinstance(first_val, list):
            # Already parsed
            df_out["subjects_list"] = df_out[subjects_col]
        else:
            # Raw string, need to parse
            df_out["subjects_list"] = df_out[subjects_col].apply(split_subjects)
    else:
        raise KeyError(
            f"Column '{subjects_col}' not found in DataFrame. Available columns: {df_out.columns.tolist()}"
        )

    df_out["subjects_norm"] = df_out["subjects_list"].apply(
        lambda xs: [normalize_subject(x) for x in xs]
    )
    df_out["has_subjects"] = df_out["subjects_norm"].apply(bool)
    df_out["n_subjects"] = df_out["subjects_norm"].apply(len)

    # 2) unique subjects (normalized)
    all_norm = pd.Series([s for xs in df_out["subjects_norm"] for s in xs])
    unique_subjects = (
        all_norm.dropna()
        .loc[lambda s: s != ""]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    # Setup checkpoint paths
    embed_checkpoint = None
    cluster_checkpoint = None
    if checkpoint_dir:
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        embed_checkpoint = checkpoint_path / "subject_embeddings.npy"
        cluster_checkpoint = checkpoint_path / "subject_clusters.npy"

    # 3) embed
    embeddings = embed_texts_openai(
        unique_subjects,
        model=openai_model,
        batch_size=batch_size,
        api_key=openai_api_key,
        checkpoint_path=embed_checkpoint,
        force_regenerate=force_regenerate,
    )

    # 4) cluster
    topic_ids = cluster_embeddings_hdbscan(
        embeddings,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        checkpoint_path=cluster_checkpoint,
        force_regenerate=force_regenerate,
    )

    # Diagnostics (printed, but keep minimal / safe)
    n_clusters = len(set(topic_ids)) - (1 if -1 in topic_ids else 0)
    noise_ratio = float((topic_ids == -1).mean())
    print(
        f"[subjects->topics] unique_subjects={len(unique_subjects)} topics={n_clusters} noise={noise_ratio:.3f}"
    )

    # 5) mapping + artifacts
    subject_to_topic = dict(zip(unique_subjects, topic_ids.tolist()))
    valid_topics = sorted([t for t in set(topic_ids.tolist()) if t != -1])
    topic_index = {t: i for i, t in enumerate(valid_topics)}
    T = len(valid_topics)

    # Topic inspection table
    topics_df = pd.DataFrame({"subject_norm": unique_subjects, "topic_id": topic_ids})
    # optional: add counts in corpus
    counts = all_norm.value_counts()
    topics_df["count_in_items"] = (
        topics_df["subject_norm"].map(counts).fillna(0).astype(int)
    )

    artifacts = TopicArtifacts(
        subject_to_topic=subject_to_topic,
        valid_topics=valid_topics,
        topic_index=topic_index,
        topics_df=topics_df,
        T=T,
    )

    # 6) map back to items
    def subjects_to_topics(subjects_norm: List[str]) -> List[int]:
        return sorted(
            {
                subject_to_topic.get(s, -1)
                for s in subjects_norm
                if subject_to_topic.get(s, -1) != -1
            }
        )

    df_out["topic_ids"] = df_out["subjects_norm"].apply(subjects_to_topics)
    df_out["has_topics"] = df_out["topic_ids"].apply(bool)
    df_out["n_topics"] = df_out["topic_ids"].apply(len)

    def make_multihot(ts: List[int]) -> np.ndarray:
        v = np.zeros(T, dtype=np.float32)
        for t in ts:
            v[topic_index[t]] = 1.0
        return v

    df_out["topic_multihot"] = df_out["topic_ids"].apply(
        make_multihot, convert_dtype=False
    )

    # invariants
    if T > 0:
        assert df_out["topic_multihot"].apply(len).nunique() == 1, (
            "All multihot vectors must have same length"
        )

    return df_out, artifacts
