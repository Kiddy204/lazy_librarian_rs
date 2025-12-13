"""Subject tag encoding using TF-IDF."""
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import json
from typing import List
from config import MIN_SUBJECT_FREQ, MAX_SUBJECT_FEATURES


def build_subject_vocabulary(
    subjects_lists: pd.Series,
    min_freq: int = MIN_SUBJECT_FREQ,
    max_features: int = MAX_SUBJECT_FEATURES
) -> List[str]:
    """
    Build vocabulary of subjects with frequency filtering.
    
    Args:
        subjects_lists: Series of subject lists
        min_freq: Minimum occurrences to include subject
        max_features: Maximum vocabulary size
        
    Returns:
        List of subjects in vocabulary
    """
    counter = Counter()
    for subjects in subjects_lists:
        counter.update(subjects)
    
    vocab = [
        subj for subj, count in counter.most_common(max_features)
        if count >= min_freq
    ]
    
    print(f"Subject vocabulary: {len(counter)} total -> {len(vocab)} after filtering (min_freq={min_freq})")
    return vocab


# Unique separator that won't appear in subjects
SUBJECT_SEPARATOR = "|||"


def subjects_to_text(subjects_list: List[str]) -> str:
    """Convert subject list to separated string for TF-IDF."""
    return SUBJECT_SEPARATOR.join(subjects_list) if subjects_list else ""


def encode_subjects_tfidf(
    df: pd.DataFrame,
    vocab: List[str]
) -> sparse.csr_matrix:
    """
    Encode subjects as TF-IDF matrix.
    
    Args:
        df: DataFrame with 'subjects_list' column, sorted by 'i'
        vocab: Subject vocabulary
        
    Returns:
        Sparse TF-IDF matrix of shape (n_items, len(vocab))
    """
    df = df.sort_values("i").copy()
    texts = df["subjects_list"].apply(subjects_to_text)
    
    # Custom tokenizer: split on our unique separator
    def tokenize(text):
        if not text:
            return []
        return [t.strip() for t in text.split(SUBJECT_SEPARATOR) if t.strip()]
    
    vectorizer = TfidfVectorizer(
        vocabulary={subj: idx for idx, subj in enumerate(vocab)},
        lowercase=True,
        tokenizer=tokenize,
        token_pattern=None,
    )
    
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    density = tfidf_matrix.nnz / np.prod(tfidf_matrix.shape)
    print(f"Subject TF-IDF matrix: {tfidf_matrix.shape}, density: {density:.4f}")
    return tfidf_matrix


def save_subject_vocab(vocab: List[str], path: str) -> None:
    """Save vocabulary to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def load_subject_vocab(path: str) -> List[str]:
    """Load vocabulary from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
