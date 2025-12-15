"""Clean and prepare item metadata."""

import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import re
from typing import List, Optional

from data_preprocessing.config import UNKNOWN_AUTHOR, UNKNOWN_PUBLISHER
from data_preprocessing.load_data import load_interactions, load_items


def normalize_text(text: Optional[str]) -> str:
    """Normalize text: strip, collapse whitespace."""
    if pd.isna(text) or text is None:
        return ""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)  # Collapse whitespace
    return text


def parse_subjects(subjects_str: Optional[str]) -> List[str]:
    """Parse semicolon-separated subjects into clean list."""
    if pd.isna(subjects_str) or not subjects_str:
        return []
    subjects = [s.strip().lower() for s in str(subjects_str).split(";")]
    subjects = [s for s in subjects if s]  # Remove empty
    return subjects


def clean_author(author: Optional[str]) -> str:
    """
    Clean author field.
    Format: "Lastname, Firstname, birth-death" -> "lastname firstname"
    """
    if pd.isna(author) or not author:
        return UNKNOWN_AUTHOR

    author = str(author).strip()
    # Remove birth-death years (e.g., "1947-", "1923-2010")
    author = re.sub(r",?\s*\d{4}-?\d{0,4}\.?$", "", author)
    # Normalize: lowercase, collapse whitespace
    author = re.sub(r"\s+", " ", author.lower().strip())

    return author if author else UNKNOWN_AUTHOR


def clean_publisher(publisher: Optional[str]) -> str:
    """Clean publisher field."""
    if pd.isna(publisher) or not publisher:
        return UNKNOWN_PUBLISHER
    return normalize_text(publisher)


def clean_title(title: str) -> str:
    """Clean title for embedding generation."""
    title = normalize_text(title)
    # Remove trailing " /" which appears in many titles
    title = re.sub(r"\s*/\s*$", "", title)
    return title


def extract_isbns(raw: str) -> list[str]:
    """Extract ISBN-10/13 candidates from a raw field like '978...; 270...; ...'."""
    if pd.isna(raw) or raw is None:
        return []
    s = str(raw)
    # keep digits and X/x, split on non alnum
    tokens = re.split(r"[^0-9Xx]+", s)
    tokens = [t.upper() for t in tokens if t]
    # keep plausible lengths
    out = [t for t in tokens if len(t) in (10, 13)]
    return out


def isbn13_from_isbn10(isbn10: str) -> str | None:
    """Convert ISBN-10 to ISBN-13 (978 prefix) when possible. Returns None if invalid format."""
    if not re.fullmatch(r"[0-9]{9}[0-9X]", isbn10):
        return None
    core = "978" + isbn10[:9]
    # compute ISBN-13 check digit
    total = 0
    for i, ch in enumerate(core):
        d = int(ch)
        total += d * (1 if i % 2 == 0 else 3)
    check = (10 - (total % 10)) % 10
    return core + str(check)


def normalize_isbn_list(raw: str) -> list[str]:
    """Return canonical ISBN-13 when possible, else keep cleaned token."""
    isbns = extract_isbns(raw)
    norm = []
    for t in isbns:
        if len(t) == 10:
            t13 = isbn13_from_isbn10(t)
            norm.append(t13 if t13 else t)
        else:
            norm.append(t)
    # unique, stable order
    return sorted(set(norm))


def build_macro_isbn(
    df: pd.DataFrame, id_col="i", isbn_col="ISBN Valid"
) -> pd.DataFrame:
    df = df.copy()

    # 1) normalize to list of isbn13/10 tokens (using your normalize_isbn_list)
    df["isbn_norm"] = df[isbn_col].apply(normalize_isbn_list)

    # 2) explode to (row_id, isbn)
    edges = df[[id_col, "isbn_norm"]].explode("isbn_norm").dropna()
    edges = edges[edges["isbn_norm"] != ""]

    # 3) Build adjacency via ISBN -> list of row_ids
    isbn_to_rows = edges.groupby("isbn_norm")[id_col].apply(list).to_dict()

    # Union-Find
    parent = {rid: rid for rid in df[id_col].tolist()}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # 4) Union rows that share an ISBN
    for _, rows in isbn_to_rows.items():
        if len(rows) > 1:
            base = rows[0]
            for r in rows[1:]:
                union(base, r)

    # 5) Assign macro id per component
    # Use smallest row id as stable representative
    root = df[id_col].map(find)
    macro_rep = root.groupby(root).transform("min")  # stable rep id
    df["macro_isbn"] = "M" + macro_rep.astype(str)

    return df


def clean_items(df: pd.DataFrame, interactions_df: List[int]) -> pd.DataFrame:
    """
    Clean all item metadata fields.

    Returns DataFrame with cleaned columns:
    - i: item ID (unchanged)
    - title_clean: cleaned title for embeddings
    - title_original: original title (preserved)
    - author_clean: normalized author
    - publisher_clean: normalized publisher
    - subjects_list: list of subject tags
    - has_subjects: boolean flag
    """
    df = df.copy()
    interacted_items = set(interactions_df["i"].unique())

    # Remove cold-start items
    n_before = len(df)
    df = df[df["i"].isin(interacted_items)]
    n_removed = n_before - len(df)
    print(f"Removed {n_removed} cold-start items ({n_removed / n_before * 100:.1f}%)")
    # Clean each field
    df["title_original"] = df["Title"]
    df["title_clean"] = df["Title"].apply(clean_title)
    df["author_clean"] = df["Author"].apply(clean_author)
    df["publisher_clean"] = df["Publisher"].apply(clean_publisher)
    df["subjects_list"] = df["Subjects"].apply(parse_subjects)
    df["has_subjects"] = df["subjects_list"].apply(lambda x: len(x) > 0)

    # df2 = build_macro_isbn(df, id_col="i", isbn_col="ISBN Valid")
    # df2["n_isbns"] = df2["isbn_norm"].apply(len)

    # print(
    #     f"Built {df2['macro_isbn'].nunique()} macro ISBN groups"
    #     f" from {df2['i'].nunique()} items."
    # )
    # print(df2[["i", "ISBN Valid", "n_isbns", "macro_isbn"]].head)

    # print(
    #     f"Built {df2['macro_isbn'].nunique()} macro ISBN groups "
    #     f"from {df2['i'].nunique()} items.\n"
    #     f"Avg ISBNs per item: {df2['n_isbns'].mean():.2f} "
    #     f"(min={df2['n_isbns'].min()}, max={df2['n_isbns'].max()})"
    # )
    # Select and reorder columns
    df_clean = df[
        [
            "i",
            "title_clean",
            "title_original",
            "author_clean",
            "publisher_clean",
            "subjects_list",
            "has_subjects",
        ]
    ].copy()

    # Validate
    assert df_clean["i"].is_unique, "Item IDs must be unique"
    assert df_clean["title_clean"].notna().all(), "All items must have titles"

    return df_clean


def get_item_stats(df: pd.DataFrame) -> dict:
    """Compute statistics for validation."""
    return {
        "n_items": len(df),
        "n_items_with_subjects": str(
            round((df["has_subjects"].sum() / len(df) * 100), 2)
        )
        + "%",
        "n_items_with_authors": str(
            round((len(df[df["author_clean"] != UNKNOWN_AUTHOR]) / len(df) * 100), 2)
        )
        + "%",
        "n_items_with_publishers": str(
            round(
                (len(df[df["publisher_clean"] != UNKNOWN_PUBLISHER]) / len(df) * 100), 2
            )
        )
        + "%",
    }


if __name__ == "__main__":
    items = load_items()
    interactions = load_interactions()
    items_clean = clean_items(items, interactions)
    item_stats = get_item_stats(items_clean)
    print(item_stats)
    print(f"columns: {items_clean.columns}")
    print(f"items_clean shape: {items_clean.shape}")
