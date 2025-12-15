"""
Clean raw Subjects -> cleaned subjects ready for clustering to create Topics.

"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional

import pandas as pd

# ----------------------------
# Subjects parsing + normalization
# ----------------------------


def split_subjects(raw: Optional[str]) -> List[str]:
    """Split semicolon-separated subjects into a list (keeps original casing/punct)."""
    if pd.isna(raw) or raw is None:
        return []
    return [p.strip() for p in str(raw).split(";") if p.strip()]


def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


def normalize_subject(s: str) -> str:
    """
    Normalize a subject string to reduce duplicates:
    - lowercase
    - strip accents
    - normalize whitespace
    - keep '--' structure but normalize spacing
    - remove most punctuation
    """
    s = str(s).strip().lower()
    s = _strip_accents(s)

    # preserve hierarchical hint "--" but normalize around it
    s = re.sub(r"\s*--\s*", " -- ", s)

    # keep alphanum, spaces, hyphen
    s = re.sub(r"[^a-z0-9\- ]+", " ", s)

    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s
