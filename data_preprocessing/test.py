# Extended verification script
import pandas as pd
import numpy as np
from scipy import sparse

# Load artifacts
interactions = pd.read_parquet("data/processed/interactions_clean.parquet")
items = pd.read_parquet("data/processed/items_clean.parquet")
embeddings = np.load("data/processed/title_embeddings.npy")
features = sparse.load_npz("data/processed/item_features.npz")

print("=" * 50)
print("DETAILED VERIFICATION")
print("=" * 50)

# 1. Interactions
print(f"\n[Interactions]")
print(f"  Rows: {len(interactions)}")
print(f"  Unique users: {interactions['u'].nunique()}")
print(f"  Unique items: {interactions['i'].nunique()}")
print(f"  Has weights: {'weight' in interactions.columns}")
print(f"  Columns: {list(interactions.columns)}")

# 2. Items alignment
print(f"\n[Items]")
print(f"  Rows: {len(items)}")
print(f"  Item ID range: {items['i'].min()} - {items['i'].max()}")
print(f"  IDs contiguous: {items['i'].nunique() == items['i'].max() + 1}")
print(f"  Columns: {list(items.columns)}")

# 3. Embeddings
print(f"\n[Embeddings]")
print(f"  Shape: {embeddings.shape}")
print(f"  Has NaN: {np.isnan(embeddings).any()}")
print(f"  Norm range: {np.linalg.norm(embeddings, axis=1).min():.3f} - {np.linalg.norm(embeddings, axis=1).max():.3f}")

# 4. Feature matrix
print(f"\n[Feature Matrix]")
print(f"  Shape: {features.shape}")
print(f"  Density: {features.nnz / np.prod(features.shape):.4f}")
print(f"  Non-zero entries: {features.nnz}")

# 5. Alignment check
print(f"\n[Alignment Check]")
print(f"  Items == Embeddings rows: {len(items) == embeddings.shape[0]}")
print(f"  Items == Features rows: {len(items) == features.shape[0]}")

# 6. Feature breakdown (estimate)
print(f"\n[Feature Breakdown]")
total_features = features.shape[1]
title_dim = embeddings.shape[1]
remaining = total_features - title_dim
print(f"  Title embeddings: {title_dim}")
print(f"  Other features: {remaining} (subjects + author + publisher)")