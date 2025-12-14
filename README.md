# Book Recommendation System - Data Quality Analysis

## Executive Summary

This report analyzes two datasets for building a hybrid book recommendation system: **interactions** (user-item behaviors) and **items** (book metadata). The data comes from what appears to be a **Swiss library system** with French and German language materials.

**Key Findings:**
- ✅ Data is clean with no orphan interactions
- ✅ Good subject coverage (85.5% of interacted items have subjects)
- ⚠️ High sparsity (99.93%) - typical for recommender systems
- ⚠️ 26.5% of interactions are repeat borrows (implicit signal strength)
- ⚠️ 14.5% of items missing subjects (cold-start challenge for content-based)

---

## 1. Interactions Dataset

### Overview
| Metric | Value |
|--------|-------|
| Total interactions | 87,047 |
| Unique (user, item) pairs | 64,003 |
| Unique users | 7,838 |
| Unique items (interacted) | 15,109 |
| Date range | Jan 2, 2023 → Oct 14, 2024 (651 days) |
| Matrix sparsity | 99.93% |

### Data Quality
| Column | Type | Missing | Notes |
|--------|------|---------|-------|
| `u` | int64 | 0 | User index (0-7837), contiguous |
| `i` | int64 | 0 | Item index (0-15290) |
| `t` | float64 | 0 | Unix timestamp |

### User Behavior Distribution

| Interactions per User | Count | Percentage |
|-----------------------|-------|------------|
| 3-5 (light users) | 3,737 | 47.7% |
| 6-10 | 1,884 | 24.0% |
| 11-20 | 1,224 | 15.6% |
| 21-50 | 744 | 9.5% |
| 50+ (power users) | 249 | 3.2% |

**Insight:** Nearly half the users have minimal history (3-5 interactions). This suggests:
- Cold-start handling is important for users
- Content-based fallback will be valuable for light users

### Item Popularity Distribution

| Interactions per Item | Count | Percentage |
|-----------------------|-------|------------|
| 1 (long tail) | 692 | 4.6% |
| 2-5 | 9,754 | 64.6% |
| 6-10 | 3,579 | 23.7% |
| 11-20 | 789 | 5.2% |
| 20+ (popular) | 295 | 2.0% |

**Top 5 Most Popular Items:**
1. Le Petit Robert (dictionary) - 380 interactions
2. Demon Slayer (manga) - 357 interactions
3. Vagabond (manga) - 305 interactions
4. Spy x Family (manga) - 257 interactions
5. L'Arabe du futur (graphic novel) - 217 interactions

**Insight:** Strong manga/graphic novel presence suggests distinct user segments.

### Repeat Interactions (Important Finding!)

| Metric | Value |
|--------|-------|
| Repeat (u,i) pairs | 10,376 pairs |
| Total repeat interactions | 23,044 (26.5% of all) |

**Distribution of repeats:**
- 2 times: 5,642 pairs
- 3 times: 2,196 pairs
- 4 times: 1,125 pairs
- 5+ times: 1,413 pairs

**Exploitation Potential:** Repeat interactions are a strong signal! Users who borrow the same item multiple times show high affinity. This can be used to:
- Weight interactions (repeat = stronger preference)
- Create an implicit rating proxy (count → score)

### Temporal Patterns

Monthly interaction volume shows:
- **Peak periods:** Jan-Mar 2023, Jan-Feb 2024
- **Low periods:** Jul-Aug (summer), recent months (Oct 2024 partial)
- **Trend:** Declining activity in 2024 (possible data collection issue?)

**Exploitation Potential:** Time-aware splitting for train/test (avoid data leakage).

---

## 2. Items Dataset

### Overview
| Metric | Value |
|--------|-------|
| Total items | 15,291 |
| Items with interactions | 15,109 (98.8%) |
| Cold-start items | 182 (1.2%) |

### Column Quality Assessment

| Column | Missing | % Missing | Unique Values | Quality |
|--------|---------|-----------|---------------|---------|
| Title | 0 | 0% | 14,576 | ✅ Excellent |
| Author | 2,653 | 17.4% | 9,357 | ⚠️ Moderate gaps |
| ISBN Valid | 723 | 4.7% | - | ✅ Good |
| Publisher | 25 | 0.2% | 4,337 | ✅ Excellent |
| Subjects | 2,223 | 14.5% | 23,305 unique tags | ⚠️ Key gap |
| i | 0 | 0% | 15,291 | ✅ Contiguous index |

### Title Analysis

- **Duplicate titles:** 514 (different editions of same work)
- Examples: "Oeuvres complètes" (35 times), "Oeuvres" (13 times)
- **Title length:** 4-569 chars, mean 49.6 chars

**Exploitation Potential:**
- Can generate embeddings from titles
- Need deduplication strategy for same-work recommendations
- Multilingual titles (French/German) - consider language-aware embeddings

### Author Analysis

- **Coverage:** 82.6% of interacted items have authors
- **No multi-author entries** (no semicolons in author field)
- **Format:** "Lastname, Firstname, birth-death" (e.g., "Cicurel, Francine, 1947-")

**Exploitation Potential:**
- Author embeddings or categorical features
- Author popularity features
- Handle missing as "Unknown" category

### Publisher Analysis

- **Excellent coverage:** 99.8%
- **Cardinality:** 4,337 unique publishers
- **Top publishers:** Gallimard (611), Flammarion (241), Albin Michel (228)

**Exploitation Potential:**
- Categorical feature for content-based
- Publisher similarity (genre proxy)

### Subjects Analysis (Most Valuable for Content-Based)

| Metric | Value |
|--------|-------|
| Items with subjects | 13,068 (85.5% of catalog) |
| Interacted items with subjects | 12,923 (85.5% of interacted) |
| Total subject tags | 49,216 |
| Unique subjects | 23,305 |
| Subjects per item | 1-85, mean 3.8 |

**Subject Richness Distribution:**
- 1 subject: 3,160 items (24%)
- 2-3 subjects: 5,303 items (41%)
- 4-6 subjects: 2,905 items (22%)
- 7+ subjects: 1,700 items (13%)

**Top Subjects:**
1. Bandes dessinées (comics) - 993
2. Schweiz (Switzerland) - 918
3. Suisse (Switzerland FR) - 446
4. Guides pratiques - 351
5. Mangas - 246

**Exploitation Potential:**
- Multi-hot encoding for subject tags
- Subject embeddings (TF-IDF or learned)
- Subject co-occurrence for item similarity
- Language detection via subjects

### Language/Locale

Evidence of **bilingual French/German collection**:
- French indicators: 3,359 occurrences
- German indicators: 1,426 occurrences

**Consideration:** May need language-aware embeddings or treat language as a feature.

---

## 3. Cross-Dataset Analysis

### Item Coverage
| Metric | Count | % |
|--------|-------|---|
| Items in catalog | 15,291 | 100% |
| Items with interactions | 15,109 | 98.8% |
| Cold-start items | 182 | 1.2% |
| Orphan interactions | 0 | 0% |

**Good news:** Nearly full coverage and no data integrity issues.

### Feature Availability for Interacted Items

| Feature | Available | % of Interacted |
|---------|-----------|-----------------|
| Title | 15,109 | 100% |
| Subjects | 12,923 | 85.5% |
| Author | 12,477 | 82.6% |
| Publisher | 15,084 | 99.8% |

---

## 4. Recommendations for Hybrid Approach

### Collaborative Filtering Component

**Data preparation:**
- Convert to implicit feedback matrix (user × item)
- Consider weighting by interaction count (repeat borrows = stronger signal)
- Time-based train/test split (last 20% of interactions for test)
- Filter users with <3 interactions already done

**Suggested algorithms:**
- Matrix Factorization (ALS for implicit feedback)
- Neural Collaborative Filtering
- LightGCN (graph-based)

### Content-Based Component

**Feature engineering priorities:**

1. **Subjects (highest value)**
   - Multi-hot encoding (23K dims) or TF-IDF
   - Can reduce dimensionality via embeddings
   - 85.5% coverage

2. **Title embeddings**
   - Use multilingual model (French/German)
   - E.g., `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
   - 100% coverage

3. **Author (categorical)**
   - Entity embedding or simple categorical encoding
   - Group rare authors into "Other"
   - 82.6% coverage

4. **Publisher (categorical)**
   - Lower priority, but good coverage
   - May capture genre patterns

### Handling Missing Data

| Feature | Strategy |
|---------|----------|
| Subjects | Use title embeddings as fallback |
| Author | "Unknown" category + rely on other features |
| For cold-start items | Pure content-based recommendation |
| For cold-start users | Popular items + content diversity |

### Suggested Hybrid Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    HYBRID MODEL                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────┐      ┌─────────────────────────┐  │
│  │ Collaborative   │      │    Content-Based        │  │
│  │ Filtering       │      │                         │  │
│  │                 │      │  ┌─────────────────┐    │  │
│  │ User embeddings │      │  │ Title embedding │    │  │
│  │ Item embeddings │      │  │ Subject TF-IDF  │    │  │
│  │ (from interactions)    │  │ Author/Publisher│    │  │
│  └────────┬────────┘      │  └────────┬────────┘    │  │
│           │               │           │             │  │
│           └───────────────┼───────────┘             │  │
│                           │                         │  │
│                    ┌──────▼──────┐                  │  │
│                    │   Fusion    │                  │  │
│                    │   Layer     │                  │  │
│                    └──────┬──────┘                  │  │
│                           │                         │  │
│                    ┌──────▼──────┐                  │  │
│                    │ Final Score │                  │  │
│                    └─────────────┘                  │  │
└─────────────────────────────────────────────────────────┘
```

---

## 5. Implemented: Data Preprocessing Pipeline

The `data_preprocessing/` module provides a complete feature engineering pipeline. All modules are production-ready and tested.

### Pipeline Architecture

```
data/
├── interactions.csv          # Raw interactions (u, i, t)
├── items.csv                 # Raw item metadata
└── processed/                # Generated by pipeline
    ├── interactions_clean.parquet
    ├── items_clean.parquet
    ├── title_embeddings.npy      # (n_items, 512) dense
    ├── subject_tfidf.npz         # (n_items, vocab_size) sparse
    ├── subject_vocab.json
    ├── author_encoder.pkl
    ├── publisher_encoder.pkl
    ├── item_features.npz         # Combined feature matrix
    ├── pipeline_stats.json
    ├── train.parquet             # Training split
    ├── val.parquet               # Validation split
    ├── test.parquet              # Test split
    ├── train_matrix.npz          # Sparse interaction matrix (weighted)
    ├── val_matrix.npz            # Sparse interaction matrix (binary)
    ├── test_matrix.npz           # Sparse interaction matrix (binary)
    └── split_stats.json          # Split statistics
```

### Module Overview

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `config.py` | Central configuration | Paths, thresholds, model settings |
| `load_data.py` | Data loading | `load_interactions()`, `load_items()` |
| `clean_interactions.py` | Interaction preprocessing | Adds datetime, interaction rank per (u,i) pair |
| `clean_items.py` | Item metadata cleaning | Title normalization, author/publisher cleaning, subject parsing |
| `encode_subjects.py` | Subject TF-IDF encoding | Vocabulary building with frequency filtering |
| `encode_categorical.py` | Author/Publisher encoding | `FrequencyAwareLabelEncoder` groups rare categories |
| `encode_embeddings.py` | Title embeddings | OpenAI API with checkpointing (local option available) |
| `build_features.py` | Feature matrix assembly | Combines all features into single sparse matrix |
| `split_data.py` | Train/val/test splitting | Temporal split, cold-start filtering, matrix creation |
| `run_pipeline.py` | Entry point | CLI with `--local-embeddings` option |

### Running the Pipeline

```bash
# Install dependencies
pip install pandas numpy scipy scikit-learn openai tqdm pyarrow

# Set OpenAI API key (for title embeddings)
export OPENAI_API_KEY="your-key-here"

# Run the full pipeline
cd data_preprocessing
python run_pipeline.py

# Or use local embeddings (no API key needed)
python run_pipeline.py --local-embeddings
```

### Configuration Options (`config.py`)

| Setting | Default | Description |
|---------|---------|-------------|
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI model for title embeddings |
| `EMBEDDING_DIM` | 512 | Embedding dimensionality |
| `MIN_SUBJECT_FREQ` | 5 | Minimum occurrences to keep a subject |
| `MAX_SUBJECT_FEATURES` | 5000 | Cap on subject vocabulary size |
| `RARE_AUTHOR_THRESHOLD` | 2 | Authors with fewer items → "__RARE_AUTHOR__" |
| `RARE_PUBLISHER_THRESHOLD` | 3 | Publishers with fewer items → "__RARE_PUBLISHER__" |

### Output Feature Matrix

The combined `item_features.npz` matrix concatenates:

```
[title_embeddings | subject_tfidf | author_onehot | publisher_onehot]
     (512 dims)    (~2000 dims)    (~4000 dims)    (~1500 dims)
```

All features are normalized and ready for similarity computation or model input.

---

## 6. Implemented: Train/Val/Test Splits

### Why Temporal Splitting?

For recommendation systems, **temporal splits** prevent data leakage:
- Training on future interactions to predict past = cheating
- Real-world deployment only has access to historical data

### Split Strategy

```
Timeline: ─────────────────────────────────────────────────────────────────►
          │          TRAIN (70%)           │   VAL (10%)   │   TEST (20%)  │
          Jan 2023                      Jan 2024         Feb 2024       Oct 2024
```

### Module: `split_data.py`

| Function | Purpose |
|----------|---------|
| `temporal_split()` | Split interactions chronologically by timestamp |
| `filter_cold_start()` | Remove users/items from eval sets not seen in training |
| `create_interaction_matrix()` | Build sparse user-item matrices (weighted or binary) |
| `get_split_stats()` | Compute statistics for validation |

### Running the Split

```bash
cd /path/to/lazy_librarian_rs
uv run python data_preprocessing/split_data.py
```

### Output Files

```
data/processed/
├── train.parquet          # Training interactions DataFrame
├── val.parquet            # Validation interactions DataFrame  
├── test.parquet           # Test interactions DataFrame
├── train_matrix.npz       # Sparse (users × items), weighted by count
├── val_matrix.npz         # Sparse (users × items), binary
├── test_matrix.npz        # Sparse (users × items), binary
└── split_stats.json       # Statistics for each split
```

### Actual Results

| Split | Interactions | Unique Pairs | Users | Items | Date Range |
|-------|-------------|--------------|-------|-------|------------|
| **Train** | 60,932 | 46,508 | 6,925 | 14,037 | Jan 2, 2023 → Jan 15, 2024 |
| **Val** | 7,371 | 6,493 | 2,297 | 5,234 | Jan 15, 2024 → Feb 27, 2024 |
| **Test** | 11,082 | 9,634 | 2,824 | 6,631 | Feb 27, 2024 → Oct 14, 2024 |

**Cold-start filtering:**
- Validation: 1,334 interactions removed (15.3%) - users/items not in training
- Test: 6,328 interactions removed (36.3%) - users/items not in training

**Note:** Higher cold-start filtering in test set is expected due to longer time gap from training data.

### Configuration (`config.py`)

| Setting | Default | Description |
|---------|---------|-------------|
| `TRAIN_RATIO` | 0.7 | Fraction of data for training |
| `VAL_RATIO` | 0.1 | Fraction of data for validation |
| `TEST_RATIO` | 0.2 | Fraction of data for testing |

---

## 7. Next Step: Evaluation Framework

### Why Proper Evaluation Matters

Recommendation systems require **ranking metrics** rather than classification metrics:
- Users see a ranked list, not individual predictions
- Position matters: relevant items at the top are more valuable
- We evaluate how well we rank the items users will actually interact with

### Core Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MAP@K** | Mean of AP@K across users | Average precision of top-K recommendations |
| **NDCG@K** | Normalized DCG | Measures ranking quality with position discounting |
| **Hit Rate@K** | % users with ≥1 hit in top-K | Basic recall measure |
| **MRR** | Mean Reciprocal Rank | How early is the first relevant item? |
| **Precision@K** | Relevant items in top-K / K | Fraction of relevant items |
| **Recall@K** | Relevant items in top-K / total relevant | Coverage of relevant items |

### Primary Metric: MAP@10

**Mean Average Precision at 10 (MAP@10)** is our primary metric because:

1. **Position-aware**: Rewards relevant items appearing earlier in the list
2. **Practical K**: Users typically browse 5-15 recommendations
3. **Interpretable**: Higher = better ranked recommendations
4. **Industry standard**: Widely used in production recommender systems

**Formula:**
```
AP@K = (1/min(K, R)) × Σᵢ (Precision@i × rel(i))

where:
  - R = total relevant items for user
  - rel(i) = 1 if item at position i is relevant, 0 otherwise
  - Precision@i = relevant items in top i / i

MAP@K = mean(AP@K) across all users
```

**Example:**
```
User has relevant items: {A, B, C}
Recommendation list: [A, X, B, Y, Z, C, ...]

Precision@1 = 1/1 = 1.0 (A is relevant)
Precision@2 = 1/2 = 0.5 (X not relevant)
Precision@3 = 2/3 = 0.67 (B is relevant)
Precision@6 = 3/6 = 0.5 (C is relevant)

AP@10 = (1/3) × (1.0×1 + 0.67×1 + 0.5×1) = 0.72
```

### Secondary Metrics

| Metric | K values | Use Case |
|--------|----------|----------|
| **NDCG@K** | 5, 10, 20 | Graded relevance (if available) |
| **Hit Rate@K** | 1, 5, 10 | Quick sanity check |
| **MRR** | - | When only first hit matters |
| **Coverage** | - | Catalog diversity |

### Implemented: `evaluation/` Module ✅

```
evaluation/
├── __init__.py           # Public API exports
├── metrics.py            # Core metric implementations
├── baselines.py          # Random, Popularity, UserKNN, ItemKNN
└── evaluate.py           # Evaluation runner with CLI
```

#### Module Overview

| File | Key Functions/Classes |
|------|----------------------|
| `metrics.py` | `precision_at_k`, `recall_at_k`, `average_precision_at_k`, `ndcg_at_k`, `hit_rate_at_k`, `mrr`, `evaluate_user` |
| `baselines.py` | `RandomRecommender`, `PopularityRecommender`, `UserKNNRecommender`, `ItemKNNRecommender` |
| `evaluate.py` | `evaluate_model`, `evaluate_all_baselines`, `create_comparison_table` |

#### Key Features

- **Efficient evaluation**: Samples candidates for faster evaluation (configurable)
- **Excludes training items**: Ensures fair evaluation by not recommending known items
- **Progress tracking**: tqdm progress bars for long evaluations
- **JSON export**: Results saved to `data/processed/baseline_results.json`
- **Comparison tables**: Automatic DataFrame generation for model comparison

#### Model Interface

All models follow a simple interface:

```python
class Recommender:
    def fit(self, train_matrix: sparse.csr_matrix) -> "Recommender":
        """Fit the model on training data."""
        ...
    
    def predict(self, user_id: int, candidate_items: np.ndarray) -> np.ndarray:
        """Predict scores for candidate items."""
        ...
```

### Actual Baseline Results ✅

Evaluated on test set (2,824 users, 500 sampled candidates per user):

| Model | MAP@10 | NDCG@10 | HR@10 | MRR | Eval Time |
|-------|--------|---------|-------|-----|-----------|
| Random | 0.0044 | 0.0095 | 0.048 | 0.026 | 6.4s |
| Popularity | 0.0367 | 0.0580 | 0.178 | 0.086 | 6.3s |
| **UserKNN-50** | **0.1822** | **0.2483** | **0.561** | **0.332** | 7.2s |
| UserKNN-100 | 0.1817 | 0.2483 | 0.557 | 0.330 | 7.5s |

**Key Insights:**
- UserKNN achieves **MAP@10 = 0.18**, already exceeding the "Good" threshold (0.10)
- Hit Rate@10 of **56%** means most users get at least one relevant recommendation
- Popularity baseline is much stronger than random (8x better MAP@10)
- UserKNN-50 slightly outperforms UserKNN-100 (more neighbors = more noise)

### Running Evaluation

```bash
cd /path/to/lazy_librarian_rs

# Run all baselines
uv run python evaluation/evaluate.py --k 5 10 20 --n-candidates 500

# Use validation set instead of test
uv run python evaluation/evaluate.py --use-val --k 5 10 20
```

### Success Criteria

| Milestone | MAP@10 Target | Status |
|-----------|---------------|--------|
| Baseline | > 0.02 | ✅ Popularity: 0.037 |
| Good | > 0.10 | ✅ UserKNN: 0.182 |
| Excellent | > 0.20 | 🔜 Next target |

---

## 8. Roadmap

| Step | Status | Description |
|------|--------|-------------|
| 1. Data preprocessing pipeline | ✅ Done | Feature extraction, encoding, normalization |
| 2. Build content features | ✅ Done | Title embeddings, subject TF-IDF, categorical |
| 3. Train/Val/Test splits | ✅ Done | Temporal splitting with cold-start filtering |
| 4. Evaluation framework | ✅ Done | MAP@10, NDCG@K, Hit Rate@K, MRR metrics |
| 5. Collaborative filtering baseline | ✅ Done | UserKNN achieves MAP@10=0.18 |
| 6. **Content-based baseline** | 🔜 Next | Item-item similarity using content features |
| 7. Hybrid fusion | ⬜ Pending | Score combination or learned fusion |
| 8. Cold-start handling | ⬜ Pending | Fallback strategies |

---

*Report generated from data analysis on uploaded datasets.*