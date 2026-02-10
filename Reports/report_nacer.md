# Document Re-Ranking Classifier: Full Project Report

## For Presentation to Professors

---

## 1. Project Overview

This project tackles the **document retrieval** problem in a Retrieval-Augmented Generation (RAG) pipeline. Given a user query, the system must find the most relevant document chunks from a large multi-domain corpus. The goal: maximize the **LCS (Longest Common Subsequence) score** between the retrieved text and the ground-truth evidence.

The approach is two-stage:
1. **BM25 Baseline** — a classical term-frequency-based retriever that fetches candidate documents.
2. **Re-Ranking Classifier** — a Random Forest model trained on 26 hand-engineered features that re-orders the BM25 candidates to push the most relevant ones to the top.

**Result: The re-ranking classifier improved the overall LCS score from 80.34% to 83.13%, a +3.5% relative improvement over the BM25 baseline.**

---

## 2. The Data Story

### 2.1 Raw Dataset

The project started with a QA dataset containing **8,498 queries** (stored in `qas_v2.json`). Each query entry contains:

| Field | Description |
|---|---|
| `ID` | Unique query identifier |
| `questions` | The natural-language question posed by a user |
| `answers` | The ground-truth answer |
| `doc_name` | The source document path (e.g., `finance/report_2023`) |
| `doc_type` | The domain category of the document |
| `evidence_context` | The exact text passage that contains the answer |
| `evidence_page_no` | The page number(s) where the evidence lives |
| `evidence_source` | The type of evidence: text, table, chart, formula, multi, or reading_order |

The queries span **9 document domains**: academic, administration, finance, law, manual, news, notes, paper, and textbook.

### 2.2 Data Cleaning (clean_data.py)

A dedicated Python script (`clean_data.py`) was applied to the raw 8,498 queries. The cleaning pipeline performed five filtering steps:

1. **Removed rows with missing values** — Any query lacking a required field (`questions`, `answers`, `doc_name`, or `evidence_context`) was dropped.
2. **Removed non-English queries** — The dataset was multilingual. A character-level heuristic detected Chinese, Korean, Japanese, Arabic, and Cyrillic characters. If more than 5% of a query's characters were non-English, it was removed.
3. **Removed too-short queries** — Queries shorter than 15 characters were dropped as likely malformed.
4. **Removed too-long queries** — Queries longer than 500 characters were dropped (likely full document pastes, not real questions).
5. **Removed noisy queries** — Queries with more than 35% special characters were filtered out.
6. **Verified source documents exist** — Each query's referenced document was checked against the actual file system in the retrieval base. Queries pointing to missing documents were removed.

**After cleaning: 7,481 queries remained** (saved as `qas_v2_clean.json`), a reduction of ~12% from the original 8,498.

### 2.3 Document Corpus

The retrieval base consists of JSON files organized by domain under `data/retrieval_base/gt/`. Each JSON file contains chunked document text with page indices. The corpus covers 7 active domains used in evaluation:

| Domain | Query Count | Description |
|---|---|---|
| Finance | 1,365 | Financial reports, statements |
| Administration | 1,322 | Administrative documents |
| Academic | 1,150 | Academic papers, research |
| Law | 1,142 | Legal texts, regulations |
| Manual | 1,107 | Product/software manuals |
| Textbook | 849 | Educational textbook content |
| News | 546 | News articles |

Evidence source types in the data:

| Evidence Type | Query Count | Description |
|---|---|---|
| Text | 3,361 | Plain text passages |
| Table | 2,053 | Tabular data |
| Formula | 1,142 | Mathematical formulas |
| Chart | 747 | Charts and figures |
| Multi | 126 | Multiple evidence types combined |
| Reading Order | 52 | Reading-order-dependent evidence |

---

## 3. The Evaluation Metric: LCS Score

The system is evaluated using the **Longest Common Subsequence (LCS) score**. This measures how much of the ground-truth evidence text is recoverable from the retrieved document chunks.

**How it works:**
1. Both the predicted text and the ground-truth text are **normalized**: lowercased, articles (a, an, the) removed, punctuation stripped, whitespace collapsed.
2. The texts are split into words.
3. A dynamic programming algorithm computes the length of the longest common subsequence between the two word sequences.
4. The LCS length is divided by the ground-truth length to produce a score between 0 and 1.

A score of 1.0 means the retrieved text contains every word of the ground truth in order. A score of 0.0 means no overlap at all. This metric rewards retrieving the right passage while being tolerant of extra surrounding text.

---

## 4. Stage 1: BM25 Baseline Retrieval

### 4.1 How BM25 Works

BM25 (Best Matching 25) is a classical information retrieval algorithm based on **term frequency** and **inverse document frequency**. It ranks documents by how well their words match the query words, with:

- **Term Frequency (TF)**: Words appearing more often in a document increase its score, but with diminishing returns (saturation).
- **Inverse Document Frequency (IDF)**: Rare words across the corpus are weighted more heavily than common words.
- **Document Length Normalization**: Longer documents are slightly penalized to avoid bias.

### 4.2 Implementation Details

The BM25 retriever is built **per domain**. For each of the 7 domains:

1. All JSON document files in that domain folder are loaded using a custom `PCJSONReader` class.
2. Documents are parsed into **nodes** (chunks) using LlamaIndex's `SimpleNodeParser` with a **chunk size of 1,024 tokens** and **no overlap** (chunk_overlap=0).
3. A LlamaIndex `BM25Retriever` index is built over those nodes.

At query time:
- The query's domain is identified from its `doc_name` field.
- The domain-specific BM25 index is searched.
- The **top-K candidates** are returned (K=5 for re-ranking candidate generation, K=2 for final baseline evaluation).

### 4.3 BM25 Baseline Results

| Metric | Value |
|---|---|
| **Overall LCS Score** | **80.34%** |

**By Domain:**

| Domain | LCS % | Query Count |
|---|---|---|
| News | 87.87 | 546 |
| Law | 85.81 | 1,142 |
| Administration | 84.81 | 1,322 |
| Manual | 84.54 | 1,107 |
| Textbook | 84.34 | 849 |
| Academic | 80.64 | 1,150 |
| Finance | 62.29 | 1,365 |

**By Evidence Source:**

| Evidence Type | LCS % | Count |
|---|---|---|
| Text | 85.54 | 3,361 |
| Formula | 80.77 | 1,142 |
| Reading Order | 76.92 | 52 |
| Table | 75.90 | 2,053 |
| Chart | 71.10 | 747 |
| Multi | 66.44 | 126 |

**Key observation:** BM25 struggles most with **finance** documents (62.29%) and **chart/table** evidence types. Pure keyword matching has trouble with structured/numerical data where the answer may not share exact terms with the query.

---

## 5. Stage 2: Re-Ranking with Random Forest Classifier

### 5.1 The Re-Ranking Strategy

Instead of replacing BM25, the re-ranker **sits on top of it**:

1. BM25 retrieves the **top 5 candidates** (CANDIDATE_K=5) per query.
2. For each candidate, **26 features** are extracted describing the query-document relationship.
3. A Random Forest classifier predicts the probability that each candidate is relevant (label=1).
4. Candidates are re-sorted by their predicted probability.
5. The **top 2** (FINAL_K=2) re-ranked documents are returned.

This is a classic **Learning-to-Rank (L2R)** approach applied as a binary classification problem.

### 5.2 Training Data Generation

For each of the 7,481 cleaned queries:
- BM25 retrieves 5 candidate documents.
- Each candidate is **labeled**:
  - **Label 1 (relevant):** if the candidate matches the correct document name AND page number, OR if its text has an LCS overlap > 0.3 with the ground-truth evidence.
  - **Label 0 (irrelevant):** otherwise.
- 26 features are extracted for each query-candidate pair.

This produces **149,620 training samples** (7,481 queries x ~20 candidates on average, accounting for some queries with fewer than 5 candidates):
- **96,051 negative samples** (label=0)
- **53,569 positive samples** (label=1)

The data is split 70/30 into training and test sets:
- **Train:** 104,734 samples
- **Test:** 44,886 samples

### 5.3 The 26 Features Explained

The features are organized into 5 groups, progressing from simple lexical matching to deep semantic understanding:

---

#### Group 1: Basic Lexical Features (Features 1-10)

These are the foundational text-matching signals:

| # | Feature Name | Description |
|---|---|---|
| 1 | **query_coverage** | Fraction of unique query words found in the document. If the query has 10 unique words and 7 appear in the doc, coverage = 0.7. *The single most important feature (importance: 0.113).* |
| 2 | **word_overlap** | Jaccard similarity — intersection of query and doc word sets divided by their union. Measures bidirectional overlap. |
| 3 | **bigram_overlap** | Fraction of query bigrams (consecutive word pairs) found in the doc. Captures phrase-level matching, not just individual words. |
| 4 | **trigram_overlap** | Same as bigram but for trigrams (3-word sequences). Even stricter phrase matching. |
| 5 | **exact_match** | Binary: 1.0 if the entire query string appears verbatim inside the document, 0.0 otherwise. |
| 6 | **term_freq** | Average frequency of query terms in the document, normalized by document length. Measures how much the document "talks about" the query topics. |
| 7 | **early_match** | Fraction of query words found in the first 50 words of the document. Relevant documents often mention key terms early. |
| 8 | **doc_len_norm** | Document length normalized to [0, 1] by dividing word count by 500. Captures document size as a signal. |
| 9 | **query_doc_ratio** | Ratio of query length to document length. Helps distinguish between short focused chunks and long generic ones. |
| 10 | **bm25_rank** | Inverse of BM25 rank position: 1/(rank+1). Encodes BM25's own confidence as a feature for the classifier. |

---

#### Group 2: Sliding Window Features (Features 11, 13, 16, 23)

These features look at **localized regions** of the document to find where query terms cluster together:

| # | Feature Name | Description |
|---|---|---|
| 11 | **min_query_coverage_window** | Best query coverage achieved by any sliding window over the document (window size = 3x query length). Finds the document region most relevant to the query. |
| 13 | **best_window_match_density** | Among all sliding windows, the highest ratio of matched query terms to window size. Measures how densely concentrated the matches are. |
| 16 | **first_complete_match_position** | Normalized position of the first window that covers 90%+ of query terms. Earlier = higher score. Documents that address the query early rank better. |
| 23 | **multi_window_coverage_count** | How many windows achieve 90%+ query coverage, normalized by 5. Documents with multiple high-coverage regions are more likely relevant. |

---

#### Group 3: Query Term Distance & Compactness Features (Features 12, 14, 15, 17)

These measure **how close together** the matched query terms are within the document:

| # | Feature Name | Description |
|---|---|---|
| 12 | **query_compactness_gain** | How much more compact the matched terms are compared to a random distribution. Higher = terms are clustered together, not scattered. |
| 14 | **avg_query_term_distance** | Average distance between consecutive matched query term positions, inverted and normalized. Closer terms = higher score. |
| 15 | **query_term_distance_variance** | Variance of distances between matched positions. Low variance means terms are evenly and closely spaced. |
| 17 | **match_span_compression_ratio** | Ratio of how compressed the match span is relative to the full document. If all matches fall in a small section, compression is high. |

---

#### Group 4: IDF-Weighted Features (Features 18, 19, 20)

These give **more weight to rare, information-rich terms** rather than common words:

| # | Feature Name | Description |
|---|---|---|
| 18 | **avg_idf_matched_terms** | Average IDF score of query terms that appear in the document. High = the document matches rare/important query words, not just common ones. |
| 19 | **max_idf_term_presence** | The highest IDF score among matched terms. Captures presence of the single most distinctive query word. |
| 20 | **idf_weighted_window_density** | IDF-weighted coverage: sum of IDF for matched terms / sum of IDF for all query terms. Prioritizes matching important words over stop-word-like terms. *Second most important feature (importance: 0.112).* |

---

#### Group 5: Advanced Composite Features (Features 21, 22, 24, 25)

These combine multiple signals for higher-level relevance judgments:

| # | Feature Name | Description |
|---|---|---|
| 21 | **length_normalized_match_strength** | Query coverage divided by a document length penalty factor. Long documents that only partially match are penalized more than short, focused chunks. |
| 22 | **answer_likeness_score** | Combines coverage with a length penalty centered at 100 words (ideal answer length). Documents close to answer length AND with good coverage score highest. |
| 24 | **near_exact_phrase_density** | Count of consecutive word-pair matches between query and document, normalized. Captures near-exact phrase reproduction beyond bigram overlap. |
| 25 | **rank_confidence_ratio** | Decay function of BM25 rank: 1/(1 + rank*0.5). Encodes the intuition that top-ranked BM25 results deserve more trust. |

---

#### Feature 26: Semantic Similarity

| # | Feature Name | Description |
|---|---|---|
| 26 | **semantic_similarity** | Cosine similarity between the query embedding and the document embedding, computed using **MiniLM-L6-v2** (a sentence-transformer model). This captures meaning-level similarity that lexical features miss entirely. *Third most important feature (importance: 0.112).* |

This is the only **neural/learned** feature. It uses a pre-trained transformer to encode both the query and document into 384-dimensional dense vectors, then measures their cosine distance. Two texts can have high semantic similarity even with zero word overlap (e.g., "automobile" vs "car").

---

### 5.4 Random Forest Classifier Configuration

| Hyperparameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 150 | Enough trees for stable predictions and variance reduction |
| `max_depth` | 15 | Deep enough to capture complex patterns from ~150K samples |
| `min_samples_leaf` | 5 | Prevents overfitting to noise clusters |
| `class_weight` | balanced | Compensates for the imbalanced label distribution (64% negative / 36% positive) |
| `random_state` | 42 | Reproducibility |

**Test set performance:**
- **Accuracy:** 75.2%
- **F1 Score:** 65.8%

### 5.5 Feature Importance Ranking

The Random Forest reveals which features matter most for distinguishing relevant from irrelevant documents:

| Rank | Feature | Importance |
|---|---|---|
| 1 | query_coverage | 0.113 |
| 2 | idf_weighted_window_density | 0.112 |
| 3 | semantic_similarity | 0.112 |
| 4 | min_query_coverage_window | 0.082 |
| 5 | best_window_match_density | 0.072 |
| 6 | bigram_overlap | 0.056 |
| 7 | near_exact_phrase_density | 0.052 |
| 8 | query_doc_ratio | 0.038 |
| 9 | rank_confidence_ratio | 0.037 |
| 10 | bm25_rank | 0.034 |

**Key insight:** The top 3 features each contribute ~11% importance and represent three different matching paradigms: pure word coverage (lexical), IDF-weighted density (statistical), and semantic similarity (neural). This diversity is what makes the re-ranker stronger than BM25 alone.

---

## 6. Results: BM25 vs. Re-Ranking Classifier

### 6.1 Overall Performance

| Method | LCS Score | Change |
|---|---|---|
| BM25 Baseline | **80.34%** | — |
| Re-Ranking Classifier | **83.13%** | **+3.5%** relative improvement |

### 6.2 Performance by Domain

| Domain | BM25 LCS % | Rerank LCS % | Improvement |
|---|---|---|---|
| Academic | 80.64 | 83.09 | +2.45 |
| Administration | 84.81 | 86.22 | +1.41 |
| Finance | 62.29 | 66.78 | +4.49 |
| Law | 85.81 | 89.78 | +3.97 |
| Manual | 84.54 | 87.25 | +2.71 |
| News | 87.87 | 88.72 | +0.85 |
| Textbook | 84.34 | 86.71 | +2.37 |

**Finance** saw the largest absolute improvement (+4.49 points), showing the re-ranker particularly helps where BM25 struggles most. **Law** also gained significantly (+3.97 points).

### 6.3 Performance by Evidence Source

| Evidence Type | BM25 LCS % | Rerank LCS % | Improvement |
|---|---|---|---|
| Text | 85.54 | 88.39 | +2.85 |
| Formula | 80.77 | 82.50 | +1.73 |
| Reading Order | 76.92 | 80.68 | +3.76 |
| Table | 75.90 | 79.19 | +3.29 |
| Chart | 71.10 | 74.07 | +2.97 |
| Multi | 66.44 | 67.41 | +0.97 |

The re-ranker improves across **every evidence type**, with the largest gains on **reading order** (+3.76), **table** (+3.29), and **chart** (+2.97) — precisely the structured evidence types where BM25's keyword matching is weakest.

---

## 7. End-to-End Data Flow

```
Raw Dataset (qas_v2.json)
    │  8,498 queries
    │
    ▼
┌─────────────────────────┐
│   Data Cleaning Script  │  clean_data.py
│   - Remove missing vals │
│   - Remove non-English  │
│   - Remove too short    │
│   - Remove too long     │
│   - Remove noisy        │
│   - Verify docs exist   │
└─────────┬───────────────┘
          │  7,481 queries
          ▼
┌─────────────────────────┐
│   Clean Dataset         │  qas_v2_clean.json
│   7,481 queries across  │
│   7 domains, 6 evidence │
│   types                 │
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│   BM25 Index Building   │  Per-domain BM25 indices
│   Document corpus split │  using LlamaIndex
│   into 1024-token chunks│
└─────────┬───────────────┘
          │
    ┌─────┴──────┐
    ▼            ▼
┌────────┐  ┌──────────────────┐
│Baseline│  │Training Data Gen │
│Eval    │  │For each query:   │
│top_k=2 │  │  BM25 top 5      │
│        │  │  Label candidates │
│LCS:    │  │  Extract 26 feats │
│80.34%  │  │  149,620 samples  │
└────────┘  └────────┬─────────┘
                     │
                     ▼
            ┌────────────────┐
            │ Train/Test     │
            │ Split (70/30)  │
            │ Train: 104,734 │
            │ Test:  44,886  │
            └───────┬────────┘
                    │
                    ▼
            ┌────────────────┐
            │ Random Forest  │
            │ 150 trees      │
            │ max_depth=15   │
            │ 26 features    │
            │ balanced class │
            │                │
            │ Acc: 75.2%     │
            │ F1:  65.8%     │
            └───────┬────────┘
                    │
                    ▼
            ┌────────────────┐
            │ Re-Rank Search │
            │ BM25 top 5 →   │
            │ RF scores →    │
            │ Return top 2   │
            │                │
            │ LCS: 83.13%   │
            │ (+3.5% gain)  │
            └────────────────┘
```

---

## 8. How the Re-Ranking Pipeline Works at Inference Time

For a new incoming query:

1. **Domain routing** — The query's domain is identified, selecting the correct BM25 index.
2. **BM25 candidate retrieval** — The domain-specific BM25 retriever returns the top 5 most keyword-relevant document chunks.
3. **Feature extraction** — For each of the 5 candidates, all 26 features are computed:
   - 10 basic lexical features (word overlap, n-grams, etc.)
   - 4 sliding window features (localized match quality)
   - 4 distance/compactness features (how clustered the matches are)
   - 3 IDF-weighted features (importance-aware matching)
   - 4 advanced composite features (length-aware, answer-likeness)
   - 1 semantic similarity feature (MiniLM neural embedding cosine similarity)
4. **Random Forest prediction** — The trained RF model outputs `predict_proba[:, 1]` (probability of relevance) for each candidate.
5. **Re-sorting** — Candidates are sorted by predicted relevance probability in descending order.
6. **Final selection** — The top 2 re-ranked candidates are returned as the final retrieved documents.

---

## 9. Why This Works: The Intuition

BM25 only considers **lexical (word-level) matching**. It has no understanding of:
- **Semantics** — "automobile" and "car" are different words to BM25.
- **Term proximity** — BM25 doesn't care if query terms are scattered across a 10,000-word document or clustered in one paragraph.
- **Answer structure** — BM25 doesn't know that a focused 100-word passage is more likely to be an answer than a 2,000-word chapter.

The 26-feature Random Forest addresses all these blind spots:
- **Semantic similarity** (feature 26) captures meaning beyond words.
- **Window and distance features** (features 11-17, 23) capture term proximity and clustering.
- **Answer likeness and length normalization** (features 21-22) capture structural fit.
- **IDF weighting** (features 18-20) ensures rare, informative terms matter more than common ones.

By combining BM25's fast candidate generation with the Random Forest's rich multi-signal re-ranking, the system gets the best of both worlds: speed and accuracy.

---

## 10. Summary of Key Numbers

| Metric | Value |
|---|---|
| Raw queries | 8,498 |
| Cleaned queries | 7,481 |
| Queries removed by cleaning | 1,017 (12%) |
| Document domains | 7 (active in evaluation) |
| Evidence types | 6 (text, table, formula, chart, multi, reading_order) |
| BM25 chunk size | 1,024 tokens |
| BM25 candidates for re-ranking | 5 |
| Final documents returned | 2 |
| Training samples generated | 149,620 |
| Features per sample | 26 |
| Random Forest trees | 150 |
| Random Forest max depth | 15 |
| Train/test split | 70% / 30% |
| RF test accuracy | 75.2% |
| RF test F1 | 65.8% |
| **BM25 baseline LCS** | **80.34%** |
| **Re-rank LCS** | **83.13%** |
| **Improvement** | **+3.5% relative (+2.79 absolute points)** |
