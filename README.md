# RAG Retrieval Analysis and Improvement

This repository contains various experiments and implementations aimed at improving the retrieval stage of a RAG (Retrieval-Augmented Generation) system. The project explores Re-ranking with Random Forest, Association Rule Mining for feature analysis, and Clustering techniques (HDBSCAN, K-means) to identify and filter high-quality document chunks.

## Repository Structure

### Notebooks

- **`reranking_classifier_nacer.ipynb`**
  - **Description**: The primary notebook implementing the BM25 baseline retrieval followed by a Random Forest re-ranking model.
  - **Key Steps**: 
    1. BM25 Baseline Retrieval.
    2. Training data generation (feature extraction).
    3. Random Forest model training and evaluation.

- **`association_rules_analysis_oumnia.ipynb`**
  - **Description**: Implements Association Rule Mining to analyze chunk quality.
  - **Goal**: Find features of retrieved chunks (e.g., length, table presence) that are strongly associated with high LCS (Longest Common Subsequence) scores.
  - **Libraries**: Uses `mlxtend` for Apriori and Association Rules.

- **`clustering_asmaa.ipynb`**
  - **Description**: Explores HDBSCAN clustering for quality filtering in the retrieval pipeline.
  - **Goal**: Identify and remove low-quality document chunks before retrieval aimed at improving performance.

- **`clustering_sarah.ipynb`**
  - **Description**: Uses K-means clustering to assess text chunk quality.
  - **Goal**: Discover natural quality tiers and patterns in document chunks to explain retrieval performance variations.

### Data (`data/`)

- `qas_v2_clean.json`, `qas_v2.json`, `qas.json`: Question Answering datasets (queries and ground truths).
- `retrieval_base/gt/`: Ground truth document corpus organized by domain (Academic, Finance, Law, etc.). Each file is a JSON containing page text.

### Output (`output/`)

- Contains generated files such as:
  - `training_data.csv` & `training_data_enhanced.csv`: Features extracted for model training.
  - `rerank_scores.csv`: Results of re-ranking.
  - `model/`: Saved models and configurations.

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Placement**
   Ensure the `data/` folder is populated with the Q&A JSON files and the `retrieval_base/gt/` corpus as per the structure above.

## Usage

Each notebook is relatively self-contained for its specific analysis task.

- To run the **Re-ranking Pipeline**, execute `reranking_classifier_nacer.ipynb`.
- To perform **Association Rule Analysis**, run `association_rules_analysis_oumnia.ipynb`.
- To explore **Clustering Experiments**, run `clustering_asmaa.ipynb` or `clustering_sarah.ipynb`.
