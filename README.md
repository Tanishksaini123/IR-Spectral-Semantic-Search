# IR Spectral Semantic Search

A self-supervised learning framework for learning robust representations of infrared (IR) spectra and performing **semantic similarity search** over chemical spectra.

The project uses a Transformer-based encoder pretrained with **contrastive learning and adversarial spectral masking** to convert an IR spectrum into a compact **256-dimensional embedding**. These embeddings can then be compared using cosine similarity to retrieve chemically relevant spectra.

---

## Overview

Infrared spectroscopy provides characteristic spectral patterns that can be used to identify and analyze chemical compounds. However, conventional spectrum matching methods generally rely on direct signal-level similarity.

This project explores a representation-learning approach:

$$\text{IR Spectrum} \longrightarrow \text{Self-Supervised Transformer Encoder} \longrightarrow \text{256-D Embedding} \longrightarrow \text{Cosine Similarity} \longrightarrow \text{Semantic Retrieval}$$

The encoder is pretrained without requiring molecular labels. After pretraining, the learned representation is evaluated on labeled IR spectral datasets using an information-retrieval setup.

---

## Key Features

- Self-supervised pretraining on large-scale IR spectral data
- Contrastive learning using **NT-Xent / SimCLR loss**
- Adversarial masking of informative spectral regions
- CNN-based spectral embedding before Transformer encoding
- Transformer encoder with collaborative attention
- 256-dimensional spectrum embeddings
- Cosine-similarity-based semantic search
- Retrieval evaluation using:
  - Recall@K
  - Precision@K
  - mAP@K
- Evaluation on OPD and FGPD labeled datasets

---

## Model Architecture

The overall architecture consists of four major stages:

```text
                    IR Spectrum
                         │
                         ▼
              ┌─────────────────────┐
              │   CNN Embedding     │
              │  + Spatial Attention│
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Transformer Encoder │
              │ Collaborative Attn. │
              └──────────┬──────────┘
                         │
                         ▼
                    Mean Pooling
                         │
                         ▼
              ┌─────────────────────┐
              │   Linear Projection │
              └──────────┬──────────┘
                         │
                         ▼
                 256-D Embedding
                         │
                         ▼
              Cosine Similarity Search
                         │
                         ▼
                Top-K Similar Spectra
```

## Self-Supervised Pretraining

The encoder is pretrained using two augmented views of the same IR spectrum.

### View Generation

The second view is generated using:
- Gaussian noise
- Random block masking
- Adversarial spectral masking

The adversarial masker learns to identify informative regions of the spectrum and selectively masks them during training.

```text
Original Spectrum
       │
       ├───────────────► View 1
       │
       │
       └──► Noise + Random Masking
                    │
                    ▼
             Adversarial Masker
                    │
                    ▼
                 View 2
```

Both views are passed through the same encoder and projection head.

The representations are optimized using NT-Xent contrastive loss, encouraging embeddings of augmented versions of the same spectrum to remain close while separating embeddings from different spectra.

### Spectral Encoder

The encoder consists of:
- CNN spectral embedding
- Spatial attention
- Positional encoding
- Transformer layers
- Collaborative attention
- Mean pooling
- Linear projection

For the configuration used in the semantic-search experiment, the encoder produces:

- **Input:** $1 \times 1648$ spectral points
- **Output:** 256-dimensional embedding

The resulting embedding provides a compact representation of the spectral characteristics learned during self-supervised pretraining.

---

## Semantic Search

After pretraining, the projection head used during contrastive learning is not required for retrieval.

The pretrained encoder is used to generate embeddings:

$$\text{Spectrum} \longrightarrow \text{Encoder} \longrightarrow \text{256-D Embedding}$$

For a query spectrum, cosine similarity is calculated against the embeddings of the dataset.

The spectra are then ranked according to similarity:

```text
Query Spectrum
      │
      ▼
256-D Query Embedding
      │
      ▼
Cosine Similarity
      │
      ▼
Rank All Dataset Embeddings
      │
      ▼
Top-K Retrieved Spectra
```

---

## Evaluation Datasets

The pretrained encoder was evaluated on two labeled datasets:

### OPD
The OPD dataset contains infrared spectra with odor-related labels.

| Property | Value |
| :--- | :--- |
| Valid spectra | 3,017 |
| Spectrum length | 1,647 |
| Embedding dimension | 256 |

### FGPD
The FGPD dataset contains infrared spectra with functional-group-related labels.

| Property | Value |
| :--- | :--- |
| Valid spectra | 8,164 |
| Spectrum length | 1,648 |
| Embedding dimension | 256 |

---

## Retrieval Metrics

The semantic search system is evaluated using standard information-retrieval metrics.

- **Recall@K:** Measures the fraction of all relevant items that were retrieved within the top $K$ results.
  $$\text{Recall@K} = \frac{\text{Relevant items retrieved in top } K}{\text{Total relevant items}}$$

- **Precision@K:** Measures the fraction of retrieved items that are relevant.
  $$\text{Precision@K} = \frac{\text{Relevant items in top } K}{K}$$

- **mAP@K:** Mean Average Precision evaluates the ranking quality of relevant results within the top $K$ retrieved items. Higher values indicate that relevant spectra tend to appear earlier in the ranked retrieval results.

---

## Results

### OPD
| Metric | @1 | @5 | @10 |
| :--- | :--- | :--- | :--- |
| Recall | 0.0023 | 0.0070 | 0.0120 |
| Precision | 0.5171 | 0.4049 | 0.3706 |
| mAP | 0.5171 | 0.3244 | 0.2619 |

### FGPD
| Metric | @1 | @5 | @10 |
| :--- | :--- | :--- | :--- |
| Recall | 0.0002 | 0.0009 | 0.0017 |
| Precision | 0.8979 | 0.8902 | 0.8913 |
| mAP | 0.8979 | 0.8484 | 0.8356 |

### Overall Comparison
| Dataset | P@1 | P@5 | P@10 | mAP@10 |
| :--- | :--- | :--- | :--- | :--- |
| OPD | 0.5171 | 0.4049 | 0.3706 | 0.2619 |
| FGPD | 0.8979 | 0.8902 | 0.8913 | 0.8356 |

The FGPD evaluation produced substantially higher precision and mAP values than the OPD evaluation under the same retrieval framework.

For complete retrieval examples and all reported metrics, see `results.md`.

### Example Semantic Search
A query spectrum can be represented as a 256-dimensional vector and compared with the entire dataset.

Example from the FGPD evaluation (Query spectrum: 7058):

| Rank | Index | Similarity | Relevant |
| :---: | :---: | :---: | :---: |
| 1 | 4623 | 0.9067 | Yes |
| 2 | 2025 | 0.8877 | Yes |
| 3 | 5032 | 0.8791 | Yes |
| 4 | 1870 | 0.8739 | Yes |
| 5 | 1725 | 0.8681 | Yes |
| 6 | 5718 | 0.8625 | Yes |
| 7 | 570 | 0.8614 | Yes |
| 8 | 6384 | 0.8596 | Yes |
| 9 | 1531 | 0.8565 | Yes |
| 10 | 2343 | 0.8559 | Yes |

---

## Repository Structure

The repository intentionally contains only the core implementation and evaluation results.

```text
IR-Spectral-Semantic-Search/
│
├── pretrain.py
│   └── Self-supervised IR spectral pretraining
│
├── search.py
│   └── Embedding generation and semantic retrieval evaluation
│
└── results.md
    └── Detailed retrieval results
```

### `pretrain.py`
This script contains the complete self-supervised pretraining pipeline, including:
- Dataset loading
- Spectrum normalization
- Random spectral masking
- Gaussian noise augmentation
- Adversarial masking
- CNN spectral embedding
- Transformer encoder
- Collaborative attention
- Contrastive NT-Xent loss
- AdamW optimization
- Learning-rate scheduling
- Validation
- Checkpointing

The pretrained encoder is saved separately so that it can later be reused for downstream representation-learning tasks.

### `search.py`
The semantic-search pipeline performs:
- Loading of the pretrained encoder
- Loading and preprocessing of evaluation spectra
- Embedding generation
- L2 normalization of embeddings
- Cosine-similarity computation
- Construction of relevance relationships from labels
- Top-K retrieval
- Recall@K calculation
- Precision@K calculation
- mAP@K calculation
- Example semantic-search visualization/output

---

## Reproducibility

The pretraining pipeline uses a fixed random seed for reproducibility.

Important configuration parameters include:
- Embedding dimension: $256$
- Transformer layers: $4$
- Attention heads: $4$
- Contrastive temperature: $0.07$
- Optimizer: AdamW
- Learning rate: $1\times 10^{-4}$
- Weight decay: $0.05$

The exact configuration used for a particular experiment is stored with the corresponding model checkpoint.

---

## Why This Project?

This project demonstrates the application of modern representation-learning techniques to scientific data rather than conventional tabular datasets.

It combines:
- Self-supervised learning
- Contrastive representation learning
- Transformers
- Attention mechanisms
- Adversarial data augmentation
- Vector embeddings
- Information retrieval
- Scientific/chemical spectral data

The resulting pipeline can serve as a foundation for applications such as:
- Similar-spectrum retrieval
- Spectral database search
- Chemical similarity exploration
- Representation-based clustering
- Downstream classification
- Few-shot learning on IR spectra
- Retrieval-augmented scientific analysis

---

## Future Improvements

Potential extensions include:
- Approximate nearest-neighbor search using FAISS
- Interactive semantic-search API
- Vector database integration
- Spectrum visualization for retrieved results
- UMAP/t-SNE visualization of learned embeddings
- Fine-tuning for specific chemical classification tasks
- Cross-dataset retrieval evaluation
- Hybrid spectral + molecular-structure retrieval
- Web interface for uploading an IR spectrum and retrieving similar spectra

---

## Tech Stack
- Python
- PyTorch
- NumPy
- SciPy
- Scikit-learn
- Transformer architectures
- Contrastive learning
- Cosine similarity
- Information retrieval metrics
