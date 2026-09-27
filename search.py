# IR SPECTRAL SEMANTIC SEARCH

import os
import json
import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ================================================================
# 1. PATHS
# ================================================================

MODEL_DIR = r"saved_models"
DATASETS = {
    "OPD": {
        "spectra": r"Datasets/OPD/spectra.npy",
        "labels": r"Datasets/OPD/labels.npy",
        "metadata": r"Datasets/OPD/metadata.csv",
    },
    "FGPD": {
        "spectra": r"Datasets/FGPD/spectra.npy",
        "labels": r"Datasets/FGPD/labels.npy",
        "metadata": r"Datasets/FGPD/metadata.csv",
    },
}

BATCH_SIZE = 256
QUERY_FRACTION = 0.30
K_VALUES = [1, 5, 10]
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# ================================================================
# 2. REPRODUCIBILITY
# ================================================================

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ================================================================
# 3. EXACT MODEL COMPONENTS
# ================================================================


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        attention = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attention


def conv_out_len(L, k, s, p):
    return (L + 2 * p - k) // s + 1


def cnn_specs(last_stride=1):
    return [(14, 3, 2), (14, 3, 0), (10, 2, 0), (10, last_stride, 0)]


class ConvEmbed(nn.Module):
    def __init__(self, num_lead, d_model, channels=(64, 128, 128, 128), last_stride=1):
        super().__init__()
        chans = [num_lead, *channels]
        layers = []
        for i, (k, s, p) in enumerate(cnn_specs(last_stride)):
            layers.extend(
                [
                    nn.Conv1d(chans[i], chans[i + 1], k, stride=s, padding=p),
                    nn.GroupNorm(8, chans[i + 1]),
                    nn.GELU(),
                ]
            )

        self.convs = nn.Sequential(*layers)
        self.attn = SpatialAttention(7)
        self.dense = nn.Linear(chans[-1], d_model)

    def forward(self, x):
        feat = self.attn(self.convs(x))
        return self.dense(feat.permute(0, 2, 1))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        T = x.size(1)
        return self.dropout(x + self.pe[:T].unsqueeze(0))


# ================================================================
# Collaborative Attention
# ================================================================


class MixingMatrixInit:
    UNIFORM = 3


class CollaborativeAttention(nn.Module):
    def __init__(
        self,
        dim_input,
        dim_value_all,
        dim_key_query_all,
        dim_output,
        num_attention_heads,
        dropout=0.1,
    ):
        super().__init__()
        self.dim_value_all = dim_value_all
        self.dim_key_query_all = dim_key_query_all
        self.num_attention_heads = num_attention_heads
        self.dim_value_per_head = dim_value_all // num_attention_heads
        self.attention_head_size = dim_key_query_all / num_attention_heads
        self.query = nn.Linear(dim_input, dim_key_query_all, bias=False)
        self.key = nn.Linear(dim_input, dim_key_query_all, bias=False)
        self.content_bias = nn.Linear(dim_input, num_attention_heads, bias=False)
        self.value = nn.Linear(dim_input, dim_value_all)
        self.m_t = nn.Parameter(
            torch.randn(num_attention_heads, dim_key_query_all) * 0.2
        )
        self.m_c = nn.Parameter(
            torch.randn(num_attention_heads, dim_key_query_all) * 0.2
        )
        self.dense = nn.Linear(dim_value_all, dim_output)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states):
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        mixed_query = query_layer[..., None, :, :] * self.m_c[..., :, None, :]
        mixed_key = key_layer[..., None, :, :] * self.m_t[..., :, None, :]
        scale = 1.0 / math.sqrt(self.attention_head_size)
        attention_scores = torch.matmul(
            mixed_query * scale, mixed_key.transpose(-1, -2)
        )
        content_bias = self.content_bias(hidden_states)
        attention_scores = (
            attention_scores + content_bias.transpose(-1, -2).unsqueeze(-2) * scale
        )
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        value_layer = self.value(hidden_states)
        B, T, D = value_layer.shape
        value_layer = value_layer.view(B, T, self.num_attention_heads, -1)
        value_layer = value_layer.permute(0, 2, 1, 3)
        context = torch.matmul(attention_probs, value_layer)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(B, T, self.dim_value_all)

        return self.dense(context)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1, key_query_dim=None):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CollaborativeAttention(
            dim_input=d_model,
            dim_value_all=d_model,
            dim_key_query_all=(key_query_dim or d_model),
            dim_output=d_model,
            num_attention_heads=num_heads,
            dropout=dropout,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout1(self.attn(self.norm1(x)))
        x = x + self.dropout2(self.ff(self.norm2(x)))
        return x


class TransformerModel(nn.Module):
    def __init__(
        self,
        num_layers,
        num_heads,
        d_model,
        ff_dim,
        embedding_dim,
        dropout,
        max_tokens,
        num_lead=1,
        stem_channels=(64, 128, 128, 128),
        stem_last_stride=1,
    ):
        super().__init__()
        self.embed = ConvEmbed(
            num_lead=num_lead,
            d_model=d_model,
            channels=stem_channels,
            last_stride=stem_last_stride,
        )
        self.embed_norm = nn.LayerNorm(d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_tokens)
        self.layers = nn.Sequential(
            *[
                TransformerEncoderLayer(d_model, num_heads, ff_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, embedding_dim)

    def forward(self, x):
        feat = self.embed_norm(self.embed(x))
        feat = self.pos_encoder(feat)
        feat = self.layers(feat)
        feat = self.final_norm(feat)
        feat = feat.mean(dim=1)

        return self.fc(feat)


# ================================================================
# 4. LOAD CONFIGURATION
# ================================================================

with open(os.path.join(MODEL_DIR, "config.json"), "r") as f:
    config = json.load(f)
print("\nLoaded configuration:")

for key in [
    "num_layers",
    "num_heads",
    "d_model",
    "embedding_dim",
    "embed_mode",
    "stem_channels",
    "stem_last_stride",
    "normalize",
]:
    if key in config:
        print(f"{key}: {config[key]}")

# ================================================================
# 5. BUILD EXACT ENCODER
# ================================================================

encoder = TransformerModel(
    num_layers=config["num_layers"],
    num_heads=config["num_heads"],
    d_model=config["d_model"],
    ff_dim=config["ff_mult"] * config["d_model"],
    embedding_dim=config["embedding_dim"],
    dropout=config["dropout"],
    max_tokens=config["max_tokens"],
    num_lead=config.get("num_lead", 1),
    stem_channels=tuple(config["stem_channels"]),
    stem_last_stride=config["stem_last_stride"],
)

# ================================================================
# 6. LOAD PRETRAINED WEIGHTS
# ================================================================

encoder_path = os.path.join(MODEL_DIR, "encoder_best.pth")
state_dict = torch.load(encoder_path, map_location="cpu")
encoder.load_state_dict(state_dict, strict=True)
encoder = encoder.to(DEVICE)
encoder.eval()
print("\nEncoder loaded successfully.")
print("Embedding dimension:", config["embedding_dim"])

# ================================================================
# 7. NORMALIZATION
# ================================================================


class Normalize:
    def __init__(self, mode="sample", mean=None, std=None):
        self.mode = mode
        self.mean = mean
        self.std = std

    def __call__(self, x):
        if self.mode == "sample":
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True)
            return (x - mean) / (std + 1e-6)

        elif self.mode == "global":
            return (x - self.mean) / self.std
        return x


normalizer_path = os.path.join(MODEL_DIR, "normalizer.pt")
norm_state = torch.load(normalizer_path, map_location="cpu")
normalizer = Normalize(
    mode=norm_state["mode"], mean=norm_state.get("mean"), std=norm_state.get("std")
)
print("Normalization:", norm_state["mode"])

# ================================================================
# 8. EMBEDDING FUNCTION
# ================================================================


@torch.no_grad()
def generate_embeddings(spectra, batch_size=256):
    embeddings = []
    for start in range(0, len(spectra), batch_size):
        batch = np.asarray(spectra[start : start + batch_size], dtype=np.float32)
        x = torch.from_numpy(batch)
        if x.ndim == 2:
            x = x.unsqueeze(1)

        # Exact same normalization used during pretraining
        normalized = []
        for sample in x:
            normalized.append(normalizer(sample))
        x = torch.stack(normalized).to(DEVICE)
        z = encoder(x)
        # VERY IMPORTANT:
        # cosine retrieval requires normalized embeddings
        z = F.normalize(z, p=2, dim=1)
        embeddings.append(z.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


# ================================================================
# 9. LABEL PROCESSING
# ================================================================


def load_labels(path):
    labels = np.load(path, allow_pickle=True)
    labels = np.asarray(labels)
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)
    return labels.astype(np.float32)


# ================================================================
# 10. RELEVANCE FUNCTION
# ================================================================


def relevance_matrix(query_labels, database_labels):
    """
    Multi-label relevance.

    Two spectra are relevant if they share
    at least ONE functional-group label.
    """
    query_bool = query_labels > 0
    database_bool = database_labels > 0
    # Intersection of functional groups
    overlap = query_bool.astype(np.int8) @ database_bool.T.astype(np.int8)
    return overlap > 0


# ================================================================
# 11. METRIC FUNCTIONS
# ================================================================


def recall_at_k(retrieved, relevant, k):
    retrieved = retrieved[:, :k]
    scores = []
    for i in range(len(relevant)):
        rel = relevant[i]
        total_relevant = rel.sum()
        if total_relevant == 0:
            continue
        hits = rel[retrieved[i]].sum()
        scores.append(hits / total_relevant)
    return np.mean(scores) if scores else 0.0


def precision_at_k(retrieved, relevant, k):
    retrieved = retrieved[:, :k]
    scores = []
    for i in range(len(relevant)):
        hits = relevant[i][retrieved[i]].sum()
        scores.append(hits / k)
    return np.mean(scores)


def average_precision(ranked_indices, relevant):
    hits = 0
    precision_sum = 0
    total_relevant = relevant.sum()
    if total_relevant == 0:
        return 0.0

    for rank, idx in enumerate(ranked_indices, start=1):
        if relevant[idx]:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / min(total_relevant, len(ranked_indices))


def map_at_k(retrieved, relevant, k):

    scores = []
    for i in range(len(relevant)):
        ranked = retrieved[i, :k]
        scores.append(average_precision(ranked, relevant[i]))
    return np.mean(scores)


# ================================================================
# 12. SEMANTIC SEARCH
# ================================================================


def semantic_search(query_embeddings, database_embeddings, top_k=10):
    # Embeddings already L2-normalized
    similarity = query_embeddings @ database_embeddings.T

    # top-k without sorting entire matrix
    top_indices = np.argpartition(-similarity, kth=top_k - 1, axis=1)[:, :top_k]

    # Sort those top-k results
    rows = np.arange(len(query_embeddings))[:, None]
    top_indices = top_indices[rows, np.argsort(-similarity[rows, top_indices], axis=1)]
    top_scores = similarity[rows, top_indices]
    return top_indices, top_scores


# ================================================================
# 13. EVALUATE DATASET
# ================================================================


def evaluate_dataset(name, spectra_path, labels_path, metadata_path):
    print("\n" + "=" * 70)
    print(f"DATASET: {name}")
    print("=" * 70)

    # Load data
    spectra = np.load(spectra_path, mmap_mode="r")
    labels = load_labels(labels_path)
    metadata = pd.read_csv(metadata_path)
    print("Spectra:", spectra.shape)
    print("Labels:", labels.shape)
    print("Metadata:", metadata.shape)
    assert len(spectra) == len(labels)
    assert len(spectra) == len(metadata)

    # Remove invalid samples
    valid = np.isfinite(np.asarray(spectra[:], dtype=np.float32)).all(
        axis=1
    ) & np.isfinite(labels).all(axis=1)
    spectra = np.asarray(spectra[valid], dtype=np.float32)
    labels = labels[valid]
    metadata = metadata.iloc[np.where(valid)[0]].reset_index(drop=True)
    print("Valid samples:", len(spectra))

    # Generate embeddings
    print("\nGenerating embeddings...")
    embeddings = generate_embeddings(spectra, BATCH_SIZE)
    print("Embedding shape:", embeddings.shape)

    # Save embeddings
    os.makedirs("semantic_search_results", exist_ok=True)
    np.save(f"semantic_search_results/{name}_embeddings.npy", embeddings)

    # Train/query split
    N = len(embeddings)
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(N)
    n_query = int(N * QUERY_FRACTION)
    query_idx = indices[:n_query]
    database_idx = indices[n_query:]
    query_embeddings = embeddings[query_idx]
    database_embeddings = embeddings[database_idx]
    query_labels = labels[query_idx]
    database_labels = labels[database_idx]

    # Relevance matrix
    print("\nBuilding relevance matrix...")
    relevant = relevance_matrix(query_labels, database_labels)

    # Search
    max_k = max(K_VALUES)
    retrieved, scores = semantic_search(query_embeddings, database_embeddings, max_k)

    # Metrics
    results = {}
    print("\nRetrieval metrics:")
    print("-" * 50)
    for k in K_VALUES:
        recall = recall_at_k(retrieved, relevant, k)
        precision = precision_at_k(retrieved, relevant, k)
        map_score = map_at_k(retrieved, relevant, k)

        results[f"Recall@{k}"] = recall
        results[f"Precision@{k}"] = precision
        results[f"mAP@{k}"] = map_score

        print(f"Recall@{k:<2}:     {recall:.4f}")
        print(f"Precision@{k:<2}:  {precision:.4f}")
        print(f"mAP@{k:<2}:        {map_score:.4f}")

    # Save retrieval results
    records = []
    for qi in range(min(100, len(query_idx))):
        original_query_index = query_idx[qi]
        for rank in range(max_k):
            database_position = retrieved[qi, rank]
            original_database_index = database_idx[database_position]
            records.append(
                {
                    "query_index": int(original_query_index),
                    "database_index": int(original_database_index),
                    "rank": rank + 1,
                    "cosine_similarity": float(scores[qi, rank]),
                    "relevant": bool(relevant[qi, database_position]),
                }
            )
    retrieval_df = pd.DataFrame(records)
    retrieval_df.to_csv(
        f"semantic_search_results/{name}_retrieval_results.csv", index=False
    )

    # Display example retrieval
    print("\nExample semantic search:")
    print("-" * 70)
    example_query = 0
    q_original = query_idx[example_query]
    print("Query spectrum index:", q_original)
    print("\nTop retrieved spectra:")

    for rank in range(min(10, max_k)):
        db_position = retrieved[example_query, rank]
        db_original = database_idx[db_position]
        print(
            f"Rank {rank + 1:2d} | "
            f"Index {db_original:5d} | "
            f"Similarity "
            f"{scores[example_query, rank]:.4f} | "
            f"Relevant: "
            f"{relevant[example_query, db_position]}"
        )

    return {
        "embeddings": embeddings,
        "metrics": results,
        "retrieved": retrieved,
        "scores": scores,
    }


# ================================================================
# 14. RUN OPD + FGPD
# ================================================================

all_results = {}
for name, paths in DATASETS.items():
    try:
        all_results[name] = evaluate_dataset(
            name=name,
            spectra_path=paths["spectra"],
            labels_path=paths["labels"],
            metadata_path=paths["metadata"],
        )

    except Exception as e:
        print(f"\nERROR processing {name}:")
        print(e)

# ================================================================
# 15. FINAL SUMMARY
# ================================================================

print("\n\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)
for name, result in all_results.items():
    print(f"\n{name}")
    for metric, value in result["metrics"].items():
        print(f"{metric:<15}: " f"{value:.4f}")
