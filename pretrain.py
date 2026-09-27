import os
import math
import copy
import json
import random
import time
from dataclasses import dataclass, asdict, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


@dataclass
class Config:
    # ---------------- data ----------------
    data_path: str = r"D:\IR Spectral Semantic Search\Datasets\45k_dataset.npy"   # Colab: copy the file to /content first, e.g. "/content/45k_dataset.npy"
    val_fraction: float = 0.05
    normalize: str = "sample"          # "sample": per-spectrum z-score | "global": per-wavenumber z-score | "none"
    # ---------------- weak (non-adversarial) augmentations ----------------
    noise_std: float = 0.05            # Gaussian noise on view 2 (in normalised units)
    num_random_masks: int = 3
    random_mask_ratio: float = 0.05    # fraction of the spectrum per random block
    # ---------------- adversarial masker ----------------
    adv_mask_ratio: float = 0.10       # fraction of PATCHES the adversary may hide (hard budget)
    adv_patch_size: int = 16           # points per patch
    adv_logit_scale: float = 3.0       # bounds the learned logits so some exploration noise always remains
    adv_start_epoch: int = 5           # before this the mask is purely random (strength 0)
    adv_ramp_epochs: int = 20          # epochs to go from random to fully adversarial
    gen_lr_mult: float = 1.0           # adversary lr = lr * gen_lr_mult
    # ---------------- encoder ----------------
    num_layers: int = 4
    num_heads: int = 4
    d_model: int = 512
    ff_mult: int = 4
    key_query_dim: Optional[int] = None  # None -> d_model (no compression). Set e.g. 256 to actually use the
                                          # compressed shared key/query space that collaborative attention is designed for
    embed_mode: str = "cnn"            # "cnn" | "linear"
    chunk_len: int = 50                # only used by embed_mode="linear"
    stem_channels: tuple = (128, 256, 256, 256)   # conv stem widths (each must be divisible by 8)
    stem_last_stride: int = 1          # 2 halves the number of tokens (76 -> 38 for L=1648) => ~2x cheaper transformer
    embedding_dim: int = 512
    dropout: float = 0.1
    max_tokens: int = 1024             # max sequence length (in tokens) the positional encoding supports
    proj_hidden_dim: int = 2048
    output_dim: int = 128
    # ---------------- optimisation ----------------
    batch_size: int = 64
    num_workers: int = 2               # Colab has ~2 vCPUs; more workers just fight for them
    in_memory: bool = True             # load the whole array into RAM as float32 (memmap over a Drive mount is extremely slow)
    use_amp: bool = True               # fp16 autocast + GradScaler (uses the T4 tensor cores); ignored on CPU
    lr: float = 1e-4
    weight_decay: float = 0.05
    temperature: float = 0.07
    num_epochs: int = 150
    warmup_epochs: int = 5
    min_lr_ratio: float = 0.01         # final lr = lr * min_lr_ratio
    grad_clip: float = 1.0
    # ---------------- early stopping (on validation loss) ----------------
    patience: int = 20
    min_delta: float = 1e-4
    es_start_epoch: int = 25           # do not count patience before the adversary is fully ramped in
    # ---------------- io / misc ----------------
    save_dir: str = r"./saved_models"     # Colab: use a Drive path (e.g. /content/drive/MyDrive/saved_models) so checkpoints survive a disconnect
    save_every: int = 10               # extra model-only snapshot every N epochs (0 = off)
    resume: Optional[str] = None       # e.g. "saved_models/last.pth"
    seed: int = 42

    # filled in from the data at runtime
    num_lead: int = 1


# Model-size presets. Only fields listed here are overridden. Params/cost are for a 1648-point spectrum.
PRESETS = {
    "original":   {},                                            # ~16.1M params, 76 tokens      (cost 1.0x)
    "small":      dict(d_model=256, ff_mult=2, embedding_dim=256, proj_hidden_dim=512,
                       stem_channels=(64, 128, 128, 128)),       # ~3M params,    76 tokens      (cost ~0.2x)
    "small_fast": dict(d_model=256, ff_mult=2, embedding_dim=256, proj_hidden_dim=512,
                       stem_channels=(64, 128, 128, 128), stem_last_stride=2),   # ~3M, 38 tokens (cost ~0.1x)
    "tiny":       dict(num_layers=3, d_model=128, ff_mult=2, embedding_dim=128, proj_hidden_dim=256,
                       stem_channels=(32, 64, 64, 64), stem_last_stride=2),      # ~0.6M, 38 tokens (cost ~0.03x)
}


def apply_preset(cfg: "Config", name: str) -> "Config":
    cfg = copy.deepcopy(cfg)
    for k, v in PRESETS[name].items():
        setattr(cfg, k, v)
    return cfg


PRESET = "small"                       # <- choose: "original" | "small" | "small_fast" | "tiny"
cfg = apply_preset(Config(), PRESET)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(cfg.seed)
torch.backends.cudnn.benchmark = True
print("device:", device, "| cuda available:", torch.cuda.is_available(), "| preset:", PRESET)
if device.type == "cpu":
    print("WARNING: training on CPU. This model needs a GPU for reasonable speed. On CPU, DataLoader workers also compete with the "
          "model for cores, so consider cfg.num_workers = 0-1 and a smaller preset.")

class Normalize:
    """mode='sample': z-score every spectrum over its own length (removes absolute intensity).
       mode='global': z-score every wavenumber using dataset statistics (keeps relative intensities).
       mode='none'  : identity."""

    def __init__(self, mode: str = "sample", mean: Optional[torch.Tensor] = None, std: Optional[torch.Tensor] = None):
        assert mode in ("sample", "global", "none"), mode
        if mode == "global":
            assert mean is not None and std is not None, "global mode needs mean/std"
        self.mode, self.mean, self.std = mode, mean, std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:   # x: (C, L)
        if self.mode == "sample":
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True)
            return (x - mean) / (std + 1e-6)
        if self.mode == "global":
            return (x - self.mean) / self.std
        return x

    def state(self):
        return {"mode": self.mode, "mean": self.mean, "std": self.std}


class GaussianNoise:
    def __init__(self, std: float = 0.05):
        self.std = std

    def __call__(self, x: torch.Tensor, g: Optional[torch.Generator] = None) -> torch.Tensor:
        if self.std <= 0:
            return x
        return x + torch.randn(x.shape, generator=g) * self.std


class MultiBlockMask:
    """Zero out `num_masks` random blocks of length `mask_ratio * L` along the LAST axis (all channels)."""

    def __init__(self, num_masks: int = 3, mask_ratio: float = 0.05):
        self.num_masks, self.mask_ratio = num_masks, mask_ratio

    def __call__(self, x: torch.Tensor, g: Optional[torch.Generator] = None) -> torch.Tensor:
        L = x.shape[-1]                                   # <-- original used len(x) == number of channels
        mask_len = max(1, int(L * self.mask_ratio))
        out = x.clone()
        for _ in range(self.num_masks):
            start = int(torch.randint(0, L - mask_len + 1, (1,), generator=g))
            out[..., start:start + mask_len] = 0.0
        return out


class SpectraDataset(Dataset):
    """Returns (x, x_aug), both of shape (C, L).
       x     : normalised spectrum (view 1, clean)
       x_aug : x with random block masks + Gaussian noise (input of the adversarial masker -> view 2)"""

    def __init__(self, source, indices, normalize: Normalize, cfg: Config, train: bool = True):
        self.source = source                              # np.ndarray (in memory) or path to an .npy file (memmap)
        self.indices = np.asarray(indices)
        self.normalize = normalize
        self.train = train
        self.seed = cfg.seed
        self.block_mask = MultiBlockMask(cfg.num_random_masks, cfg.random_mask_ratio)
        self.noise = GaussianNoise(cfg.noise_std)
        self._X = None

    @property
    def X(self):
        if isinstance(self.source, np.ndarray):           # in-memory (shared with forked workers, no copy)
            return self.source
        if self._X is None:                               # otherwise a memmap, opened lazily per worker
            self._X = np.load(self.source, mmap_mode="r")
        return self._X

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        x = np.array(self.X[idx], dtype=np.float32)      # copy out of the memmap
        if x.ndim == 1:
            x = x[None, :]
        x = self.normalize(torch.from_numpy(x))
        g = None if self.train else torch.Generator().manual_seed(self.seed + idx)
        x_aug = self.noise(self.block_mask(x, g), g)
        return x, x_aug


def inspect_data(X, n_check=2000):
    print(f"shape={X.shape} dtype={X.dtype}")
    sample = np.asarray(X[:n_check], dtype=np.float32)
    assert np.isfinite(sample).all(), "NaN/Inf found in the first rows of the dataset"
    print(f"first {len(sample)} rows: min={sample.min():.4g} max={sample.max():.4g} mean={sample.mean():.4g} std={sample.std():.4g}")
    num_lead = 1 if X.ndim == 2 else X.shape[1]
    return X.shape[0], num_lead, X.shape[-1]


def build_normalizer(cfg: Config, X, train_idx, n_stats: int = 5000) -> Normalize:
    if cfg.normalize != "global":
        return Normalize(cfg.normalize)
    rng = np.random.default_rng(cfg.seed)
    sel = np.sort(rng.choice(train_idx, size=min(n_stats, len(train_idx)), replace=False))
    arr = np.asarray(X[sel], dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, None, :]
    mean, std = arr.mean(0), arr.std(0)
    std[std < 1e-8] = 1.0
    return Normalize("global", torch.from_numpy(mean), torch.from_numpy(std))


def make_loaders(cfg: Config):
    if cfg.in_memory:                                     # one sequential read, float32 copy (306 MB for your 46k x 1648 array)
        t0 = time.time()
        X = np.load(cfg.data_path).astype(np.float32, copy=False)
        print(f"loaded {X.shape} into RAM as float32 in {time.time() - t0:.1f}s")
    else:
        X = np.load(cfg.data_path, mmap_mode="r")
    source = X if cfg.in_memory else cfg.data_path
    N, C, L = inspect_data(X)
    cfg.num_lead = C
    perm = np.random.default_rng(cfg.seed).permutation(N)
    n_val = max(1, int(N * cfg.val_fraction))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    normalizer = build_normalizer(cfg, X, train_idx)
    train_ds = SpectraDataset(source, train_idx, normalizer, cfg, train=True)
    val_ds = SpectraDataset(source, val_idx, normalizer, cfg, train=False)
    kw = dict(num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(),
              persistent_workers=cfg.num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False, **kw)
    print(f"train={len(train_ds)}  val={len(val_ds)}  channels={C}  length={L}")
    return train_loader, val_loader, normalizer, L

class SpatialAttention(nn.Module):
    """CBAM-style attention over positions (channel-pooled avg/max -> conv -> sigmoid)."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=kernel_size // 2)

    def forward(self, x):                                  # (B, C, T)
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


def conv_out_len(L, k, s, p):
    return (L + 2 * p - k) // s + 1


def cnn_specs(last_stride: int = 1):
    return [(14, 3, 2), (14, 3, 0), (10, 2, 0), (10, last_stride, 0)]   # (kernel, stride, padding)


def cnn_num_tokens(L: int, last_stride: int = 1) -> int:
    for k, s, p in cnn_specs(last_stride):
        L = conv_out_len(L, k, s, p)
    return L


class ConvEmbed(nn.Module):
    """Strided conv stem. Every conv is followed by GroupNorm + GELU (the original stack had no nonlinearity)."""

    def __init__(self, num_lead: int, d_model: int, channels=(128, 256, 256, 256), last_stride: int = 1):
        super().__init__()
        chans = [num_lead, *channels]
        assert all(c % 8 == 0 for c in chans[1:]), "stem channels must be divisible by 8 (GroupNorm groups)"
        layers = []
        for i, (k, s, p) in enumerate(cnn_specs(last_stride)):
            layers += [nn.Conv1d(chans[i], chans[i + 1], k, stride=s, padding=p),
                       nn.GroupNorm(8, chans[i + 1]), nn.GELU()]
        self.convs = nn.Sequential(*layers)
        self.attn = SpatialAttention(kernel_size=7)
        self.dense = nn.Linear(chans[-1], d_model)

    def forward(self, x):                                  # (B, C, L) -> (B, T, d_model)
        feat = self.attn(self.convs(x))
        return self.dense(feat.permute(0, 2, 1))


class LinearEmbed(nn.Module):
    """Non-overlapping chunks of `chunk_len` points -> linear projection."""

    def __init__(self, num_lead: int, chunk_len: int, d_model: int):
        super().__init__()
        self.chunk_len = chunk_len
        self.proj = nn.Linear(num_lead * chunk_len, d_model)

    def num_tokens(self, L):
        return math.ceil(L / self.chunk_len)

    def forward(self, x):                                  # (B, C, L) -> (B, T, d_model)
        B, C, L = x.shape
        pad = (-L) % self.chunk_len
        if pad:
            x = F.pad(x, (0, pad))
        x = x.unfold(-1, self.chunk_len, self.chunk_len)   # (B, C, T, chunk)
        x = x.permute(0, 2, 1, 3).reshape(B, -1, C * self.chunk_len)
        return self.proj(x)


class PositionalEncoding(nn.Module):
    """Sinusoidal encoding (pytorch.org transformer tutorial)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even"
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):                                  # (B, T, d_model)
        T = x.size(1)
        if T > self.pe.size(0):
            raise ValueError(f"Sequence has {T} tokens but positional encoding supports {self.pe.size(0)}; raise cfg.max_tokens")
        return self.dropout(x + self.pe[:T].unsqueeze(0))

from enum import Enum


class MixingMatrixInit(Enum):
    CONCATENATE = 1
    ALL_ONES = 2
    UNIFORM = 3     # NB: despite the name this is N(0, scale^2), as in the reference implementation


class CollaborativeAttention(nn.Module):
    def __init__(self, dim_input, dim_value_all, dim_key_query_all, dim_output, num_attention_heads,
                 output_attentions, attention_probs_dropout_prob, use_dense_layer, use_layer_norm,
                 mixing_initialization: MixingMatrixInit = MixingMatrixInit.UNIFORM):
        super().__init__()
        if dim_value_all % num_attention_heads != 0:
            raise ValueError(f"Value dimension ({dim_value_all}) should be divisible by number of heads ({num_attention_heads})")
        if not use_dense_layer and dim_value_all != dim_output:
            raise ValueError(f"Output dimension ({dim_output}) should equal value dimension ({dim_value_all}) if no dense layer is used")

        self.dim_input = dim_input
        self.dim_value_all = dim_value_all
        self.dim_key_query_all = dim_key_query_all
        self.dim_output = dim_output
        self.num_attention_heads = num_attention_heads
        self.output_attentions = output_attentions
        self.mixing_initialization = mixing_initialization
        self.use_dense_layer = use_dense_layer
        self.use_layer_norm = use_layer_norm

        self.dim_value_per_head = dim_value_all // num_attention_heads
        self.attention_head_size = dim_key_query_all / num_attention_heads   # need not be an integer

        self.query = nn.Linear(dim_input, dim_key_query_all, bias=False)
        self.key = nn.Linear(dim_input, dim_key_query_all, bias=False)
        self.content_bias = nn.Linear(dim_input, num_attention_heads, bias=False)
        self.value = nn.Linear(dim_input, dim_value_all)
        self.m_t = self.init_mixing_matrix()
        self.m_c = self.init_mixing_matrix()

        self.dense = nn.Linear(dim_value_all, dim_output) if use_dense_layer else nn.Sequential()
        self.dropout = nn.Dropout(attention_probs_dropout_prob)
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(dim_value_all)

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None):
        from_sequence = hidden_states
        to_sequence = hidden_states
        if encoder_hidden_states is not None:
            to_sequence = encoder_hidden_states
            attention_mask = encoder_attention_mask

        query_layer = self.query(from_sequence)
        key_layer = self.key(to_sequence)
        mixed_query = query_layer[..., None, :, :] * self.m_c[..., :, None, :]     # (B, H, T, Dk)
        mixed_key = key_layer[..., None, :, :] * self.m_t[..., :, None, :]
        scale = 1.0 / math.sqrt(self.attention_head_size)                           # applied before the matmul: identical maths, avoids fp16 overflow
        attention_scores = torch.matmul(mixed_query * scale, mixed_key.transpose(-1, -2))   # (B, H, T, T)
        content_bias = self.content_bias(to_sequence)                               # (B, T, H)
        attention_scores = attention_scores + content_bias.transpose(-1, -2).unsqueeze(-2) * scale

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = self.dropout(F.softmax(attention_scores, dim=-1))
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        value_layer = self.transpose_for_scores(self.value(to_sequence))
        context_layer = torch.matmul(attention_probs, value_layer).permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(*context_layer.size()[:-2], self.dim_value_all)
        context_layer = self.dense(context_layer)

        if self.use_layer_norm:
            context_layer = self.layer_norm(from_sequence + context_layer)
        if self.output_attentions:
            return (context_layer, attention_probs)
        return (context_layer,)

    def transpose_for_scores(self, x):
        x = x.view(*x.size()[:-1], self.num_attention_heads, -1)
        return x.permute(0, 2, 1, 3)

    def init_mixing_matrix(self, scale=0.2):
        mixing = torch.zeros(self.num_attention_heads, self.dim_key_query_all)
        if self.mixing_initialization is MixingMatrixInit.CONCATENATE:
            dim_head = int(math.ceil(self.dim_key_query_all / self.num_attention_heads))
            for i in range(self.num_attention_heads):
                mixing[i, i * dim_head:(i + 1) * dim_head] = 1.0
        elif self.mixing_initialization is MixingMatrixInit.ALL_ONES:
            mixing.fill_(1.0)
        elif self.mixing_initialization is MixingMatrixInit.UNIFORM:
            mixing.normal_(std=scale)
        else:
            raise ValueError(f"Unknown mixing matrix initialization: {self.mixing_initialization}")
        return nn.Parameter(mixing)


class TransformerEncoderLayer(nn.Module):
    """Pre-LN block with collaborative attention."""

    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1, key_query_dim=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CollaborativeAttention(
            dim_input=d_model, dim_value_all=d_model, dim_key_query_all=key_query_dim or d_model,
            dim_output=d_model, num_attention_heads=num_heads, output_attentions=False,
            attention_probs_dropout_prob=dropout, use_dense_layer=True, use_layer_norm=False,
            mixing_initialization=MixingMatrixInit.UNIFORM)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ff_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_dim, d_model))
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout1(self.attn(self.norm1(x))[0])
        x = x + self.dropout2(self.ff(self.norm2(x)))
        return x


class TransformerModel(nn.Module):
    def __init__(self, num_layers, num_heads, d_model, ff_dim, embed_mode, num_lead=1, chunk_len=50,
                 embedding_dim=512, dropout=0.1, max_tokens=1024, key_query_dim=None,
                 stem_channels=(128, 256, 256, 256), stem_last_stride=1):
        super().__init__()
        self.embedding_dim = embedding_dim
        if embed_mode == "linear":
            self.embed = LinearEmbed(num_lead, chunk_len, d_model)
        elif embed_mode == "cnn":
            self.embed = ConvEmbed(num_lead, d_model, stem_channels, stem_last_stride)
        else:
            raise NotImplementedError(f"Embedding model `{embed_mode}` is not implemented")

        self.embed_norm = nn.LayerNorm(d_model)             # replaces the old hard clamp(-2, 2)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_tokens)
        self.layers = nn.Sequential(*[TransformerEncoderLayer(d_model, num_heads, ff_dim, dropout, key_query_dim)
                                      for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model)             # Pre-LN stacks need a final norm before pooling
        self.fc = nn.Linear(d_model, embedding_dim)

    def forward(self, x):                                   # (B, C, L) -> (B, embedding_dim)
        feat = self.embed_norm(self.embed(x))
        feat = self.layers(self.pos_encoder(feat))
        feat = self.final_norm(feat).mean(dim=1)            # mean pool over tokens
        return self.fc(feat)


def build_encoder(cfg: Config) -> TransformerModel:
    return TransformerModel(cfg.num_layers, cfg.num_heads, cfg.d_model, cfg.ff_mult * cfg.d_model, cfg.embed_mode,
                            num_lead=cfg.num_lead, chunk_len=cfg.chunk_len, embedding_dim=cfg.embedding_dim,
                            dropout=cfg.dropout, max_tokens=cfg.max_tokens, key_query_dim=cfg.key_query_dim,
                            stem_channels=tuple(cfg.stem_channels), stem_last_stride=cfg.stem_last_stride)


def num_tokens(cfg: Config, L: int) -> int:
    return cnn_num_tokens(L, cfg.stem_last_stride) if cfg.embed_mode == "cnn" else math.ceil(L / cfg.chunk_len)


def encoder_name(cfg: Config) -> str:
    return f"transformer_l{cfg.num_layers}_h{cfg.num_heads}_d{cfg.d_model}_{cfg.embed_mode}"


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -grad


class AdversarialMasker(nn.Module):
    def __init__(self, in_ch=1, patch_size=16, mask_ratio=0.10, logit_scale=3.0):
        super().__init__()
        self.patch_size, self.mask_ratio, self.logit_scale = patch_size, mask_ratio, logit_scale
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, 7, padding=3), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv1d(32, 64, 5, padding=2), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv1d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU())
        self.head = nn.Conv1d(64, 1, 1)

    def forward(self, x, strength: float = 1.0):
        """x: (B, C, L). Returns (masked x, fraction of patches hidden)."""
        B, C, L = x.shape
        ps = self.patch_size
        P = math.ceil(L / ps)
        h = F.pad(self.net(x), (0, P * ps - L))
        h = F.avg_pool1d(h, ps)                                         # (B, 64, P): one feature vector per patch
        raw = self.head(h).squeeze(1).float()                           # (B, P), fp32 even under autocast
        logits = self.logit_scale * torch.tanh(raw / self.logit_scale)  # bounded -> Gumbel noise never becomes irrelevant

        u = torch.rand_like(logits).clamp_(1e-6, 1 - 1e-6)
        noisy = strength * logits + (-torch.log(-torch.log(u)))         # Gumbel-perturbed scores
        K = max(1, int(round(self.mask_ratio * P)))
        hard = torch.zeros_like(noisy).scatter_(1, noisy.topk(K, dim=1).indices, 1.0)
        soft = torch.sigmoid(noisy)
        m = hard + (soft - soft.detach())                               # straight-through: forward = hard, backward = soft
        m = GradReverse.apply(m)                                        # masker ascends the loss the encoder descends
        m = m.repeat_interleave(ps, dim=1)[:, :L].unsqueeze(1)          # (B, 1, L), shared across channels
        return x * (1.0 - m), hard.mean().detach()


def simclr_loss_fn(z1, z2, temperature=0.07):
    """NT-Xent. z1, z2: (B, H). Returns (loss, top-1 retrieval accuracy of the positive)."""
    B = z1.size(0)
    z = F.normalize(torch.cat([z1, z2], dim=0).float(), dim=1)
    sim = z @ z.T / temperature
    eye = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, torch.finfo(sim.dtype).min)
    labels = (torch.arange(2 * B, device=z.device) + B) % (2 * B)
    loss = F.cross_entropy(sim, labels)
    acc = (sim.argmax(dim=1) == labels).float().mean().detach()
    return loss, acc


class BaseModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.encoder = build_encoder(cfg)
        self.projector = nn.Sequential(nn.Linear(cfg.embedding_dim, cfg.proj_hidden_dim), nn.ReLU(),
                                       nn.Linear(cfg.proj_hidden_dim, cfg.output_dim))

    def forward(self, x):
        feats = self.encoder(x)
        return {"feats": feats, "z": self.projector(feats)}


class AdvMaskModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.base_model = BaseModel(cfg)
        self.masker = AdversarialMasker(cfg.num_lead, cfg.adv_patch_size, cfg.adv_mask_ratio, cfg.adv_logit_scale)
        self.temperature = cfg.temperature
        self.use_amp = cfg.use_amp

    def forward(self, x_orig, x_aug, adv_strength: Optional[float] = None):
        """adv_strength=None -> no masker (validation); otherwise adversarial masking with that strength."""
        amp = self.use_amp and x_orig.is_cuda
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            if adv_strength is None:
                view2, mask_frac = x_aug, torch.zeros((), device=x_aug.device)
            else:
                view2, mask_frac = self.masker(x_aug, adv_strength)
            B = x_orig.size(0)
            z = self.base_model(torch.cat([x_orig, view2], dim=0))["z"]  # one pass; no batch-stat layers in the encoder
        loss, acc = simclr_loss_fn(z[:B].float(), z[B:].float(), self.temperature)   # loss always in fp32
        return loss, acc, mask_frac

def build_optimizer(model: AdvMaskModel, cfg: Config):
    decay, no_decay, gen = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("masker."):
            gen.append(p)
        elif p.ndim < 2 or n.endswith(("m_t", "m_c")):     # norms, biases, mixing matrices
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [{"params": decay, "weight_decay": cfg.weight_decay, "lr": cfg.lr},
              {"params": no_decay, "weight_decay": 0.0, "lr": cfg.lr},
              {"params": gen, "weight_decay": 0.0, "lr": cfg.lr * cfg.gen_lr_mult}]
    print(f"param groups: decay={sum(p.numel() for p in decay):,}  no_decay={sum(p.numel() for p in no_decay):,}  masker={sum(p.numel() for p in gen):,}")
    return torch.optim.AdamW(groups, betas=(0.9, 0.999))


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    total = cfg.num_epochs * steps_per_epoch
    warmup = max(1, cfg.warmup_epochs * steps_per_epoch)

    def lr_lambda(step):                                   # linear warm-up, then ONE cosine decay (no restarts)
        if step < warmup:
            return (step + 1) / warmup
        p = (step - warmup) / max(1, total - warmup)
        return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def adv_strength_at(epoch: int, cfg: Config) -> float:
    if epoch < cfg.adv_start_epoch:
        return 0.0
    return min(1.0, (epoch - cfg.adv_start_epoch + 1) / max(1, cfg.adv_ramp_epochs))


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, epoch, cfg):
    model.train()
    strength = adv_strength_at(epoch, cfg)
    tot = acc_tot = mf_tot = 0.0
    with tqdm(loader, desc=f"Epoch {epoch + 1}/{cfg.num_epochs} [adv={strength:.2f}]", unit="batch", leave=False) as pbar:
        for i, (x, x_aug) in enumerate(pbar):
            x, x_aug = x.to(device, non_blocking=True), x_aug.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss, acc, mf = model(x, x_aug, adv_strength=strength)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1}, batch {i}")
            scaler.scale(loss).backward()
            if cfg.grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()                               # per step
            tot += loss.item(); acc_tot += acc.item(); mf_tot += mf.item()
            pbar.set_postfix(loss=tot / (i + 1), acc=acc_tot / (i + 1))
    n = len(loader)
    return {"train_loss": tot / n, "train_acc": acc_tot / n, "mask_frac": mf_tot / n, "adv_strength": strength}


@torch.no_grad()
def evaluate(model, loader, device):
    """Deterministic validation loss on the non-adversarial views (clean vs. random-block-masked + noise)."""
    model.eval()
    tot = acc_tot = 0.0
    count = 0
    for x, x_aug in loader:
        x, x_aug = x.to(device), x_aug.to(device)
        loss, acc, _ = model(x, x_aug, adv_strength=None)
        b = x.size(0)
        tot += loss.item() * b; acc_tot += acc.item() * b; count += b
    return {"val_loss": tot / count, "val_acc": acc_tot / count}


def make_scaler(cfg: Config):
    enabled = bool(cfg.use_amp and torch.cuda.is_available())
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):                    # older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

def run_sanity_checks(cfg: Config, L: int = 1648):
    print("---- sanity checks ----")
    # 1) random block mask really masks
    x = torch.randn(1, L)
    out = MultiBlockMask(cfg.num_random_masks, cfg.random_mask_ratio)(x)
    frac = (out == 0).float().mean().item()
    assert frac > 0.05, f"random masking is (almost) a no-op: {frac:.3f}"
    print(f"[ok] MultiBlockMask hides {frac * 100:.1f}% of a (1, {L}) spectrum")

    # 2) gradient reversal
    t = torch.ones(3, requires_grad=True)
    GradReverse.apply(t).sum().backward()
    assert torch.all(t.grad == -1)
    print("[ok] GradReverse flips the gradient sign")

    # 3) shapes / tokens
    tmp = copy.deepcopy(cfg)
    tmp.num_lead = 1
    model = AdvMaskModel(tmp).eval()
    n_tok = num_tokens(tmp, L)
    assert n_tok <= tmp.max_tokens, f"{n_tok} tokens > max_tokens={tmp.max_tokens}"
    xb = torch.randn(4, 1, L)
    assert model.base_model.encoder(xb).shape == (4, tmp.embedding_dim)
    print(f"[ok] encoder output OK; {n_tok} tokens per spectrum (max_tokens={tmp.max_tokens})")

    # 4) adversarial masker: exact budget, masks change the signal
    masked, frac_patches = model.masker(xb, strength=1.0)
    K = max(1, int(round(tmp.adv_mask_ratio * math.ceil(L / tmp.adv_patch_size))))
    assert not torch.equal(masked, xb)
    assert abs(frac_patches.item() - K / math.ceil(L / tmp.adv_patch_size)) < 1e-6
    print(f"[ok] masker hides exactly {K} patches ({frac_patches.item() * 100:.1f}%) per spectrum")

    # 5) both parts receive gradients, masker gradient is non-zero
    model.train()
    loss, acc, _ = model(xb, xb + 0.05 * torch.randn_like(xb), adv_strength=1.0)
    loss.backward()
    g_enc = model.base_model.encoder.fc.weight.grad.abs().sum().item()
    g_gen = sum(p.grad.abs().sum().item() for p in model.masker.parameters() if p.grad is not None)
    assert g_enc > 0 and g_gen > 0, (g_enc, g_gen)
    print(f"[ok] gradients flow: encoder |g|={g_enc:.3e}, masker |g|={g_gen:.3e}; loss={loss.item():.3f}")
    print("---- all checks passed ----")


run_sanity_checks(cfg)

def benchmark(c: Config, L: int = 1648, steps: int = 3, n_train: int = 44128):
    c = copy.deepcopy(c); c.num_lead = 1
    model = AdvMaskModel(c).to(device).train()
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=c.lr)
    x = torch.randn(c.batch_size, 1, L, device=device)
    times = []
    for i in range(steps + 1):                              # first step = warm-up, not timed
        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.time()
        loss, _, _ = model(x, x + 0.05 * torch.randn_like(x), adv_strength=1.0)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if device.type == "cuda": torch.cuda.synchronize()
        if i > 0: times.append(time.time() - t0)
    s = sum(times) / len(times)
    epoch_min = s * (n_train // c.batch_size) / 60
    print(f"{n_params / 1e6:6.2f}M params | {num_tokens(c, L):3d} tokens | {s:6.2f} s/step | ~{epoch_min:7.1f} min/epoch (compute only)")


def benchmark_loader(c: Config, n_batches: int = 10):
    """Time the input pipeline alone. If this is much larger than the s/step above, data loading is your bottleneck."""
    train_loader, _, _, _ = make_loaders(copy.deepcopy(c))
    it = iter(train_loader)
    next(it)                                                # first batch includes worker start-up
    t0 = time.time()
    for _ in range(n_batches):
        next(it)
    print(f"data loading: {(time.time() - t0) / n_batches:.3f} s/batch")


RUN_LOADER_BENCHMARK = False                                # set True to run (loads your data once)
if RUN_LOADER_BENCHMARK:
    benchmark_loader(cfg)

RUN_BENCHMARK = False                                       # set True to run
if RUN_BENCHMARK:
    for name in PRESETS:
        print(f"{name:11s}", end=" ")
        benchmark(apply_preset(cfg, name))

def train(cfg: Config):
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)
    train_loader, val_loader, normalizer, L = make_loaders(cfg)

    n_tok = num_tokens(cfg, L)
    assert n_tok <= cfg.max_tokens, f"{n_tok} tokens > cfg.max_tokens={cfg.max_tokens}"
    print(f"{encoder_name(cfg)}: {n_tok} tokens per spectrum")

    model = AdvMaskModel(cfg).to(device)
    print(f"total parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = make_scaler(cfg)

    torch.save(normalizer.state(), os.path.join(cfg.save_dir, "normalizer.pt"))
    with open(os.path.join(cfg.save_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    history, best_val, no_improve, start_epoch = [], float("inf"), 0, 0
    if cfg.resume:
        ck = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"]); scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck.get("scaler", {}))
        history, best_val, no_improve, start_epoch = ck["history"], ck["best_val"], ck["no_improve"], ck["epoch"] + 1
        print(f"resumed from {cfg.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, cfg.num_epochs):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, epoch, cfg)
        va = evaluate(model, val_loader, device)
        rec = {"epoch": epoch + 1, **tr, **va, "lr": optimizer.param_groups[0]["lr"], "time_s": time.time() - t0}
        history.append(rec)

        improved = va["val_loss"] < best_val - cfg.min_delta
        if improved:
            best_val, no_improve = va["val_loss"], 0
            atomic_save({"model": model.state_dict(), "cfg": asdict(cfg), "epoch": epoch, "val_loss": best_val},
                        os.path.join(cfg.save_dir, "best.pth"))
            atomic_save(model.base_model.encoder.state_dict(), os.path.join(cfg.save_dir, "encoder_best.pth"))
        elif epoch + 1 > cfg.es_start_epoch:
            no_improve += 1

        print(f"ep {epoch + 1:3d} | train {tr['train_loss']:.4f} (acc {tr['train_acc']:.3f}, adv {tr['adv_strength']:.2f}) | "
              f"val {va['val_loss']:.4f} (acc {va['val_acc']:.3f}) | lr {rec['lr']:.2e} | {rec['time_s']:.0f}s"
              + ("  * best" if improved else f"  (no improvement: {no_improve})"))

        atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                     "epoch": epoch, "best_val": best_val, "no_improve": no_improve, "history": history, "cfg": asdict(cfg)},
                    os.path.join(cfg.save_dir, "last.pth"))
        if cfg.save_every and (epoch + 1) % cfg.save_every == 0:
            atomic_save(model.state_dict(), os.path.join(cfg.save_dir, f"model_epoch_{epoch + 1}.pth"))
        with open(os.path.join(cfg.save_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=1)

        if no_improve >= cfg.patience:
            print(f"Early stopping at epoch {epoch + 1} (best val loss {best_val:.4f})")
            break
    return model, history


if __name__ == "__main__" or True:      # notebook: just run
    model, history = train(cfg)

try:
    import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train (adversarial views)")
    ax[0].plot(ep, [h["val_loss"] for h in history], label="val (clean vs random-masked)")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("NT-Xent"); ax[0].legend()
    ax[1].plot(ep, [h["train_acc"] for h in history], label="train")
    ax[1].plot(ep, [h["val_acc"] for h in history], label="val")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("top-1 positive retrieval"); ax[1].legend()
    plt.tight_layout(); plt.show()
except ImportError:
    print("matplotlib not installed - see saved_models/history.json")

def load_encoder(save_dir=None, device=device):
    save_dir = save_dir or cfg.save_dir
    with open(os.path.join(save_dir, "config.json")) as f:
        c = Config(**json.load(f))
    enc = build_encoder(c).to(device)
    enc.load_state_dict(torch.load(os.path.join(save_dir, "encoder_best.pth"), map_location=device))
    st = torch.load(os.path.join(save_dir, "normalizer.pt"))
    return enc.eval(), Normalize(st["mode"], st["mean"], st["std"])


@torch.no_grad()
def embed_spectra(encoder, normalizer, X: np.ndarray, batch_size=256, device=device):
    """X: (N, L) or (N, C, L) numpy array -> (N, embedding_dim) numpy array."""
    outs = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(np.asarray(X[i:i + batch_size], dtype=np.float32))
        if xb.ndim == 2:
            xb = xb[:, None, :]
        xb = torch.stack([normalizer(s) for s in xb]).to(device)
        outs.append(encoder(xb).cpu())
    return torch.cat(outs).numpy()

# Example:
encoder, normalizer = load_encoder()
emb = embed_spectra(encoder, normalizer, np.load(cfg.data_path, mmap_mode="r")[:1000])