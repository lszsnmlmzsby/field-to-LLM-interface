"""Shared components used by the current multi-source field workflow."""
from __future__ import annotations
import math
from collections.abc import Mapping, Sequence
from typing import Any
import torch
from torch import nn

def build_patch_encoder_config(
    *,
    patch_encoder_cfg: Mapping[str, Any] | None,
    field_keys: Sequence[str],
    patch_size: int,
) -> dict[str, Any]:
    cfg = dict(patch_encoder_cfg or {})
    model_cfg = dict(cfg.get("model") or {})
    patch_size = int(patch_size)
    multipliers = [int(value) for value in model_cfg.get("channel_multipliers", [1, 2])]
    down_factor = 2 ** len(multipliers)
    if patch_size % down_factor != 0:
        raise ValueError(
            "patch_size must be divisible by the patch encoder downsampling factor. "
            f"Got patch_size={patch_size}, down_factor={down_factor} from channel_multipliers={multipliers}."
        )
    model_defaults = {
        "name": "conv_token_autoencoder_2d",
        "input_size": [patch_size, patch_size],
        "base_channels": 32,
        "channel_multipliers": multipliers,
        "num_res_blocks": 1,
        "latent_dim": 128,
        "latent_grid": [patch_size // down_factor, patch_size // down_factor],
        "dropout": 0.0,
        "norm": "group",
        "activation": "gelu",
        "output_activation": "identity",
    }
    for key, value in model_defaults.items():
        model_cfg.setdefault(key, value)
    model_cfg["in_channels"] = int(model_cfg.get("in_channels", len(field_keys)))
    model_cfg["out_channels"] = int(model_cfg.get("out_channels", len(field_keys)))
    if model_cfg["in_channels"] != len(field_keys) or model_cfg["out_channels"] != len(field_keys):
        raise ValueError(
            "patch_encoder.model in_channels/out_channels must match the selected fields. "
            f"Got in_channels={model_cfg['in_channels']}, out_channels={model_cfg['out_channels']}, "
            f"fields={list(field_keys)}."
        )
    normalization_cfg = dict(
        cfg.get(
            "normalization",
            {
                "mode": "zscore",
                "scope": "channel",
                "stats_path": None,
                "clip_min": None,
                "clip_max": None,
            },
        )
        or {}
    )
    return {
        "model": model_cfg,
        "data": {
            "dataset": {
                "hdf5_dataset_key": str(field_keys[0]) if len(field_keys) == 1 else None,
                "hdf5_dataset_keys": list(field_keys) if len(field_keys) > 1 else [],
                "normalization": normalization_cfg,
            }
        },
    }

def sinusoidal_position_encoding(length: int, dim: int) -> torch.Tensor:
    """Return a deterministic [length, dim] sinusoidal position table."""
    if int(length) <= 0 or int(dim) <= 0:
        raise ValueError(f"Position encoding length and dim must be positive, got {length} and {dim}.")
    positions = torch.arange(int(length), dtype=torch.float32).unsqueeze(1)
    frequency_count = (int(dim) + 1) // 2
    frequencies = torch.exp(
        torch.arange(frequency_count, dtype=torch.float32)
        * (-math.log(10000.0) / max(frequency_count - 1, 1))
    )
    angles = positions * frequencies.unsqueeze(0)
    encoding = torch.empty(int(length), frequency_count * 2, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(angles)
    encoding[:, 1::2] = torch.cos(angles)
    return encoding[:, : int(dim)]

def sinusoidal_2d_position_encoding(height: int, width: int, dim: int) -> torch.Tensor:
    """Encode row and column independently and flatten positions in row-major order."""
    if int(height) <= 0 or int(width) <= 0:
        raise ValueError(f"Spatial dimensions must be positive, got {(height, width)}.")
    row_dim = int(dim) // 2
    col_dim = int(dim) - row_dim
    if row_dim <= 0 or col_dim <= 0:
        raise ValueError(f"2D position encoding requires dim >= 2, got {dim}.")
    rows = sinusoidal_position_encoding(int(height), row_dim)
    cols = sinusoidal_position_encoding(int(width), col_dim)
    row_grid = rows[:, None, :].expand(int(height), int(width), row_dim)
    col_grid = cols[None, :, :].expand(int(height), int(width), col_dim)
    return torch.cat([row_grid, col_grid], dim=-1).reshape(1, int(height) * int(width), int(dim))

class SpatialTransformerBlock(nn.Module):
    """Contextualize spatial tokens without replacing their position-specific states."""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=int(heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(dim * 4, dim),
            nn.Dropout(float(dropout)),
        )
        self.capture_attention = False
        self.last_self_attention_weights: torch.Tensor | None = None
        self.last_attention_weights: torch.Tensor | None = None

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.self_norm(tokens)
        attended, weights = self.self_attn(
            normalized,
            normalized,
            normalized,
            need_weights=bool(self.capture_attention),
            average_attn_weights=False,
        )
        self.last_self_attention_weights = (
            weights.detach().float().cpu()
            if self.capture_attention and weights is not None
            else None
        )
        self.last_attention_weights = self.last_self_attention_weights
        tokens = tokens + attended
        return tokens + self.ffn(self.ffn_norm(tokens))
