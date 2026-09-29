"""Query-independent, row-major packing; padded slots are never real cells."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def block_shape(value=(1, 1)) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2 or any(type(n) is not int or n <= 0 for n in value):
        raise ValueError("memory.block_shape must contain two positive integers")
    return tuple(value)


def pack_blocks(field: torch.Tensor, shape) -> tuple[torch.Tensor, torch.Tensor]:
    """[B,C,H,W] -> [B,ceil(H/ph)*ceil(W/pw),ph*pw,C], plus slot mask.

    Blocks and slots both run left-to-right, then top-to-bottom. Zero padding
    exists only on the bottom/right and is accompanied by an explicit mask.
    """
    ph, pw = block_shape(shape)
    if field.ndim != 4 or min(field.shape) < 1:
        raise ValueError("Packing requires nonempty [B,C,H,W]")
    b, c, h, w = field.shape
    gh, gw = (h + ph - 1) // ph, (w + pw - 1) // pw
    def pack(x):
        x = F.pad(x, (0, gw * pw - w, 0, gh * ph - h))
        return x.reshape(b, x.shape[1], gh, ph, gw, pw).permute(0, 2, 4, 3, 5, 1).reshape(b, gh * gw, ph * pw, -1)
    values = pack(field)
    valid = pack(field.new_ones((b, 1, h, w))).squeeze(-1).bool()
    return values, valid


def masked_reconstruction(prediction, target, valid):
    error = F.smooth_l1_loss(prediction.float(), target.float(), reduction="none")
    return (error * valid).sum() / valid.sum().clamp_min(1)
