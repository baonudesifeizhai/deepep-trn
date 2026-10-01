"""Row codec: payload columns plus bit-packed routing metadata in one row.

Packing metadata into the payload row keeps each dispatch to a single
collective. int32/fp32 values are stored as pairs of 16-bit columns and decoded
bit-exactly, the same trick nkilib's permute_routed_tokens uses.

Row layout, in 16-bit columns::

    [0, H)            payload
    [H, H+2K)         top-k weights, fp32
    [H+2K, H+4K)      top-k global expert ids, int32 (-1 = masked)
    [H+4K, H+4K+2)    source-local token index, int32
    [H+4K+2, W)       zero padding up to META_ALIGN
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

META_ALIGN = 32
SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)


def _cols(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return t.contiguous().view(dtype)


@dataclass(frozen=True)
class RowCodec:
    hidden: int
    topk: int
    dtype: torch.dtype = torch.bfloat16

    def __post_init__(self):
        if self.dtype not in SUPPORTED_DTYPES:
            raise ValueError(f'payload dtype must be one of {SUPPORTED_DTYPES}, got {self.dtype}')

    @property
    def meta_cols(self) -> int:
        need = 4 * self.topk + 2
        return (need + META_ALIGN - 1) // META_ALIGN * META_ALIGN

    @property
    def width(self) -> int:
        return self.hidden + self.meta_cols

    def encode(self, x: torch.Tensor, topk_idx: torch.Tensor | None,
               topk_weights: torch.Tensor | None, token_index: torch.Tensor) -> torch.Tensor:
        n, H, K = x.shape[0], self.hidden, self.topk
        rows = torch.zeros(n, self.width, dtype=self.dtype)
        rows[:, :H] = x
        if topk_weights is not None:
            rows[:, H:H + 2 * K] = _cols(topk_weights.to(torch.float32), self.dtype)
        idx = topk_idx.to(torch.int32) if topk_idx is not None else torch.full((n, K), -1, dtype=torch.int32)
        rows[:, H + 2 * K:H + 4 * K] = _cols(idx, self.dtype)
        rows[:, H + 4 * K:H + 4 * K + 2] = _cols(token_index.to(torch.int32)[:, None], self.dtype)
        return rows

    def decode(self, rows: torch.Tensor):
        """Returns (x, topk_idx int64, topk_weights fp32, token_index int64)."""
        H, K = self.hidden, self.topk
        x = rows[:, :H]
        weights = _cols(rows[:, H:H + 2 * K], torch.float32)
        idx = _cols(rows[:, H + 2 * K:H + 4 * K], torch.int32).to(torch.int64)
        token = _cols(rows[:, H + 4 * K:H + 4 * K + 2], torch.int32)[:, 0].to(torch.int64)
        return x, idx, weights, token
