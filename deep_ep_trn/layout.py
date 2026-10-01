"""Routing layout helpers (vectorised torch, device-agnostic)."""
from __future__ import annotations

import torch


def rank_of_expert(topk_idx: torch.Tensor, num_experts: int, num_ranks: int) -> torch.Tensor:
    """[T, K] owning rank of every routed expert, -1 where ``topk_idx`` is masked."""
    if num_experts % num_ranks:
        raise ValueError(f'num_experts={num_experts} must be divisible by num_ranks={num_ranks}')
    local = num_experts // num_ranks
    return torch.where(topk_idx >= 0, topk_idx // local, torch.full_like(topk_idx, -1))


def get_dispatch_layout(topk_idx: torch.Tensor, num_experts: int, num_ranks: int):
    """DeepEP ``get_dispatch_layout`` for a single node.

    Returns:
        num_tokens_per_rank: [R] int32, tokens sent to each rank (one per token-rank pair).
        num_tokens_per_expert: [E] int32, routed (token, expert) pairs per expert.
        is_token_in_rank: [T, R] bool.
    """
    topk_idx = topk_idx.to(torch.int64)
    owner = rank_of_expert(topk_idx, num_experts, num_ranks)
    is_token_in_rank = (owner[:, :, None] == torch.arange(num_ranks)[None, None, :]).any(1)
    num_tokens_per_rank = is_token_in_rank.sum(0).to(torch.int32)
    routed = topk_idx[topk_idx >= 0]
    num_tokens_per_expert = torch.bincount(routed, minlength=num_experts).to(torch.int32)
    return num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank


def send_plan(is_token_in_rank: torch.Tensor):
    """Destination-major send order: (token ids [N], rows per destination [R])."""
    dst, token = is_token_in_rank.t().nonzero(as_tuple=True)
    counts = is_token_in_rank.sum(0).to(torch.int64)
    return token, counts


def pair_plan(owner: torch.Tensor, num_ranks: int):
    """(token, k) pairs ordered by destination rank, then token, then k.

    Returns (token ids [P], k ids [P], pairs per destination [R]).
    """
    keys = []
    for dst in range(num_ranks):
        t, k = (owner == dst).nonzero(as_tuple=True)
        keys.append((t, k))
    token = torch.cat([t for t, _ in keys]) if keys else torch.empty(0, dtype=torch.int64)
    k = torch.cat([k for _, k in keys]) if keys else torch.empty(0, dtype=torch.int64)
    counts = torch.tensor([t.numel() for t, _ in keys], dtype=torch.int64)
    return token, k, counts


def slot_index(counts: torch.Tensor, slot_rows: int) -> torch.Tensor:
    """Row positions of the valid entries of a slotted buffer, source-major."""
    pos = torch.arange(slot_rows)[None, :]
    valid = pos < counts[:, None]
    src, row = valid.nonzero(as_tuple=True)
    return src * slot_rows + row


def exclusive_cumsum(counts: torch.Tensor) -> torch.Tensor:
    return torch.cumsum(counts, 0) - counts
