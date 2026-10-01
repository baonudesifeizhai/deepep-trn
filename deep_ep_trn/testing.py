"""Helpers shared by tests/ and bench/. Not part of the stable Buffer API."""
from __future__ import annotations

import torch

# Correctness tests use all kinds; benchmarks use the ones that keep T fixed.
ROUTINGS = ('random', 'balanced', 'zipf', 'hot_rank0', 'ragged', 'masked')


def strided_ep_groups(world_size: int, ep: int) -> list[list[int]]:
    """EP groups of ``ep`` ranks at stride ``world_size // ep``.

    ``world=4, ep=2`` gives ``[[0, 2], [1, 3]]``: TP-adjacent ranks land in
    different EP groups, the TP2 x EP2 layout used by sglang-neuron.
    """
    if world_size % ep:
        raise ValueError(f'world size {world_size} is not divisible by ep {ep}')
    stride = world_size // ep
    return [list(range(g, world_size, stride)) for g in range(stride)]


def make_routing(kind: str, tokens: int, experts: int, topk: int, ep: int, ep_rank: int, seed: int):
    """Synthetic router output: (topk_idx [T, K] int64, topk_weights [T, K] fp32).

    random     uniform top-k
    balanced   every token hits every rank evenly
    zipf       expert popularity ~ 1/rank, so EP rank 0 is overloaded
    hot_rank0  every token routes only to rank 0's experts
    ragged     random, but T shrinks with ep_rank and the last rank is idle (T=0)
    masked     random with ~30% of entries set to -1
    """
    g = torch.Generator().manual_seed(seed)
    E, K = experts, topk
    T = tokens
    if kind == 'ragged':
        T = 0 if ep_rank == ep - 1 else max(1, tokens - ep_rank * (tokens // ep))
    if kind == 'balanced':
        idx = (torch.arange(T)[:, None] * 4 + torch.arange(K)[None, :] * (E // K)) % E
    elif kind == 'hot_rank0':
        local = E // ep
        if local < K:
            raise ValueError('hot_rank0 needs experts / ep >= topk')
        idx = (torch.arange(T)[:, None] + torch.arange(K)[None, :]) % local
    elif kind == 'zipf':
        logits = -torch.log(torch.arange(1, E + 1, dtype=torch.float32))
        gumbel = -torch.log(-torch.log(torch.rand(T, E, generator=g).clamp(1e-9, 1 - 1e-9)))
        idx = (logits + gumbel).topk(K, dim=-1).indices
    elif kind in ('random', 'ragged', 'masked'):
        idx = torch.rand(T, E, generator=g).topk(K, dim=-1).indices
    else:
        raise ValueError(f'unknown routing {kind!r}; expected one of {ROUTINGS}')
    if kind == 'masked':
        idx = torch.where(torch.rand(T, K, generator=g) < 0.3, torch.full_like(idx, -1), idx)
    weights = torch.rand(T, K, generator=g)
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
    return idx.to(torch.int64), weights


def expert_scale(global_expert: torch.Tensor) -> torch.Tensor:
    """Deterministic per-expert 'compute' that is exact in bf16."""
    return 1.0 + (global_expert % 8).to(torch.float32) / 8.0


def bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.element_size() == 2:
        return torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))
    return torch.equal(a, b)
