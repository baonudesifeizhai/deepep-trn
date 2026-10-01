"""Shared setup for torchrun correctness tests (run with --nproc-per-node=4)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.distributed as dist

from deep_ep_trn.testing import bitwise_equal, expert_scale, make_routing, strided_ep_groups  # noqa: F401

KINDS = ('balanced', 'random', 'hot_rank0', 'ragged', 'masked')


def parse_args(description: str):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--backend', choices=['nki', 'gloo'], default='nki')
    p.add_argument('--ep', type=int, default=4, help='EP group size; world is split into world/ep groups')
    p.add_argument('--tokens', type=int, default=24, help='tokens per rank (ragged shrinks this)')
    p.add_argument('--max-tokens', type=int, default=32, help='static per-rank token capacity')
    p.add_argument('--hidden', type=int, default=2048)
    p.add_argument('--experts', type=int, default=128)
    p.add_argument('--topk', type=int, default=8)
    p.add_argument('--kinds', default=','.join(KINDS))
    return p.parse_args()


def init(backend: str):
    if backend == 'nki':
        os.environ.setdefault('NEURON_PLATFORM_TARGET_OVERRIDE', 'trn2')
        import torch_neuronx  # noqa: F401
        dist.init_process_group('neuron')
    else:
        dist.init_process_group('gloo')
    torch.set_num_threads(1)
    return dist.get_rank(), dist.get_world_size(), dist.new_group(backend='gloo')


def make_buffer(backend: str, ep: int, max_tokens: int):
    from deep_ep_trn import Buffer
    rank = dist.get_rank()
    groups = strided_ep_groups(dist.get_world_size(), ep)
    mine = peers = None
    for ranks in groups:  # every rank must create every group, in order
        group = dist.new_group(ranks=ranks, backend='gloo')
        if rank in ranks:
            mine, peers = group, ranks
    buf = Buffer(mine, backend=backend, replica_groups=groups, num_max_tokens_per_rank=max_tokens)
    return buf, peers


def gather_peers(obj, peers, control):
    everyone = [None] * dist.get_world_size()
    dist.all_gather_object(everyone, (dist.get_rank(), obj), group=control)
    data = dict(everyone)
    return [data[p] for p in peers]


def to_device(t: torch.Tensor, backend: str) -> torch.Tensor:
    if backend != 'nki':
        return t
    return t.to(torch.device('neuron', int(os.environ.get('LOCAL_RANK', 0))))
