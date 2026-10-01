"""trn2 transport: one NKI ``ncc.all_to_all_v`` per exchange.

Constraints of the installed Neuron stack (see docs/trn2-constraints.md):

- Intra-node groups only support ``has_rdispls=False``: incoming chunks land in
  equally spaced slots of ``dst.numel() / EP`` elements.
- Without receive displacements the driver requires ``dst`` to be exactly
  ``EP * src`` ("output size ... must be rank_n * input size"), so every slot is
  as large as the whole send buffer. Callers should keep send capacity tight.
- nkilib's ``permute_a2av``/``unpermute_a2av`` are Trn3-only (EP >= 8 across
  >= 2 devices), so this module carries its own kernel.
- The kernel targets LNC=2 (two physical cores per logical rank).

Only one replica-group layout per process is supported: NKI resolves module
globals, not closure cells, when it traces the kernel.
"""
from __future__ import annotations

import os

import torch
import torch_neuronx  # noqa: F401  registers the neuron device and compile backend
from torch_neuronx import nki_op, wrap_nki
import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from ..profiling import timer
from .base import Transport

_TILE = 128
_LNC2_CORES = (0, 1)
_REPLICA_GROUP = None
_REPLICA_GROUP_LISTS = None


def _copy_rows(dst, src, rows, width, dtype):
    for i in range((rows + _TILE - 1) // _TILE):
        lo, hi = i * _TILE, min(i * _TILE + _TILE, rows)
        tile = nl.ndarray((hi - lo, width), dtype=dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=tile, src=src[lo:hi, :])
        nisa.dma_copy(dst=dst[lo:hi, :], src=tile)


@wrap_nki
@nki.jit
def _a2av_fixed_slot_kernel(src, meta_in, slot_rows):
    """[cap, W] packed rows + [3, EP] int32 element metadata -> [EP * slot_rows, W]."""
    cap, width = src.shape
    n = meta_in.shape[1]
    dtype = src.dtype
    # Collectives must use internal HBM staging, not graph IO tensors.
    send = nl.ndarray((cap, width), dtype=dtype, buffer=nl.shared_hbm)
    _copy_rows(send, src, cap, width, dtype)
    meta_sb = nl.ndarray((3, n), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=meta_sb, src=meta_in)
    meta_u = nl.ndarray((3, n), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=meta_u, src=meta_sb)
    meta = nl.ndarray((3, n), dtype=nl.uint32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=meta, src=meta_u)
    recv = nl.ndarray((n * slot_rows, width), dtype=dtype, buffer=nl.shared_hbm)
    nisa.core_barrier(send, _LNC2_CORES)
    nisa.core_barrier(meta, _LNC2_CORES)
    ncc.all_to_all_v(srcs=[send], dsts=[recv], replica_group=_REPLICA_GROUP,
                     metadata_tensor=meta, recv_counts_known=False, has_rdispls=False)
    nisa.core_barrier(recv, _LNC2_CORES)
    nisa.core_barrier(meta, _LNC2_CORES)
    out = nl.ndarray((n * slot_rows, width), dtype=dtype, buffer=nl.shared_hbm)
    _copy_rows(out, recv, n * slot_rows, width, dtype)
    meta_out = nl.ndarray((3, n), dtype=nl.uint32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=meta_u, src=meta)
    nisa.dma_copy(dst=meta_out, src=meta_u)
    return out, meta_out


@nki_op('deep_ep_trn::a2av_fixed_slot', mutates_args={})
def _a2av_op(src: torch.Tensor, meta: torch.Tensor, slot_rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    return _a2av_fixed_slot_kernel[(2,)](src, meta, slot_rows)


def _install_replica_group(groups: list[list[int]]) -> None:
    global _REPLICA_GROUP, _REPLICA_GROUP_LISTS
    if _REPLICA_GROUP_LISTS is not None and _REPLICA_GROUP_LISTS != groups:
        raise RuntimeError(f'replica groups already set to {_REPLICA_GROUP_LISTS}; '
                           f'one layout per process is supported, got {groups}')
    _REPLICA_GROUP, _REPLICA_GROUP_LISTS = ReplicaGroup(groups), groups


class NkiA2avTransport(Transport):
    """Fixed-slot all_to_all_v on Neuron devices.

    Args:
        replica_groups: every EP group in the world, e.g. ``[[0, 2], [1, 3]]``.
        global_rank: this process's rank in the default process group.
        device: neuron device; defaults to ``neuron:$LOCAL_RANK``.
        short_slots: request ``slot_rows < send capacity``. The trn2 driver rejects
            this today; kept so tools/probe_transport.py can re-check new runtimes.
    """

    def __init__(self, replica_groups, global_rank: int, device=None, short_slots=False):
        groups = [list(map(int, g)) for g in replica_groups]
        mine = [g for g in groups if global_rank in g]
        if len(mine) != 1:
            raise ValueError(f'rank {global_rank} must be in exactly one of {groups}')
        if int(os.environ.get('NEURON_LOGICAL_NC_CONFIG', '2')) != 2:
            raise RuntimeError('the a2av kernel is written for LNC=2')
        _install_replica_group(groups)
        self.rank = mine[0].index(global_rank)
        self.num_ranks = len(mine[0])
        self.device = device or torch.device('neuron', int(os.environ.get('LOCAL_RANK', 0)))
        self.short_slots = short_slots
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
        self._fn = torch.compile(_a2av_op, backend='neuron', fullgraph=True, dynamic=False)

    def slot_rows(self, send_capacity: int, max_rows_per_peer: int) -> int:
        return max_rows_per_peer if self.short_slots else send_capacity

    @staticmethod
    def _meta(counts: torch.Tensor, width: int) -> torch.Tensor:
        displs = torch.cumsum(counts, 0) - counts
        return torch.stack([counts * width, displs * width, torch.zeros_like(counts)]).to(torch.int32)

    def all_to_all_fixed_slot(self, send, counts, slot_rows):
        counts = counts.to(torch.int64).cpu()
        self._check(send, counts, slot_rows)
        width = send.shape[1]
        with timer.stage('h2d'):
            d_send, d_meta = send.to(self.device), self._meta(counts, width).to(self.device)
        with timer.stage('kernel'):
            recv, meta_out = self._fn(d_send, d_meta, slot_rows)
        with timer.stage('d2h'):
            recv, meta_out = recv.cpu(), meta_out.cpu()
        return recv, meta_out[2].to(torch.int64) // width

    def resident_call(self, capacity: int, width: int, counts: torch.Tensor, slot_rows: int,
                      dtype: torch.dtype = torch.bfloat16):
        """Benchmark hook: a zero-argument call running only the kernel on device-resident buffers."""
        counts = counts.to(torch.int64).cpu()
        d_send = torch.zeros(capacity, width, dtype=dtype).to(self.device)
        d_meta = self._meta(counts, width).to(self.device)
        return lambda: self._fn(d_send, d_meta, slot_rows)
