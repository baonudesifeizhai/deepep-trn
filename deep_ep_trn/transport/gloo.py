"""CPU reference transport over a gloo process group.

Same fixed-slot contract as the NKI backend, so dispatch/combine semantics can be
tested without a Neuron device. Not intended to be fast.
"""
from __future__ import annotations

import torch
import torch.distributed as dist

from ..profiling import timer
from .base import Transport


class GlooTransport(Transport):
    def __init__(self, group: dist.ProcessGroup):
        self.group = group
        self.rank = dist.get_rank(group)
        self.num_ranks = dist.get_world_size(group)

    def slot_rows(self, send_capacity: int, max_rows_per_peer: int) -> int:
        return max_rows_per_peer

    def all_to_all_fixed_slot(self, send, counts, slot_rows):
        counts = counts.to(torch.int64).cpu()
        self._check(send, counts, slot_rows)
        with timer.stage('kernel'):
            return self._exchange(send, counts, slot_rows)

    def _exchange(self, send, counts, slot_rows):
        recv_counts = torch.empty_like(counts)
        dist.all_to_all_single(recv_counts, counts, group=self.group)
        width = send.shape[1]
        # gloo has no bf16 all-to-all; move raw bytes.
        payload = send[:int(counts.sum())].contiguous().view(torch.uint8)
        packed = torch.empty(int(recv_counts.sum()), 2 * width, dtype=torch.uint8)
        dist.all_to_all_single(packed, payload, output_split_sizes=recv_counts.tolist(),
                               input_split_sizes=counts.tolist(), group=self.group)
        packed = packed.view(send.dtype)
        recv = torch.zeros(self.num_ranks * slot_rows, width, dtype=send.dtype)
        offset = 0
        for src, n in enumerate(recv_counts.tolist()):
            recv[src * slot_rows:src * slot_rows + n] = packed[offset:offset + n]
            offset += n
        return recv, recv_counts
