"""Transport contract shared by every backend.

Every dispatch/combine in deep_ep_trn reduces to one collective: each rank sends
rows packed contiguously by destination (``counts[d]`` rows for peer ``d``, in
peer order) and receives them into equally sized per-source slots. That is the
only all-to-all-v shape trn2 supports inside a node (``has_rdispls=False``).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Transport(ABC):
    rank: int       # position of this process inside its EP group
    num_ranks: int  # EP group size

    @abstractmethod
    def slot_rows(self, send_capacity: int, max_rows_per_peer: int) -> int:
        """Rows per source slot the backend will use for this buffer shape."""

    @abstractmethod
    def all_to_all_fixed_slot(self, send: torch.Tensor, counts: torch.Tensor,
                              slot_rows: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Exchange packed rows.

        Args:
            send: [capacity, W] 16-bit CPU tensor; the first ``counts.sum()`` rows
                are packed by destination rank.
            counts: [num_ranks] int64 rows destined for each peer.
            slot_rows: value returned by :meth:`slot_rows` for this shape. Must be
                identical on every rank of the group.

        Returns:
            recv: [num_ranks * slot_rows, W] CPU tensor; rows from source ``s`` are
                at ``[s * slot_rows, s * slot_rows + recv_counts[s])``.
            recv_counts: [num_ranks] int64 rows received from each source.
        """

    def _check(self, send: torch.Tensor, counts: torch.Tensor, slot_rows: int) -> None:
        if send.dim() != 2 or send.element_size() != 2:
            raise ValueError(f'send must be a 2-D 16-bit tensor, got {send.dtype} {tuple(send.shape)}')
        if counts.shape != (self.num_ranks,):
            raise ValueError(f'counts must be [{self.num_ranks}], got {tuple(counts.shape)}')
        if int(counts.sum()) > send.shape[0]:
            raise ValueError(f'{int(counts.sum())} packed rows exceed send capacity {send.shape[0]}')
        if counts.numel() and int(counts.max()) > slot_rows:
            # Receivers cannot detect this; the collective would overrun the slot.
            raise ValueError(f'{int(counts.max())} rows for one peer exceed slot_rows={slot_rows}')
