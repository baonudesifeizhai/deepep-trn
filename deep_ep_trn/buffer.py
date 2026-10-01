"""DeepEP-compatible ``Buffer`` for AWS Trainium (single node).

Mirrors the deep_ep Python API so MoE layers written against DeepEP can switch
backends. Every exchange is one fixed-slot all-to-all-v (see transport/base.py).

Status (v0): routing, packing and unpacking run in torch on the host; only the
collectives run on the Neuron device. Moving them into NKI kernels is the next
step (docs/roadmap.md). FP8 payloads, inter-node groups and async overlap are
not implemented yet and raise ``NotImplementedError``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.distributed as dist

from .codec import META_ALIGN, RowCodec
from .profiling import timer
from .layout import (exclusive_cumsum, get_dispatch_layout, pair_plan, rank_of_expert,
                     send_plan, slot_index)
from .transport import GlooTransport, NkiA2avTransport, Transport


class EventOverlap:
    """API parity only: Neuron work is ordered on the device queue, so there is nothing to wait on."""

    def __init__(self, event=None, extra_tensors=None):
        self.event = event
        self.extra_tensors = extra_tensors

    def current_stream_wait(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        pass


@dataclass
class Config:
    """Placeholder for deep_ep.Config; SM/chunk tuning has no trn counterpart yet."""
    num_sms: int = 0


@dataclass
class DispatchHandle:
    num_tokens: int
    topk: int
    send_token_idx: torch.Tensor    # [N_send] source token of each sent row, destination-major
    send_counts: torch.Tensor       # [R] rows sent to each rank
    recv_counts: torch.Tensor       # [R] rows received from each rank
    recv_src_idx: torch.Tensor      # [N_recv] source-local token index of each received row
    is_token_in_rank: torch.Tensor  # [T, R]


@dataclass
class LowLatencyHandle:
    num_tokens: int
    num_max_dispatch_tokens_per_rank: int
    hidden: int
    num_experts: int
    topk: int
    src_info: torch.Tensor       # [EL, R*M] int32 source token of each packed row, -1 = empty
    layout_range: torch.Tensor   # [EL, R] int64 (start << 32) | count of each source's block
    pair_expert: torch.Tensor    # [P] local expert of each received (row, k) pair, (src, row, k) order
    pair_pos: torch.Tensor       # [P] position of that pair in recv_x[expert]
    pairs_per_src: torch.Tensor  # [R]


def _no_hook() -> None:
    pass


class Buffer:
    """Expert-parallel communication buffer.

    Args:
        group: EP process group. With ``backend='gloo'`` it must support gloo.
        num_nvl_bytes, num_rdma_bytes, low_latency_mode, **deepep_kwargs: accepted
            for deep_ep source compatibility and ignored; trn buffers are static
            compiled tensors sized by ``num_max_tokens_per_rank``.
        num_max_tokens_per_rank: static token capacity per rank for normal-mode
            dispatch. Every rank must pass the same value.
        backend: ``'nki'`` (trn2 device) or ``'gloo'`` (CPU reference).
        replica_groups: every EP group in the world for the NKI backend, e.g.
            ``[[0, 2], [1, 3]]``. Defaults to ``[ranks of group]`` when the group
            spans the whole world.
        transport: use this transport instead of building one from ``backend``.
    """

    def __init__(self, group: dist.ProcessGroup, num_nvl_bytes: int = 0, num_rdma_bytes: int = 0,
                 low_latency_mode: bool = False, *, num_max_tokens_per_rank: int = 128,
                 backend: str = 'nki', replica_groups=None, transport: Optional[Transport] = None,
                 **deepep_kwargs):
        self.group = group
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        if transport is None:
            if backend == 'gloo':
                transport = GlooTransport(group)
            elif backend == 'nki':
                ranks = dist.get_process_group_ranks(group)
                if replica_groups is None:
                    if len(ranks) != dist.get_world_size():
                        raise ValueError('pass replica_groups when the EP group is a subgroup')
                    replica_groups = [ranks]
                transport = NkiA2avTransport(replica_groups, dist.get_rank())
            else:
                raise ValueError(f'unknown backend {backend!r}')
        if transport.num_ranks != dist.get_world_size(group):
            raise ValueError(f'transport has {transport.num_ranks} ranks, group has {dist.get_world_size(group)}')
        self.transport = transport
        self.rank = transport.rank
        self.group_size = transport.num_ranks
        # Shape of the most recent exchange; lets benchmarks replay the bare kernel.
        self.last_exchange: Optional[dict] = None

    # ------------------------------------------------------------------ deep_ep statics
    @staticmethod
    def set_num_sms(new_num_sms: int) -> None:
        pass

    @staticmethod
    def get_dispatch_config(num_ranks: int) -> Config:
        return Config()

    @staticmethod
    def get_combine_config(num_ranks: int) -> Config:
        return Config()

    @staticmethod
    def get_low_latency_rdma_size_hint(num_max_dispatch_tokens_per_rank: int, hidden: int,
                                       num_ranks: int, num_experts: int) -> int:
        return 0

    def destroy(self) -> None:
        pass

    # ------------------------------------------------------------------ helpers
    def _exchange(self, rows: torch.Tensor, order: Optional[torch.Tensor], counts: torch.Tensor,
                  capacity: int, max_rows_per_peer: int):
        """Pack ``rows[order]`` into a [capacity, W] buffer, exchange, return valid rows + metadata."""
        n = int(counts.sum())
        with timer.stage('pack'):
            send = torch.zeros(capacity, rows.shape[1], dtype=rows.dtype)
            send[:n] = rows if order is None else rows[order]
        slot = self.transport.slot_rows(capacity, max_rows_per_peer)
        self.last_exchange = dict(capacity=capacity, width=rows.shape[1], counts=counts.clone(),
                                  slot_rows=slot, dtype=rows.dtype)
        recv, recv_counts = self.transport.all_to_all_fixed_slot(send, counts, slot)
        with timer.stage('unpack'):
            flat = slot_index(recv_counts, slot)
            return recv[flat], recv_counts, flat // slot

    def _check_tokens(self, num_tokens: int, limit: int) -> None:
        if num_tokens > limit:
            raise ValueError(f'{num_tokens} tokens exceed the per-rank capacity {limit}')

    # ------------------------------------------------------------------ normal mode
    def get_dispatch_layout(self, topk_idx: torch.Tensor, num_experts: int, previous_event=None,
                            async_finish: bool = False, allocate_on_comm_stream: bool = False):
        per_rank, per_expert, in_rank = get_dispatch_layout(topk_idx.cpu(), num_experts, self.group_size)
        device = topk_idx.device
        return per_rank.to(device), None, per_expert.to(device), in_rank.to(device), EventOverlap()

    def dispatch(self, x, handle: Optional[DispatchHandle] = None, num_tokens_per_rank=None,
                 num_tokens_per_rdma_rank=None, is_token_in_rank=None, num_tokens_per_expert=None,
                 topk_idx=None, topk_weights=None, expert_alignment: int = 1, num_worst_tokens: int = 0,
                 config=None, previous_event=None, async_finish: bool = False,
                 allocate_on_comm_stream: bool = False):
        """Send each token once to every rank that owns one of its experts.

        Received rows are ordered by source rank, then source token index.
        Returns (recv_x, recv_topk_idx, recv_topk_weights,
        num_recv_tokens_per_expert_list, handle, event); the topk outputs are
        None for a cached dispatch (``handle`` given).
        """
        if isinstance(x, tuple):
            raise NotImplementedError('FP8 (x, scales) dispatch is not implemented on trn yet')
        if num_worst_tokens:
            raise NotImplementedError('num_worst_tokens is not supported')
        device = x.device
        cached = handle is not None
        if not cached and (topk_idx is None or num_tokens_per_expert is None):
            raise ValueError('dispatch needs topk_idx and num_tokens_per_expert (see get_dispatch_layout)')
        with timer.stage('in_d2h'):
            x_h = x.detach().cpu()
            if not cached:
                topk_h = topk_idx.cpu().to(torch.int64)
                weights_h = topk_weights.cpu() if topk_weights is not None else None
                in_rank = is_token_in_rank.cpu() if is_token_in_rank is not None else None
        T, H = x_h.shape
        R, M = self.group_size, self.num_max_tokens_per_rank
        self._check_tokens(T, M)
        with timer.stage('plan'):
            if cached:
                K, in_rank, topk_h, weights_h = handle.topk, handle.is_token_in_rank, None, None
            else:
                num_experts = num_tokens_per_expert.numel()
                K = topk_h.shape[1]
                if in_rank is None:
                    in_rank = get_dispatch_layout(topk_h, num_experts, R)[2]
            token, counts = send_plan(in_rank)
        codec = RowCodec(H, K, x_h.dtype)
        with timer.stage('pack'):
            rows = codec.encode(x_h, topk_h, weights_h, torch.arange(T))
        recv_rows, recv_counts, _ = self._exchange(rows, token, counts, M * min(K, R), M)
        with timer.stage('unpack'):
            recv_x, recv_idx, recv_w, recv_src = codec.decode(recv_rows)
            new_handle = DispatchHandle(T, K, token, counts, recv_counts, recv_src, in_rank)
            if not cached:
                EL = num_experts // R
                lo = self.rank * EL
                local = (recv_idx >= lo) & (recv_idx < lo + EL)
                recv_topk_idx = torch.where(local, recv_idx - lo, torch.full_like(recv_idx, -1))
                recv_topk_weights = torch.where(local, recv_w, torch.zeros_like(recv_w))
                per_expert = torch.bincount(recv_topk_idx[local], minlength=EL)
                if expert_alignment > 1:
                    per_expert = (per_expert + expert_alignment - 1) // expert_alignment * expert_alignment
        with timer.stage('out_h2d'):
            recv_x = recv_x.to(device)
            if cached:
                return recv_x, None, None, None, new_handle, EventOverlap()
            out_w = recv_topk_weights.to(topk_weights.dtype).to(device) if topk_weights is not None else None
            return (recv_x, recv_topk_idx.to(topk_idx.dtype).to(device), out_w, per_expert.tolist(),
                    new_handle, EventOverlap())

    def combine(self, x, handle: DispatchHandle, topk_weights=None, bias=None, config=None,
                previous_event=None, async_finish: bool = False, allocate_on_comm_stream: bool = False):
        """Return every received row to its source and sum per token.

        Returns (combined_x, combined_topk_weights, event).
        """
        if bias is not None:
            raise NotImplementedError('combine bias is not supported')
        device = x.device
        with timer.stage('in_d2h'):
            x_h = x.detach().cpu()
            w_h = topk_weights.detach().cpu() if topk_weights is not None else None
        N, H = x_h.shape
        if N != int(handle.recv_counts.sum()):
            raise ValueError(f'combine got {N} rows, dispatch delivered {int(handle.recv_counts.sum())}')
        R, M, T = self.group_size, self.num_max_tokens_per_rank, handle.num_tokens
        K = topk_weights.shape[1] if topk_weights is not None else 0
        extra = (2 * K + META_ALIGN - 1) // META_ALIGN * META_ALIGN
        with timer.stage('pack'):
            rows = torch.zeros(N, H + extra, dtype=x_h.dtype)
            rows[:, :H] = x_h
            if K:
                rows[:, H:H + 2 * K] = w_h.to(torch.float32).contiguous().view(x_h.dtype)
        back, back_counts, _ = self._exchange(rows, None, handle.recv_counts, R * M, M)
        if not torch.equal(back_counts, handle.send_counts):
            raise RuntimeError(f'combine returned {back_counts.tolist()} rows, expected {handle.send_counts.tolist()}')
        with timer.stage('unpack'):
            combined = torch.zeros(T, H).index_add_(0, handle.send_token_idx, back[:, :H].float())
            combined = combined.to(x_h.dtype)
            combined_w = None
            if K:
                w = back[:, H:H + 2 * K].contiguous().view(torch.float32)
                combined_w = torch.zeros(T, K).index_add_(0, handle.send_token_idx, w).to(topk_weights.dtype)
        with timer.stage('out_h2d'):
            return (combined.to(device), combined_w.to(device) if combined_w is not None else None,
                    EventOverlap())

    # ------------------------------------------------------------------ low-latency mode
    def low_latency_dispatch(self, x, topk_idx, num_max_dispatch_tokens_per_rank: int, num_experts: int,
                             cumulative_local_expert_recv_stats=None, dispatch_wait_recv_cost_stats=None,
                             use_fp8: bool = True, round_scale: bool = False, use_ue8m0: bool = False,
                             async_finish: bool = False, return_recv_hook: bool = False):
        """Dispatch into the DeepEP-LL expert-major layout.

        Wire format is rank-deduplicated (one row per token and destination rank,
        carrying its top-k ids); the receiver expands rows per local expert.
        Returns (recv_x [EL, R*M, H], recv_count [EL] int32, handle, event, hook).
        """
        if use_fp8:
            raise NotImplementedError('FP8 low-latency dispatch is not implemented on trn yet; pass use_fp8=False')
        device = x.device
        with timer.stage('in_d2h'):
            x_h, topk_h = x.detach().cpu(), topk_idx.cpu().to(torch.int64)
        T, H = x_h.shape
        R, M = self.group_size, num_max_dispatch_tokens_per_rank
        self._check_tokens(T, M)
        EL = num_experts // R
        lo = self.rank * EL
        K = topk_h.shape[1]
        with timer.stage('plan'):
            owner = rank_of_expert(topk_h, num_experts, R)
            in_rank = (owner[:, :, None] == torch.arange(R)[None, None, :]).any(1)
            token, counts = send_plan(in_rank)
        codec = RowCodec(H, K, x_h.dtype)
        with timer.stage('pack'):
            rows = codec.encode(x_h, topk_h, None, torch.arange(T))
        recv_rows, _, src_rank = self._exchange(rows, token, counts, M * min(K, R), M)
        with timer.stage('unpack'):
            out = self._ll_expand(recv_rows, src_rank, codec, T, M, H, num_experts, K, EL, lo)
        recv_x, per_expert, handle = out
        hook = _no_hook if return_recv_hook else None
        with timer.stage('out_h2d'):
            return recv_x.to(device), per_expert.to(torch.int32).to(device), handle, EventOverlap(), hook

    def _ll_expand(self, recv_rows, src_rank, codec, T, M, H, num_experts, K, EL, lo):
        """Expand rank-deduplicated rows into the [EL, R*M, H] expert-major layout."""
        R = self.group_size
        xr, ridx, _, rtok = codec.decode(recv_rows)
        row, k = ((ridx >= lo) & (ridx < lo + EL)).nonzero(as_tuple=True)  # (src, row, k) order
        expert = ridx[row, k] - lo
        P = row.numel()
        order = torch.argsort(expert * max(1, recv_rows.shape[0] * K) + row * K + k)
        per_expert = torch.bincount(expert, minlength=EL)
        pos = torch.empty(P, dtype=torch.int64)
        pos[order] = torch.arange(P) - exclusive_cumsum(per_expert)[expert[order]]

        recv_x = torch.zeros(EL, R * M, H, dtype=xr.dtype)
        recv_x[expert, pos] = xr[row]
        src_info = torch.full((EL, R * M), -1, dtype=torch.int32)
        src_info[expert, pos] = rtok[row].to(torch.int32)
        pair_src = src_rank[row]
        block = torch.zeros(EL, R, dtype=torch.int64).index_put_(
            (expert, pair_src), torch.ones(P, dtype=torch.int64), accumulate=True)
        start = torch.cumsum(block, 1) - block
        handle = LowLatencyHandle(T, M, H, num_experts, K, src_info, (start << 32) | block,
                                  expert, pos, torch.bincount(pair_src, minlength=R))
        return recv_x, per_expert, handle

    def low_latency_combine(self, x, topk_idx, topk_weights, handle: LowLatencyHandle,
                            use_logfmt: bool = False, zero_copy: bool = False, async_finish: bool = False,
                            return_recv_hook: bool = False, out: Optional[torch.Tensor] = None):
        """Return expert outputs per (token, expert) pair; the source applies top-k weights.

        Returns (combined_x [T, H], event, hook).
        """
        if use_logfmt:
            raise NotImplementedError('LogFMT combine is not implemented on trn yet')
        device = x.device
        with timer.stage('in_d2h'):
            x_h = x.detach().cpu()
            topk_h = topk_idx.cpu().to(torch.int64)
            w_h = topk_weights.detach().cpu().to(torch.float32)
        R, M, K = self.group_size, handle.num_max_dispatch_tokens_per_rank, handle.topk
        EL = handle.num_experts // R
        with timer.stage('pack'):
            y = x_h[handle.pair_expert, handle.pair_pos]
        per_peer = M * min(K, EL)
        back, back_counts, _ = self._exchange(y, None, handle.pairs_per_src, R * per_peer, per_peer)
        with timer.stage('plan'):
            owner = rank_of_expert(topk_h, handle.num_experts, R)
            tok, kk, expected = pair_plan(owner, R)
        if not torch.equal(back_counts, expected):
            raise RuntimeError(f'combine returned {back_counts.tolist()} pairs, expected {expected.tolist()}')
        with timer.stage('unpack'):
            w = w_h[tok, kk]
            combined = torch.zeros(handle.num_tokens, handle.hidden).index_add_(0, tok, back.float() * w[:, None])
            combined = combined.to(x_h.dtype)
        with timer.stage('out_h2d'):
            combined = combined.to(device)
            if out is not None:
                out.copy_(combined)
                combined = out
        hook = _no_hook if return_recv_hook else None
        return combined, EventOverlap(), hook
