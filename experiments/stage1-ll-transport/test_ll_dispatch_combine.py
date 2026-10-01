"""DeepEP-LL-style dispatch/combine on one trn2 chip via NKI all_to_all_v.

Stage 1 of the DeepEP-on-trn2 validation: transport semantics only.

On non-NeuronSwitch instances (trn2), intra-node all_to_all_v only supports
has_rdispls=False, i.e. every source rank owns an equally sized receive slot.
That is the DeepEP low-latency buffer shape, so this probe implements:

  dispatch: each source packs one row per (token, destination rank) -- rank-level
            dedup as in DeepEP normal -- with routing metadata (top-k weights,
            global expert ids, source token index) bit-packed as extra columns;
  expert:   emulated on CPU on the receiving rank (y = x * sum_local w_k*s(e_k));
  combine:  the reduced row is returned to its source's slot and summed there.

Routing/packing and the expert run on CPU; only the two collectives run on the
Neuron device. Run with torchrun --nproc-per-node=4 on four logical cores.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import time
import traceback

parser = argparse.ArgumentParser()
parser.add_argument('--ep', type=int, choices=[2, 4], required=True)
parser.add_argument('--tokens', type=int, default=None,
                    help='tokens per EP rank (default: 64 // EP, i.e. global B64)')
parser.add_argument('--hidden', type=int, default=2048)
parser.add_argument('--experts', type=int, default=128)
parser.add_argument('--topk', type=int, default=8)
parser.add_argument('--bench', action='store_true')
parser.add_argument('--rounds', type=int, default=5)
parser.add_argument('--iters', type=int, default=100)
parser.add_argument('--output', required=True)
args = parser.parse_args()
outdir = Path(args.output)
outdir.mkdir(parents=True, exist_ok=True)
os.environ['NKI_ENABLE_TRACE_CACHE'] = '0'
os.environ['TORCH_NEURONX_NEFF_CACHE_DIR'] = str(outdir / 'neff_cache')
os.environ.setdefault('NEURON_PLATFORM_TARGET_OVERRIDE', 'trn2')

import torch
import torch.distributed as dist
import torch_neuronx
from torch.distributed._functional_collectives import all_reduce
from torch_neuronx import nki_op, wrap_nki
import nki
import nki.language as nl
import nki.isa as nisa
import nki.collectives as ncc
from nki.collectives import ReplicaGroup

EP = args.ep
T = args.tokens or 64 // EP
H, E, K = args.hidden, args.experts, args.topk
EL = E // EP
S_CAP = T * min(K, EP)  # dispatch send rows: at most one row per (token, dst rank)
META_COLS = 64          # >= 4K + 4 bf16 columns of packed routing metadata
W = H + META_COLS
GROUPS = [[0, 1, 2, 3]] if EP == 4 else [[0, 2], [1, 3]]
# NKI resolves globals, not Python closure cells, when tracing kernels.
EP_GROUP = ReplicaGroup(GROUPS)
TILE = 128
BF16, I32, F32 = torch.bfloat16, torch.int32, torch.float32
assert 4 * K + 4 <= META_COLS and E % EP == 0


def _copy_rows(dst, src, rows, width, dtype):
    for i in range((rows + TILE - 1) // TILE):
        lo, hi = i * TILE, min(i * TILE + TILE, rows)
        tile = nl.ndarray((hi - lo, width), dtype=dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=tile, src=src[lo:hi, :])
        nisa.dma_copy(dst=dst[lo:hi, :], src=tile)


@wrap_nki
@nki.jit
def a2av_fixed_slot(src, meta_in):
    """[R, W] rows + [3, EP] int32 element metadata -> [EP * R, W] slotted rows.

    Sender s's chunk lands at rows [s * R, s * R + count); row 2 of the
    returned metadata holds the per-sender receive counts in elements.
    """
    R, Wd = src.shape
    n = meta_in.shape[1]
    dtype = src.dtype
    # Collectives must use internal HBM staging, not graph IO tensors.
    send = nl.ndarray((R, Wd), dtype=dtype, buffer=nl.shared_hbm)
    _copy_rows(send, src, R, Wd, dtype)
    meta_sb = nl.ndarray((3, n), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=meta_sb, src=meta_in)
    meta_u = nl.ndarray((3, n), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=meta_u, src=meta_sb)
    meta = nl.ndarray((3, n), dtype=nl.uint32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=meta, src=meta_u)
    # has_rdispls=False: dst capacity must be EP * src capacity on this runtime.
    recv = nl.ndarray((n * R, Wd), dtype=dtype, buffer=nl.shared_hbm)
    nisa.core_barrier(send, (0, 1))
    nisa.core_barrier(meta, (0, 1))
    ncc.all_to_all_v(srcs=[send], dsts=[recv], replica_group=EP_GROUP,
                     metadata_tensor=meta, recv_counts_known=False, has_rdispls=False)
    nisa.core_barrier(recv, (0, 1))
    nisa.core_barrier(meta, (0, 1))
    out = nl.ndarray((n * R, Wd), dtype=dtype, buffer=nl.shared_hbm)
    _copy_rows(out, recv, n * R, Wd, dtype)
    meta_out = nl.ndarray((3, n), dtype=nl.uint32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=meta_u, src=meta)
    nisa.dma_copy(dst=meta_out, src=meta_u)
    return out, meta_out


@nki_op('deepep_trn2::a2av_fixed_slot', mutates_args={})
def a2av_op(src: torch.Tensor, meta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return a2av_fixed_slot[(2,)](src, meta)


def routes(kind, ep_rank):
    g = torch.Generator().manual_seed(20261001 + ep_rank)
    n_valid = T
    if kind == 'random':
        ids = torch.rand(T, E, generator=g).topk(K, dim=-1).indices
    elif kind == 'balanced':
        ids = (torch.arange(T)[:, None] * 4 + torch.arange(K)[None, :] * (E // K)) % E
    elif kind == 'hot_rank0':
        # Every token routes only to rank 0's experts: max skew, zero counts elsewhere.
        ids = (torch.arange(T)[:, None] + torch.arange(K)[None, :]) % EL
    elif kind == 'ragged':
        # Variable tokens per rank; the last rank is idle (sends nothing).
        ids = torch.rand(T, E, generator=g).topk(K, dim=-1).indices
        n_valid = 0 if ep_rank == EP - 1 else max(1, T - ep_rank * (T // EP))
    else:
        raise ValueError(kind)
    weights = torch.rand(T, K, generator=g)
    weights /= weights.sum(-1, keepdim=True)
    return ids.to(I32), weights, n_valid


def as_bf16_cols(t):
    return t.contiguous().view(BF16)


def pack_dispatch(x, ids, weights, n_valid, ep_rank):
    rows = torch.zeros(T, W, dtype=BF16)
    rows[:, :H] = x
    rows[:, H:H + 2 * K] = as_bf16_cols(weights.to(F32))
    rows[:, H + 2 * K:H + 4 * K] = as_bf16_cols(ids)
    rows[:, H + 4 * K:H + 4 * K + 2] = as_bf16_cols(torch.arange(T, dtype=I32)[:, None])
    rows[:, H + 4 * K + 2:H + 4 * K + 4] = as_bf16_cols(torch.full((T, 1), ep_rank, dtype=I32))
    dst_of = ids // EL
    lists = [[t for t in range(n_valid) if bool((dst_of[t] == d).any())] for d in range(EP)]
    send = torch.zeros(S_CAP, W, dtype=BF16)
    counts, displs, off = [], [], 0
    for d in range(EP):
        if lists[d]:
            send[off:off + len(lists[d])] = rows[torch.tensor(lists[d])]
        counts.append(len(lists[d]))
        displs.append(off)
        off += len(lists[d])
    meta = torch.tensor([[c * W for c in counts], [o * W for o in displs], [0] * EP], dtype=I32)
    return send, meta, lists


def expert_scale(ids):
    return 1.0 + (ids % 8).to(F32) / 8.0


def emulate_experts(recv, recv_rows, ep_rank):
    """Receiver side: reduce local experts per row, lay results out per sender slot."""
    out = torch.zeros(EP * S_CAP, H, dtype=BF16)
    for s in range(EP):
        n = recv_rows[s]
        if not n:
            continue
        blk = recv[s * S_CAP:s * S_CAP + n]
        w = blk[:, H:H + 2 * K].contiguous().view(F32)
        ids = blk[:, H + 2 * K:H + 4 * K].contiguous().view(I32)
        coef = (expert_scale(ids) * w * ((ids // EL) == ep_rank)).sum(-1, keepdim=True)
        out[s * S_CAP:s * S_CAP + n] = (blk[:, :H].float() * coef).to(BF16)
    meta = torch.tensor([[n * H for n in recv_rows], [s * S_CAP * H for s in range(EP)],
                         [0] * EP], dtype=I32)
    return out, meta


def errors(actual, ref):
    diff = (actual - ref).abs()
    rel = float(diff.mean() / ref.abs().mean().clamp_min(1e-9))
    return dict(max_abs=float(diff.max()), relative_mean_abs=rel,
                passed=bool(torch.allclose(actual, ref, rtol=2e-2, atol=2e-3) and rel < 1e-2))


def main():
    torch.set_num_threads(1)
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    assert world == 4, 'run on all four logical NeuronCores of the trn2 chip'
    dist.init_process_group('neuron')
    control = dist.new_group(ranks=list(range(world)), backend='gloo')
    ep_group = None
    for ranks in GROUPS:
        group = dist.new_group(ranks=ranks, backend='neuron')
        if rank in ranks:
            ep_group, peers = group, ranks
    ep_rank = peers.index(rank)
    device = torch.device(f'neuron:{int(os.environ["LOCAL_RANK"])}')
    fn = torch.compile(a2av_op, backend='neuron', fullgraph=True, dynamic=False)

    result = dict(rank=rank, ep_rank=ep_rank, ep=EP, groups=GROUPS, tokens_per_rank=T, H=H,
                  experts=E, local_experts=EL, top_k=K, send_capacity_rows=S_CAP, row_width=W,
                  layout='DeepEP-LL fixed slot per source (has_rdispls=False), rank-dedup rows',
                  versions={p: importlib.metadata.version(p) for p in
                            ('torch', 'torch-neuronx', 'neuronx-cc', 'nki')},
                  cases={})
    path = outdir / f'rank-{rank}-result.json'
    bench_inputs = None
    try:
        for kind in ('balanced', 'random', 'hot_rank0', 'ragged'):
            ids, weights, n_valid = routes(kind, ep_rank)
            x = torch.randn(T, H, generator=torch.Generator().manual_seed(rank), dtype=BF16)
            send, meta, lists = pack_dispatch(x, ids, weights, n_valid, ep_rank)
            gathered = [None] * world
            dist.all_gather_object(gathered, (rank, send, meta), group=control)
            peer_data = {r: (s, m) for r, s, m in gathered if r in peers}

            t0 = time.perf_counter()
            recv, rmeta = (v.cpu() for v in fn(send.to(device), meta.to(device)))
            dispatch_first_s = time.perf_counter() - t0
            recv_counts = rmeta[2].to(torch.int64)
            exp_counts = torch.tensor([int(peer_data[p][1][0, ep_rank]) for p in peers])
            assert torch.equal(recv_counts, exp_counts), (kind, 'dispatch counts', recv_counts, exp_counts)
            for s, p in enumerate(peers):
                psend, pmeta = peer_data[p]
                n, off = int(pmeta[0, ep_rank]) // W, int(pmeta[1, ep_rank]) // W
                assert torch.equal(recv[s * S_CAP:s * S_CAP + n].view(torch.int16),
                                   psend[off:off + n].view(torch.int16)), (kind, 'dispatch payload from', p)
            recv_rows = [int(c) // W for c in recv_counts]

            ret_send, ret_meta = emulate_experts(recv, recv_rows, ep_rank)
            t0 = time.perf_counter()
            ret, cmeta = (v.cpu() for v in fn(ret_send.to(device), ret_meta.to(device)))
            combine_first_s = time.perf_counter() - t0
            exp_ret = torch.tensor([len(lists[d]) * H for d in range(EP)])
            assert torch.equal(cmeta[2].to(torch.int64), exp_ret), (kind, 'combine counts', cmeta[2], exp_ret)
            slot = EP * S_CAP
            out = torch.zeros(T, H)
            for d in range(EP):
                if lists[d]:
                    out[torch.tensor(lists[d])] += ret[d * slot:d * slot + len(lists[d])].float()
            ref = x.float() * (expert_scale(ids) * weights).sum(-1, keepdim=True)
            ref[n_valid:] = 0
            err = errors(out, ref)
            result['cases'][kind] = dict(
                n_valid=n_valid, send_counts_rows=[len(l) for l in lists], recv_counts_rows=recv_rows,
                dispatch_payload='bitwise_equal', combine=err,
                first_call_s=dict(dispatch=dispatch_first_s, combine=combine_first_s))
            print(f'RANK={rank} CASE={kind} send={[len(l) for l in lists]} recv={recv_rows} '
                  f'combine_rel={err["relative_mean_abs"]:.2e} passed={err["passed"]}', flush=True)
            assert err['passed'], (kind, err)
            if kind == 'random':
                bench_inputs = (send, meta, ret_send, ret_meta)
        result['status'] = 'correctness_pass'

        if args.bench:
            def measure(call):
                for _ in range(20):
                    call()
                torch.neuron.synchronize()
                rounds = []
                for _ in range(args.rounds):
                    dist.barrier(group=control)
                    torch.neuron.synchronize()
                    start = time.perf_counter()
                    for _ in range(args.iters):
                        call()
                    torch.neuron.synchronize()
                    elapsed = (time.perf_counter() - start) * 1e6 / args.iters
                    times = [None] * world
                    dist.all_gather_object(times, elapsed, group=control)
                    rounds.append(times)
                critical = [max(r) for r in rounds]
                return dict(median_us=statistics.median(critical), min_us=min(critical),
                            max_us=max(critical), per_rank_rounds_us=rounds,
                            timing='amortized synchronized host-wall; max across ranks per round')

            d_in = tuple(v.to(device) for v in bench_inputs[:2])
            c_in = tuple(v.to(device) for v in bench_inputs[2:])
            ar_in = torch.randn(T * EP, H, dtype=BF16).to(device)
            ar_fn = torch.compile(lambda v: all_reduce(v, 'sum', ep_group), backend='neuron',
                                  fullgraph=True, dynamic=False)
            ar_fn(ar_in)
            result['bench'] = dict(
                case='random',
                dispatch_a2av=measure(lambda: fn(*d_in)),
                combine_a2av=measure(lambda: fn(*c_in)),
                baseline_allreduce_ep_group=dict(shape=[T * EP, H], **measure(lambda: ar_fn(ar_in))))
            if rank == 0:
                print('BENCH ' + json.dumps({k: v['median_us'] for k, v in result['bench'].items()
                                             if isinstance(v, dict)}), flush=True)
    except Exception as exc:
        result.update(status='failed', error_type=type(exc).__name__, error=str(exc)[-4000:])
        traceback.print_exc()
        raise
    finally:
        path.write_text(json.dumps(result, indent=2) + '\n')
    dist.barrier(group=control)
    if rank == 0:
        print('LL_DISPATCH_COMBINE_PASS', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
