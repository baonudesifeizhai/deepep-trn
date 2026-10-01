"""Low-latency dispatch/combine vs an all-gather reference.

torchrun --standalone --nproc-per-node=4 tests/dist/test_low_latency.py --backend nki --ep 4
"""
from common import (bitwise_equal, expert_scale, gather_peers, init, make_buffer, make_routing,
                    parse_args, to_device)

import torch
import torch.distributed as dist


def main():
    args = parse_args(__doc__)
    rank, world, control = init(args.backend)
    buf, peers = make_buffer(args.backend, args.ep, args.max_tokens)
    R, E, K, H, M = args.ep, args.experts, args.topk, args.hidden, args.max_tokens
    EL, me = E // R, buf.rank
    lo = me * EL
    for i, kind in enumerate(args.kinds.split(',')):
        idx, w = make_routing(kind, args.tokens, E, K, R, me, seed=2000 + rank)
        T = idx.shape[0]
        x = torch.randn(T, H, generator=torch.Generator().manual_seed(rank), dtype=torch.bfloat16)
        hook_mode = bool(i % 2)
        recv_x, recv_count, handle, _, hook = buf.low_latency_dispatch(
            to_device(x, args.backend), to_device(idx, args.backend), M, E,
            use_fp8=False, return_recv_hook=hook_mode)
        if hook_mode:
            hook()
        recv_x, recv_count = recv_x.cpu(), recv_count.cpu()
        assert recv_x.shape == (EL, R * M, H) and recv_count.dtype == torch.int32

        peer_data = gather_peers((x, idx), peers, control)
        for e in range(EL):
            rows, toks, blocks = [], [], []
            for px, pidx in peer_data:
                t = (pidx == lo + e).any(1).nonzero().flatten()
                rows.append(px[t])
                toks.append(t)
                blocks.append(t.numel())
            rows, toks = torch.cat(rows), torch.cat(toks)
            n = int(recv_count[e])
            assert n == rows.shape[0], (kind, e, n, rows.shape[0])
            assert bitwise_equal(recv_x[e, :n], rows), (kind, 'recv_x', e)
            assert torch.equal(handle.src_info[e, :n].long(), toks), (kind, 'src_info', e)
            counts = (handle.layout_range[e] & 0xFFFFFFFF).tolist()
            starts = (handle.layout_range[e] >> 32).tolist()
            assert counts == blocks and starts == [sum(blocks[:s]) for s in range(R)], (kind, 'layout', e)

        # Emulated experts: each local expert scales its rows; the source applies weights.
        scale = expert_scale(torch.arange(EL) + lo)[:, None, None]
        y = (recv_x.float() * scale).to(torch.bfloat16)
        combined, _, hook = buf.low_latency_combine(
            to_device(y, args.backend), to_device(idx, args.backend), to_device(w, args.backend),
            handle, return_recv_hook=hook_mode)
        if hook_mode:
            hook()
        expert_out = (x.float()[:, None, :] * expert_scale(idx)[:, :, None]).to(torch.bfloat16).float()
        ref = (expert_out * (w * (idx >= 0))[:, :, None]).sum(1)
        diff = (combined.cpu().float() - ref).abs()
        rel = float(diff.mean() / ref.abs().mean().clamp_min(1e-9)) if T else 0.0
        assert torch.allclose(combined.cpu().float(), ref, rtol=1e-2, atol=1e-3) and rel < 5e-3, (kind, rel)
        print(f'RANK={rank} low_latency {kind}: T={T} recv_count_sum={int(recv_count.sum())} '
              f'combine_rel={rel:.2e} PASS', flush=True)
    dist.barrier(group=control)
    if rank == 0:
        print(f'LOW_LATENCY_PASS backend={args.backend} ep={args.ep}', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
