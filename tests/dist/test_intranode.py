"""Normal-mode dispatch/combine vs an all-gather reference.

torchrun --standalone --nproc-per-node=4 tests/dist/test_intranode.py --backend nki --ep 4
"""
from common import (bitwise_equal, expert_scale, gather_peers, init, make_buffer, make_routing,
                    parse_args, to_device)

import torch
import torch.distributed as dist

from deep_ep_trn.layout import rank_of_expert


def naive_layout(idx, experts, ep):
    local = experts // ep
    in_rank = torch.zeros(idx.shape[0], ep, dtype=torch.bool)
    per_expert = torch.zeros(experts, dtype=torch.int64)
    for t in range(idx.shape[0]):
        for e in idx[t].tolist():
            if e >= 0:
                in_rank[t, e // local] = True
                per_expert[e] += 1
    return in_rank.sum(0), per_expert, in_rank


def main():
    args = parse_args(__doc__)
    rank, world, control = init(args.backend)
    buf, peers = make_buffer(args.backend, args.ep, args.max_tokens)
    R, E, K, H = args.ep, args.experts, args.topk, args.hidden
    EL, me = E // R, buf.rank
    lo = me * EL
    for kind in args.kinds.split(','):
        idx, w = make_routing(kind, args.tokens, E, K, R, me, seed=1000 + rank)
        T = idx.shape[0]
        x = torch.randn(T, H, generator=torch.Generator().manual_seed(rank), dtype=torch.bfloat16)
        d_x, d_idx, d_w = (to_device(v, args.backend) for v in (x, idx, w))

        ntr, _, nte, itr, _ = buf.get_dispatch_layout(d_idx, E)
        ref_ntr, ref_nte, ref_itr = naive_layout(idx, E, R)
        assert torch.equal(ntr.cpu().long(), ref_ntr) and torch.equal(nte.cpu().long(), ref_nte)
        assert torch.equal(itr.cpu(), ref_itr), kind

        recv_x, recv_idx, recv_w, per_expert, handle, _ = buf.dispatch(
            d_x, topk_idx=d_idx, topk_weights=d_w, num_tokens_per_rank=ntr,
            is_token_in_rank=itr, num_tokens_per_expert=nte)
        recv_x, recv_idx, recv_w = recv_x.cpu(), recv_idx.cpu(), recv_w.cpu()

        # Reference: rows from each source rank, in source token order.
        exp_x, exp_idx, exp_w = [], [], []
        for px, pidx, pw in gather_peers((x, idx, w), peers, control):
            owner = rank_of_expert(pidx, E, R)
            rows = (owner == me).any(1).nonzero().flatten()
            local = owner[rows] == me
            exp_x.append(px[rows])
            exp_idx.append(torch.where(local, pidx[rows] - lo, torch.full_like(pidx[rows], -1)))
            exp_w.append(torch.where(local, pw[rows], torch.zeros_like(pw[rows])))
        exp_x, exp_idx, exp_w = torch.cat(exp_x), torch.cat(exp_idx), torch.cat(exp_w)
        assert bitwise_equal(recv_x, exp_x), (kind, 'recv_x')
        assert torch.equal(recv_idx, exp_idx), (kind, 'recv_topk_idx')
        assert torch.equal(recv_w, exp_w), (kind, 'recv_topk_weights')
        exp_per_expert = torch.bincount(exp_idx[exp_idx >= 0], minlength=EL).tolist()
        assert per_expert == exp_per_expert, (kind, per_expert, exp_per_expert)

        cached_x, *_ = buf.dispatch(d_x, handle=handle)
        assert bitwise_equal(cached_x.cpu(), recv_x), (kind, 'cached dispatch')

        # Identity combine: every token comes back once per rank it visited.
        combined, combined_w, _ = buf.combine(to_device(recv_x, args.backend), handle,
                                              topk_weights=to_device(recv_w, args.backend))
        ref = (x.float() * ref_itr.sum(1, keepdim=True)).to(torch.bfloat16)
        assert torch.equal(combined.cpu(), ref), (kind, 'identity combine')  # value-equal: -0 == +0
        assert torch.equal(combined_w.cpu(), torch.where(idx >= 0, w, torch.zeros_like(w))), (kind, 'weights')

        # Emulated experts: receiver reduces its local experts, combine sums across ranks.
        coef = (expert_scale(recv_idx + lo) * recv_w * (recv_idx >= 0)).sum(-1, keepdim=True)
        y = (recv_x.float() * coef).to(torch.bfloat16)
        combined, _, _ = buf.combine(to_device(y, args.backend), handle)
        ref = x.float() * (expert_scale(idx) * w * (idx >= 0)).sum(-1, keepdim=True)
        diff = (combined.cpu().float() - ref).abs()
        rel = float(diff.mean() / ref.abs().mean().clamp_min(1e-9)) if T else 0.0
        assert torch.allclose(combined.cpu().float(), ref, rtol=2e-2, atol=2e-3) and rel < 1e-2, (kind, rel)
        print(f'RANK={rank} intranode {kind}: T={T} sent={handle.send_counts.tolist()} '
              f'recv={handle.recv_counts.tolist()} expert_combine_rel={rel:.2e} PASS', flush=True)
    dist.barrier(group=control)
    if rank == 0:
        print(f'INTRANODE_PASS backend={args.backend} ep={args.ep}', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
