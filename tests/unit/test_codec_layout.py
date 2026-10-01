"""CPU-only unit tests: python -m pytest tests/unit"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
import torch
import torch.distributed as dist

from deep_ep_trn import Buffer, RowCodec, get_dispatch_layout
from deep_ep_trn.layout import pair_plan, rank_of_expert, send_plan, slot_index


def test_codec_roundtrip_is_bit_exact():
    g = torch.Generator().manual_seed(0)
    T, H, K = 37, 256, 8
    x = torch.randn(T, H, generator=g).to(torch.bfloat16)
    idx = torch.randint(-1, 128, (T, K), generator=g)
    w = torch.randn(T, K, generator=g)
    w[0, 0], w[1, 1] = float('inf'), -0.0
    token = torch.arange(T) * 1_000_003
    codec = RowCodec(H, K)
    assert codec.width % 32 == 0 and codec.width >= H + 4 * K + 2
    rx, ridx, rw, rtok = codec.decode(codec.encode(x, idx, w, token))
    assert torch.equal(rx.view(torch.int16), x.view(torch.int16))
    assert torch.equal(ridx, idx) and torch.equal(rtok, token)
    assert torch.equal(rw.view(torch.int32), w.view(torch.int32))


def test_layout_matches_naive():
    g = torch.Generator().manual_seed(1)
    T, E, K, R = 50, 64, 6, 4
    idx = torch.rand(T, E, generator=g).topk(K, dim=-1).indices
    idx[::7, 2] = -1
    per_rank, per_expert, in_rank = get_dispatch_layout(idx, E, R)
    naive = torch.zeros(T, R, dtype=torch.bool)
    for t in range(T):
        for e in idx[t].tolist():
            if e >= 0:
                naive[t, e // (E // R)] = True
    assert torch.equal(in_rank, naive)
    assert torch.equal(per_rank.long(), naive.sum(0))
    assert torch.equal(per_expert.long(), torch.bincount(idx[idx >= 0], minlength=E))
    token, counts = send_plan(in_rank)
    expected = torch.cat([naive[:, r].nonzero().flatten() for r in range(R)])
    assert torch.equal(token, expected) and torch.equal(counts, naive.sum(0))
    tok, kk, pcounts = pair_plan(rank_of_expert(idx, E, R), R)
    assert int(pcounts.sum()) == int((idx >= 0).sum())
    assert torch.equal(rank_of_expert(idx[tok, kk], E, R), torch.repeat_interleave(torch.arange(R), pcounts))


def test_slot_index():
    counts = torch.tensor([2, 0, 3])
    assert slot_index(counts, 4).tolist() == [0, 1, 8, 9, 10]


@pytest.fixture(scope='module')
def single_rank_group(tmp_path_factory):
    path = tmp_path_factory.mktemp('pg') / 'store'
    dist.init_process_group('gloo', init_method=f'file://{path}', rank=0, world_size=1)
    yield dist.group.WORLD
    dist.destroy_process_group()


def test_single_rank_roundtrip(single_rank_group):
    g = torch.Generator().manual_seed(2)
    T, H, E, K, M = 20, 128, 16, 4, 32
    x = torch.randn(T, H, generator=g).to(torch.bfloat16)
    idx = torch.rand(T, E, generator=g).topk(K, dim=-1).indices
    w = torch.rand(T, K, generator=g)
    buf = Buffer(single_rank_group, backend='gloo', num_max_tokens_per_rank=M)
    ntr, _, nte, itr, _ = buf.get_dispatch_layout(idx, E)
    rx, ridx, rw, per_expert, handle, _ = buf.dispatch(
        x, topk_idx=idx, topk_weights=w, num_tokens_per_rank=ntr, is_token_in_rank=itr,
        num_tokens_per_expert=nte)
    assert torch.equal(rx, x) and torch.equal(ridx, idx) and torch.equal(rw, w)
    assert per_expert == nte.tolist()
    combined, cw, _ = buf.combine(rx, handle, topk_weights=rw)
    assert torch.equal(combined, x) and torch.equal(cw, w)

    recv_x, count, ll_handle, _, _ = buf.low_latency_dispatch(x, idx, M, E, use_fp8=False)
    assert recv_x.shape == (E, M, H) and int(count.sum()) == T * K
    out, _, _ = buf.low_latency_combine(recv_x, idx, w, ll_handle)
    ref = (x.float() * w.sum(-1, keepdim=True)).to(torch.bfloat16)
    assert torch.allclose(out.float(), ref.float(), rtol=1e-2, atol=1e-3)

    with pytest.raises(NotImplementedError):
        buf.low_latency_dispatch(x, idx, M, E)  # deep_ep defaults to FP8
    with pytest.raises(ValueError):
        buf.dispatch(torch.zeros(M + 1, H, dtype=torch.bfloat16), topk_idx=idx.repeat(2, 1)[:M + 1],
                     num_tokens_per_expert=nte)


def test_testing_helpers():
    from deep_ep_trn.testing import ROUTINGS, make_routing, strided_ep_groups
    assert strided_ep_groups(4, 2) == [[0, 2], [1, 3]] and strided_ep_groups(4, 4) == [[0, 1, 2, 3]]
    T, E, K, R = 24, 128, 8, 4
    for kind in ROUTINGS:
        for ep_rank in range(R):
            idx, w = make_routing(kind, T, E, K, R, ep_rank, seed=ep_rank)
            n = idx.shape[0]
            assert idx.shape == (n, K) and w.shape == (n, K) and n <= T
            assert not n or (int(idx.max()) < E and int(idx.min()) >= -1)
            if kind not in ('masked', 'ragged'):
                assert n == T and (idx >= 0).all()
            if kind == 'hot_rank0':
                assert (idx < E // R).all()
    assert make_routing('ragged', T, E, K, R, R - 1, 0)[0].shape[0] == 0
