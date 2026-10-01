"""DeepEP-style op benchmark: normal (dispatch/combine) and low-latency modes.

For every (mode, preset, tokens, routing) case:
  1. correctness check with identity experts (a fast wrong answer is worthless);
  2. first-call time (includes NEFF compile when the cache is cold);
  3. end-to-end latency of dispatch, combine and dispatch+combine;
  4. per-stage breakdown (in_d2h / plan / pack / h2d / kernel / d2h / unpack / out_h2d);
  5. device-resident kernel latency replaying the exact exchange shapes;
  6. bytes: DeepEP-definition logical bytes, actual wire bytes, receive-buffer bytes.

torchrun --standalone --nproc-per-node=4 bench/bench_ops.py --ep 4 --modes normal,low_latency
"""
from common import ROUTINGS, PRESETS, Ctx, append_jsonl, base_parser, capture_env, csv_list, gbps, make_routing

import torch

from deep_ep_trn.profiling import timer

BF16 = 2  # bytes


def stats(values):
    return dict(min=min(values), max=max(values), mean=round(sum(values) / len(values), 1))


def breakdown(ctx, call, calls: int) -> dict:
    timer.enabled, timer.sync = True, ctx.sync
    timer.reset()
    for _ in range(calls):
        call()
    mine = timer.per_call_us(calls)
    timer.enabled, timer.sync = False, None
    timer.reset()
    everyone = ctx.gather(mine)
    stages = sorted({k for d in everyone for k in d})
    return {s: round(max(d.get(s, 0.0) for d in everyone), 1) for s in stages}


def resident(ctx, ex, args, sync_each=False):
    if ctx.backend != 'nki':
        return None
    call = ctx.transport.resident_call(ex['capacity'], ex['width'], ex['counts'], ex['slot_rows'], ex['dtype'])
    return ctx.measure(call, args.rounds, args.max_iters, args.target_round_s, sync_each=sync_each)


def bench_case(ctx, args, mode, preset, T, routing):
    H, E, K = PRESETS[preset]
    R = ctx.ep
    seed = 7919 * T + 31 * ctx.rank + ROUTINGS.index(routing)
    idx, w = make_routing(routing, T, E, K, R, ctx.ep_rank, seed)
    x = torch.randn(T, H, generator=torch.Generator().manual_seed(seed), dtype=torch.bfloat16)
    d_x, d_idx, d_w = ctx.to_dev(x), ctx.to_dev(idx), ctx.to_dev(w)
    buf = ctx.buffer(T)
    measure = lambda call: ctx.measure(call, args.rounds, args.max_iters, args.target_round_s)

    if mode == 'normal':
        ntr, _, nte, itr, _ = buf.get_dispatch_layout(d_idx, E)

        def dispatch():
            return buf.dispatch(d_x, topk_idx=d_idx, topk_weights=d_w, num_tokens_per_rank=ntr,
                                is_token_in_rank=itr, num_tokens_per_expert=nte)

        out, first_d = ctx.first_call(dispatch)
        recv_x, handle = out[0], out[4]
        ex_d = dict(buf.last_exchange)

        def combine():
            return buf.combine(recv_x, handle)

        (combined, _, _), first_c = ctx.first_call(combine)
        ex_c = dict(buf.last_exchange)

        def both():
            r = dispatch()
            return buf.combine(r[0], r[4])

        ref = (x.float() * itr.cpu().sum(1, keepdim=True)).to(torch.bfloat16)
        passed, detail = bool(torch.equal(combined.cpu(), ref)), 'combine(dispatch(x)) == x * ranks_visited'
        recv_rows, send_rows = int(handle.recv_counts.sum()), int(handle.send_counts.sum())
        logical_d = logical_c = recv_rows * H * BF16           # deep_ep: dispatch recv bytes
        wire_d = send_rows * ex_d['width'] * BF16
        wire_c = recv_rows * ex_c['width'] * BF16
        load = recv_rows
    else:
        def dispatch():
            return buf.low_latency_dispatch(d_x, d_idx, T, E, use_fp8=False)

        out, first_d = ctx.first_call(dispatch)
        recv_x, handle = out[0], out[2]
        ex_d = dict(buf.last_exchange)

        def combine():
            return buf.low_latency_combine(recv_x, d_idx, d_w, handle)

        (combined, _, _), first_c = ctx.first_call(combine)
        ex_c = dict(buf.last_exchange)

        def both():
            r = dispatch()
            return buf.low_latency_combine(r[0], d_idx, d_w, r[2])

        ref = x.float() * (w * (idx >= 0)).sum(-1, keepdim=True)
        got = combined.cpu().float()
        passed = bool(torch.allclose(got, ref, rtol=1e-2, atol=1e-3))
        detail = 'low_latency_combine(low_latency_dispatch(x)) ~= x * sum(topk_weights)'
        selections = int((idx >= 0).sum())
        logical_d = logical_c = selections * H * BF16         # deep_ep LL: bytes per selected expert
        wire_d = int(ex_d['counts'].sum()) * ex_d['width'] * BF16
        wire_c = int(ex_c['counts'].sum()) * ex_c['width'] * BF16
        load = int(out[1].cpu().sum())

    record = dict(
        bench='ops', mode=mode, preset=preset, hidden=H, experts=E, topk=K, ep=R, tokens=T,
        routing=routing, backend=ctx.backend,
        check=dict(passed=passed, detail=detail),
        first_call_s=dict(dispatch=round(first_d, 3), combine=round(first_c, 3)),
        e2e_us=dict(dispatch=measure(dispatch), combine=measure(combine), dispatch_combine=measure(both)),
        breakdown_us=dict(dispatch=breakdown(ctx, dispatch, 3), combine=breakdown(ctx, combine, 3)),
        resident_us=dict(dispatch=resident(ctx, ex_d, args), combine=resident(ctx, ex_c, args)),
        resident_latency_us=dict(dispatch=resident(ctx, ex_d, args, True), combine=resident(ctx, ex_c, args, True)),
    )
    peers = ctx.peer_values(dict(logical_d=logical_d, logical_c=logical_c, wire_d=wire_d, wire_c=wire_c,
                                 load=load, passed=passed))
    mean = lambda k: sum(p[k] for p in peers) / len(peers)
    record['check']['passed'] = all(p['passed'] for p in peers)
    record['recv_load'] = stats([p['load'] for p in peers])
    record['recv_load']['imbalance'] = round(record['recv_load']['max'] / max(record['recv_load']['mean'], 1e-9), 3)
    record['bytes_per_rank'] = dict(
        logical_dispatch=mean('logical_d'), logical_combine=mean('logical_c'),
        wire_dispatch=mean('wire_d'), wire_combine=mean('wire_c'),
        recv_buffer_dispatch=R * ex_d['slot_rows'] * ex_d['width'] * BF16,
        recv_buffer_combine=R * ex_c['slot_rows'] * ex_c['width'] * BF16)
    b = record['bytes_per_rank']
    record['logical_gbps'] = dict(
        e2e_dispatch=gbps(b['logical_dispatch'], record['e2e_us']['dispatch']['median_us']),
        e2e_combine=gbps(b['logical_combine'], record['e2e_us']['combine']['median_us']))
    if record['resident_us']['dispatch']:
        record['logical_gbps'].update(
            resident_dispatch=gbps(b['logical_dispatch'], record['resident_us']['dispatch']['median_us']),
            resident_combine=gbps(b['logical_combine'], record['resident_us']['combine']['median_us']))
    return record


def main():
    p = base_parser(__doc__)
    p.add_argument('--modes', type=csv_list(str), default=['normal', 'low_latency'])
    p.add_argument('--routings', type=csv_list(str), default=['random'])
    args = p.parse_args()
    ctx = Ctx(args.backend, args.ep)
    if ctx.rank == 0:
        capture_env(args.out_dir)
    for mode in args.modes:
        for preset in args.presets:
            for T in args.tokens:
                for routing in args.routings:
                    rec = bench_case(ctx, args, mode, preset, T, routing)
                    if ctx.rank == 0:
                        append_jsonl(args.out_dir, f'ops_ep{args.ep}.jsonl', rec)
                        e2e, res = rec['e2e_us'], rec['resident_us']
                        lat = rec['resident_latency_us']
                        res_txt = (f" kernel d/c={res['dispatch']['median_us']}/{res['combine']['median_us']}us"
                                   f" (latency {lat['dispatch']['median_us']}/{lat['combine']['median_us']}us)"
                                   if res['dispatch'] else '')
                        print(f"[ops] {mode} {preset} ep={args.ep} T={T} {routing}: "
                              f"check={'PASS' if rec['check']['passed'] else 'FAIL'} "
                              f"e2e d/c={e2e['dispatch']['median_us']}/{e2e['combine']['median_us']}us"
                              f"{res_txt} imbalance={rec['recv_load']['imbalance']}", flush=True)
    ctx.finish()


if __name__ == '__main__':
    main()
