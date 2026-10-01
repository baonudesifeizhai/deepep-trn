"""Raw fixed-slot all-to-all-v: latency/bandwidth curve of the NKI kernel itself.

Sweeps rows-per-peer x row width x fill ratio. ``fill`` < 1 sends fewer rows than
the static capacity: if time tracks capacity rather than payload, the kernel's
fixed-size buffer copies (not the wire) dominate.

torchrun --standalone --nproc-per-node=4 bench/bench_transport.py --ep 4
"""
from common import PRESETS, Ctx, append_jsonl, base_parser, capture_env, csv_list, gbps

import torch

from deep_ep_trn.codec import RowCodec


def main():
    p = base_parser(__doc__)
    p.add_argument('--rows', type=csv_list(int), default=[8, 32, 128, 512], help='rows per peer')
    p.add_argument('--fills', type=csv_list(float), default=[1.0, 0.25])
    p.add_argument('--no-host', action='store_true', help='skip the variant with host<->device copies')
    args = p.parse_args()
    if args.backend != 'nki':
        raise SystemExit('bench_transport measures the NKI kernel; use --backend nki')
    ctx = Ctx(args.backend, args.ep)
    if ctx.rank == 0:
        capture_env(args.out_dir)
    t, R = ctx.transport, ctx.ep
    widths = sorted({RowCodec(PRESETS[n][0], PRESETS[n][2]).width for n in args.presets})
    for width in widths:
        for rows in args.rows:
            for fill in args.fills:
                n = max(1, round(rows * fill))
                cap = R * rows
                counts = torch.full((R,), n, dtype=torch.int64)
                slot = t.slot_rows(cap, rows)
                call = t.resident_call(cap, width, counts, slot)
                _, first = ctx.first_call(call)
                res = ctx.measure(call, args.rounds, args.max_iters, args.target_round_s)
                lat = ctx.measure(call, args.rounds, args.max_iters, args.target_round_s, sync_each=True)
                payload = int(counts.sum()) * width * 2
                rec = dict(bench='transport', ep=R, width=width, rows_per_peer=rows, fill=fill,
                           send_capacity_rows=cap, slot_rows=slot, first_call_s=round(first, 3),
                           resident_us=res, resident_latency_us=lat, payload_bytes=payload,
                           off_rank_bytes=(R - 1) * n * width * 2,
                           recv_buffer_bytes=R * slot * width * 2,
                           payload_gbps=gbps(payload, res['median_us']),
                           off_rank_gbps=gbps((R - 1) * n * width * 2, res['median_us']))
                if not args.no_host:
                    send = torch.zeros(cap, width, dtype=torch.bfloat16)
                    host = ctx.measure(lambda: t.all_to_all_fixed_slot(send, counts, slot),
                                       args.rounds, args.max_iters, args.target_round_s)
                    rec['with_host_copies_us'] = host
                if ctx.rank == 0:
                    append_jsonl(args.out_dir, f'transport_ep{R}.jsonl', rec)
                    host_txt = f" with_host={rec['with_host_copies_us']['median_us']}us" if not args.no_host else ''
                    print(f"[transport] ep={R} W={width} rows/peer={rows} fill={fill}: "
                          f"kernel={res['median_us']}us latency={lat['median_us']}us "
                          f"payload={rec['payload_gbps']}GB/s{host_txt}",
                          flush=True)
    ctx.finish()


if __name__ == '__main__':
    main()
