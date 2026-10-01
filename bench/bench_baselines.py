"""Communication baselines an EP MoE layer could use instead of dispatch/combine.

All device-resident (inputs on device, outputs left there), compiled with
torch.compile(backend='neuron') over the EP group's neuron process group:

  all_reduce        [R*T, H]          current sglang-neuron path (DP1 attention, all-expert MoE)
  all_gather        [T, H] -> [R*T, H] first half of DP-attention + gather-EP
  reduce_scatter    [R*T, H] -> [T, H] second half
  ag_rs             all_gather + reduce_scatter in one graph
  all_to_all_single [R*T, H] dense equal splits (capacity-padded dispatch, no routing)

torchrun --standalone --nproc-per-node=4 bench/bench_baselines.py --ep 4
"""
from common import PRESETS, Ctx, append_jsonl, base_parser, capture_env

import torch
from torch.distributed import _functional_collectives as funcol


def main():
    args = base_parser(__doc__).parse_args()
    if args.backend != 'nki':
        raise SystemExit('baselines use the neuron process group; use --backend nki')
    ctx = Ctx(args.backend, args.ep)
    if ctx.rank == 0:
        capture_env(args.out_dir)
    g, R = ctx.ep_device_group, ctx.ep

    def compiled(fn):
        return torch.compile(fn, backend='neuron', fullgraph=True, dynamic=False)

    ops = {
        'all_reduce': (compiled(lambda v: funcol.all_reduce(v, 'sum', g)), lambda T, H: [(R * T, H)]),
        'all_gather': (compiled(lambda v: funcol.all_gather_tensor(v, 0, g)), lambda T, H: [(T, H)]),
        'reduce_scatter': (compiled(lambda v: funcol.reduce_scatter_tensor(v, 'sum', 0, g)),
                           lambda T, H: [(R * T, H)]),
        'ag_rs': (compiled(lambda a, b: (funcol.all_gather_tensor(a, 0, g),
                                         funcol.reduce_scatter_tensor(b, 'sum', 0, g))),
                  lambda T, H: [(T, H), (R * T, H)]),
        'all_to_all_single': (compiled(lambda v: funcol.all_to_all_single(v, None, None, g)),
                              lambda T, H: [(R * T, H)]),
    }
    for preset in args.presets:
        H = PRESETS[preset][0]
        for T in args.tokens:
            for name, (fn, shapes) in ops.items():
                inputs = [ctx.to_dev(torch.randn(*s, dtype=torch.bfloat16)) for s in shapes(T, H)]
                rec = dict(bench='baseline', op=name, preset=preset, hidden=H, ep=R, tokens=T,
                           shapes=[list(s) for s in shapes(T, H)])
                try:
                    _, first = ctx.first_call(lambda: fn(*inputs))
                    call = lambda: fn(*inputs)
                    rec.update(first_call_s=round(first, 3),
                               resident_us=ctx.measure(call, args.rounds, args.max_iters, args.target_round_s),
                               resident_latency_us=ctx.measure(call, args.rounds, args.max_iters,
                                                               args.target_round_s, sync_each=True))
                except Exception as exc:  # unsupported collectives are a result, not a crash
                    rec['error'] = f'{type(exc).__name__}: {str(exc).splitlines()[0][:300]}'
                if ctx.rank == 0:
                    append_jsonl(args.out_dir, f'baselines_ep{R}.jsonl', rec)
                    val = (f"{rec['resident_us']['median_us']}us latency={rec['resident_latency_us']['median_us']}us"
                           if 'resident_us' in rec else rec['error'])
                    print(f'[baseline] {name} {preset} ep={R} T={T}: {val}', flush=True)
    ctx.finish()


if __name__ == '__main__':
    main()
