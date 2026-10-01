# Development

## Repository layout

See [architecture.md](architecture.md) for layers and data flow.

```
deep_ep_trn/
  buffer.py            Buffer: normal + low-latency modes
  codec.py             row codec: payload + bit-packed routing metadata
  layout.py            routing layout, vectorised torch
  profiling.py         optional per-stage timers for benchmarks
  testing.py           helpers shared by tests and bench
  transport/
    base.py            fixed-slot all-to-all-v contract
    nki_a2av.py        trn2 NKI kernel (nki.collectives.all_to_all_v)
    gloo.py            CPU reference
scripts/               container env, one-shot test runner
tests/unit/            CPU unit tests (pytest)
tests/dist/            torchrun correctness tests
bench/                 benchmark suite
tools/probe_transport.py  probe which slot shapes the runtime accepts
experiments/           archived stage-1 prototype
results/               benchmark results, one directory per run
docs/                  architecture, constraints, benchmark, roadmap
```

## Tests

Run on a trn2 host. The container is `sgl-plugin-dev`, with this repo mounted at `/workspace_user/user/deepep-trn`.

```bash
# Full suite: unit + gloo/nki × normal/LL × EP4/EP2
sudo docker exec -w /workspace_user/user/deepep-trn sgl-plugin-dev bash scripts/run_tests_trn2.sh

# Single test
sudo docker exec -w /workspace_user/user/deepep-trn sgl-plugin-dev bash -lc 'source scripts/env_trn2.sh;
  python -m torch.distributed.run --standalone --nproc-per-node=4 tests/dist/test_intranode.py --backend nki --ep 4'
```
