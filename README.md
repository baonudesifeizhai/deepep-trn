# deep_ep_trn

DeepEP-style MoE expert-parallel dispatch / combine for AWS Trainium (trn2).

This is an independent implementation, not a wrapper: it does not import or call deep_ep. It mirrors [deep_ep](https://github.com/deepseek-ai/DeepEP)'s Python API (same method names and arguments) so that MoE code written against DeepEP can switch backends. On trn2 the data moves through a custom NKI kernel built on `nki.collectives.all_to_all_v`.

## Status (v0.1, 2026-10-01)

| | |
|---|---|
| ✅ Normal mode | `get_dispatch_layout` / `dispatch` (incl. cached dispatch) / `combine` |
| ✅ Low-latency mode | `low_latency_dispatch` / `low_latency_combine` (BF16) |
| ✅ Backends | `nki`: trn2 single node, LNC=2<br>`gloo`: CPU reference for tests |
| ✅ Correctness | Unit tests + torchrun tests: EP=4 and EP=2 (TP2×EP2 groups), 5 routing distributions, passing on both `nki` and `gloo` |
| ✅ Benchmark suite | See [docs/benchmark.md](docs/benchmark.md) |
| 🚧 v0 data path | Routing, packing and unpacking run in torch on the host; only the collectives run on the Neuron device. Moving them on-device is next ([roadmap](docs/roadmap.md)) |
| ❌ Not yet | FP8, inter-node, real async/hook overlap, `num_worst_tokens`, combine bias, LogFMT |

## Usage

```python
from deep_ep_trn import Buffer

buf = Buffer(ep_group, backend='nki', num_max_tokens_per_rank=128)

# Normal mode
ntr, _, nte, itr, _ = buf.get_dispatch_layout(topk_idx, num_experts)
recv_x, recv_topk_idx, recv_topk_weights, tokens_per_expert, handle, _ = buf.dispatch(
    x, topk_idx=topk_idx, topk_weights=topk_weights,
    num_tokens_per_rank=ntr, is_token_in_rank=itr, num_tokens_per_expert=nte)
y = experts(recv_x, recv_topk_idx, recv_topk_weights)   # reduce this rank's experts per row
out, _, _ = buf.combine(y, handle)

# Low-latency mode
recv_x, recv_count, handle, _, hook = buf.low_latency_dispatch(
    x, topk_idx, num_max_dispatch_tokens_per_rank=128, num_experts=num_experts, use_fp8=False)
expert_out = grouped_experts(recv_x, recv_count)         # [EL, R*M, H]
out, _, _ = buf.low_latency_combine(expert_out, topk_idx, topk_weights, handle)
```

When the EP group is a subgroup (e.g. `[[0, 2], [1, 3]]` for TP2×EP2), pass `replica_groups=` listing every EP group in the world.

### Differences from deep_ep

- **Constructor**: byte-size arguments such as `num_nvl_bytes` / `num_rdma_bytes` are ignored. trn buffers are compile-time static shapes, so `num_max_tokens_per_rank` is required and must be identical on every rank.
- **FP8 in LL mode**: deep_ep's `low_latency_dispatch` defaults to `use_fp8=True`; here that raises `NotImplementedError`, so pass `use_fp8=False` explicitly.
- **Wire format**: one row per (token, destination rank), with top-k weights, expert ids and the source token index bit-packed at the end of the row, so each dispatch is a single collective. LL dispatch is also rank-deduplicated and the receiver expands rows into `[EL, R*M, H]` (deep_ep LL sends once per (token, expert)), so fewer bytes go over the wire.
- **LL combine**: same semantics as deep_ep: one row per (token, expert) goes back, and the source applies top-k weights and sums.
- **Receive order**: normal mode orders rows by (source rank, source token). In LL mode each expert's rows follow the same order, and `handle.layout_range` gives each source's range (`start << 32 | count`).
- **Events and hooks**: `EventOverlap` is a stub and hooks are no-ops; Neuron work is already ordered on the device queue.

## Layout

Architecture (layers, data flow, capacities, extension points): [docs/architecture.md](docs/architecture.md).

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

## Docs

| | |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Layers, data flow, capacities, handles, extension points |
| [docs/trn2-constraints.md](docs/trn2-constraints.md) | Measured trn2 platform constraints, e.g. fixed-slot `all_to_all_v` and `dst = EP × src` |
| [docs/benchmark.md](docs/benchmark.md) | Benchmark methodology, results and conclusions |
| [docs/roadmap.md](docs/roadmap.md) | Next stages |
