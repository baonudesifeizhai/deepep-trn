# deep_ep_trn

DeepEP-style MoE expert-parallel dispatch / combine for AWS Trainium (trn2), with a Python API that mirrors [deep_ep](https://github.com/deepseek-ai/DeepEP).

在 AWS Trainium（trn2）上做 DeepEP 风格的 MoE expert-parallel dispatch / combine，Python API 对齐 [deep_ep](https://github.com/deepseek-ai/DeepEP)。

## Status / 状态（v0.1, 2026-10-01）

| | |
|---|---|
| ✅ Normal mode / normal 模式 | `get_dispatch_layout` / `dispatch` (incl. cached dispatch / 含 cached dispatch) / `combine` |
| ✅ Low-latency mode / low-latency 模式 | `low_latency_dispatch` / `low_latency_combine` (BF16) |
| ✅ Backends / 后端 | `nki`: trn2 single node, LNC=2 / trn2 单机，LNC=2<br>`gloo`: CPU reference for tests / CPU 参考实现，测试用 |
| ✅ Correctness / 正确性 | Unit tests + torchrun tests: EP=4 and EP=2 (TP2×EP2 groups), 5 routing distributions, passing on both `nki` and `gloo`<br>单测 + torchrun 分布式测试：EP=4、EP=2（TP2×EP2 分组），5 种路由，`nki` 和 `gloo` 都通过 |
| ✅ Benchmark suite / Benchmark 框架 | See [docs/benchmark.md](docs/benchmark.md) / 见 [docs/benchmark.md](docs/benchmark.md) |
| 🚧 v0 data path / v0 数据路径 | Routing, packing and unpacking run in torch on the host; only the collectives run on the Neuron device. Moving them on-device is next ([roadmap](docs/roadmap.md))<br>路由、打包、解包目前在 host 上（torch），只有集合通信在设备上；下一步把它们挪到设备上（[roadmap](docs/roadmap.md)） |
| ❌ Not yet / 未实现 | FP8, inter-node, real async/hook overlap, `num_worst_tokens`, combine bias, LogFMT<br>FP8、跨机、真正的 async/hook overlap、`num_worst_tokens`、combine bias、LogFMT |

## Usage / 用法

```python
from deep_ep_trn import Buffer

buf = Buffer(ep_group, backend='nki', num_max_tokens_per_rank=128)

# Normal mode / normal 模式
ntr, _, nte, itr, _ = buf.get_dispatch_layout(topk_idx, num_experts)
recv_x, recv_topk_idx, recv_topk_weights, tokens_per_expert, handle, _ = buf.dispatch(
    x, topk_idx=topk_idx, topk_weights=topk_weights,
    num_tokens_per_rank=ntr, is_token_in_rank=itr, num_tokens_per_expert=nte)
y = experts(recv_x, recv_topk_idx, recv_topk_weights)   # reduce this rank's experts per row / 每行先归约本 rank 的 expert
out, _, _ = buf.combine(y, handle)

# Low-latency mode / low-latency 模式
recv_x, recv_count, handle, _, hook = buf.low_latency_dispatch(
    x, topk_idx, num_max_dispatch_tokens_per_rank=128, num_experts=num_experts, use_fp8=False)
expert_out = grouped_experts(recv_x, recv_count)         # [EL, R*M, H]
out, _, _ = buf.low_latency_combine(expert_out, topk_idx, topk_weights, handle)
```

When the EP group is a subgroup (e.g. `[[0, 2], [1, 3]]` for TP2×EP2), pass `replica_groups=` listing every EP group in the world.

EP 组是子组时（例如 TP2×EP2 的 `[[0, 2], [1, 3]]`），要传 `replica_groups=` 列出世界里所有 EP 组。

### Differences from deep_ep / 与 deep_ep 的差异

- **Constructor / 构造参数**: byte-size arguments such as `num_nvl_bytes` / `num_rdma_bytes` are ignored. trn buffers are compile-time static shapes, so `num_max_tokens_per_rank` is required and must be identical on every rank.

  `num_nvl_bytes` / `num_rdma_bytes` 等字节参数会被忽略。trn 的 buffer 是编译期静态 shape，所以必须传 `num_max_tokens_per_rank`，且所有 rank 的值相同。
- **FP8 in LL mode / LL 模式 FP8**: deep_ep's `low_latency_dispatch` defaults to `use_fp8=True`; here that raises `NotImplementedError`, so pass `use_fp8=False` explicitly.

  deep_ep 的 `low_latency_dispatch` 默认 `use_fp8=True`，这里会报 `NotImplementedError`，需要显式传 `use_fp8=False`。
- **Wire format / 线上格式**: one row per (token, destination rank), with top-k weights, expert ids and the source token index bit-packed at the end of the row, so each dispatch is a single collective. LL dispatch is also rank-deduplicated and the receiver expands rows into `[EL, R*M, H]` (deep_ep LL sends once per (token, expert)), so fewer bytes go over the wire.

  每个 (token, 目的 rank) 只发一行，top-k 权重、expert id、源 token 下标按位打包在行尾，所以一次 dispatch 只要一次集合通信。LL dispatch 也按 rank 去重，由接收端展开成 `[EL, R*M, H]`（deep_ep LL 是每个 (token, expert) 发一次），线上字节更少。
- **LL combine**: same semantics as deep_ep: one row per (token, expert) goes back, and the source applies top-k weights and sums.

  语义与 deep_ep 相同：每个 (token, expert) 回传一行，由源端乘权重后求和。
- **Receive order / 接收顺序**: normal mode orders rows by (source rank, source token). In LL mode each expert's rows follow the same order, and `handle.layout_range` gives each source's range (`start << 32 | count`).

  normal 模式按 (源 rank, 源 token) 排列。LL 模式下每个 expert 内也按这个顺序，`handle.layout_range` 给出每个源的区间（`start << 32 | count`）。
- **Events and hooks / 事件与 hook**: `EventOverlap` is a stub and hooks are no-ops; Neuron work is already ordered on the device queue.

  `EventOverlap` 是空壳，hook 是 no-op；Neuron 的执行本身在设备队列上有序。

## Layout / 目录

Architecture (layers, data flow, capacities, extension points): [docs/architecture.md](docs/architecture.md).

架构说明（分层、数据流、容量、扩展点）见 [docs/architecture.md](docs/architecture.md)。

```
deep_ep_trn/
  buffer.py            Buffer: normal + low-latency modes / 两种模式
  codec.py             row codec: payload + bit-packed routing metadata / 行编码
  layout.py            routing layout, vectorised torch / 路由布局
  profiling.py         optional per-stage timers for benchmarks / 可选的分阶段计时
  testing.py           helpers shared by tests and bench / tests 与 bench 共用的辅助函数
  transport/
    base.py            fixed-slot all-to-all-v contract / 固定槽位 all-to-all-v 契约
    nki_a2av.py        trn2 NKI kernel (ncc.all_to_all_v)
    gloo.py            CPU reference / CPU 参考实现
scripts/               container env, one-shot test runner / 容器环境变量、一键正确性测试
tests/unit/            CPU unit tests (pytest) / CPU 单测
tests/dist/            torchrun correctness tests / torchrun 正确性测试
bench/                 benchmark suite / benchmark 框架
tools/probe_transport.py  probe which slot shapes the runtime accepts / 探测 runtime 接受的槽位 shape
experiments/           archived stage-1 prototype / 归档的 stage-1 原型
results/               benchmark results, one directory per run / benchmark 结果，每次运行一个目录
docs/                  architecture, constraints, benchmark, roadmap / 架构、约束、benchmark、路线图
```

## Tests / 测试

Run on a trn2 host. The container is `sgl-plugin-dev`, with this repo mounted at `/workspace_user/user/deepep-trn`.

在 trn2 宿主机上执行。容器是 `sgl-plugin-dev`，本仓库挂载在 `/workspace_user/user/deepep-trn`。

```bash
# Full suite / 全套：unit + gloo/nki × normal/LL × EP4/EP2
sudo docker exec -w /workspace_user/user/deepep-trn sgl-plugin-dev bash scripts/run_tests_trn2.sh

# Single test / 单项
sudo docker exec -w /workspace_user/user/deepep-trn sgl-plugin-dev bash -lc 'source scripts/env_trn2.sh;
  python -m torch.distributed.run --standalone --nproc-per-node=4 tests/dist/test_intranode.py --backend nki --ep 4'
```

## Docs / 文档

| | |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Layers, data flow, capacities, handles, extension points / 分层、数据流、容量、handle、扩展点 |
| [docs/trn2-constraints.md](docs/trn2-constraints.md) | Measured trn2 platform constraints, e.g. fixed-slot `all_to_all_v` and `dst = EP × src` / 实测的 trn2 平台约束 |
| [docs/benchmark.md](docs/benchmark.md) | Benchmark methodology, results and conclusions / benchmark 方法、结果和结论 |
| [docs/roadmap.md](docs/roadmap.md) | Next stages / 后续阶段 |
