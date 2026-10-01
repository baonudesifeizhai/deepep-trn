# deep_ep_trn

在 AWS Trainium（trn2）上做 DeepEP 风格的 MoE expert-parallel dispatch / combine，Python API 对齐 [deep_ep](https://github.com/deepseek-ai/DeepEP)。

## 状态（v0.1，2026-10-01）

| | |
|---|---|
| ✅ normal 模式 | `get_dispatch_layout` / `dispatch`（含 cached dispatch）/ `combine` |
| ✅ low-latency 模式 | `low_latency_dispatch` / `low_latency_combine`（BF16） |
| ✅ 后端 | `nki`（trn2 单机，LNC=2）、`gloo`（CPU 参考实现，测试用） |
| ✅ 正确性 | 单测 + torchrun 分布式测试：EP=4、EP=2（TP2×EP2 分组），5 种路由（均衡 / 随机 / 全打到一个 rank / 各 rank token 数不同且有空闲 rank / 部分 top-k 被 mask），nki 和 gloo 都通过 |
| ✅ Benchmark 框架 | ops（normal/LL）/ transport 曲线 / baselines / 分阶段拆分 / 汇总报表，见 [docs/benchmark.md](docs/benchmark.md) |
| ⚠️ 性能 | 路由、打包、解包目前在 host 上（torch CPU），只有集合通信在设备上，端到端是毫秒级。见下面的 Benchmark 和 [roadmap](docs/roadmap.md) |
| ❌ 未实现 | FP8、跨机（internode）、真正的 async/hook overlap、`num_worst_tokens`、combine bias、LogFMT |

## 用法

```python
from deep_ep_trn import Buffer

buf = Buffer(ep_group, backend='nki', num_max_tokens_per_rank=128)

# normal 模式
ntr, _, nte, itr, _ = buf.get_dispatch_layout(topk_idx, num_experts)
recv_x, recv_topk_idx, recv_topk_weights, tokens_per_expert, handle, _ = buf.dispatch(
    x, topk_idx=topk_idx, topk_weights=topk_weights,
    num_tokens_per_rank=ntr, is_token_in_rank=itr, num_tokens_per_expert=nte)
y = experts(recv_x, recv_topk_idx, recv_topk_weights)       # 每行先归约本 rank 的 expert
out, _, _ = buf.combine(y, handle)

# low-latency 模式
recv_x, recv_count, handle, _, hook = buf.low_latency_dispatch(
    x, topk_idx, num_max_dispatch_tokens_per_rank=128, num_experts=num_experts, use_fp8=False)
expert_out = grouped_experts(recv_x, recv_count)             # [EL, R*M, H]
out, _, _ = buf.low_latency_combine(expert_out, topk_idx, topk_weights, handle)
```

EP 组是子组时（例如 TP2×EP2 的 `[[0, 2], [1, 3]]`），要传 `replica_groups=` 列出世界里所有 EP 组。

### 与 deep_ep 的差异

- **构造参数**：`num_nvl_bytes` / `num_rdma_bytes` 等字节参数会被忽略。trn 的 buffer 是编译期静态 shape，需要 `num_max_tokens_per_rank`，且所有 rank 的值必须相同。
- **LL 模式 FP8**：deep_ep 的 `low_latency_dispatch` 默认 `use_fp8=True`，这里会报 `NotImplementedError`，需要显式传 `use_fp8=False`。
- **线上格式**：每个 (token, 目的 rank) 只发一行，top-k 权重、expert id、源 token 下标按位打包在行尾，所以一次 dispatch 只要一次集合通信。LL dispatch 也按 rank 去重，由接收端展开成 `[EL, R*M, H]`（deep_ep LL 是每个 (token, expert) 发一次），线上字节更少。
- **LL combine**：语义与 deep_ep 相同，每个 (token, expert) 回传一行，由源端乘权重后求和。
- **接收顺序**：normal 模式按 (源 rank, 源 token) 排列。LL 模式下每个 expert 内也按 (源 rank, 源 token) 排列，`handle.layout_range` 给出每个源的区间（`start << 32 | count`）。
- **事件与 hook**：`EventOverlap` 是空壳，hook 是 no-op（Neuron 的执行本身在设备队列上有序）。

## 目录

架构说明（分层、数据流、容量、扩展点）见 [docs/architecture.md](docs/architecture.md)。

```
deep_ep_trn/
  buffer.py            Buffer：normal + low-latency 两种模式
  codec.py             行编码：payload + 按位打包的路由元数据
  layout.py            路由布局（向量化 torch 实现）
  profiling.py         可选的分阶段计时（benchmark 用）
  testing.py           tests 和 bench 共用的辅助函数（合成路由、EP 分组）
  transport/
    base.py            固定槽位 all-to-all-v 契约
    nki_a2av.py        trn2 NKI kernel（ncc.all_to_all_v）
    gloo.py            CPU 参考实现
scripts/               容器环境变量（env_trn2.sh）、一键正确性测试（run_tests_trn2.sh）
tests/unit/            CPU 单测（pytest）
tests/dist/            torchrun 正确性测试（normal / low-latency）
bench/                 benchmark 框架：ops / transport / baselines / report / run_suite.sh
tools/probe_transport.py  探测 runtime 接受哪些槽位 shape
experiments/           stage-1 原型及结果
results/               benchmark 结果（每次运行一个目录）
docs/                  架构、平台约束、benchmark 方法、路线图
```

## 测试

在 trn2 宿主机上执行（容器 `sgl-plugin-dev`，本仓库挂载在 `/workspace_user/user/deepep-trn`）：

```bash
# 全套：单测 + gloo/nki × normal/LL × EP4/EP2
sudo docker exec -w /workspace_user/user/deepep-trn sgl-plugin-dev bash scripts/run_tests_trn2.sh

# 单项
sudo docker exec -w /workspace_user/user/deepep-trn sgl-plugin-dev bash -lc 'source scripts/env_trn2.sh;
  python -m torch.distributed.run --standalone --nproc-per-node=4 tests/dist/test_intranode.py --backend nki --ep 4'
```

## Benchmark

框架说明见 [docs/benchmark.md](docs/benchmark.md)。一键运行：

```bash
sudo docker exec -w /workspace_user/user/deepep-trn sgl-plugin-dev bash bench/run_suite.sh default   # 或 quick
```

最近一次完整结果：[results/20261001-default/summary.md](results/20261001-default/summary.md)。

- 环境：trn2.3xlarge，BF16。
- 覆盖：EP4 / EP2 × normal / LL × qwen3-30b-a3b、deepseek-v3 × T=16–128 × 4 种路由。
- 正确性：36 个 case 全部 PASS。

### 关键数字（qwen3-30b-a3b，EP4，随机路由，单位 µs）

| T/rank | normal 端到端 d / c | normal kernel 摊销 d / c | normal kernel 单次延迟 d / c | LL 端到端 d / c | LL kernel 单次延迟 d / c |
|---|---|---|---|---|---|
| 16 | 2877 / 1931 | 205 / 206 | 432 / 401 | 5728 / 8006 | 416 / 491 |
| 64 | 6480 / 4253 | 213 / 241 | 482 / 482 | 29169 / 33470 | 482 / 750 |
| 128 | 11576 / 7257 | 249 / 245 | 539 / 531 | 54991 / 64606 | 535 / 1094 |

对照：EP 组内常驻设备的集合通信（摊销 / 单次延迟，T=64）：

| all_reduce [256, 2048] | ag_rs | all_to_all_single | deep_ep_trn normal d+c |
|---|---|---|---|
| 161 / 422 | 190 / 423 | 157 / 350 | 454 / 964 |

### 结论

1. **host 侧占端到端的约 90–99.7%。** 包括打包、解包和 host↔device 拷贝，LL 模式在 host 上构造 `[EL, R*M, H]` 尤其重。让数据常驻设备是第一优先级（roadmap stage 2）。
2. **单芯片上任何集合通信都有约 135–230 µs（摊销）或 300–510 µs（单次延迟）的固定开销。** 我们的 kernel 比裸 `all_to_all_single` 多约 40–60 µs（qwen3），deepseek-v3 T=64 时多约 125 µs。B≤128 时基本受延迟主导。
3. **dispatch + combine 是两次独立 launch。** 在小 batch 下比单图的 all_reduce 或 AG+RS 慢约 2.3–2.8 倍，所以单芯片、decode 尺寸下，a2a 路线要靠和 expert 计算融合进同一个 NEFF 才有机会。
4. **kernel 带宽最高约 12–16 GB/s**（每个 peer 512 行）。填充率降到 25% 时，耗时只减少约 20–40%，说明 kernel 内按容量做的整块 buffer 拷贝是主要成本，因为驱动要求 dst = EP × src。
5. **LL combine 的回传 buffer 被放大了 R × min(K, EL) 倍。** T=128 时单次延迟约 1.1 ms，比 dispatch 大一倍。接收端先归约（roadmap stage 2.5）可以消掉这部分。
6. **路由偏斜**（zipf、hot_rank0）下：
   - normal 模式的 kernel 时间变化在 ±10% 以内，因为固定容量 buffer 本来就按最坏情况开。
   - LL combine 在 hot_rank0 下慢约 36%，因为 rank 0 的回传量最大。

## 平台约束

详见 [docs/trn2-constraints.md](docs/trn2-constraints.md)。最关键的两条：

- trn2 单机只支持固定槽位的 `all_to_all_v`。
- 驱动要求 `dst = EP × src`。
