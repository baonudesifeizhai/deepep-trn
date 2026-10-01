# Benchmark

The suite follows DeepEP's own `tests/test_intranode.py` / `test_low_latency.py`: every case is checked for correctness before it is timed, and bytes are counted the way DeepEP counts them. trn needs two more things: a per-stage breakdown and a device-resident replay of the kernel, so host overhead can be separated from the communication itself.

对标 DeepEP 自带的 `tests/test_intranode.py` / `test_low_latency.py`：每个 case 先校验正确性，再测性能，字节按 DeepEP 的口径统计。trn 上还需要两样东西：分阶段拆分，以及常驻设备的 kernel 回放，用来把 host 开销和通信本身分开。

## Running / 运行

```bash
# On the trn2 host; quick takes ~5 min, default ~1 hour
# 在 trn2 宿主机上执行；quick 约 5 分钟，default 约 1 小时
sudo docker exec -w /workspace_user/user/deepep-trn sgl-plugin-dev bash bench/run_suite.sh quick
sudo docker exec -w /workspace_user/user/deepep-trn sgl-plugin-dev bash bench/run_suite.sh default
# Output / 产物: results/<RUN_ID>/{env.json, *.jsonl, summary.md, ops.csv, *.log}
```

A single benchmark (inside the container, after `source scripts/env_trn2.sh`):

单项运行（容器内先 `source scripts/env_trn2.sh`）：

```bash
python -m torch.distributed.run --standalone --nproc-per-node=4 bench/bench_ops.py \
    --ep 4 --modes normal,low_latency --presets qwen3-30b-a3b,deepseek-v3 \
    --tokens 16,32,64,128 --routings random,zipf --out-dir results/my-run
python bench/report.py results/my-run
```

`bench_ops.py --backend gloo` runs without a Neuron device. It is only for debugging the benchmark code itself.

`bench_ops.py --backend gloo` 可以在没有 Neuron 设备的机器上跑，只用来调试 benchmark 代码本身。

## Components / 组成

| Script / 脚本 | Measures / 测什么 | Output / 输出 |
|---|---|---|
| `bench_ops.py` | Normal and LL dispatch / combine / d+c: correctness, first-call time (incl. compile), end-to-end, stage breakdown, device-resident kernel (pipelined and per-call latency), bytes<br>normal 和 LL 的 dispatch / combine / d+c：正确性、首次调用耗时（含编译）、端到端、分阶段拆分、常驻 kernel（摊销与单次延迟）、字节数 | `ops_ep{R}.jsonl` |
| `bench_transport.py` | Raw a2av kernel: rows per peer × row width × fill ratio, device-resident and with host copies<br>裸 a2av kernel：每个 peer 的行数 × 行宽 × 填充率，分别测常驻设备和含 host 拷贝 | `transport_ep{R}.jsonl` |
| `bench_baselines.py` | all_reduce, all_gather, reduce_scatter, ag_rs, all_to_all_single on the EP group (neuron PG, device-resident)<br>EP 组上的 all_reduce、all_gather、reduce_scatter、ag_rs、all_to_all_single（neuron PG，常驻设备） | `baselines_ep{R}.jsonl` |
| `report.py` | Aggregates a run into tables / 汇总成表 | `summary.md`, `ops.csv` |
| `common.py` | Presets, timing, environment capture / 模型预设、计时、环境记录 | — |

Model presets `(hidden, experts, top-k)` / 模型预设：`qwen3-30b-a3b (2048, 128, 8)`, `qwen3-235b-a22b (4096, 128, 8)`, `deepseek-v3 (7168, 256, 8)`.

Routing distributions / 路由分布：

| | |
|---|---|
| `random` | Uniform top-k / 均匀 top-k |
| `balanced` | Every token hits every rank evenly / 每个 token 均匀落到各 rank |
| `zipf` | Expert popularity ~ 1/rank, rank 0 overloaded / expert 热度服从 1/rank，rank 0 过载 |
| `hot_rank0` | All tokens route only to rank 0 / 全部 token 只路由到 rank 0 |

## Metrics / 指标口径

- **Timing / 计时**: synchronized host wall-clock. Each round is barrier → N calls → one device sync. The round time is the max across ranks (the critical path) divided by N. We report the median, min and max of 5 rounds. N is adaptive, targeting ~0.5 s per round.

  同步的 host 墙钟时间。每轮先 barrier，再连续调用 N 次，最后做一次 device sync；取各 rank 的最大值（即关键路径）除以 N，报告 5 轮的中位数、最小值、最大值。N 自适应，每轮目标约 0.5 s。
- **e2e / 端到端**: calls the `Buffer` API, including host packing and host↔device copies. This is what a user sees.

  调用 `Buffer` 的 API，包含 host 打包和 host↔device 拷贝，也就是用户实际看到的时间。
- **kernel (pipelined) / kernel 摊销**: `Transport.resident_call` replays the exact exchange shape on device-resident data, issued back to back. This is an amortized throughput number.

  用 `Transport.resident_call` 回放这次交换的精确 shape，数据常驻设备，连续发射。这是摊销后的吞吐口径。
- **kernel latency / kernel 单次延迟**: same, but with a sync after every call. A decode step waits on each layer's communication, so this is the number that matters there.

  同上，但每次调用后都 sync。decode 时每层都要等通信完成，看的是这个数。
- **Stage breakdown / 分阶段拆分**: syncs at every stage boundary (`deep_ep_trn.profiling.timer`). Because of those syncs the stages add up to more than e2e, so use them for proportions only.

  每个阶段边界都 sync（`deep_ep_trn.profiling.timer`）。因为插入了 sync，各阶段之和会大于 e2e，只用来看比例。

  | Stage / 阶段 | Meaning / 含义 |
  |---|---|
  | `in_d2h` | Inputs to host / 输入拷回 host |
  | `plan` | Routing plan / 路由规划 |
  | `pack` | Row packing / 行打包 |
  | `h2d` | Send buffer to device / 发送 buffer 上设备 |
  | `kernel` | a2av |
  | `d2h` | Receive buffer to host / 接收 buffer 回 host |
  | `unpack` | Unpacking / 解包 |
  | `out_h2d` | Outputs back to the caller's device / 输出拷回调用方设备 |
- **Bytes / 字节**:
  - logical: DeepEP's definition. Normal mode is received rows × H × 2; LL mode is selected (token, expert) pairs × H × 2, counted as BF16 regardless of FP8.

    logical：按 DeepEP 口径。normal 模式是接收行数 × H × 2；LL 模式是被选中的 (token, expert) 数 × H × 2，与 FP8 无关，统一按 BF16 计。
  - wire: rows actually sent × row width (incl. metadata columns) × 2.

    wire：实际发出的行数 × 行宽（含元数据列）× 2。
  - recv buffer: size of the receive buffer. The driver requires `EP × send capacity`, so it is usually much larger than the payload.

    recv buffer：接收 buffer 的大小。驱动要求它是 `EP × 发送容量`，所以通常远大于 payload。
- **Load imbalance / 负载不均**: max / mean of received load across the EP group.

  EP 组内各 rank 接收量的 max / mean。

## Results / 结果（2026-10-01）

Full report / 完整报告：[results/20261001-default/summary.md](../results/20261001-default/summary.md)

- Environment / 环境：trn2.3xlarge, BF16。
- Coverage / 覆盖：EP4 / EP2 × normal / LL × qwen3-30b-a3b, deepseek-v3 × T = 16–128 × 4 routings。
- Correctness / 正确性：all 36 cases PASS / 36 个 case 全部 PASS。

### Key numbers / 关键数字

qwen3-30b-a3b, EP4, random routing, µs. d / c = dispatch / combine.

qwen3-30b-a3b，EP4，随机路由，单位 µs。d / c 表示 dispatch / combine。

| T/rank | normal e2e d / c | normal kernel pipelined d / c | normal kernel latency d / c | LL e2e d / c | LL kernel latency d / c |
|---|---|---|---|---|---|
| 16 | 2877 / 1931 | 205 / 206 | 432 / 401 | 5728 / 8006 | 416 / 491 |
| 64 | 6480 / 4253 | 213 / 241 | 482 / 482 | 29169 / 33470 | 482 / 750 |
| 128 | 11576 / 7257 | 249 / 245 | 539 / 531 | 54991 / 64606 | 535 / 1094 |

Device-resident collectives on the EP group for comparison (pipelined / latency, T=64):

对照：EP 组内常驻设备的集合通信（摊销 / 单次延迟，T=64）：

| all_reduce [256, 2048] | ag_rs | all_to_all_single | deep_ep_trn normal d+c |
|---|---|---|---|
| 161 / 422 | 190 / 423 | 157 / 350 | 454 / 964 |

### Conclusions / 结论

1. **The host side is ~90–99.7% of e2e.** That covers packing, unpacking and host↔device copies; LL mode building `[EL, R*M, H]` on the host is the heaviest. Keeping data on the device is the top priority (roadmap stage 2).

   **host 侧占端到端的约 90–99.7%。** 包括打包、解包和 host↔device 拷贝，LL 模式在 host 上构造 `[EL, R*M, H]` 尤其重。让数据常驻设备是第一优先级（roadmap stage 2）。
2. **Every on-chip collective has a fixed cost of ~135–230 µs pipelined, or ~300–510 µs per call.** Our kernel adds ~40–60 µs over a bare `all_to_all_single` for qwen3, and ~125 µs for deepseek-v3 at T=64. At B ≤ 128 everything is latency-bound.

   **单芯片上任何集合通信都有约 135–230 µs（摊销）或 300–510 µs（单次延迟）的固定开销。** 我们的 kernel 比裸 `all_to_all_single` 多约 40–60 µs（qwen3），deepseek-v3 T=64 时多约 125 µs。B ≤ 128 时基本受延迟主导。
3. **Dispatch + combine are two separate launches.** At small batch they are ~2.3–2.8× slower than a single-graph all_reduce or AG+RS. On one chip at decode sizes, the a2a path needs to be fused with expert compute into one NEFF to compete.

   **dispatch + combine 是两次独立 launch。** 小 batch 下比单图的 all_reduce 或 AG+RS 慢约 2.3–2.8 倍。单芯片、decode 尺寸下，a2a 路线要和 expert 计算融合进同一个 NEFF 才有机会。
4. **Kernel bandwidth peaks at ~12–16 GB/s** (512 rows per peer). Cutting the fill ratio to 25% only reduces time by ~20–40%: whole-buffer copies sized by capacity dominate, because the driver requires dst = EP × src.

   **kernel 带宽最高约 12–16 GB/s**（每个 peer 512 行）。填充率降到 25% 时，耗时只减少约 20–40%，说明按容量做的整块 buffer 拷贝是主要成本，因为驱动要求 dst = EP × src。
5. **The LL combine return buffer is amplified R × min(K, EL) times.** At T=128 its per-call latency is ~1.1 ms, about twice dispatch. Reducing on the receiver first (roadmap stage 2.5) removes this.

   **LL combine 的回传 buffer 被放大了 R × min(K, EL) 倍。** T=128 时单次延迟约 1.1 ms，是 dispatch 的两倍。接收端先归约（roadmap stage 2.5）可以消掉这部分。
6. **Routing skew / 路由偏斜**（zipf, hot_rank0）:
   - Normal-mode kernel time stays within ±10%, because fixed-capacity buffers are already sized for the worst case.

     normal 模式的 kernel 时间变化在 ±10% 以内，因为固定容量 buffer 本来就按最坏情况开。
   - LL combine is ~36% slower under hot_rank0, since rank 0 returns the most rows.

     LL combine 在 hot_rank0 下慢约 36%，因为 rank 0 的回传量最大。

## Caveats / 注意

- First-call time includes NEFF compilation and drops a lot once the cache is warm.

  首次调用耗时包含 NEFF 编译，cache 热了之后会快很多。
- trn2.3xlarge is a single chip with 4 logical cores, so all traffic stays on-chip. Inter-node (EFA) performance is not measured here.

  trn2.3xlarge 只有单芯片 4 个逻辑核，所有数据都在芯片内搬运；跨机（EFA）的性能这里测不到。
- Baselines and kernel replays are device-resident and directly comparable. e2e includes host overhead and is not comparable to them.

  baselines 和 kernel 回放都是常驻设备的数，可以直接对比；e2e 包含 host 开销，不能拿来和它们对比。
