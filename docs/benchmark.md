# Benchmark 框架

对标 DeepEP 自带的 `tests/test_intranode.py` / `test_low_latency.py`：每个 case 先校验正确性，再测性能，并按 DeepEP 的口径统计字节。trn 上额外需要两样东西，一是分阶段拆分，二是常驻设备的 kernel 回放，用来把"host 开销"和"通信本身"分开。

## 运行

```bash
# trn2 宿主机上执行；quick 大约 5 分钟，default 为几十分钟
sudo docker exec -w /workspace_user/user/deepep-trn sgl-plugin-dev bash bench/run_suite.sh quick
sudo docker exec -w /workspace_user/user/deepep-trn sgl-plugin-dev bash bench/run_suite.sh default
# 产物在 results/<RUN_ID>/：env.json、*.jsonl、summary.md、ops.csv、各项 .log
```

单项运行（容器内先 `source scripts/env_trn2.sh`）：

```bash
python -m torch.distributed.run --standalone --nproc-per-node=4 bench/bench_ops.py \
    --ep 4 --modes normal,low_latency --presets qwen3-30b-a3b,deepseek-v3 \
    --tokens 16,32,64,128 --routings random,zipf --out-dir results/my-run
python bench/report.py results/my-run
```

`bench_ops.py --backend gloo` 可以在没有 Neuron 设备的机器上跑，只用来调试 benchmark 代码本身。

## 组成

| 脚本 | 测什么 | 输出 |
|---|---|---|
| `bench_ops.py` | normal 和 LL 两种模式的 dispatch / combine / d+c：正确性、首次调用耗时（含编译）、端到端、分阶段拆分、常驻 kernel（摊销与单次延迟）、字节数 | `ops_ep{R}.jsonl` |
| `bench_transport.py` | 裸 a2av kernel：每个 peer 的行数 × 行宽 × 填充率；分别测常驻设备和含 host 拷贝两种 | `transport_ep{R}.jsonl` |
| `bench_baselines.py` | EP 组上 all_reduce、all_gather、reduce_scatter、ag_rs、all_to_all_single（neuron PG，常驻设备） | `baselines_ep{R}.jsonl` |
| `report.py` | 汇总成表 | `summary.md`、`ops.csv` |
| `common.py` | 模型预设、路由分布、计时、环境记录 | — |

模型预设 `(hidden, experts, top-k)`：`qwen3-30b-a3b (2048, 128, 8)`、`qwen3-235b-a22b (4096, 128, 8)`、`deepseek-v3 (7168, 256, 8)`。

路由分布：

- `random`：均匀 top-k。
- `balanced`：每个 token 均匀落到各 rank。
- `zipf`：expert 热度服从 1/rank，rank 0 过载。
- `hot_rank0`：全部 token 只路由到 rank 0。

## 指标口径

- **计时**：同步的 host 墙钟时间。每轮先 barrier，再连续调用 N 次，最后做一次 device sync，取各 rank 的最大值（即关键路径）除以 N；报告 5 轮的中位数、最小值、最大值。N 自适应，每轮目标约 0.5 s。
- **e2e**：调用 `Buffer` 的 API，包含 host 打包和 host↔device 拷贝，也就是真实用户看到的时间。
- **kernel（pipelined）**：用 `Transport.resident_call` 回放这次交换的精确 shape，数据常驻设备，连续发射。这是摊销后的吞吐口径。
- **kernel latency**：同上，但每次调用后都 sync。这是单次延迟，decode 时每层都要等通信完成，看的是这个数。
- **分阶段（breakdown）**：每个阶段边界都 sync（`deep_ep_trn.profiling.timer`）。阶段包括：

  | 阶段 | 含义 |
  |---|---|
  | `in_d2h` | 输入拷回 host |
  | `plan` | 路由规划 |
  | `pack` | 行打包 |
  | `h2d` | 发送 buffer 上设备 |
  | `kernel` | a2av |
  | `d2h` | 接收 buffer 回 host |
  | `unpack` | 解包 |
  | `out_h2d` | 输出拷回调用方设备 |

  因为插入了 sync，各阶段之和会大于 e2e，只用于看比例。
- **字节**：
  - logical：按 DeepEP 口径。normal 模式是接收行数 × H × 2；LL 模式是被选中的 (token, expert) 数 × H × 2，与 FP8 无关，统一按 BF16 计。
  - wire：实际发出的行数 × 行宽（含元数据列）× 2。
  - recv buffer：接收 buffer 的大小。驱动要求它是 `EP × 发送容量`，所以通常远大于 payload。
- **load imbalance**：EP 组内各 rank 接收量的 max / mean。

## 注意

- 首次调用耗时包含 NEFF 编译，cache 热了之后会快很多。
- trn2.3xlarge 只有单芯片 4 个逻辑核，所有数据都在芯片内搬运；跨机（EFA）的性能这里测不到。
- baselines 和 kernel 回放都是常驻设备的数，可以直接对比；e2e 包含 host 开销，不能拿来和它们对比。
