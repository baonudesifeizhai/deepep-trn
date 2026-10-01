# Roadmap

## v0.1（当前）
API 对齐 deep_ep，正确性已在 trn2 上验证。路由、打包、解包在 host 上做（torch CPU），只有集合通信跑在设备上。

## Stage 2：数据常驻设备（性能的第一优先级）
v0 端到端耗时为毫秒级，几乎全部花在 host 打包和 host↔device 拷贝上；常驻设备的 kernel 本身只要约 200–300 µs。

1. 打包和解包在设备上做，每个 op 编译成一个 NEFF：元数据构建 → 行打包 → a2av → 解包。
   可参考 nkilib：`build_all2all_dispatch_metadata`、`permute_routed_tokens`、`topk_reduce`。
2. 去掉 kernel 里两次 SBUF 中转：打包直接写进集合通信的发送 buffer，解包直接读接收 buffer。
3. 把拷贝和打包拆给 LNC=2 的两个物理核。
4. kernel 只拷出有效行（按 recv counts 做动态 DMA），不再整块拷 `EP × 容量` 的接收 buffer。
   benchmark 显示填充率 25% 时耗时只降 20–40%，说明容量拷贝是主要成本。
5. LL combine 在接收端先归约：dispatch 时顺带发 top-k 权重，回传时每个 (token, rank) 只发一行。
   回传 buffer 能缩小 `min(K, EL)` 倍。这是 trn 扩展，需要显式开关。
6. 评估用 `reduce_scatter_v` 做 combine（≤16 KiB/rank 的限制只适合很小的 B）。
7. Buffer 层的 shape 预热（把会用到的 shape 提前编译），并持久化 NEFF cache。

## Stage 3：接 expert 计算
- 接收布局对接 sglang-neuron / nkilib 的 selective MoE kernel（不走 all_expert）。
- 单层 MoE 端到端，对比当前 all_expert + all-reduce（B64 单层 945 µs），按 token 数扫描找交叉点。
- dispatch → expert → combine 放进同一个 NEFF。单芯片上每次集合通信 launch 有约 300–500 µs 的单次延迟，
  两次独立 launch 在 decode 尺寸下比单图的 AG+RS 慢约 2.5 倍（`results/20261001-default`）。

## Stage 4：接入 sglang
- 在 sglang-neuron 实现 Neuron dispatcher，对齐 `token_dispatcher/deepep.py` 的 `dispatch_a/b`、`combine_a/b` 和 `DeepEPNormal/LL*` 输出格式。
- 前提是 attention DP：每个 rank 持有不同的 token。

## Stage 5：跨机与其它
- 跨机组（≥2 台 trn2.48xlarge，EFA）上用 `has_rdispls=True` 做 DeepEP normal 式的紧凑传输。
- 机内 + 机间两级 dispatch（对应 DeepEP 的 NVLink + RDMA）。
- FP8 payload（`(x, scales)`），LogFMT combine。
- 真正的 async / recv hook overlap。
