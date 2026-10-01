# DeepEP 在 trn2 上的特化验证

机器：trn2.3xlarge（1 颗 Trainium2，LNC=2 → 4 个逻辑 NeuronCore，96 GB HBM）
容器：`sgl-plugin-dev`（torch-neuronx 2.14.3 / nki 0.7.0b1 / neuronx-cc 2.0.404830）
本目录在容器内路径：`/workspace_user/artifacts/deepep-trn2-20261001`

## 平台约束（决定了"特化"必须长什么样）

| 事实 | 出处 | 影响 |
|---|---|---|
| trn2 单机组内 `ncc.all_to_all_v` 只支持 `has_rdispls=False`：每个源 rank 在接收端占一个等长固定槽位（`dst.numel()/EP`） | `nki/collectives/_ops.py` docstring | 单机只能用 **DeepEP-LL 式固定槽位 buffer**；DeepEP normal 式的紧凑 packed 接收（rdispls）只在跨机组可用 |
| 本 runtime 下 `has_rdispls=False` 时要求 dst 容量 = EP × src 容量 | 之前 probe-raw-v3 实测 | combine 的接收 buffer 会被放大 EP 倍（stage 2 再探测能否缩小） |
| nkilib `permute_a2av / unpermute_a2av` 只支持 Trn3 LNC=2、EP≥8、≥2 个 device | `a2av_train_utils._validate_trn3_a2av_group_size` | 现成 kernel 在 trn2.3xlarge 上直接 assert 失败，需要自己写 trn2 版本 |
| `reduce_scatter_v`：仅 LNC=2、每组正好 4 rank（单芯片）、每 rank 输出 ≤16 KiB | 同上 docstring | 可作为 combine 的备选原语（stage 2 评估） |
| 当前 sglang-neuron EP 路径是 `all_expert=True` + all-reduce（attention DP1，每个 rank 持有全部 token） | `sglang_neuron/nki/moe.py`、`bench_tp_ep.py` | 只有 attention 改成 DP（各 rank 持有不同 token）后，a2a dispatch/combine 才有意义 |

## 分阶段计划

- [x] **Stage 0 能力探测**（`../moe-ep-transport-b64-20261001`）：原始 `all_to_all_v` EP=2 正确；nkilib a2av_train kernel 在 trn2 上被拒。
- [x] **Stage 1 LL 式传输语义**（本目录 `test_ll_dispatch_combine.py`）：dispatch → 接收端模拟 expert → combine 回源端求和，端到端校验。路由打包、expert 在 CPU 上做，只有两次集合通信在设备上跑。
- [ ] **Stage 2 设备端 dispatch/combine kernel**：路由元数据（counts/displs/send indices）在设备上从 top-k 生成（参考 nkilib `build_all2all_dispatch_metadata`、`permute_routed_tokens`、`topk_reduce`）；去掉 SBUF 双重中转，直接写进集合通信 buffer；把拷贝拆给 LNC2 的两个物理核；探测 combine 接收 buffer 能否小于 EP×src；评估 `reduce_scatter_v` 做 combine。
- [ ] **Stage 3 接 expert 计算**：接收槽位 → selective MoE kernel（非 all_expert）；单层 MoE 端到端对比当前 all_expert + all-reduce 基线（B64 单层 945 µs，见 stage 0 baseline）；按 token 数扫描找交叉点。
- [ ] **Stage 4 接入 sglang**：在 sglang-neuron 里实现 Neuron 版 dispatcher（对齐 `token_dispatcher/deepep.py` 的 `dispatch_a/b`、`combine_a/b` 和 `DeepEPLLDispatchOutput` 格式），并需要 attention DP。
- [ ] **Stage 5 跨机**（需要 ≥2 台 trn2.48xlarge + EFA，本机做不了）：跨机组上用 `has_rdispls=True` 做 DeepEP normal 式紧凑传输——这是 DeepEP 真正产生价值的场景。

## Stage 1 结果（2026-10-01）

配置：H=2048，E=128，top-k=8，BF16；每 rank 的 token 数 = 64/EP（全局 B64）。每个 (token, 目的 rank) 只发一行（rank 级去重），top-k 权重 / 全局 expert id / 源 token 下标按位打包在每行末尾的 64 列里。

正确性：EP=4（组 `[[0,1,2,3]]`）和 EP=2（组 `[[0,2],[1,3]]`）都通过 `balanced / random / hot_rank0 / ragged` 四种路由——dispatch 负载逐位一致，runtime 回写的 recv counts 与期望一致，combine 相对误差 ≤1.5e-3（bf16 量级）。覆盖了 0 count 的 peer、全部打到一个 rank、各 rank token 数不同、以及完全空闲的源 rank。

耗时（host 墙钟摊销，每轮取各 rank 最大值，5 轮 × 100 次的中位数，`random` 路由）：

| | EP=4，T=16/rank | EP=2，T=32/rank |
|---|---|---|
| dispatch a2av | 213.7 µs | 173.6 µs |
| combine a2av | 215.9 µs | 170.8 µs |
| 基线：EP 组内 all-reduce [64, 2048] | 166.0 µs | 141.5 µs |

解读注意：

- 这是单独 launch 的 NEFF，包含每次调用的 host launch 开销，以及 kernel 内两次 SBUF 中转拷贝，不是纯传输时间；需要用 `neuron-profile` 拆开看。
- B64 的负载很小（约 256 KB），完全受延迟主导；dispatch + combine（约 430 µs @EP4）比一次 all-reduce 慢。所以这组数**不能**说明 a2a 路线不行，只说明在 B64 单芯片上，固定开销是主要矛盾。下一步要按 token 数扫描。

## 复现

```bash
sudo docker exec sgl-plugin-dev bash -lc 'bash /workspace_user/artifacts/deepep-trn2-20261001/run_stage1.sh 4 2'
# 结果：stage1-ep{4,2}/rank-*-result.json，日志：stage1-ep{4,2}.log，退出码：status.txt
```
