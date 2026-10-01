# trn2 平台约束（实测）

环境：trn2.3xlarge（1 颗 Trainium2，LNC=2 → 4 个逻辑核），torch-neuronx 2.14.3、nki 0.7.0b1、neuronx-cc 2.0.404830、aws-neuronx-runtime-lib 2.x.79381、aws-neuronx-collectives 2.x.78358。

| # | 约束 | 证据 | 框架里的处理 |
|---|---|---|---|
| 1 | 单机组内 `ncc.all_to_all_v` 只支持 `has_rdispls=False`：来自源 `s` 的数据固定落在 `dst[s * dst.numel()/EP : ...]` | `nki/collectives/_ops.py` 的 docstring："has_rdispls=True is currently only supported on inter-node replica groups" | 所有交换都走固定槽位（`transport/base.py`） |
| 2 | 没有 rdispls 时，驱动要求 `dst` 正好是 `EP × src` | `tools/probe_transport.py --short-slots 1` 加载失败：`Invalid variable-size collective instruction without recv_displ: output size 811008 bytes must be rank_n * input size (4 * 811008 = 3244032 bytes)` | `NkiA2avTransport.slot_rows` 返回整个发送容量；buffer 层把发送容量压到最小（dispatch 为 `M·min(K,R)`，combine 为 `R·M`） |
| 3 | nkilib `permute_a2av / unpermute_a2av` 只支持 Trn3 LNC=2，且 EP≥8、≥2 个 device | `a2av_train_utils._validate_trn3_a2av_group_size`；stage-0 的 dispatch/combine probe 被 assert 拒绝 | 自写 `_a2av_fixed_slot_kernel` |
| 4 | 集合通信的 src/dst 必须是 kernel 内部的 `shared_hbm` tensor，不能直接用 graph IO tensor | stage-0 probe-raw v2 和 v3 的对比 | kernel 先拷进内部 HBM，交换完再拷出 |
| 5 | `reduce_scatter_v`：仅 LNC=2，每组正好 4 rank（单芯片），每 rank 输出 ≤16 KiB | `_ops.py` 的 docstring | 还没用；B 很小时可作为 combine 的备选 |
| 6 | kernel 按 LNC=2 编写（`grid=(2,)`，`core_barrier((0,1))`） | — | `NkiA2avTransport` 检查 `NEURON_LOGICAL_NC_CONFIG` |
| 7 | NKI 解析的是模块全局变量，不是闭包变量 | stage-0 probe 的注释 | 每个进程只能有一套 replica-group 布局（`_install_replica_group`） |
| 8 | 每次 host↔device 往返约 0.7–3.5 ms（驱动缺少 async IO，`nrta_tensor_read/write` 走回退路径） | 运行日志 WARN；`bench_buffer.py` 中 transport 与 kernel_resident 的差值 | v0 的主要瓶颈；roadmap stage 2 |

重新验证新 runtime：

```bash
python -m torch.distributed.run --standalone --nproc-per-node=4 tools/probe_transport.py --ep 4 --short-slots 1
```
