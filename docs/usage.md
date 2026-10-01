# Usage

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

## Differences from deep_ep

- **Constructor**: byte-size arguments such as `num_nvl_bytes` / `num_rdma_bytes` are ignored. trn buffers are compile-time static shapes, so `num_max_tokens_per_rank` is required and must be identical on every rank.
- **FP8 in LL mode**: deep_ep's `low_latency_dispatch` defaults to `use_fp8=True`; here that raises `NotImplementedError`, so pass `use_fp8=False` explicitly.
- **Wire format**: one row per (token, destination rank), with top-k weights, expert ids and the source token index bit-packed at the end of the row, so each dispatch is a single collective. LL dispatch is also rank-deduplicated and the receiver expands rows into `[EL, R*M, H]` (deep_ep LL sends once per (token, expert)), so fewer bytes go over the wire.
- **LL combine**: same semantics as deep_ep: one row per (token, expert) goes back, and the source applies top-k weights and sums.
- **Receive order**: normal mode orders rows by (source rank, source token). In LL mode each expert's rows follow the same order, and `handle.layout_range` gives each source's range (`start << 32 | count`).
- **Events and hooks**: `EventOverlap` is a stub and hooks are no-ops; Neuron work is already ordered on the device queue.
