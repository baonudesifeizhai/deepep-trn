# trn2 platform constraints (measured)

Environment: trn2.3xlarge (1 Trainium2 chip, LNC=2 → 4 logical cores), torch-neuronx 2.14.3, nki 0.7.0b1, neuronx-cc 2.0.404830, aws-neuronx-runtime-lib 2.x.79381, aws-neuronx-collectives 2.x.78358.

| # | Constraint | Evidence | How the framework handles it |
|---|---|---|---|
| 1 | Intra-node `ncc.all_to_all_v` only supports `has_rdispls=False`: data from source `s` always lands at `dst[s * dst.numel()/EP : ...]` | Docstring in `nki/collectives/_ops.py`: "has_rdispls=True is currently only supported on inter-node replica groups" | Every exchange uses fixed slots (`transport/base.py`) |
| 2 | Without rdispls, the driver requires `dst` to be exactly `EP × src` | `tools/probe_transport.py --short-slots 1` fails at load: `Invalid variable-size collective instruction without recv_displ: output size 811008 bytes must be rank_n * input size (4 * 811008 = 3244032 bytes)` | `NkiA2avTransport.slot_rows` returns the whole send capacity; the buffer layer keeps send capacity minimal (`M·min(K,R)` for dispatch, `R·M` for combine) |
| 3 | nkilib `permute_a2av / unpermute_a2av` only support Trn3 LNC=2 with EP ≥ 8 across ≥ 2 devices | `a2av_train_utils._validate_trn3_a2av_group_size`; the stage-0 dispatch/combine probes were rejected by that assert | Own kernel, `_a2av_fixed_slot_kernel` |
| 4 | Collective src/dst must be kernel-internal `shared_hbm` tensors, not graph IO tensors | Comparison of stage-0 probe-raw v2 and v3 | The kernel copies into internal HBM first and copies out after the exchange |
| 5 | `reduce_scatter_v`: LNC=2 only, exactly 4 ranks per group (one chip), ≤ 16 KiB output per rank | Docstring in `_ops.py` | Not used yet; a candidate for combine when B is very small |
| 6 | The kernel is written for LNC=2 (`grid=(2,)`, `core_barrier((0,1))`) | — | `NkiA2avTransport` checks `NEURON_LOGICAL_NC_CONFIG` |
| 7 | NKI resolves module globals, not closure cells | Comment in the stage-0 probe | One replica-group layout per process (`_install_replica_group`) |
| 8 | Each host↔device round trip costs ~0.7–3.5 ms (the driver lacks async IO, so `nrta_tensor_read/write` take a fallback path) | WARN lines in the run logs; in `bench/bench_transport.py`, the gap between `kernel` and `with host copies` | The main v0 bottleneck; roadmap stage 2 |

To re-check on a new runtime:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=4 tools/probe_transport.py --ep 4 --short-slots 1
```
