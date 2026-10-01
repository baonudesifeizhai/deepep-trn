# Validating a DeepEP specialization on trn2 (stage 1, archived)

> Archived snapshot from 2026-10-01. Later findings supersede parts of it. Notably, the receive buffer cannot be smaller than `EP × src` (the driver rejects it); see `docs/trn2-constraints.md`.

Machine: trn2.3xlarge (1 Trainium2 chip, LNC=2 → 4 logical NeuronCores, 96 GB HBM)
Container: `sgl-plugin-dev` (torch-neuronx 2.14.3 / nki 0.7.0b1 / neuronx-cc 2.0.404830)
Original path inside the container: `/workspace_user/artifacts/deepep-trn2-20261001`

## Platform constraints (these decide what the specialization has to look like)

| Fact | Source | Impact |
|---|---|---|
| Intra-node `ncc.all_to_all_v` on trn2 only supports `has_rdispls=False`: each source rank gets one fixed, equally sized slot on the receiver (`dst.numel()/EP`) | `nki/collectives/_ops.py` docstring | A single node can only use a **DeepEP-LL-style fixed-slot buffer**; DeepEP-normal-style compact packed receives (rdispls) only work on inter-node groups |
| On this runtime, `has_rdispls=False` requires dst capacity = EP × src capacity | Earlier probe-raw-v3 measurement | The combine receive buffer is amplified EP× (stage 2 was to probe whether it can shrink) |
| nkilib `permute_a2av / unpermute_a2av` only support Trn3 LNC=2, EP ≥ 8, ≥ 2 devices | `a2av_train_utils._validate_trn3_a2av_group_size` | The stock kernels assert on trn2.3xlarge, so a trn2 version has to be written |
| `reduce_scatter_v`: LNC=2 only, exactly 4 ranks per group (one chip), ≤ 16 KiB output per rank | Same docstring | A possible alternative primitive for combine (to evaluate in stage 2) |
| The current sglang-neuron EP path is `all_expert=True` + all-reduce (attention DP1, every rank holds all tokens) | `sglang_neuron/nki/moe.py`, `bench_tp_ep.py` | a2a dispatch/combine only makes sense once attention is DP (each rank holds different tokens) |

## Staged plan

- [x] **Stage 0, capability probe** (`../moe-ep-transport-b64-20261001`): raw `all_to_all_v` is correct at EP=2; the nkilib a2av_train kernels are rejected on trn2.
- [x] **Stage 1, LL-style transport semantics** (`test_ll_dispatch_combine.py` here): dispatch → expert emulated on the receiver → combine summed back at the source, checked end to end. Routing, packing and the expert run on the CPU; only the two collectives run on the device.
- [ ] **Stage 2, device-side dispatch/combine kernels**: generate routing metadata (counts / displs / send indices) on the device from top-k (references: nkilib `build_all2all_dispatch_metadata`, `permute_routed_tokens`, `topk_reduce`); remove the double SBUF staging and write straight into the collective buffers; split copies across the two physical cores of LNC2; probe whether the combine receive buffer can be smaller than EP×src; evaluate `reduce_scatter_v` for combine.
- [ ] **Stage 3, expert compute**: receive slots → selective MoE kernel (not all_expert); end-to-end single MoE layer against the current all_expert + all-reduce baseline (945 µs per layer at B64, see the stage-0 baseline); sweep token counts to find the crossover.
- [ ] **Stage 4, sglang integration**: a Neuron dispatcher in sglang-neuron matching `dispatch_a/b`, `combine_a/b` and the `DeepEPLLDispatchOutput` format in `token_dispatcher/deepep.py`; requires attention DP.
- [ ] **Stage 5, inter-node** (needs ≥ 2 × trn2.48xlarge + EFA; not possible on this machine): DeepEP-normal-style compact transfers with `has_rdispls=True` on inter-node groups. This is where DeepEP delivers its real value.

## Stage 1 results (2026-10-01)

Config: H=2048, E=128, top-k=8, BF16; tokens per rank = 64/EP (global B64). One row per (token, destination rank) (rank-level dedup), with top-k weights / global expert ids / source token index bit-packed into the last 64 columns of each row.

Correctness: EP=4 (group `[[0,1,2,3]]`) and EP=2 (groups `[[0,2],[1,3]]`) pass all four routings (`balanced / random / hot_rank0 / ragged`):

- dispatch payloads match bit for bit;
- the recv counts written back by the runtime match expectations;
- combine relative error ≤ 1.5e-3 (bf16 level).

Covered edge cases: peers with zero count, everything routed to one rank, different token counts per rank, and a fully idle source rank.

Latency: amortized host wall-clock, max across ranks per round, median of 5 rounds × 100 calls, `random` routing.

| | EP=4, T=16/rank | EP=2, T=32/rank |
|---|---|---|
| dispatch a2av | 213.7 µs | 173.6 µs |
| combine a2av | 215.9 µs | 170.8 µs |
| Baseline: all-reduce [64, 2048] on the EP group | 166.0 µs | 141.5 µs |

Caveats:

- Each op is a separately launched NEFF, so the numbers include per-call host launch overhead and the two SBUF staging copies inside the kernel. They are not pure transfer time; `neuron-profile` is needed to separate them.
- The B64 payload is tiny (~256 KB) and fully latency-bound. dispatch + combine (~430 µs at EP4) is slower than one all-reduce. These numbers do **not** show the a2a path is a dead end; they only show that at B64 on one chip, fixed overhead is the main problem. Next step: sweep token counts.

## Reproduce

```bash
sudo docker exec sgl-plugin-dev bash -lc 'bash /workspace_user/artifacts/deepep-trn2-20261001/run_stage1.sh 4 2'
# Results: stage1-ep{4,2}/rank-*-result.json; logs: stage1-ep{4,2}.log; exit codes: status.txt
```
