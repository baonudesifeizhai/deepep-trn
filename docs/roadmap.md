# Roadmap

## v0.1 (current)
The API mirrors deep_ep and correctness is verified on trn2. Routing, packing and unpacking run on the host (torch CPU); only the collectives run on the device.

## Stage 2: keep data on the device (top performance priority)
v0 end-to-end latency is in milliseconds and almost all of it goes to host packing and host↔device copies; the device-resident kernel itself takes only ~200–300 µs.

1. Pack and unpack on the device, compiling each op into one NEFF: metadata build → row packing → a2av → unpacking.
   Useful nkilib references: `build_all2all_dispatch_metadata`, `permute_routed_tokens`, `topk_reduce`.
2. Remove the two SBUF staging copies in the kernel: pack straight into the collective's send buffer and unpack straight from the receive buffer.
3. Split copies and packing across the two physical cores of LNC=2.
4. Copy out only the valid rows (dynamic DMA driven by recv counts) instead of the whole `EP × capacity` receive buffer.
   The benchmarks show a 25% fill ratio only cuts time by 20–40%, so capacity-sized copies are the main cost.
5. Reduce LL combine on the receiver first: send the top-k weights along with dispatch, then return one row per (token, rank).
   This shrinks the return buffer by `min(K, EL)`×. It is a trn extension and needs an explicit opt-in.
6. Evaluate `reduce_scatter_v` for combine (its ≤ 16 KiB/rank limit only fits very small B).
7. Shape warm-up in the Buffer layer (precompile the shapes that will be used) and a persistent NEFF cache.

## Stage 3: wire up expert compute
- Feed the receive layout into the sglang-neuron / nkilib selective MoE kernel (not all_expert).
- End-to-end single MoE layer against the current all_expert + all-reduce path (945 µs per layer at B64), sweeping token counts to find the crossover.
- Put dispatch → expert → combine in one NEFF. Each on-chip collective launch has ~300–500 µs of per-call latency, and two separate launches are ~2.3–2.8× slower than a single-graph AG+RS at decode sizes (`results/20261001-default`).

## Stage 4: integrate with sglang
- Implement a Neuron dispatcher in sglang-neuron matching `dispatch_a/b`, `combine_a/b` and the `DeepEPNormal/LL*` output formats in `token_dispatcher/deepep.py`.
- Requires attention DP, so that each rank holds different tokens.

## Stage 5: inter-node and more
- On inter-node groups (≥ 2 × trn2.48xlarge, EFA), use `has_rdispls=True` for DeepEP-normal-style compact transfers.
- Two-level dispatch, intra-node + inter-node (DeepEP's NVLink + RDMA counterpart).
- FP8 payloads (`(x, scales)`), LogFMT combine.
- Real async / recv-hook overlap.
