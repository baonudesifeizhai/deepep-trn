#!/usr/bin/env bash
# Stage 1: DeepEP-LL-style dispatch/combine transport on one trn2 chip.
# Usage (host): sudo docker exec sgl-plugin-dev bash -lc 'bash /workspace_user/artifacts/deepep-trn2-20261001/run_stage1.sh [ep...]'
set -uo pipefail
ROOT=/workspace_user/artifacts/deepep-trn2-20261001
cd "$ROOT"
[ -f /opt/torch-neuronx/.venv/bin/activate ] && source /opt/torch-neuronx/.venv/bin/activate
export OMP_NUM_THREADS=1 NEURON_RT_VISIBLE_CORES=0-3 NEURON_RT_EXEC_TIMEOUT=60
export NKI_ENABLE_TRACE_CACHE=0 NEURON_PLATFORM_TARGET_OVERRIDE=trn2
unset TORCH_NEURONX_NEFF_CACHE_DIR
for ep in "${@:-4 2}"; do
    timeout --signal=TERM --kill-after=20s 600s python -m torch.distributed.run --standalone --nproc-per-node=4 \
        test_ll_dispatch_combine.py --ep "$ep" --bench --output "$ROOT/stage1-ep$ep" > "$ROOT/stage1-ep$ep.log" 2>&1
    echo "stage1_ep${ep}_exit=$?" | tee -a "$ROOT/status.txt"
done
