#!/usr/bin/env bash
# Benchmark suite on a trn2 host. From the host:
#   sudo docker exec -w /workspace_user/user/deepep-trn sgl-plugin-dev bash bench/run_suite.sh [quick|default]
# Results: results/<RUN_ID>/{env.json,*.jsonl,summary.md,ops.csv}; logs alongside.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env_trn2.sh
SUITE=${1:-default}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$SUITE}
OUT=results/$RUN_ID
mkdir -p "$OUT"
fail=0
run() {
    local name=$1; shift
    timeout "${BENCH_TIMEOUT:-3600}" python -m torch.distributed.run --standalone --nproc-per-node=4 "$@" \
        --out-dir "$OUT" > "$OUT/$name.log" 2>&1
    local rc=$?
    echo "$name exit=$rc"
    grep -h -E '^\[(ops|transport|baseline)\]' "$OUT/$name.log" | sed 's/^/    /'
    [ "$rc" -eq 0 ] || fail=1
}
case "$SUITE" in
quick)
    run transport_ep4 bench/bench_transport.py --ep 4 --rows 8,128 --fills 1.0
    run baselines_ep4 bench/bench_baselines.py --ep 4 --tokens 16,64
    run ops_ep4 bench/bench_ops.py --ep 4 --tokens 16,64
    ;;
default)
    for ep in 4 2; do
        run transport_ep$ep bench/bench_transport.py --ep $ep --presets qwen3-30b-a3b,deepseek-v3
        run baselines_ep$ep bench/bench_baselines.py --ep $ep --presets qwen3-30b-a3b,deepseek-v3 \
            --tokens 16,32,64,128
        run ops_ep${ep}_sweep bench/bench_ops.py --ep $ep --tokens 16,32,64,128
        run ops_ep${ep}_routing bench/bench_ops.py --ep $ep --tokens 64 --routings balanced,zipf,hot_rank0
        run ops_ep${ep}_dsv3 bench/bench_ops.py --ep $ep --presets deepseek-v3 --tokens 16,64
    done
    ;;
*)
    echo "unknown suite $SUITE (quick|default)"; exit 2 ;;
esac
python bench/report.py "$OUT"
exit $fail
