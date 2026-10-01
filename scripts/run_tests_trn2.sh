#!/usr/bin/env bash
# Full correctness suite on a trn2 host (unit + gloo + nki, EP 4 and 2). From the host:
#   sudo docker exec -w /workspace_user/user/deepep-trn sgl-plugin-dev bash scripts/run_tests_trn2.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env_trn2.sh
LOG=${LOG_DIR:-/tmp/deep_ep_trn_logs}
mkdir -p "$LOG"
fail=0
run() {
    local name=$1; shift
    timeout 900 "$@" > "$LOG/$name.log" 2>&1
    local rc=$?
    echo "$name exit=$rc"
    [ "$rc" -eq 0 ] || fail=1
}
run unit python -m pytest -q -p no:cacheprovider tests/unit
for backend in gloo nki; do
    for test in intranode low_latency; do
        for ep in 4 2; do
            run "${test}_${backend}_ep${ep}" python -m torch.distributed.run --standalone \
                --nproc-per-node=4 tests/dist/test_${test}.py --backend "$backend" --ep "$ep"
        done
    done
done
echo "logs: $LOG"
exit $fail
