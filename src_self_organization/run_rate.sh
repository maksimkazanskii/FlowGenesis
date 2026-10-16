#!/usr/bin/env bash
# Rate x initial-size ablation for the rebuilt train_neurogenesis_ablation.py.
# Run from anywhere: bash src_self_organization/run_rate.sh [start|stop]
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

RATES=(5 1 2.5 10)
N0_VALUES=(20 40 80 160)
SEEDS=(0 1 2 3 4)
DATASET="${DATASET:-fashionmnist}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/neurogenesis/rate_ablations}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"
RUN_DIR="${OUTPUT_ROOT}/.run_control"
PID_FILE="$RUN_DIR/main.pid"
CHILD_FILE="$RUN_DIR/children.pid"

stop_all() {
    if [[ -f "$CHILD_FILE" ]]; then
        while IFS= read -r pid; do
            [[ -n "$pid" ]] || continue
            # Children are subshells; kill their process group descendants via pkill -P.
            pkill -TERM -P "$pid" 2>/dev/null || true
            kill -TERM "$pid" 2>/dev/null || true
        done < "$CHILD_FILE"
    fi
    if [[ -f "$PID_FILE" ]]; then
        pid="$(cat "$PID_FILE")"
        [[ "$pid" == "$$" ]] || kill -TERM "$pid" 2>/dev/null || true
    fi
}

if [[ "${1:-start}" == stop ]]; then
    if [[ ! -f "$PID_FILE" ]]; then
        echo "No recorded rate-ablation run in $RUN_DIR"
        exit 0
    fi
    stop_all
    echo "Stop requested."
    exit 0
fi
if [[ "${1:-start}" != start ]]; then
    echo "Usage: bash $0 [start|stop]" >&2
    exit 2
fi

mkdir -p "$RUN_DIR" "$LOG_DIR"
if [[ -f "$PID_FILE" ]]; then
    old_pid="$(cat "$PID_FILE")"
    if kill -0 "$old_pid" 2>/dev/null; then
        echo "Already running with PID $old_pid; use: bash $0 stop" >&2
        exit 1
    fi
fi
printf '%s\n' "$$" > "$PID_FILE"
: > "$CHILD_FILE"
children=()
cleanup() {
    trap - EXIT INT TERM
    for pid in "${children[@]-}"; do
        pkill -TERM -P "$pid" 2>/dev/null || true
        kill -TERM "$pid" 2>/dev/null || true
    done
    rm -f "$PID_FILE" "$CHILD_FILE"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

printf 'Dataset=%s | output=%s | total=%s runs\n' "$DATASET" "$OUTPUT_ROOT" "$((${#RATES[@]} * ${#N0_VALUES[@]} * ${#SEEDS[@]}))"
for seed in "${SEEDS[@]}"; do
    for rate in "${RATES[@]}"; do
        echo "=== Starting rate=$rate seed=$seed; ${#N0_VALUES[@]} parallel N0 runs ==="
        children=()
        : > "$CHILD_FILE"
        for n0 in "${N0_VALUES[@]}"; do
            log="$LOG_DIR/rate_${rate}_N${n0}_seed${seed}.log"
            (
                set -o pipefail
                python -u -m src_self_organization.train_neurogenesis_ablation \
                    --dataset "$DATASET" \
                    --hidden_neurons "$n0" \
                    --seed "$seed" \
                    --ablation rate \
                    --value "$rate" \
                    --output_dir "$OUTPUT_ROOT" \
                    2>&1 | tee "$log" | sed -u "s/^/[rate=$rate N0=$n0 seed=$seed] /"
            ) &
            pid=$!
            children+=("$pid")
            echo "$pid" >> "$CHILD_FILE"
            echo "Started rate=$rate N0=$n0 seed=$seed PID=$pid"
        done
        failed=0
        for pid in "${children[@]-}"; do
            wait "$pid" || failed=1
        done
        children=()
        : > "$CHILD_FILE"
        if (( failed )); then
            echo "ERROR: one or more runs failed for rate=$rate seed=$seed. See $LOG_DIR" >&2
            exit 1
        fi
        echo "=== Completed rate=$rate seed=$seed ==="
    done
done
echo "All $((${#RATES[@]} * ${#N0_VALUES[@]} * ${#SEEDS[@]})) rate x N0 experiments completed."