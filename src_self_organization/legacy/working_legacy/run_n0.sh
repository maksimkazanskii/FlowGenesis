#!/usr/bin/env bash

# N0 ablation:
#   N0 = 10, 20, 40, 80, 160
#   seeds = 0, 1, 2, 3, 4, 5
#
# Five N0 values run in parallel for each seed.
# Seeds are processed sequentially.
# Output is printed live with [N0=... seed=...] prefixes.
#
# Stop everything with:
#   bash src_self_organization/run_n0.sh stop

set -u

N0_VALUES=(10 20 40 80 160)
SEEDS=(0 1 2 3 4 5)
OUTPUT_DIR="N0_ablation"

PID_FILE=".n0_ablation.pid"
CHILD_PID_FILE=".n0_ablation.children"

stop_all() {
    echo "Stopping N0 ablation..."

    if [[ -f "$CHILD_PID_FILE" ]]; then
        while read -r PID; do
            if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
                kill "$PID" 2>/dev/null || true
            fi
        done < "$CHILD_PID_FILE"
    fi

    if [[ -f "$PID_FILE" ]]; then
        PID="$(cat "$PID_FILE" 2>/dev/null || true)"
        if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
            kill "$PID" 2>/dev/null || true
        fi
    fi

    rm -f "$PID_FILE" "$CHILD_PID_FILE"
    echo "Stop signal sent to all recorded N0 ablation processes."
    exit 0
}

if [[ "${1:-}" == "stop" ]]; then
    stop_all
fi

if [[ -f "$PID_FILE" ]]; then
    OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "N0 ablation already appears to be running (PID $OLD_PID)."
        echo "Stop it with: bash $0 stop"
        exit 1
    fi
fi

echo "$$" > "$PID_FILE"
: > "$CHILD_PID_FILE"

cleanup() {
    # Stop any children still alive if the master exits/interrupted.
    if [[ -f "$CHILD_PID_FILE" ]]; then
        while read -r PID; do
            if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
                kill "$PID" 2>/dev/null || true
            fi
        done < "$CHILD_PID_FILE"
    fi
    rm -f "$PID_FILE" "$CHILD_PID_FILE"
}
trap cleanup EXIT INT TERM

echo "============================================================"
echo "N0 ABLATION"
echo "N0 values : ${N0_VALUES[*]}"
echo "Seeds     : ${SEEDS[*]}"
echo "Output arg: $OUTPUT_DIR"
echo "============================================================"

for seed in "${SEEDS[@]}"; do
    echo
    echo "============================================================"
    echo "STARTING SEED $seed"
    echo "============================================================"

    pids=()

    for n0 in "${N0_VALUES[@]}"; do
        python -u -m src_self_organization.train_neurogenesis_ablation \
            --hidden_neurons "$n0" \
            --seed "$seed" \
            --output_dir "$OUTPUT_DIR" \
            > >(sed -u "s/^/[N0=$n0 seed=$seed] /") 2>&1 &
        child_pid=$!
        pids+=("$child_pid")
        echo "$child_pid" >> "$CHILD_PID_FILE"
    done

    failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            failed=1
        fi
    done

    if [[ "$failed" -ne 0 ]]; then
        echo "ERROR: at least one run failed for seed $seed."
        exit 1
    fi

    : > "$CHILD_PID_FILE"
    echo "=== SEED $seed FINISHED ==="
done

echo
echo "============================================================"
echo "ALL N0 ABLATION RUNS FINISHED"
echo "============================================================"