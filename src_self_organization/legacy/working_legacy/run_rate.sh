#!/usr/bin/env bash

# FlowGenesis structural-rate x initial-size grid ablation
#
#   rate = max_births_per_epoch = max_deaths_per_epoch
#   rate values: 2, 4, 8, 16, 32
#   N0 values:   10, 20, 40, 80, 160
#   seeds:       0, 1, 2, 3, 4
#
# For each seed and rate, all five N0 values run in parallel.
# Rates and seeds are processed sequentially.
#
# Stop with:
#   bash src_self_organization/run_rate.sh stop

set -u

RATE_VALUES=(2 4 8 16 32)
N0_VALUES=(10 20 40 80 160)
SEEDS=(0 1 2 3 4)

OUTPUT_ROOT="rate_ablations"

PID_FILE=".rate_ablation.pid"
CHILD_PID_FILE=".rate_ablation.children"

stop_all() {
    echo "Stopping rate x N0 ablation..."

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
    echo "Stop signal sent to all recorded processes."
    exit 0
}

if [[ "${1:-}" == "stop" ]]; then
    stop_all
fi

if [[ -f "$PID_FILE" ]]; then
    OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Rate x N0 ablation already appears to be running (PID $OLD_PID)."
        echo "Stop it with: bash $0 stop"
        exit 1
    fi
fi

echo "$$" > "$PID_FILE"
: > "$CHILD_PID_FILE"

cleanup() {
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
echo "STRUCTURAL RATE x N0 GRID ABLATION"
echo "Rates : ${RATE_VALUES[*]}"
echo "N0    : ${N0_VALUES[*]}"
echo "Seeds : ${SEEDS[*]}"
echo "Total : $((${#RATE_VALUES[@]} * ${#N0_VALUES[@]} * ${#SEEDS[@]})) runs"
echo "============================================================"

for seed in "${SEEDS[@]}"; do

    echo
    echo "############################################################"
    echo "SEED $seed"
    echo "############################################################"

    for rate in "${RATE_VALUES[@]}"; do

        echo
        echo "============================================================"
        echo "STARTING rate=$rate, seed=$seed"
        echo "N0 values in parallel: ${N0_VALUES[*]}"
        echo "============================================================"

        pids=()

        for n0 in "${N0_VALUES[@]}"; do

            python -u -m src_self_organization.train_neurogenesis_ablation \
                --hidden_neurons "$n0" \
                --seed "$seed" \
                --ablation rate \
                --value "$rate" \
                --output_dir "${OUTPUT_ROOT}/value_${rate}" \
                > >(sed -u "s/^/[rate=$rate N0=$n0 seed=$seed] /") 2>&1 &

            child_pid=$!
            pids+=("$child_pid")
            echo "$child_pid" >> "$CHILD_PID_FILE"

            echo "Started rate=$rate N0=$n0 seed=$seed PID=$child_pid"
        done

        failed=0

        for pid in "${pids[@]}"; do
            if ! wait "$pid"; then
                failed=1
            fi
        done

        if [[ "$failed" -ne 0 ]]; then
            echo "ERROR: at least one run failed for rate=$rate seed=$seed."
            exit 1
        fi

        : > "$CHILD_PID_FILE"

        echo "=== rate=$rate seed=$seed FINISHED ==="
    done

    echo
    echo "=== ALL RATES FOR SEED $seed FINISHED ==="
done

echo
echo "============================================================"
echo "ALL RATE x N0 GRID ABLATION RUNS FINISHED"
echo "============================================================"
