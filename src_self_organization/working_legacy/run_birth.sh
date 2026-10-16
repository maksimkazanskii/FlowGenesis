#!/bin/bash

set -u

# Birth-threshold ablation:
# 0.0625 -> 0.125 -> 0.25 (default) -> 0.50
BIRTH_THRESHOLDS=(0.0625 0.125 0.25 0.50)
SEEDS=(0 1 2 3 4)

PID_FILE="/tmp/flowgenesis_birth_threshold_ablation.pids"

# ------------------------------------------------------------
# STOP
# ------------------------------------------------------------

if [[ "${1:-}" == "stop" ]]; then
    echo "Stopping birth-threshold ablation..."

    if [[ -f "$PID_FILE" ]]; then
        while read -r pid; do
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null || true
            fi
        done < "$PID_FILE"

        rm -f "$PID_FILE"
        echo "Stop signal sent to all recorded birth-threshold processes."
    else
        echo "No recorded birth-threshold processes found."
    fi

    exit 0
fi

# ------------------------------------------------------------
# PREVENT ACCIDENTAL DUPLICATE LAUNCH
# ------------------------------------------------------------

if [[ -f "$PID_FILE" ]]; then
    alive=0

    while read -r pid; do
        if kill -0 "$pid" 2>/dev/null; then
            alive=1
            break
        fi
    done < "$PID_FILE"

    if [[ "$alive" -eq 1 ]]; then
        echo "Birth-threshold ablation already appears to be running."
        echo "Stop it with:"
        echo "  bash src_self_organization/run_birth_threshold.sh stop"
        exit 1
    else
        rm -f "$PID_FILE"
    fi
fi

# ------------------------------------------------------------
# START
# ------------------------------------------------------------

: > "$PID_FILE"

echo "============================================================"
echo "BIRTH-THRESHOLD ABLATION"
echo "Thresholds : ${BIRTH_THRESHOLDS[*]}"
echo "Seeds      : ${SEEDS[*]}"
echo "Output     : data/neurogenesis/birth_ablations/"
echo "============================================================"

# Run all threshold values in parallel for one seed.
# Finish that seed before starting the next seed.
for seed in "${SEEDS[@]}"; do

    echo
    echo "============================================================"
    echo "STARTING SEED $seed"
    echo "============================================================"

    pids=()

    for threshold in "${BIRTH_THRESHOLDS[@]}"; do

        LOG="birth_threshold_${threshold}_seed_${seed}.log"

        python -u -m src_self_organization.train_neurogenesis_ablation \
            --seed "$seed" \
            --ablation birth \
            --value "$threshold" \
            --output_dir "birth_ablations/value_${threshold}" \
            > >(sed -u "s/^/[birth=${threshold} seed=${seed}] /" | tee "$LOG") \
            2>&1 &

        pid=$!
        pids+=("$pid")
        echo "$pid" >> "$PID_FILE"

        echo "Started threshold=$threshold seed=$seed PID=$pid"
    done

    failed=0

    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            failed=1
        fi
    done

    if [[ "$failed" -ne 0 ]]; then
        echo "ERROR: at least one run failed for seed $seed."
        rm -f "$PID_FILE"
        exit 1
    fi

    echo "Finished seed $seed."
done

rm -f "$PID_FILE"

echo
echo "============================================================"
echo "BIRTH-THRESHOLD ABLATION COMPLETE"
echo "============================================================"
