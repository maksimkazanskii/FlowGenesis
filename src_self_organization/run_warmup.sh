#!/bin/bash

set -u

WARMUPS=(12)
SEEDS=( 4)

PID_FILE="/tmp/flowgenesis_warmup_ablation.pids"

# ------------------------------------------------------------
# STOP
# ------------------------------------------------------------

if [[ "${1:-}" == "stop" ]]; then
    echo "Stopping warmup ablation..."

    if [[ -f "$PID_FILE" ]]; then
        while read -r pid; do
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null || true
            fi
        done < "$PID_FILE"

        rm -f "$PID_FILE"
        echo "Stop signal sent to all recorded warmup processes."
    else
        echo "No recorded warmup processes found."
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
        echo "Warmup ablation already appears to be running."
        echo "Stop it with:"
        echo "  bash src_self_organization/run_warmup.sh stop"
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
echo "WARMUP ABLATION"
echo "Warmup values : ${WARMUPS[*]}"
echo "Seeds         : ${SEEDS[*]}"
echo "Output        : data/neurogenesis/warmup_ablations/"
echo "============================================================"


# Run all warmup values in parallel for one seed.
# Finish that seed before starting the next seed.
for seed in "${SEEDS[@]}"; do

    echo
    echo "============================================================"
    echo "STARTING SEED $seed"
    echo "============================================================"

    pids=()

    for warmup in "${WARMUPS[@]}"; do

        LOG="warmup_${warmup}_seed_${seed}.log"

        python -u -m src_self_organization.train_neurogenesis_ablation \
            --seed "$seed" \
            --ablation warmup \
            --value "$warmup" \
            --output_dir "warmup_ablations/value_${warmup}" \
            > >(sed -u "s/^/[warmup=${warmup} seed=${seed}] /" | tee "$LOG") \
            2>&1 &

        pid=$!
        pids+=("$pid")
        echo "$pid" >> "$PID_FILE"

        echo "Started warmup=$warmup seed=$seed PID=$pid"
    done

    # Wait for all four warmup values for this seed
    for pid in "${pids[@]}"; do
        wait "$pid"
    done

    echo "=== SEED $seed FINISHED ==="

done


rm -f "$PID_FILE"

echo
echo "============================================================"
echo "WARMUP ABLATION COMPLETE"
echo "============================================================"