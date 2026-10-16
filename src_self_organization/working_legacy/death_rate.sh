#!/bin/bash
set -u

# Death-threshold ablation.
# Other parameters remain at Config defaults.
# Config default death threshold is 0.01.
DEATH_THRESHOLDS=(0.0025 0.005 0.01 0.02 0.04)
SEEDS=(0 1 2 3 4)

PID_FILE="/tmp/flowgenesis_death_threshold_ablation.pids"

if [[ "${1:-}" == "stop" ]]; then
    echo "Stopping death-threshold ablation..."
    if [[ -f "$PID_FILE" ]]; then
        while read -r pid; do
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null || true
            fi
        done < "$PID_FILE"
        rm -f "$PID_FILE"
    fi
    exit 0
fi

if [[ -f "$PID_FILE" ]]; then
    alive=0
    while read -r pid; do
        if kill -0 "$pid" 2>/dev/null; then
            alive=1
            break
        fi
    done < "$PID_FILE"
    if [[ "$alive" -eq 1 ]]; then
        echo "Death-threshold ablation already appears to be running."
        echo "Stop it with: bash src_self_organization/run_death_threshold.sh stop"
        exit 1
    fi
    rm -f "$PID_FILE"
fi

: > "$PID_FILE"

echo "============================================================"
echo "DEATH-THRESHOLD ABLATION"
echo "Thresholds : ${DEATH_THRESHOLDS[*]}"
echo "Seeds      : ${SEEDS[*]}"
echo "Other parameters: Config defaults"
echo "============================================================"

for seed in "${SEEDS[@]}"; do
    echo "=== STARTING SEED $seed ==="
    pids=()

    for threshold in "${DEATH_THRESHOLDS[@]}"; do
        LOG="death_threshold_${threshold}_seed_${seed}.log"

        python -u -m src_self_organization.train_neurogenesis_ablation \
            --seed "$seed" \
            --ablation death \
            --value "$threshold" \
            --output_dir "death_ablations/value_${threshold}" \
            > >(sed -u "s/^/[death=${threshold} seed=${seed}] /" | tee "$LOG") \
            2>&1 &

        pid=$!
        pids+=("$pid")
        echo "$pid" >> "$PID_FILE"
    done

    failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then failed=1; fi
    done

    if [[ "$failed" -ne 0 ]]; then
        echo "ERROR: at least one run failed for seed $seed."
        rm -f "$PID_FILE"
        exit 1
    fi

    echo "=== SEED $seed FINISHED ==="
done

rm -f "$PID_FILE"
echo "============================================================"
echo "DEATH-THRESHOLD ABLATION COMPLETE"
echo "============================================================"

