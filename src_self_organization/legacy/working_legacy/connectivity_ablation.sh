#!/usr/bin/env bash

set -u

N0_VALUES=(10 20 40 80 160)
BETA_VALUES=(0.4)
SEEDS=(0 1 2 3 4)

ROOT="data/neurogenesis/beta_ablation"

mkdir -p "$ROOT"

echo "============================================================"
echo "CONNECTIVITY / BETA ABLATION"
echo "N0 values   : ${N0_VALUES[*]}"
echo "Beta values : ${BETA_VALUES[*]}"
echo "Seeds       : ${SEEDS[*]}"
echo "Total runs  : 100"
echo "Output      : $ROOT"
echo "============================================================"

for beta in "${BETA_VALUES[@]}"; do

    echo
    echo "============================================================"
    echo "BETA = $beta"
    echo "============================================================"

    for seed in "${SEEDS[@]}"; do

        echo
        echo "------------------------------------------------------------"
        echo "Starting beta=$beta seed=$seed"
        echo "------------------------------------------------------------"

        pids=()

        # Run all five N0 values in parallel
        for n0 in "${N0_VALUES[@]}"; do

            echo "[START] beta=$beta N0=$n0 seed=$seed"

            python -u -m src_self_organization.train_neurogenesis_ablation \
                --hidden_neurons "$n0" \
                --seed "$seed" \
                --ablation beta \
                --value "$beta" \
                --output_dir "${ROOT}/beta_${beta}" \
                > >(sed -u "s/^/[beta=$beta N0=$n0 seed=$seed] /") 2>&1 &

            pids+=("$!")
        done

        # Wait for all five N0 runs
        failed=0

        for pid in "${pids[@]}"; do
            if ! wait "$pid"; then
                failed=1
            fi
        done

        if [[ "$failed" -ne 0 ]]; then
            echo
            echo "ERROR: at least one run failed:"
            echo "beta=$beta seed=$seed"
            exit 1
        fi

        echo
        echo "[DONE] beta=$beta seed=$seed"

    done
done

echo
echo "============================================================"
echo "ALL CONNECTIVITY ABLATION RUNS FINISHED"
echo "Total runs: 100"
echo "============================================================"