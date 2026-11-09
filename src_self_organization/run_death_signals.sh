#!/usr/bin/env bash
set -Eeuo pipefail
cd /Users/maksimkazanskii2/Desktop/interferenceNN
OUTPUT_ROOT="./results/death_selection"
mkdir -p "$OUTPUT_ROOT/logs"
for regime in original unrestricted_random shuffle_activity shuffle_flux shuffle_both; do
  echo "========== START $regime =========="
  pids=()
  for seed in 0 1 2 3 4; do
    (
      python -u -m src_self_organization.train_flowgenesis_death_signals \
        --dataset fashionmnist --hidden_neurons 20 --seed "$seed" \
        --ablation death_selection --value "$regime" \
        --output_dir "$OUTPUT_ROOT" 2>&1 \
      | awk -v prefix="[$regime seed=$seed] " '{print prefix $0; fflush()}' \
      | tee "$OUTPUT_ROOT/logs/${regime}_seed_${seed}.log"
    ) &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  if ((failed)); then echo "ERROR in $regime" >&2; exit 1; fi
done
echo "ALL 25 RUNS COMPLETED"
