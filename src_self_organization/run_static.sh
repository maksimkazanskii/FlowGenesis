#!/usr/bin/env bash
set -Eeuo pipefail

SEEDS=(0 1 2 3 4)
N0=20
OUTPUT_ROOT="data/neurogenesis/static_ablations"
LOG_DIR="$OUTPUT_ROOT/logs"

mkdir -p "$LOG_DIR"

pids=()

for SEED in "${SEEDS[@]}"; do
  echo "Starting static N0=$N0, seed=$SEED"

  (
    python -u -m src_self_organization.train_neurogenesis_ablation \
      --dataset fashionmnist \
      --hidden_neurons "$N0" \
      --seed "$SEED" \
      --ablation regime \
      --value static \
      --output_dir "$OUTPUT_ROOT" \
      > "$LOG_DIR/static_N${N0}_seed${SEED}.log" 2>&1
  ) &

  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done

if (( failed )); then
  echo "ERROR: At least one experiment failed."
  exit 1
fi

echo "All 5 static experiments completed."