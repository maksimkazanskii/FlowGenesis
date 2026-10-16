#!/usr/bin/env bash
set -o pipefail

for SEED in 0 1 2 3 4; do
  echo "======================================"
  echo "Starting seed=$SEED (N0=20,40,80,160)"
  echo "======================================"

  pids=()
  for N in 20 40 80 160; do
    echo "Starting N0=$N, seed=$SEED"
    (
      set -o pipefail
      python -u -m src_self_organization.train_neurogenesis \
        --dataset fashionmnist \
        --hidden_neurons "$N" \
        --seed "$SEED" \
        2>&1 | sed -u "s/^/[N0=$N seed=$SEED] /" \
        | tee "fashionmnist_N${N}_seed${SEED}.log"
    ) &
    pids+=("$!")
  done

  failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  if (( failed )); then
    echo "ERROR: At least one experiment failed for seed=$SEED; stopping." >&2
    exit 1
  fi
  echo "Completed seed=$SEED"
done

echo "All 20 experiments completed successfully"
